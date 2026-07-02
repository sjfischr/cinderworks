"""Image upscaling — built-in Lanczos plus optional model-based enlarge.

Two tiers, mirroring how Forge/ComfyUI treat "enlarge":

1. Lanczos — a resampling filter built into PIL. No model, no VRAM,
   instant. Good for modest enlargements; softens fine detail.
2. Real-ESRGAN 4x — an ESRGAN-family super-resolution model loaded via
   `spandrel` (the same loader Forge and ComfyUI use). Requires a one-time
   ~67 MB download (see downloader.UPSCALER_FILES) and the `spandrel`
   package. Sharp, detail-preserving 4x.

Model inference is tiled (512 px tiles with overlap) so arbitrarily large
inputs never spike VRAM, and GPU residency is coordinated through the
shared VRAMManager like every other GPU consumer.

This module is importable without torch/spandrel (lazy imports).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Method display names (UI-facing)
METHOD_LANCZOS = "Lanczos (built-in, fast)"
METHOD_REALESRGAN = "Real-ESRGAN 4x (model, sharpest)"

# The upscaler model file — downloaded flat into Config.MODEL_DIR by the
# downloader (see downloader.UPSCALER_FILES for the HF source).
UPSCALER_MODEL_FILE = "RealESRGAN_x4.pth"

# VRAM estimate for the upscaler tenant: the RRDBNet model itself is
# ~67 MB; tiled inference bounds activations to a few hundred MB. 1.5 GB
# is a comfortable ceiling that coexists with the resident transformer.
UPSCALER_VRAM_BYTES = 1_500_000_000

# Tiling parameters — bound VRAM regardless of input size
_TILE_SIZE = 512
_TILE_OVERLAP = 16

# Loaded model cache (one upscaler at a time)
_model_cache: dict[str, Any] = {}

SCALE_MIN, SCALE_MAX = 1.0, 4.0


def list_methods() -> list[str]:
    """Return available upscale method names (Lanczos always first)."""
    return [METHOD_LANCZOS, METHOD_REALESRGAN]


def model_file_path() -> Path:
    """Where the Real-ESRGAN weights live once downloaded."""
    from studio.config import Config

    return Config.MODEL_DIR / UPSCALER_MODEL_FILE


def model_available() -> bool:
    """True if the Real-ESRGAN weights are on disk."""
    return model_file_path().is_file()


def upscale(image_path: str | Path, method: str, scale: float) -> Path:
    """Upscale an image and save the result next to the outputs.

    Args:
        image_path: Source image file.
        method: One of list_methods().
        scale: Target scale factor (1.0–4.0). Model-based upscaling runs
            at the model's native factor and is Lanczos-resized to the
            exact requested scale afterwards.

    Returns:
        Path to the upscaled image.

    Raises:
        ValueError: Unknown method, bad scale, or missing input.
        RuntimeError: Model requested but weights/spandrel missing.
    """
    from PIL import Image

    src = Path(image_path)
    if not src.is_file():
        raise ValueError(f"Image not found: {src}")
    if not (SCALE_MIN <= scale <= SCALE_MAX):
        raise ValueError(
            f"Scale must be between {SCALE_MIN} and {SCALE_MAX}, got {scale}"
        )

    image = Image.open(src).convert("RGB")

    if method == METHOD_LANCZOS:
        result = _upscale_lanczos(image, scale)
    elif method == METHOD_REALESRGAN:
        result = _upscale_realesrgan(image, scale)
    else:
        raise ValueError(f"Unknown upscale method: '{method}'")

    from studio.config import Config

    out_dir = Config.OUTPUT_DIR / "upscaled"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}_x{scale:g}_{int(time.time())}.png"
    result.save(out_path)
    log.info(
        "Upscaled %s -> %s (%s, x%g, %dx%d)",
        src.name,
        out_path.name,
        method,
        scale,
        *result.size,
    )
    return out_path


def _upscale_lanczos(image: Any, scale: float) -> Any:
    """Plain Lanczos resample — no model, no GPU."""
    from PIL import Image

    new_size = (round(image.width * scale), round(image.height * scale))
    return image.resize(new_size, Image.LANCZOS)


def _upscale_realesrgan(image: Any, scale: float) -> Any:
    """Model-based upscale via spandrel, tiled, VRAM-tenant coordinated."""
    model_path = model_file_path()
    if not model_path.is_file():
        raise RuntimeError(
            "The Real-ESRGAN upscaler model is not downloaded yet — "
            "go to the Models tab and click 'Download Upscaler', or "
            "switch to the Lanczos method."
        )
    try:
        import spandrel  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The 'spandrel' package is required for model-based "
            "upscaling. Run: pip install spandrel"
        ) from exc

    import torch

    from studio.core.vram_manager import Tenant, get_vram_manager

    model = _get_upscaler_model(model_path)
    vram_mgr = get_vram_manager()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load() -> None:
        model.to(device)

    def _unload() -> None:
        model.to("cpu")
        if device == "cuda":
            torch.cuda.empty_cache()

    tenant = Tenant(
        name="upscaler",
        estimated_bytes=UPSCALER_VRAM_BYTES,
        load_fn=_load,
        unload_fn=_unload,
    )

    vram_mgr.acquire(tenant)
    try:
        upscaled = _run_tiled(model.model, image, model.scale, device)
    finally:
        vram_mgr.release(tenant)

    # The model runs at its native factor (4x); trim to the exact
    # requested scale with Lanczos if they differ.
    if abs(model.scale - scale) > 1e-6:
        target = (round(image.width * scale), round(image.height * scale))
        from PIL import Image

        upscaled = upscaled.resize(target, Image.LANCZOS)
    return upscaled


def _get_upscaler_model(model_path: Path) -> Any:
    """Load (and cache) the spandrel model descriptor from disk."""
    key = str(model_path)
    if key in _model_cache:
        return _model_cache[key]

    from spandrel import ModelLoader

    log.info("Loading upscaler model: %s", model_path.name)
    descriptor = ModelLoader().load_from_file(key)
    descriptor.model.eval()
    _model_cache[key] = descriptor
    log.info(
        "Upscaler loaded: %s (native scale x%d)",
        descriptor.architecture.name,
        descriptor.scale,
    )
    return descriptor


def _run_tiled(model: Any, image: Any, model_scale: int, device: str) -> Any:
    """Run the upscaler over the image in overlapping tiles.

    Bounds peak VRAM to one tile's activations regardless of input size.
    Overlap regions are cropped on stitch so tile seams never show.
    """
    import numpy as np
    import torch
    from PIL import Image

    src = np.asarray(image).astype(np.float32) / 255.0  # HWC, 0-1
    h, w = src.shape[:2]
    out = np.zeros((h * model_scale, w * model_scale, 3), dtype=np.float32)

    step = _TILE_SIZE - 2 * _TILE_OVERLAP
    with torch.inference_mode():
        for top in range(0, h, step):
            for left in range(0, w, step):
                # Tile bounds with overlap, clamped to the image
                t0 = max(0, top - _TILE_OVERLAP)
                l0 = max(0, left - _TILE_OVERLAP)
                t1 = min(h, top + step + _TILE_OVERLAP)
                l1 = min(w, left + step + _TILE_OVERLAP)

                tile = src[t0:t1, l0:l1]
                tensor = (
                    torch.from_numpy(tile)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )
                result = model(tensor)[0].permute(1, 2, 0).clamp(0, 1)
                result_np = result.float().cpu().numpy()

                # Crop the overlap back off (except at image edges)
                crop_t = (top - t0) * model_scale
                crop_l = (left - l0) * model_scale
                inner_h = min(step, h - top) * model_scale
                inner_w = min(step, w - left) * model_scale
                out[
                    top * model_scale : top * model_scale + inner_h,
                    left * model_scale : left * model_scale + inner_w,
                ] = result_np[
                    crop_t : crop_t + inner_h,
                    crop_l : crop_l + inner_w,
                ]

    return Image.fromarray((out * 255.0).round().astype(np.uint8))


def _clear_model_cache() -> None:
    """Clear the upscaler model cache. For testing only."""
    _model_cache.clear()
