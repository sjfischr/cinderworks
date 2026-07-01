"""Krea 2 Turbo Backend — load→encode→offload→load→sample→decode.

Implements the full generation pipeline for Krea 2 Turbo using:
- Qwen3-VL-4B text encoder (12-layer hidden-state aggregation)
- Krea 2 Turbo DiT (Euler flow sampling)
- Qwen-Image VAE (tiled decode option)

All GPU moves go through vram_manager — nothing calls .to('cuda') directly.
This module is importable without torch/CUDA (lazy imports inside functions).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Turbo Defaults (fixed for Phase 1)
# ---------------------------------------------------------------------------

TURBO_STEPS = 8
TURBO_CFG = 1.0  # CFG disabled for Turbo
TURBO_MU_SHIFT = 1.15  # Fixed mu/shift for Turbo

# Parameter bounds
STEPS_MIN, STEPS_MAX = 1, 100
SEED_MIN, SEED_MAX = 0, 2**32 - 1
WIDTH_MIN, WIDTH_MAX = 512, 2048
HEIGHT_MIN, HEIGHT_MAX = 512, 2048
SIZE_MULTIPLE = 64
BATCH_SIZE_MIN, BATCH_SIZE_MAX = 1, 16
BATCH_COUNT_MIN, BATCH_COUNT_MAX = 1, 100

# Hidden-state layers used for text encoding (12 selected layers)
ENCODER_LAYERS = 12

# VRAM estimates (bytes)
TEXT_ENCODER_BYTES = 4_000_000_000  # ~4 GB (fp8_scaled)
VAE_BYTES = 500_000_000  # ~0.5 GB
DIT_VRAM_TIERS = {
    "bf16": 25_000_000_000,  # ~25 GB
    "fp8_scaled": 13_000_000_000,  # ~13 GB
}


# ---------------------------------------------------------------------------
# Generation Parameters (dataclass for internal param handling)
# ---------------------------------------------------------------------------


@dataclass
class GenerationParams:
    """Validated and resolved generation parameters.

    All defaults match Turbo_Defaults from the design doc.
    """

    prompt: str
    steps: int = TURBO_STEPS
    cfg: float = TURBO_CFG
    mu_shift: float = TURBO_MU_SHIFT
    width: int = 1024
    height: int = 1024
    precision: str = "bf16"
    batch_size: int = 1
    batch_count: int = 1
    seed: int | None = None  # None = generate random

    @property
    def total_images(self) -> int:
        """Total images produced = batch_size × batch_count."""
        return self.batch_size * self.batch_count


# ---------------------------------------------------------------------------
# Parameter Validation
# ---------------------------------------------------------------------------


def validate_params(params: dict[str, Any]) -> GenerationParams:
    """Validate and resolve generation parameters from a raw dict.

    Applies Turbo_Defaults for omitted parameters. Rejects invalid values
    with plain-language error messages.

    Args:
        params: Raw parameter dict from the UI/registry.

    Returns:
        A validated GenerationParams instance.

    Raises:
        ValueError: If any parameter is out of bounds or invalid.
    """
    prompt = params.get("prompt", "")
    if not prompt or not prompt.strip():
        raise ValueError("A prompt is required — please enter a text description of the image you want to generate.")

    steps = params.get("steps", TURBO_STEPS)
    if not isinstance(steps, int) or steps < STEPS_MIN or steps > STEPS_MAX:
        raise ValueError(
            f"Steps must be an integer between {STEPS_MIN} and {STEPS_MAX}, got {steps}"
        )

    seed = params.get("seed", None)
    if seed is not None:
        if not isinstance(seed, int) or seed < SEED_MIN or seed > SEED_MAX:
            raise ValueError(
                f"Seed must be an integer between {SEED_MIN} and {SEED_MAX}, got {seed}"
            )

    width = params.get("width", 1024)
    if not isinstance(width, int) or width < WIDTH_MIN or width > WIDTH_MAX:
        raise ValueError(
            f"Width must be between {WIDTH_MIN} and {WIDTH_MAX}, got {width}"
        )
    if width % SIZE_MULTIPLE != 0:
        raise ValueError(
            f"Width must be a multiple of {SIZE_MULTIPLE}, got {width}"
        )

    height = params.get("height", 1024)
    if not isinstance(height, int) or height < HEIGHT_MIN or height > HEIGHT_MAX:
        raise ValueError(
            f"Height must be between {HEIGHT_MIN} and {HEIGHT_MAX}, got {height}"
        )
    if height % SIZE_MULTIPLE != 0:
        raise ValueError(
            f"Height must be a multiple of {SIZE_MULTIPLE}, got {height}"
        )

    precision = params.get("precision", "bf16")
    if precision not in ("bf16", "fp8_scaled"):
        raise ValueError(
            f"Precision must be 'bf16' or 'fp8_scaled', got '{precision}'"
        )

    batch_size = params.get("batch_size", 1)
    if not isinstance(batch_size, int) or batch_size < BATCH_SIZE_MIN or batch_size > BATCH_SIZE_MAX:
        raise ValueError(
            f"Batch size must be between {BATCH_SIZE_MIN} and {BATCH_SIZE_MAX}, got {batch_size}"
        )

    batch_count = params.get("batch_count", 1)
    if not isinstance(batch_count, int) or batch_count < BATCH_COUNT_MIN or batch_count > BATCH_COUNT_MAX:
        raise ValueError(
            f"Batch count must be between {BATCH_COUNT_MIN} and {BATCH_COUNT_MAX}, got {batch_count}"
        )

    cfg = params.get("cfg", TURBO_CFG)
    mu_shift = params.get("mu_shift", TURBO_MU_SHIFT)

    return GenerationParams(
        prompt=prompt.strip(),
        steps=steps,
        cfg=cfg,
        mu_shift=mu_shift,
        width=width,
        height=height,
        precision=precision,
        batch_size=batch_size,
        batch_count=batch_count,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Seed Resolution
# ---------------------------------------------------------------------------


def resolve_seed(seed: int | None) -> int:
    """Resolve the base seed for generation.

    If a seed is provided, use it directly. Otherwise generate a random
    seed within the valid range.

    Args:
        seed: Explicit seed or None for random.

    Returns:
        The resolved seed value.
    """
    if seed is not None:
        return seed
    return random.randint(SEED_MIN, SEED_MAX)


# ---------------------------------------------------------------------------
# Inference Pipeline (placeholder implementations for real weights)
# ---------------------------------------------------------------------------


def _load_text_encoder(precision: str) -> Any:
    """Load Qwen3-VL-4B text encoder.

    In production, this loads the model from disk using transformers.
    Currently a placeholder that returns a sentinel object.
    """
    log.info("Loading Qwen3-VL-4B text encoder (precision: %s)", precision)
    # Placeholder: real implementation loads from safetensors
    return {"type": "text_encoder", "model": "qwen3vl_4b", "precision": precision}


def _encode_prompt(encoder: Any, prompt: str) -> Any:
    """Encode prompt using baked template + 12-layer hidden-state aggregation.

    In production:
    1. Wraps prompt in Krea's baked system/user template
    2. Tokenizes and encodes through Qwen3-VL
    3. Extracts hidden states from the 12 selected layers
    4. Aggregates them into the conditioning tensor

    Currently returns a placeholder embedding dict.
    """
    log.info("Encoding prompt with %d-layer aggregation", ENCODER_LAYERS)
    # Placeholder: real implementation does multi-layer hidden-state aggregation
    return {
        "type": "text_embedding",
        "prompt": prompt,
        "layers": ENCODER_LAYERS,
    }


def _load_dit(precision: str) -> Any:
    """Load Krea 2 Turbo DiT at the specified precision.

    In production, loads the diffusion transformer from safetensors.
    Currently a placeholder.
    """
    log.info("Loading Krea 2 Turbo DiT (precision: %s)", precision)
    return {"type": "dit", "model": "krea2_turbo", "precision": precision}


def _sample(
    dit: Any,
    embeddings: Any,
    *,
    steps: int,
    cfg: float,
    mu_shift: float,
    width: int,
    height: int,
    seed: int,
    batch_size: int,
    step_callback: Any = None,
) -> Any:
    """Euler flow sampling with Turbo parameters.

    In production:
    - Euler integration of the flow-matching ODE
    - 8 steps (Turbo), CFG 1.0 (disabled), fixed mu/shift 1.15
    - Resolution-aware latent initialization

    Currently returns a placeholder latent tensor representation.
    """
    log.info(
        "Sampling: %d steps, CFG %.1f, mu_shift %.2f, %dx%d, seed %d, batch %d",
        steps, cfg, mu_shift, width, height, seed, batch_size,
    )
    # Simulate step-by-step sampling with callbacks
    for step in range(steps):
        if step_callback is not None:
            step_callback(step + 1, steps)

    # Placeholder latents
    return {
        "type": "latents",
        "shape": (batch_size, 16, height // 8, width // 8),
        "seed": seed,
    }


def _load_vae() -> Any:
    """Load Qwen-Image VAE for latent decoding.

    In production, loads AutoencoderKLQwenImage from safetensors.
    Currently a placeholder.
    """
    log.info("Loading Qwen-Image VAE")
    return {"type": "vae", "model": "qwen_image_vae"}


def _decode_latents(vae: Any, latents: Any, *, tiled: bool = True) -> list[Any]:
    """Decode latents with Qwen-Image VAE.

    In production:
    - Decodes latent tensor to pixel space
    - Supports tiled decode for VRAM headroom
    - Returns list of PIL images

    Currently returns placeholder image objects.
    """
    batch_size = latents["shape"][0] if isinstance(latents, dict) else 1
    log.info("Decoding %d latents (tiled=%s)", batch_size, tiled)
    # Placeholder: return mock image objects
    return [
        {"type": "image", "index": i, "seed": latents.get("seed", 0) + i}
        for i in range(batch_size)
    ]


def _save_image(image: Any, output_path: Path) -> Path:
    """Save a decoded image to disk.

    In production, saves a PIL image as PNG.
    Currently creates a placeholder file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Placeholder: in real implementation this saves a PIL Image
    # For now just create the parent directory structure
    log.info("Saving image to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main Generation Entry Point
# ---------------------------------------------------------------------------


def generate(params: dict[str, Any]) -> Generator[str | dict, None, None]:
    """Generate images using Krea 2 Turbo.

    This is the main entry point called by the registry. It implements the
    full load→encode→offload→load→sample→decode sequence.

    The function is a generator that yields:
    - Progress strings (encoding, sampling steps, decoding)
    - A final dict with images and resolved params

    All GPU moves go through vram_manager — never .to('cuda') directly.

    Args:
        params: Raw generation parameters dict.

    Yields:
        str: Progress update messages.
        dict: Final result with images and resolved params (last yield).

    Raises:
        ValueError: If parameters are invalid (empty prompt, out of bounds).
    """
    # Lazy imports — only when actually generating
    from studio.core.vram_manager import Tenant, VRAMManager

    # --- 1. Validate parameters ---
    gen_params = validate_params(params)

    # --- 2. Resolve seed ---
    base_seed = resolve_seed(gen_params.seed)
    log.info("Resolved base seed: %d", base_seed)

    # --- 3. Get vram_manager instance ---
    # In production, this would be a singleton. For now, create or get from params.
    vram_mgr: VRAMManager = params.get("_vram_manager") or VRAMManager()

    # --- 4. Generation loop (batch_count sequential batches) ---
    all_images: list[Path] = []
    all_seeds: list[int] = []
    image_index = 0

    for batch_num in range(gen_params.batch_count):
        # --- 4a. Text encoding ---
        yield f"Encoding prompt... (batch {batch_num + 1}/{gen_params.batch_count})"

        # Create text encoder tenant
        encoder_model = None

        def _load_encoder() -> None:
            nonlocal encoder_model
            encoder_model = _load_text_encoder(gen_params.precision)

        def _unload_encoder() -> None:
            nonlocal encoder_model
            encoder_model = None

        text_encoder_tenant = Tenant(
            name="text_encoder",
            estimated_bytes=TEXT_ENCODER_BYTES,
            load_fn=_load_encoder,
            unload_fn=_unload_encoder,
        )

        # Acquire text encoder → load → encode → release
        vram_mgr.acquire(text_encoder_tenant)
        embeddings = _encode_prompt(encoder_model, gen_params.prompt)
        vram_mgr.release(text_encoder_tenant)

        # --- 4b. Diffusion sampling ---
        dit_model = None

        def _load_dit_model() -> None:
            nonlocal dit_model
            dit_model = _load_dit(gen_params.precision)

        def _unload_dit_model() -> None:
            nonlocal dit_model
            dit_model = None

        dit_tenant = Tenant(
            name="dit",
            estimated_bytes=DIT_VRAM_TIERS.get(gen_params.precision, DIT_VRAM_TIERS["bf16"]),
            load_fn=_load_dit_model,
            unload_fn=_unload_dit_model,
        )

        # Compute per-batch seed: image i uses base_seed + image_index
        batch_base_seed = base_seed + image_index

        # Acquire DiT → sample
        vram_mgr.acquire(dit_tenant)

        # Sampling with step-by-step progress
        step_messages: list[str] = []

        def _on_step(current: int, total: int) -> None:
            step_messages.append(f"Sampling step {current}/{total}")

        latents = _sample(
            dit_model,
            embeddings,
            steps=gen_params.steps,
            cfg=gen_params.cfg,
            mu_shift=gen_params.mu_shift,
            width=gen_params.width,
            height=gen_params.height,
            seed=batch_base_seed,
            batch_size=gen_params.batch_size,
            step_callback=_on_step,
        )

        vram_mgr.release(dit_tenant)

        # Yield sampling progress (after release so we don't hold GPU during yield)
        for msg in step_messages:
            yield msg

        # --- 4c. VAE decoding ---
        yield f"Decoding... (batch {batch_num + 1}/{gen_params.batch_count})"

        # VAE is small enough to share with CPU — no tenant needed in Phase 1
        # (kept in CPU, tiled decode reduces peak VRAM)
        vae = _load_vae()
        images = _decode_latents(vae, latents, tiled=True)

        # --- 4d. Save images ---
        from studio.config import Config

        for i, img in enumerate(images):
            img_seed = base_seed + image_index
            output_path = Config.OUTPUT_DIR / f"krea2_{base_seed}" / f"{image_index:04d}.png"
            saved_path = _save_image(img, output_path)
            all_images.append(saved_path)
            all_seeds.append(img_seed)
            image_index += 1

    # --- 5. Final result ---
    yield {
        "images": all_images,
        "seeds": all_seeds,
        "base_seed": base_seed,
        "params": {
            "prompt": gen_params.prompt,
            "steps": gen_params.steps,
            "cfg": gen_params.cfg,
            "mu_shift": gen_params.mu_shift,
            "width": gen_params.width,
            "height": gen_params.height,
            "precision": gen_params.precision,
            "batch_size": gen_params.batch_size,
            "batch_count": gen_params.batch_count,
            "seed": base_seed,
        },
        "total_images": gen_params.total_images,
    }
