"""Krea 2 Turbo Backend — real inference via diffusers Krea2Pipeline.

Implements the full generation pipeline for Krea 2 Turbo using:
- Krea2Pipeline from diffusers (handles text encoding, sampling, VAE decode)
- Qwen3-VL-4B text encoder (12-layer hidden-state aggregation) — managed internally by pipeline
- Krea 2 Turbo DiT (Euler flow sampling, 8 steps, CFG disabled)
- Qwen-Image VAE — managed internally by pipeline

All GPU moves go through vram_manager — nothing calls .to('cuda') directly.
This module is importable without torch/CUDA (lazy imports inside functions).
"""

from __future__ import annotations

import logging
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Turbo Defaults (fixed for Phase 1)
# ---------------------------------------------------------------------------

TURBO_STEPS = 8
TURBO_CFG = 0.0  # CFG disabled for Turbo (Krea convention: 0.0 = no guidance)
TURBO_MU_SHIFT = 1.15  # Fixed mu/shift for Turbo (pinned by is_distilled=True)

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

# VRAM estimates (bytes) — used for pre-checks and tenant registration.
# The diffusers-format repo ships the Qwen3-VL-4B text encoder in bf16;
# at fp8_scaled precision we apply layerwise fp8 casting to it (same
# technique as the transformer), matching the ~5 GB footprint of the
# Comfy-Org fp8_scaled single file that the ComfyUI ecosystem treats as
# the standard way to run this encoder.
TEXT_ENCODER_VRAM_TIERS = {
    "bf16": 8_500_000_000,  # ~8.5 GB (bf16, diffusers format)
    "fp8_scaled": 4_800_000_000,  # ~4.8 GB (fp8 storage, norm/embed kept bf16)
}
VAE_BYTES = 500_000_000  # ~0.5 GB
DIT_VRAM_TIERS = {
    # bf16 transformer weights are 24.76 GiB (~26.6 GB decimal) — this
    # does NOT fit on a 24 GB card once the WDDM reserve and activations
    # are accounted for. The vram_manager will refuse it there, which is
    # the honest failure mode (the alternative is sysmem-fallback thrash).
    "bf16": 25_000_000_000,
    # fp8_scaled uses layerwise casting: weights stored float8_e4m3fn
    # (~12.4 GB), upcast to bf16 per-layer for compute. Norm/modulation
    # layers stay bf16 (diffusers skips them by default for quality).
    "fp8_scaled": 13_000_000_000,
}

# Activation/intermediate-buffer estimate for the transformer forward pass
# (attention + MLP intermediates, SDPA scratch, latent/hidden-state
# tensors). This is on top of the static weight footprint in
# DIT_VRAM_TIERS above and scales with resolution and batch_size — it's
# the part of VRAM usage that a fixed per-precision constant misses, and
# the reason a generation can still spill even when the weights alone
# fit. Baseline is a conservative estimate at 1024x1024/batch_size=1;
# not profiled against the real model, so it deliberately rounds up
# rather than down (an over-estimate causes an early, honest refusal;
# an under-estimate causes the silent WDDM spill this module exists to
# prevent).
_ACTIVATION_BASELINE_BYTES = 2_000_000_000  # ~2 GB at 1024x1024, batch_size=1
_ACTIVATION_REFERENCE_PIXELS = 1024 * 1024

# HuggingFace model ID for Krea 2 Turbo (diffusers format)
_HF_MODEL_ID = "krea/Krea-2-Turbo"

# Module-level pipeline cache (loaded once on first generate, reused after)
_pipeline_cache: dict[str, Any] = {}


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
    precision: str = "fp8_scaled"
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

    precision = params.get("precision", "fp8_scaled")
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


def estimate_activation_bytes(width: int, height: int, batch_size: int) -> int:
    """Estimate transformer activation/intermediate-buffer VRAM (bytes).

    Scales linearly with pixel count (sequence length grows with the
    latent's spatial size) and with batch_size (each item in a batch
    gets its own activation buffers, run simultaneously). See
    _ACTIVATION_BASELINE_BYTES for why this is a conservative estimate
    rather than a profiled figure.

    Args:
        width: Output image width in pixels.
        height: Output image height in pixels.
        batch_size: Number of images generated simultaneously.

    Returns:
        Estimated activation VRAM footprint in bytes.
    """
    pixel_ratio = (width * height) / _ACTIVATION_REFERENCE_PIXELS
    return int(_ACTIVATION_BASELINE_BYTES * pixel_ratio * batch_size)


# ---------------------------------------------------------------------------
# VRAM telemetry — the ground truth the estimates above are checked against
# ---------------------------------------------------------------------------


def _dump_thread_stacks(reason: str) -> None:
    """Write all-thread Python stack traces to a file and the log.

    The decisive diagnostic for a silent stall: the worker thread's
    stack names the exact diffusers/torch frame it is sitting in
    (a .to() copy, an attention kernel, a casting hook, a lock).
    """
    try:
        import faulthandler

        from studio.config import Config

        diag_dir = Config.OUTPUT_DIR / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        dump_path = diag_dir / f"stacks_{int(time.time())}.txt"
        with open(dump_path, "w", encoding="utf-8") as f:
            faulthandler.dump_traceback(file=f, all_threads=True)
        stacks = dump_path.read_text(encoding="utf-8")
        log.warning("%s — all-thread stack dump (%s):\n%s", reason, dump_path, stacks)
        _log_gpu_activity()
    except Exception:
        log.exception("Thread stack dump failed")


def _log_gpu_activity() -> None:
    """Log instantaneous GPU utilization via nvidia-smi.

    Distinguishes the two stall modes a Python stack cannot: kernels
    executing glacially (GPU util high — WDDM paging / slow path) vs a
    true wedge (GPU util ~0% — deadlocked sync). Best-effort; missing
    nvidia-smi is logged and ignored.
    """
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,"
                "memory.used,memory.total,power.draw,pstate,temperature.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            log.warning(
                "GPU activity [util.gpu, util.mem, mem.used, mem.total, "
                "power, pstate, temp]: %s",
                result.stdout.strip(),
            )
        else:
            log.info("nvidia-smi query failed: %s", result.stderr.strip())
    except Exception as exc:
        log.info("nvidia-smi unavailable: %s", exc)


def _log_vram_snapshot(tag: str) -> None:
    """Log a point-in-time CUDA memory snapshot under a tag.

    Reports torch's allocated/peak/reserved alongside the driver's
    free/total (mem_get_info). On Windows/WDDM the driver numbers are
    dedicated VRAM only: if driver-free approaches zero while torch
    keeps allocating, the overflow is going into shared system memory
    over PCIe — the sysmem-spill tar pit. That condition is warned
    explicitly because nothing else in the stack reports it.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.memory_reserved()
        log.info(
            "VRAM[%s]: torch allocated %.2f GB (peak %.2f GB), reserved "
            "%.2f GB | driver free %.2f GB of %.2f GB dedicated",
            tag,
            allocated / 1e9,
            peak / 1e9,
            reserved / 1e9,
            free / 1e9,
            total / 1e9,
        )
        if free < 500_000_000:
            log.warning(
                "VRAM[%s]: dedicated VRAM nearly exhausted (%.2f GB free) — "
                "any further allocation spills into WDDM shared system "
                "memory and generation speed collapses",
                tag,
                free / 1e9,
            )
    except Exception:
        log.debug("VRAM snapshot for '%s' failed", tag, exc_info=True)


def _log_component_memory(pipe: Any) -> None:
    """Log per-component parameter memory, dtype mix, and device placement.

    This is the direct check on whether layerwise fp8 casting actually
    took effect: a cast component shows most bytes as float8_e4m3fn with
    a small bf16 remainder (skipped norm/embed modules). If a component
    that should be cast shows all-bf16, the casting silently failed and
    every VRAM estimate derived from it is wrong.
    """
    try:
        for name in ("transformer", "text_encoder", "vae"):
            component = getattr(pipe, name, None)
            if component is None and name == "text_encoder":
                # Parked encoders are detached from the pipeline
                component = getattr(pipe, _PARKED_TE_ATTR, None)
            if component is None or not hasattr(component, "parameters"):
                continue
            bytes_by_dtype: dict[str, int] = {}
            devices: set[str] = set()
            total_bytes = 0
            for p in component.parameters():
                nbytes = p.numel() * p.element_size()
                key = str(p.dtype).removeprefix("torch.")
                bytes_by_dtype[key] = bytes_by_dtype.get(key, 0) + nbytes
                devices.add(str(p.device))
                total_bytes += nbytes
            dtype_str = ", ".join(
                f"{dt}: {b / 1e9:.2f} GB"
                for dt, b in sorted(bytes_by_dtype.items(), key=lambda kv: -kv[1])
            )
            log.info(
                "Component '%s': %.2f GB total on %s (%s)",
                name,
                total_bytes / 1e9,
                "/".join(sorted(devices)),
                dtype_str,
            )
    except Exception:
        log.debug("Component memory breakdown failed", exc_info=True)


# ---------------------------------------------------------------------------
# Attention dispatch patch — GQA + mask vs torch fused kernels
# ---------------------------------------------------------------------------


def _patch_krea2_gqa_attention() -> None:
    """Expand KV heads to match Q heads in Krea2's attention dispatch.

    Krea2 uses grouped-query attention: 48 query heads sharing 12 KV
    heads, plus a text attention mask. On torch 2.7 no fused SDPA
    backend accepts that combination — flash rejects arbitrary masks,
    mem-efficient rejects GQA — so stock SDPA silently falls back to
    MATH, which materializes the fp32 attention matrix (48 heads x
    4608^2 x 4 bytes ~= 4 GB per call, 28 blocks per step). Observed in
    the field: dedicated VRAM filled to the ceiling, WDDM demoted weight
    pages to shared system memory, and sampling crawled indefinitely.

    The remedy is the one torch's own rejection message suggests:
    repeat_interleave KV from 12 to 48 heads before the kernel (~57 MB
    at 1024x1024 — noise) so the mask-compatible mem-efficient kernel
    engages. repeat_interleave matches GQA semantics (each KV head
    serves a consecutive group of query heads).

    Patches the dispatch_attention_fn binding inside the Krea2
    transformer module only — no other model's attention is touched.
    Idempotent. Layout note: the Krea2 processor hands (B, S, H, D)
    tensors to dispatch, so heads live on dim 2.
    """
    from diffusers.models.transformers import transformer_krea2 as _tk

    if getattr(_tk, "_cinderworks_gqa_patched", False):
        return

    _orig_dispatch = _tk.dispatch_attention_fn

    def _gqa_expanding_dispatch(query, key, value, *args, **kwargs):
        if (
            query.ndim == 4
            and key.ndim == 4
            and query.shape[2] != key.shape[2]
            and key.shape[2] != 0
            and query.shape[2] % key.shape[2] == 0
        ):
            repeat = query.shape[2] // key.shape[2]
            key = key.repeat_interleave(repeat, dim=2)
            value = value.repeat_interleave(repeat, dim=2)
            # Heads now match — GQA handling in the kernel is moot
            kwargs["enable_gqa"] = False
        return _orig_dispatch(query, key, value, *args, **kwargs)

    _tk.dispatch_attention_fn = _gqa_expanding_dispatch
    _tk._cinderworks_gqa_patched = True
    log.info(
        "Patched Krea2 attention dispatch: KV heads expanded to match "
        "query heads (GQA + mask is unsupported by torch fused kernels)"
    )


# ---------------------------------------------------------------------------
# Pipeline Loading (lazy, cached)
# ---------------------------------------------------------------------------


def _get_pipeline(precision: str = "fp8_scaled", full_gpu_resident: bool = False) -> Any:
    """Get or load the Krea2Pipeline at the requested precision (lazy, cached).

    On first call, loads from a local directory if available, otherwise
    falls back to downloading from HuggingFace. Subsequent calls with the
    same precision and residency mode return the cached pipeline
    immediately. The cache is keyed by (precision, mode) so switching
    bf16 <-> fp8_scaled, or crossing the full-GPU-fit threshold, reloads
    correctly.

    Local path priority:
    1. Config.MODEL_DIR / "krea2-turbo-diffusers" (pre-downloaded via `hf download`)
    2. HuggingFace hub auto-download from "krea/Krea-2-Turbo"

    Precision handling:
    - bf16: weights kept in bfloat16 (24.76 GiB transformer — does not fit
      a 24 GB card; the vram_manager pre-flight refuses it there).
    - fp8_scaled: layerwise casting on the transformer AND text encoder —
      weights stored as float8_e4m3fn (transformer ~12.4 GB, encoder
      ~4.8 GB) and upcast to bf16 per layer during compute. Norm layers
      (and the encoder's embedding table) stay bf16 for quality. Works
      directly from the bf16 weights on disk; no separate download.

    Residency mode (full_gpu_resident):
    - True: caller has already checked the *combined* footprint (text
      encoder + transformer + VAE) fits the usable VRAM budget. The whole
      pipeline is moved to the GPU once via `pipe.to("cuda")` and stays
      there for the duration of generation — no per-component CPU<->PCIe
      shuttling.
    - False: the combined footprint doesn't fit, so we fall back to
      `enable_model_cpu_offload()`, which keeps at most one component
      GPU-resident at a time (encode -> offload encoder -> load
      transformer -> sample -> offload -> load VAE -> decode) via
      accelerate hooks. This is slower (PCIe transfer per component, per
      call) but is the only way to run a working set that doesn't fit
      resident all at once.

    Returns:
        A loaded Krea2Pipeline instance ready for inference.

    Raises:
        RuntimeError: If the pipeline cannot be loaded.
    """
    mode = "gpu" if full_gpu_resident else "offload"
    cache_key = f"pipe:{precision}:{mode}"
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]

    # Evict any other cached pipeline so we don't hold two full weight
    # sets in system RAM (different precision or residency mode).
    for stale_key in [k for k in _pipeline_cache if k.startswith("pipe:")]:
        log.info("Evicting cached pipeline '%s' (precision/mode change)", stale_key)
        del _pipeline_cache[stale_key]
    import gc

    gc.collect()

    import torch
    from diffusers import Krea2Pipeline
    from studio.config import Config

    # Check for local diffusers-format weights first
    local_path = Config.MODEL_DIR / "krea2-turbo-diffusers"
    if local_path.is_dir() and (local_path / "model_index.json").is_file():
        log.info("Loading Krea2Pipeline from local path: %s", local_path)
        source = str(local_path)
    else:
        log.info(
            "Loading Krea2Pipeline from '%s' (first use — this may download ~36 GB)",
            _HF_MODEL_ID,
        )
        source = _HF_MODEL_ID

    try:
        log.info(
            "Environment: torch %s | cuda %s | %s (sm_%d%d) | diffusers %s",
            torch.__version__,
            torch.version.cuda,
            torch.cuda.get_device_name(0),
            *torch.cuda.get_device_capability(0),
            __import__("diffusers").__version__,
        )
        log.info(
            "SDPA backends enabled: flash=%s, mem_efficient=%s, math=%s, cudnn=%s",
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
            getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: "n/a")(),
        )
    except Exception:
        pass

    _patch_krea2_gqa_attention()

    pipe = Krea2Pipeline.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
    )

    if precision == "fp8_scaled":
        log.info(
            "Applying layerwise fp8 casting to transformer "
            "(storage=float8_e4m3fn, compute=bfloat16)"
        )
        pipe.transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )

        # Also fp8-cast the Qwen3-VL-4B text encoder (~8.5 GB bf16 ->
        # ~4.8 GB). Running this encoder at fp8 is the ComfyUI ecosystem
        # standard (qwen3vl_4b_fp8_scaled.safetensors is THE file every
        # workflow ships). The encoder is a transformers Qwen3VLModel,
        # not a diffusers ModelMixin, so use the generic hooks utility
        # rather than enable_layerwise_casting. Norm and embedding
        # modules stay bf16: norms are quality-critical, and the
        # embedding table is a lookup whose fp8 savings aren't worth the
        # precision loss at the very start of the conditioning path.
        log.info(
            "Applying layerwise fp8 casting to text encoder "
            "(storage=float8_e4m3fn, compute=bfloat16, norm/embed kept bf16)"
        )
        from diffusers.hooks import apply_layerwise_casting

        apply_layerwise_casting(
            pipe.text_encoder,
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
            skip_modules_pattern=("norm", "embed"),
        )

    # VAE slicing: decode one image at a time so a batch doesn't
    # multiply the decode-stage activation peak by batch_size. No effect
    # on single-image batches; guarded because not every VAE class
    # implements slicing.
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
        log.info("VAE slicing enabled (per-image decode)")
    else:
        log.warning(
            "VAE %s does not support slicing — batch decode will peak "
            "at batch_size x single-image activation cost",
            type(pipe.vae).__name__,
        )

    if full_gpu_resident:
        log.info(
            "Working set fits usable VRAM budget — loading transformer "
            "and VAE GPU-resident; text encoder parks in system RAM "
            "between encodes"
        )
        pipe.to("cuda")
        # The text encoder only runs once per generation (we pre-encode
        # and pass prompt_embeds to the pipeline), so it does NOT need
        # to occupy VRAM during the long sampling phase. Telemetry showed
        # that keeping it resident pushed dedicated VRAM to ~85%
        # occupancy before sampling even started — on WDDM the driver
        # responds to that pressure by silently migrating pages to
        # shared system memory.
        #
        # Parking it on CPU is not enough: DiffusionPipeline resolves
        # its execution device from an UNORDERED set of component
        # modules, and if the parked encoder wins that lottery the whole
        # pipeline "executes" on CPU (observed in the field — the device
        # guard fired and undid the parking). So the encoder is fully
        # DETACHED from the pipeline while parked and held under a
        # private attribute; the text_encoder tenant reattaches it just
        # for the encode. With only CUDA-resident modules attached,
        # device resolution is deterministic.
        _park_text_encoder(pipe)
    else:
        log.info(
            "Working set exceeds usable VRAM budget — using "
            "model_cpu_offload (components swap CPU<->GPU per call)"
        )
        pipe.enable_model_cpu_offload()

    log.info("Krea2Pipeline loaded (precision=%s, mode=%s)", precision, mode)
    _log_component_memory(pipe)
    _log_vram_snapshot("after pipeline load")
    _pipeline_cache[cache_key] = pipe
    return pipe


def _clear_pipeline_cache() -> None:
    """Clear the pipeline cache. For testing only."""
    _pipeline_cache.clear()


# Attribute under which the detached text encoder is held while parked.
# Underscore-prefixed so DiffusionPipeline's __setattr__ config plumbing
# and components discovery never see it as a pipeline module.
_PARKED_TE_ATTR = "_cinderworks_parked_text_encoder"


def _park_text_encoder(pipe: Any) -> None:
    """Move the text encoder to CPU and detach it from the pipeline.

    Detaching (pipe.text_encoder = None) is what makes the pipeline's
    device inference ignore the parked encoder — see the comment at the
    call site in _get_pipeline. No-op if already parked.
    """
    te = getattr(pipe, "text_encoder", None)
    if te is None:
        return
    te.to("cpu")
    pipe.text_encoder = None
    setattr(pipe, _PARKED_TE_ATTR, te)
    import torch

    # Return freed blocks to the driver — WDDM pressure counts
    # reserved-but-unused pages too.
    torch.cuda.empty_cache()
    log.info("Text encoder parked in system RAM (detached from pipeline)")


def _unpark_text_encoder(pipe: Any) -> None:
    """Reattach the parked text encoder and move it to the GPU.

    No-op if the encoder is already attached (offload mode, where the
    accelerate hooks own placement and parking never happens).
    """
    if getattr(pipe, "text_encoder", None) is not None:
        return
    te = getattr(pipe, _PARKED_TE_ATTR, None)
    if te is None:
        return
    te.to("cuda")
    pipe.text_encoder = te
    setattr(pipe, _PARKED_TE_ATTR, None)
    log.info("Text encoder reattached and moved to GPU for encode")


# ---------------------------------------------------------------------------
# Main Generation Entry Point
# ---------------------------------------------------------------------------


def generate(params: dict[str, Any]) -> Generator[str | dict, None, None]:
    """Generate images using Krea 2 Turbo.

    This is the main entry point called by the registry. It uses
    Krea2Pipeline from diffusers for the full inference sequence:
    text encoding → sampling → VAE decoding.

    The function is a generator that yields:
    - Progress strings (encoding, sampling steps, decoding)
    - A final dict with images and resolved params

    When the full working set (text encoder + transformer + VAE) fits the
    usable VRAM budget, the pipeline loads fully GPU-resident via
    `pipe.to("cuda")` — no offload traffic. Otherwise GPU moves go through
    the pipeline's cpu_offload mechanism (which internally handles the
    encode→offload→sample→decode sequence). Either way, vram_manager
    handles top-level tenant coordination.

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
    # The app-wide singleton: budget detected once per process, tenant
    # state persists across generations. A fresh manager per generation
    # would re-measure the budget with our own resident cache on the
    # card and mistake it for foreign usage.
    from studio.core.vram_manager import get_vram_manager

    vram_mgr: VRAMManager = params.get("_vram_manager") or get_vram_manager()

    # --- 4. Check if we're in test/mock mode ---
    # If no CUDA is available or _mock_inference is set, use the lightweight
    # stub path so tests work without GPU/model weights.
    use_real_inference = params.get("_real_inference", None)
    if use_real_inference is None:
        # Auto-detect: use real inference only if torch+CUDA are available
        try:
            import torch
            use_real_inference = torch.cuda.is_available()
        except ImportError:
            use_real_inference = False

    if use_real_inference:
        yield from _generate_real(gen_params, base_seed, vram_mgr, params)
    else:
        yield from _generate_stub(gen_params, base_seed, vram_mgr, params)


# ---------------------------------------------------------------------------
# Real Inference Path (GPU + Krea2Pipeline)
# ---------------------------------------------------------------------------


def _generate_real(
    gen_params: GenerationParams,
    base_seed: int,
    vram_mgr: Any,
    raw_params: dict[str, Any],
) -> Generator[str | dict, None, None]:
    """Real inference path using Krea2Pipeline.

    Physical flow (resident mode): the transformer+VAE live on the GPU
    across generations; the text encoder parks in system RAM. Each
    generation moves the encoder up, encodes the prompt ONCE, moves it
    back down, then samples every batch from the pre-computed
    prompt_embeds. This keeps the long sampling phase at minimum VRAM
    pressure — on WDDM, sustained high occupancy makes the driver
    silently migrate pages to shared system memory even without an OOM.

    In offload mode the accelerate hooks own component placement; the
    pre-encoded embeds still mean the text encoder is onloaded once per
    generation instead of once per batch.
    """
    import torch
    from studio.core.vram_manager import Tenant

    # Reset the peak-allocation counter so VRAM snapshots in this run
    # report this generation's true peak, not a stale one.
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    _log_vram_snapshot("before pipeline load")

    dit_bytes = DIT_VRAM_TIERS.get(gen_params.precision, DIT_VRAM_TIERS["fp8_scaled"])
    text_encoder_bytes = TEXT_ENCODER_VRAM_TIERS.get(
        gen_params.precision, TEXT_ENCODER_VRAM_TIERS["fp8_scaled"]
    )
    activation_bytes = estimate_activation_bytes(
        gen_params.width, gen_params.height, gen_params.batch_size
    )
    dit_peak_bytes = dit_bytes + activation_bytes

    # Peak VRAM moments in resident mode. The text encoder runs once per
    # generation (we pre-encode and hand prompt_embeds to the pipeline),
    # so it never coexists with sampling activations — but during the
    # brief encode it DOES coexist with the resident transformer+VAE.
    _ENCODE_OVERHEAD = 500_000_000  # encoder forward activations, ~0.5 GB
    encode_peak_bytes = dit_bytes + VAE_BYTES + text_encoder_bytes + _ENCODE_OVERHEAD
    sample_peak_bytes = dit_bytes + VAE_BYTES + activation_bytes
    resident_peak_bytes = max(encode_peak_bytes, sample_peak_bytes)

    # --- Pre-flight VRAM check BEFORE any weights move -----------------
    # The irreducible need (even under model_cpu_offload) is the
    # transformer weights at the chosen precision PLUS the activation
    # buffers for the sampling forward pass — the latter scales with
    # resolution and batch_size (R6.4's per-image-footprint × batch_size
    # requirement), which a flat weight-only estimate misses. If that
    # won't fit usable VRAM, refuse now with a plain-language message
    # (R6.4/R7.5) rather than letting the Windows driver spill weights
    # into shared system memory and thrash the copy engine for an hour.
    if not vram_mgr.can_fit(dit_peak_bytes):
        from studio.core.vram_manager import VRAMError

        raise VRAMError(
            f"Not enough VRAM for {gen_params.precision} precision at "
            f"{gen_params.width}x{gen_params.height}, batch size "
            f"{gen_params.batch_size} — needs about "
            f"{dit_bytes / 1e9:.0f} GB for the model plus "
            f"{activation_bytes / 1e9:.1f} GB for sampling buffers. "
            f"Try a smaller batch size, lower resolution, or fp8_scaled "
            f"precision."
        )

    # Keep the transformer+VAE resident when the worst peak moment fits.
    # The text encoder is shuttled to GPU only for the encode, so the
    # budget question is max(encode peak, sampling peak), not the sum of
    # everything at once. Fall back to enable_model_cpu_offload's
    # sequential swap only when even that doesn't fit.
    full_gpu_resident = vram_mgr.can_fit(resident_peak_bytes)
    log.info(
        "VRAM plan: encode peak %.1f GB, sampling peak %.1f GB, "
        "resident=%s",
        encode_peak_bytes / 1e9,
        sample_peak_bytes / 1e9,
        full_gpu_resident,
    )

    pipeline_ref: dict[str, Any] = {"pipe": None}

    def _load_text_encoder() -> None:
        # First acquire also loads the pipeline itself (lazy).
        pipeline_ref["pipe"] = _get_pipeline(gen_params.precision, full_gpu_resident)
        if full_gpu_resident:
            _unpark_text_encoder(pipeline_ref["pipe"])
        # In offload mode the accelerate hooks own placement — nothing to do.

    def _unload_text_encoder() -> None:
        if full_gpu_resident and pipeline_ref["pipe"] is not None:
            _park_text_encoder(pipeline_ref["pipe"])

    text_encoder_tenant = Tenant(
        name="text_encoder",
        estimated_bytes=text_encoder_bytes,
        load_fn=_load_text_encoder,
        unload_fn=_unload_text_encoder,
    )
    # The transformer+VAE stay physically resident across generations in
    # resident mode (that's the point of the cache); the tenant exists so
    # the app-wide ledger reflects who owns the GPU during sampling.
    dit_tenant = Tenant(
        name="dit",
        estimated_bytes=dit_peak_bytes,
        load_fn=lambda: None,
        unload_fn=lambda: None,
    )

    # --- Encode ONCE per generation ------------------------------------
    # The prompt is identical for every batch; only seeds differ. Encoding
    # up front means the text encoder does exactly one GPU round-trip per
    # generation instead of one per batch, and the pipeline call skips
    # its internal text-encoder forward entirely when given prompt_embeds.
    vram_mgr.acquire(text_encoder_tenant)

    yield "Loading pipeline..."
    pipe = _get_pipeline(gen_params.precision, full_gpu_resident)

    yield "Encoding prompt..."
    encode_device = torch.device("cuda") if full_gpu_resident else None
    with torch.inference_mode():
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            prompt=gen_params.prompt,
            device=encode_device,
            num_images_per_prompt=1,  # __call__ duplicates to batch_size
        )
        negative_embeds = None
        negative_embeds_mask = None
        if gen_params.cfg and gen_params.cfg > 1.0:
            negative_embeds, negative_embeds_mask = pipe.encode_prompt(
                prompt="",
                device=encode_device,
                num_images_per_prompt=1,
            )
    _log_vram_snapshot("after encode")

    vram_mgr.release(text_encoder_tenant)
    vram_mgr.acquire(dit_tenant)

    if full_gpu_resident:
        # With the encoder detached, every module still attached to the
        # pipeline is CUDA-resident, so device resolution must be cuda.
        # If it isn't, something is structurally wrong — refuse rather
        # than silently sample on CPU (a multi-hour hang, the dishonest
        # failure mode).
        exec_device = pipe._execution_device
        log.info("Pipeline execution device for sampling: %s", exec_device)
        if exec_device.type != "cuda":
            raise RuntimeError(
                f"Pipeline resolved to execution device '{exec_device}' "
                f"even with the text encoder detached — refusing to "
                f"sample on CPU. This indicates a diffusers device-"
                f"resolution change; please report this log."
            )

    # --- Generation loop ---
    all_images: list[Path] = []
    all_seeds: list[int] = []
    image_index = 0

    from studio.config import Config

    # Sentinel for the progress queue
    _DONE = object()

    for batch_num in range(gen_params.batch_count):
        batch_base_seed = base_seed + image_index

        # Build per-image generators for deterministic seeds
        generators = [
            torch.Generator("cuda").manual_seed(batch_base_seed + i)
            for i in range(gen_params.batch_size)
        ]

        # --- Live progress: run the pipeline in a worker thread and ---
        # --- stream step updates through a queue (same pattern as   ---
        # --- the downloader). The Gradio generator drains the queue ---
        # --- so the UI shows real forward movement per step (R5.8). ---
        progress_q: queue.Queue = queue.Queue()
        outcome: dict[str, Any] = {}
        step_clock = {"last": time.perf_counter()}

        def _step_callback(
            pipe_self: Any, step: int, timestep: Any, callback_kwargs: dict
        ) -> dict:
            # Per-step wall time is the single most useful thrash
            # indicator: resident fp8 sampling on this class of card
            # should be well under a second per step; tens of seconds
            # means weights are streaming over PCIe (WDDM spill).
            now = time.perf_counter()
            step_seconds = now - step_clock["last"]
            step_clock["last"] = now
            log.info(
                "Sampling step %d/%d took %.2fs",
                step + 1,
                gen_params.steps,
                step_seconds,
            )
            progress_q.put(
                f"Sampling step {step + 1}/{gen_params.steps} "
                f"(batch {batch_num + 1}/{gen_params.batch_count}) "
                f"— {step_seconds:.1f}s/step"
            )
            return callback_kwargs

        def _worker() -> None:
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                step_clock["last"] = time.perf_counter()
                log.info(
                    "Worker entering pipeline call (batch %d/%d)",
                    batch_num + 1,
                    gen_params.batch_count,
                )
                # prompt_embeds instead of prompt: the pipeline skips its
                # internal text-encoder forward (the encoder is parked on
                # CPU by now) and duplicates the embeds to batch_size via
                # num_images_per_prompt.
                #
                # The sdpa_kernel context BANS the math attention backend
                # for the whole sampling+decode call. Math attention
                # materializes the full fp32 attention matrix (multi-GB at
                # ~4600 tokens x 24 heads, 28 blocks per step) — observed
                # in the field filling dedicated VRAM to 23.7/24.5 GiB and
                # forcing WDDM to demote weight pages to shared system
                # memory (GPU pegged at "100%" but ~116 W / 1% memory
                # throughput). Flash/mem-efficient compute the same result
                # in ~100 MB tiles. If neither can handle the inputs,
                # torch raises an error naming the exact constraint —
                # the honest failure instead of an hour-long crawl.
                with sdpa_kernel(
                    [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
                ):
                    outcome["result"] = pipe(
                        prompt_embeds=prompt_embeds,
                        prompt_embeds_mask=prompt_embeds_mask,
                        negative_prompt_embeds=negative_embeds,
                        negative_prompt_embeds_mask=negative_embeds_mask,
                        height=gen_params.height,
                        width=gen_params.width,
                        num_inference_steps=gen_params.steps,
                        guidance_scale=gen_params.cfg,
                        num_images_per_prompt=gen_params.batch_size,
                        generator=generators if len(generators) > 1 else generators[0],
                        callback_on_step_end=_step_callback,
                    )
                _log_vram_snapshot(f"after sampling batch {batch_num + 1}")
            except Exception as exc:  # noqa: BLE001 — re-raised on main thread
                outcome["error"] = exc
            finally:
                progress_q.put(_DONE)

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        # Watchdog: turbo steps on a resident fp8 transformer should be
        # sub-second. If NOTHING arrives for 30s, the worker is stalled
        # inside a torch/diffusers call — dump every thread's stack so
        # the log names the exact frame instead of us guessing.
        _WATCHDOG_TIMEOUT_S = 30.0
        _MAX_STACK_DUMPS = 3
        stall_dumps = 0
        stalled_seconds = 0.0
        while True:
            try:
                msg = progress_q.get(timeout=_WATCHDOG_TIMEOUT_S)
            except queue.Empty:
                if not worker.is_alive():
                    # Worker died without posting _DONE (shouldn't happen
                    # — the finally block posts it — but never spin).
                    log.error("Sampling worker died without completing")
                    break
                stalled_seconds += _WATCHDOG_TIMEOUT_S
                if stall_dumps < _MAX_STACK_DUMPS:
                    stall_dumps += 1
                    _dump_thread_stacks(
                        f"No sampling progress for {stalled_seconds:.0f}s "
                        f"(batch {batch_num + 1})"
                    )
                yield (
                    f"⚠️ Sampling stalled — {stalled_seconds:.0f}s without "
                    f"completing a step (diagnostic stack dump in log)"
                )
                continue
            stalled_seconds = 0.0
            if msg is _DONE:
                break
            yield msg

        worker.join()
        if "error" in outcome:
            # Surface on the main thread so the handler error boundary
            # catches it and maps it to a plain-language message.
            raise outcome["error"]

        result = outcome["result"]

        yield f"Decoding... (batch {batch_num + 1}/{gen_params.batch_count})"

        # Save images
        for i, img in enumerate(result.images):
            img_seed = base_seed + image_index
            output_path = Config.OUTPUT_DIR / f"krea2_{base_seed}" / f"{image_index:04d}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
            log.info("Saved image to %s", output_path)
            all_images.append(output_path)
            all_seeds.append(img_seed)
            image_index += 1

    _log_vram_snapshot("after decode/save")

    vram_mgr.release(dit_tenant)

    # --- Final result ---
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


# ---------------------------------------------------------------------------
# Stub Inference Path (no GPU — for testing and development)
# ---------------------------------------------------------------------------


def _generate_stub(
    gen_params: GenerationParams,
    base_seed: int,
    vram_mgr: Any,
    raw_params: dict[str, Any],
) -> Generator[str | dict, None, None]:
    """Stub inference path for testing without GPU/model weights.

    Follows the same acquire/release sequence as real inference to
    maintain test compatibility (encode→offload→sample→decode order).
    """
    from studio.core.vram_manager import Tenant

    all_images: list[Path] = []
    all_seeds: list[int] = []
    image_index = 0

    for batch_num in range(gen_params.batch_count):
        # --- Text encoding (stub) ---
        yield f"Encoding prompt... (batch {batch_num + 1}/{gen_params.batch_count})"

        encoder_model = None

        def _load_encoder() -> None:
            nonlocal encoder_model
            encoder_model = {"type": "text_encoder", "precision": gen_params.precision}

        def _unload_encoder() -> None:
            nonlocal encoder_model
            encoder_model = None

        text_encoder_tenant = Tenant(
            name="text_encoder",
            estimated_bytes=TEXT_ENCODER_VRAM_TIERS.get(
                gen_params.precision, TEXT_ENCODER_VRAM_TIERS["fp8_scaled"]
            ),
            load_fn=_load_encoder,
            unload_fn=_unload_encoder,
        )

        vram_mgr.acquire(text_encoder_tenant)
        # Simulate encoding
        embeddings = {"type": "text_embedding", "prompt": gen_params.prompt, "layers": ENCODER_LAYERS}
        vram_mgr.release(text_encoder_tenant)

        # --- Diffusion sampling (stub) ---
        dit_model = None

        def _load_dit_model() -> None:
            nonlocal dit_model
            dit_model = {"type": "dit", "precision": gen_params.precision}

        def _unload_dit_model() -> None:
            nonlocal dit_model
            dit_model = None

        dit_tenant = Tenant(
            name="dit",
            estimated_bytes=DIT_VRAM_TIERS.get(gen_params.precision, DIT_VRAM_TIERS["fp8_scaled"]),
            load_fn=_load_dit_model,
            unload_fn=_unload_dit_model,
        )

        batch_base_seed = base_seed + image_index

        vram_mgr.acquire(dit_tenant)

        # Simulate sampling steps
        for step in range(gen_params.steps):
            yield f"Sampling step {step + 1}/{gen_params.steps}"

        vram_mgr.release(dit_tenant)

        # --- VAE decoding (stub) ---
        yield f"Decoding... (batch {batch_num + 1}/{gen_params.batch_count})"

        # --- Save images (stub — create placeholder paths) ---
        from studio.config import Config

        for i in range(gen_params.batch_size):
            img_seed = base_seed + image_index
            output_path = Config.OUTPUT_DIR / f"krea2_{base_seed}" / f"{image_index:04d}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            all_images.append(output_path)
            all_seeds.append(img_seed)
            image_index += 1

    # --- Final result ---
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
