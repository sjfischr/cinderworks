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
            "Combined working set fits usable VRAM budget — loading "
            "pipeline fully GPU-resident (no offload)"
        )
        pipe.to("cuda")
    else:
        log.info(
            "Combined working set exceeds usable VRAM budget — using "
            "model_cpu_offload (components swap CPU<->GPU per call)"
        )
        pipe.enable_model_cpu_offload()

    log.info("Krea2Pipeline loaded (precision=%s, mode=%s)", precision, mode)
    _pipeline_cache[cache_key] = pipe
    return pipe


def _clear_pipeline_cache() -> None:
    """Clear the pipeline cache. For testing only."""
    _pipeline_cache.clear()


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
    vram_mgr: VRAMManager = params.get("_vram_manager") or VRAMManager()

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

    Loads the pipeline on first use, then generates images using the
    diffusers Krea2Pipeline with proper progress callbacks.
    """
    import torch
    from studio.core.vram_manager import Tenant

    dit_bytes = DIT_VRAM_TIERS.get(gen_params.precision, DIT_VRAM_TIERS["fp8_scaled"])
    text_encoder_bytes = TEXT_ENCODER_VRAM_TIERS.get(
        gen_params.precision, TEXT_ENCODER_VRAM_TIERS["fp8_scaled"]
    )
    activation_bytes = estimate_activation_bytes(
        gen_params.width, gen_params.height, gen_params.batch_size
    )
    dit_peak_bytes = dit_bytes + activation_bytes
    combined_bytes = text_encoder_bytes + dit_peak_bytes + VAE_BYTES

    # --- Pre-flight VRAM check BEFORE any weights move -----------------
    # The peak GPU-resident load under model_cpu_offload is the
    # transformer weights at the chosen precision PLUS the activation
    # buffers for the sampling forward pass — the latter scales with
    # resolution and batch_size (R6.4's per-image-footprint × batch_size
    # requirement), which a flat weight-only estimate misses. If the
    # combined peak won't fit usable VRAM, refuse now with a
    # plain-language message (R6.4/R7.5) rather than letting the Windows
    # driver spill weights into shared system memory and thrash the copy
    # engine for an hour.
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

    # Does the WHOLE working set (text encoder + transformer + VAE +
    # activation buffers) fit resident together? If so there's no reason
    # to pay per-component CPU<->GPU offload traffic on every
    # generation — load once with pipe.to("cuda") and leave it resident.
    # Only fall back to enable_model_cpu_offload's sequential swap when
    # the combined footprint genuinely doesn't fit (e.g. a smaller card,
    # or a large batch/resolution eating the headroom).
    full_gpu_resident = vram_mgr.can_fit(combined_bytes)

    pipeline_ref: dict[str, Any] = {"pipe": None}

    def _load_pipeline() -> None:
        pipeline_ref["pipe"] = _get_pipeline(gen_params.precision, full_gpu_resident)

    def _unload_pipeline() -> None:
        # Pipeline stays cached in system RAM/GPU between calls; nothing
        # to move here (offload hooks, or the resident .to("cuda") state,
        # are left as-is for reuse on the next generation).
        pass

    if full_gpu_resident:
        # Single tenant sized to the combined footprint — the whole
        # pipeline stays GPU-resident for the duration of generation, so
        # there's no text_encoder -> dit handoff to account for.
        pipeline_tenant = Tenant(
            name="pipeline",
            estimated_bytes=combined_bytes,
            load_fn=_load_pipeline,
            unload_fn=_unload_pipeline,
        )
        vram_mgr.acquire(pipeline_tenant)

        yield "Loading pipeline..."
        pipe = _get_pipeline(gen_params.precision, full_gpu_resident)
    else:
        # Working set doesn't fit resident together — mirror the physical
        # encode->offload->sample order that the accelerate cpu_offload
        # hooks enforce inside the pipeline, keeping the app-wide VRAM
        # ledger accurate for Phase 3/4 tenants.
        text_encoder_tenant = Tenant(
            name="text_encoder",
            estimated_bytes=text_encoder_bytes,
            load_fn=_load_pipeline,
            unload_fn=_unload_pipeline,
        )
        dit_tenant = Tenant(
            name="dit",
            estimated_bytes=dit_peak_bytes,
            load_fn=lambda: None,
            unload_fn=lambda: None,
        )

        vram_mgr.acquire(text_encoder_tenant)

        yield "Loading pipeline..."
        pipe = _get_pipeline(gen_params.precision, full_gpu_resident)

        vram_mgr.release(text_encoder_tenant)
        vram_mgr.acquire(dit_tenant)

    # --- Generation loop ---
    all_images: list[Path] = []
    all_seeds: list[int] = []
    image_index = 0

    from studio.config import Config

    # Sentinel for the progress queue
    _DONE = object()

    for batch_num in range(gen_params.batch_count):
        yield f"Encoding prompt... (batch {batch_num + 1}/{gen_params.batch_count})"

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

        def _step_callback(
            pipe_self: Any, step: int, timestep: Any, callback_kwargs: dict
        ) -> dict:
            progress_q.put(
                f"Sampling step {step + 1}/{gen_params.steps} "
                f"(batch {batch_num + 1}/{gen_params.batch_count})"
            )
            return callback_kwargs

        def _worker() -> None:
            try:
                outcome["result"] = pipe(
                    prompt=gen_params.prompt,
                    height=gen_params.height,
                    width=gen_params.width,
                    num_inference_steps=gen_params.steps,
                    guidance_scale=gen_params.cfg,
                    num_images_per_prompt=gen_params.batch_size,
                    generator=generators if len(generators) > 1 else generators[0],
                    callback_on_step_end=_step_callback,
                )
            except Exception as exc:  # noqa: BLE001 — re-raised on main thread
                outcome["error"] = exc
            finally:
                progress_q.put(_DONE)

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        while True:
            msg = progress_q.get()
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

    # Release whichever tenant is resident
    if full_gpu_resident:
        vram_mgr.release(pipeline_tenant)
    else:
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
