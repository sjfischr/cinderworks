"""System readiness checks for Cinderworks Studio.

Mirrors BeatBunny worker/system_check.py pattern:
- check_cuda_status() → bool
- check_model_status() → dict[str, bool]
- is_ready_to_generate() → bool
- get_system_status_text() → str
- get_readiness_banner() → dict (Gradio update)

Key constraints:
- CUDA detection must NOT import torch at module level (lazy import only)
- File presence check uses size threshold (≥ 90% of expected = present)
- Partial/incomplete files are treated as not-present (R3.6)
- Readiness requires: CUDA + at least one diffusion checkpoint + text encoder + VAE
"""

from __future__ import annotations

from pathlib import Path

from studio.config import Config


# ---------------------------------------------------------------------------
# Module-level readiness state (persists across calls, set once on startup)
# ---------------------------------------------------------------------------

_readiness_state: dict[str, object] = {
    "cuda_available": None,  # None = not yet checked, bool after check
}


# ---------------------------------------------------------------------------
# Expected model files and their minimum sizes (90% of approximate expected)
# ---------------------------------------------------------------------------

# Model file definitions: (filename, min_bytes)
# Sizes are 90% of the approximate expected to allow slight variation
_MODEL_FILES: dict[str, tuple[str, int]] = {
    "diffusion_fp8": ("krea2_turbo_fp8_scaled.safetensors", int(13e9 * 0.9)),
    "diffusion_bf16": ("krea2_turbo_bf16.safetensors", int(25e9 * 0.9)),
    "text_encoder": ("qwen3vl_4b_fp8_scaled.safetensors", int(4e9 * 0.9)),
    "vae": ("qwen_image_vae.safetensors", int(0.5e9 * 0.9)),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_cuda_status() -> bool:
    """Detect CUDA availability without triggering model loads.

    Imports torch lazily so this module can be imported without
    CUDA initialization or heavy ML library side effects.
    Stores the result in module-level readiness state.
    """
    global _readiness_state

    try:
        import torch
        available = torch.cuda.is_available()
    except (ImportError, Exception):
        available = False

    _readiness_state["cuda_available"] = available
    return available


def check_model_status() -> dict[str, bool]:
    """Check presence and size validity of each model file.

    Returns a dict keyed by component name with bool indicating
    whether the file is present AND meets the minimum size threshold.
    A file that exists but is too small (partial download) is treated
    as not-present per R3.6.
    """
    status: dict[str, bool] = {}

    for component, (filename, min_size) in _MODEL_FILES.items():
        filepath = Config.MODEL_DIR / filename
        status[component] = _file_is_valid(filepath, min_size)

    return status


def is_ready_to_generate() -> bool:
    """True only when CUDA is available AND all required model files are present.

    Required files: at least ONE diffusion checkpoint (fp8 OR bf16) +
    text encoder + VAE. All must pass size validation.
    """
    # Use stored CUDA state if available, otherwise check now
    cuda_ok = _readiness_state.get("cuda_available")
    if cuda_ok is None:
        cuda_ok = check_cuda_status()

    if not cuda_ok:
        return False

    model_status = check_model_status()

    # Need at least one diffusion checkpoint
    has_diffusion = model_status["diffusion_fp8"] or model_status["diffusion_bf16"]
    has_encoder = model_status["text_encoder"]
    has_vae = model_status["vae"]

    return has_diffusion and has_encoder and has_vae


def get_system_status_text() -> str:
    """Plain-language summary of all readiness conditions.

    Lists every condition and its status. Returns a multi-line string
    suitable for display in a status panel.
    """
    lines: list[str] = []

    # CUDA status
    cuda_ok = _readiness_state.get("cuda_available")
    if cuda_ok is None:
        cuda_ok = check_cuda_status()

    if cuda_ok:
        lines.append("✅ CUDA GPU detected")
    else:
        lines.append("❌ No CUDA GPU detected")

    # Model file status
    model_status = check_model_status()

    # Diffusion checkpoint (need at least one)
    if model_status["diffusion_fp8"] or model_status["diffusion_bf16"]:
        if model_status["diffusion_fp8"] and model_status["diffusion_bf16"]:
            lines.append("✅ Krea 2 Turbo model ready (fp8 + bf16)")
        elif model_status["diffusion_fp8"]:
            lines.append("✅ Krea 2 Turbo model ready (fp8)")
        else:
            lines.append("✅ Krea 2 Turbo model ready (bf16)")
    else:
        lines.append("❌ Krea 2 Turbo model not downloaded yet")

    # Text encoder
    if model_status["text_encoder"]:
        lines.append("✅ Text encoder ready")
    else:
        lines.append("❌ Text encoder not downloaded yet")

    # VAE
    if model_status["vae"]:
        lines.append("✅ VAE ready")
    else:
        lines.append("❌ VAE not downloaded yet")

    return "\n".join(lines)


def get_readiness_banner() -> dict:
    """Return a Gradio-compatible update dict for the readiness banner.

    When system is ready: banner is hidden (visible=False).
    When not ready: banner is visible with all unmet conditions listed.

    Returns a dict with 'visible' and 'value' keys suitable for
    gr.update() or direct component state assignment.
    """
    unmet: list[str] = []

    # CUDA check
    cuda_ok = _readiness_state.get("cuda_available")
    if cuda_ok is None:
        cuda_ok = check_cuda_status()

    if not cuda_ok:
        unmet.append("No CUDA GPU detected")

    # Model files
    model_status = check_model_status()

    if not (model_status["diffusion_fp8"] or model_status["diffusion_bf16"]):
        unmet.append("Krea 2 Turbo model not downloaded yet")

    if not model_status["text_encoder"]:
        unmet.append("Text encoder not downloaded yet")

    if not model_status["vae"]:
        unmet.append("VAE not downloaded yet")

    if not unmet:
        return {"visible": False, "value": ""}

    banner_text = "⚠️ Not ready to generate:\n" + "\n".join(f"• {reason}" for reason in unmet)
    return {"visible": True, "value": banner_text}


def startup_cuda_check() -> None:
    """Run CUDA detection once on startup and store the result.

    Called during app initialization. Continues regardless of outcome
    so the UI remains accessible when CUDA is absent (R1.5).
    """
    check_cuda_status()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _file_is_valid(filepath: Path, min_size: int) -> bool:
    """Check that a file exists and meets the minimum size threshold.

    A file that exists but is smaller than min_size is treated as
    a partial/incomplete download and considered not-present.
    """
    if not filepath.is_file():
        return False

    actual_size = filepath.stat().st_size
    return actual_size >= min_size
