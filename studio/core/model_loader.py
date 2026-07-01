"""Lazy model loader — loads model components only on first generate.

Design invariants:
- Importing this module does NOT touch CUDA or trigger weight loading.
- Torch and model-loading libraries are imported lazily inside functions.
- Loaded components are cached keyed by (model_id, precision).
- First generate triggers exactly one load; subsequent calls reuse the cache.
- Actual GPU placement is delegated to vram_manager — this module never
  calls .to('cuda') directly.

Phase 1: works with the krea2 backend. The interface is generic (model_id,
precision) so additional backends slot in unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# Cache: (model_id, precision) → loaded model components dict
_cache: dict[tuple[str, str], dict[str, Any]] = {}


def get_or_load(model_id: str, precision: str) -> dict[str, Any]:
    """Get cached model components, or load them on first call.

    This is the single entry point for obtaining loaded model components.
    On first call for a given (model_id, precision) pair, it lazily imports
    torch / safetensors, loads the weights, and delegates GPU placement to
    vram_manager. Subsequent calls return the cached components immediately.

    Args:
        model_id: The model identifier (e.g. 'krea2-turbo').
        precision: The precision variant (e.g. 'bf16', 'fp8_scaled').

    Returns:
        A dict of loaded model components. The exact keys depend on the
        model backend (e.g. 'dit', 'text_encoder', 'vae' for krea2).

    Raises:
        KeyError: If model_id is not found in the registry.
        RuntimeError: If loading fails for any reason.
    """
    key = (model_id, precision)

    if key in _cache:
        log.debug("Cache hit for %s", key)
        return _cache[key]

    log.info("Loading model components for %s (first use)", key)
    components = _load_components(model_id, precision)
    _cache[key] = components
    log.info("Model components cached for %s", key)
    return components


def is_loaded(model_id: str, precision: str) -> bool:
    """Check whether model components are already cached.

    Args:
        model_id: The model identifier.
        precision: The precision variant.

    Returns:
        True if components for (model_id, precision) are in the cache.
    """
    return (model_id, precision) in _cache


def clear_cache() -> None:
    """Clear all cached model components. For testing only."""
    _cache.clear()
    log.debug("Model loader cache cleared")


def _load_components(model_id: str, precision: str) -> dict[str, Any]:
    """Load model components for the given model and precision.

    This function:
    1. Resolves model metadata from the registry.
    2. Lazily imports torch (no top-level CUDA touch).
    3. Loads checkpoint/weight files (Phase 1: placeholder — real weight
       loading comes with the krea2 backend implementation).
    4. Delegates GPU placement to vram_manager (components stay on CPU
       until vram_manager.acquire() is called by the backend).

    Args:
        model_id: The model identifier.
        precision: The precision variant.

    Returns:
        Dict of loaded model components.

    Raises:
        KeyError: If model_id is not found in the registry.
        RuntimeError: If loading fails.
    """
    # Lazy imports — these must NOT be at module level
    from studio.models.registry import get_meta
    from studio.core.vram_manager import VRAMManager

    entry = get_meta(model_id)  # raises KeyError if unknown

    log.info(
        "Resolved model '%s' (%s) — checkpoints: %s, precision: %s",
        entry.display_name,
        model_id,
        entry.checkpoints,
        precision,
    )

    # Select the correct checkpoint file for the requested precision
    checkpoint_file = _select_checkpoint(entry.checkpoints, precision)

    # Build the components dict.
    # Phase 1: With placeholder inference the "loading" doesn't load real
    # weights, but the structure, lazy-import guards, and caching are real.
    # The backend (krea2.py) will use vram_manager to place these on GPU
    # at generation time.
    components: dict[str, Any] = {
        "model_id": model_id,
        "precision": precision,
        "checkpoint_file": checkpoint_file,
        "vae_file": entry.vae,
        "text_encoder_file": entry.text_encoder,
        "sampler_defaults": dict(entry.sampler_defaults),
        "vram_tiers": dict(entry.vram_tiers),
        "loaded": True,
    }

    log.info(
        "Components prepared for '%s' @ %s (checkpoint: %s)",
        model_id,
        precision,
        checkpoint_file,
    )

    return components


def _select_checkpoint(checkpoints: list[str], precision: str) -> str:
    """Select the checkpoint file matching the requested precision.

    Convention: checkpoint filenames contain the precision string
    (e.g. 'krea2_turbo_fp8_scaled.safetensors' for fp8_scaled).

    Args:
        checkpoints: List of available checkpoint filenames.
        precision: The requested precision.

    Returns:
        The matching checkpoint filename.

    Raises:
        RuntimeError: If no checkpoint matches the requested precision.
    """
    for cp in checkpoints:
        if precision in cp:
            return cp

    raise RuntimeError(
        f"No checkpoint found for precision '{precision}' "
        f"in available checkpoints: {checkpoints}"
    )
