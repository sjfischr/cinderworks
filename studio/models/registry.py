"""Model-agnostic registry — the single routing layer for all model access.

The shell never imports a backend module directly. All model access
(metadata, load, generate) goes through this registry. Backends are
imported lazily on first access and guarded: a failing backend records
its reason and stays marked unavailable without re-attempting.

Phase 1: exactly one entry — Krea 2 Turbo. No stubs.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Generator

log = logging.getLogger(__name__)


@dataclass
class RegistryEntry:
    """Metadata and configuration for a registered model backend.

    Designed so a second backend can be added by providing a new entry
    and backend module, with no changes to the UI shell.
    """

    model_id: str  # 'krea2-turbo'
    display_name: str
    backend_module: str  # 'studio.models.backends.krea2'
    checkpoints: list[str] = field(default_factory=list)  # filenames
    vae: str = ""
    text_encoder: str = ""
    sampler_defaults: dict[str, Any] = field(default_factory=dict)  # steps, cfg, mu_shift
    precision_options: list[str] = field(default_factory=list)  # ['bf16', 'fp8_scaled']
    vram_tiers: dict[str, int] = field(default_factory=dict)  # precision → estimated bytes


# ---------------------------------------------------------------------------
# Registry data — Phase 1: one entry only
# ---------------------------------------------------------------------------

_REGISTRY: list[RegistryEntry] = [
    RegistryEntry(
        model_id="krea2-turbo",
        display_name="Krea 2 Turbo",
        backend_module="studio.models.backends.krea2",
        checkpoints=[
            "krea2_turbo_fp8_scaled.safetensors",
            "krea2_turbo_bf16.safetensors",
        ],
        vae="qwen_image_vae.safetensors",
        text_encoder="qwen3vl_4b_fp8_scaled.safetensors",
        sampler_defaults={"steps": 8, "cfg": 0.0, "mu_shift": 1.15},
        precision_options=["bf16", "fp8_scaled"],
        vram_tiers={
            # Keep in sync with DIT_VRAM_TIERS in backends/krea2.py.
            # bf16 transformer weights are 24.76 GiB — does not fit a
            # 24 GB card once the WDDM reserve is accounted for.
            "bf16": 25_000_000_000,
            "fp8_scaled": 13_000_000_000,
        },
    ),
]

# Lookup index for O(1) access by model_id
_REGISTRY_INDEX: dict[str, RegistryEntry] = {entry.model_id: entry for entry in _REGISTRY}

# ---------------------------------------------------------------------------
# Lazy backend import state
# ---------------------------------------------------------------------------

# Cached backend modules: model_id → module (after successful import)
_backend_cache: dict[str, Any] = {}

# Unavailable backends: model_id → plain-language reason string
_unavailable_backends: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_models() -> list[RegistryEntry]:
    """Return all registered model entries.

    Phase 1: returns exactly one entry (Krea 2 Turbo).
    """
    return list(_REGISTRY)


def get_meta(model_id: str) -> RegistryEntry:
    """Get metadata for a specific model by its ID.

    Args:
        model_id: The model identifier (e.g. 'krea2-turbo').

    Returns:
        The RegistryEntry for the requested model.

    Raises:
        KeyError: If model_id is not found in the registry.
    """
    if model_id not in _REGISTRY_INDEX:
        raise KeyError(f"Unknown model: '{model_id}'")
    return _REGISTRY_INDEX[model_id]


def run_generation(model_id: str, params: dict[str, Any]) -> Generator[str | dict, None, None]:
    """Run generation through the backend for the given model.

    Lazily imports the backend module on first call. If the backend is
    unavailable (failed import or prior failure), refuses with the
    recorded reason.

    Args:
        model_id: The model identifier (e.g. 'krea2-turbo').
        params: Generation parameters to pass to the backend.

    Yields:
        Progress strings (e.g. "Encoding prompt...", "Sampling step 3/8")
        or a final dict with the generation result.

    Raises:
        KeyError: If model_id is not found in the registry.
        RuntimeError: If the backend is unavailable.
    """
    # Validate model exists in registry
    if model_id not in _REGISTRY_INDEX:
        raise KeyError(f"Unknown model: '{model_id}'")

    # Refuse if backend was previously marked unavailable
    if model_id in _unavailable_backends:
        reason = _unavailable_backends[model_id]
        raise RuntimeError(
            f"Backend for '{model_id}' is unavailable: {reason}"
        )

    # Lazy import of backend module
    backend = _get_backend(model_id)

    # Delegate to the backend's generate function
    yield from backend.generate(params)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_backend(model_id: str) -> Any:
    """Lazily import and cache the backend module for a model.

    On first access, attempts to import the backend module specified in
    the registry entry. If the import fails, records the reason and marks
    the backend as unavailable — it will NOT be re-attempted.

    Args:
        model_id: The model identifier.

    Returns:
        The imported backend module.

    Raises:
        RuntimeError: If the import fails (also marks backend unavailable).
    """
    # Return cached module if already successfully imported
    if model_id in _backend_cache:
        return _backend_cache[model_id]

    entry = _REGISTRY_INDEX[model_id]
    module_path = entry.backend_module

    try:
        log.info("Importing backend module '%s' for model '%s'", module_path, model_id)
        module = importlib.import_module(module_path)
    except Exception as e:
        # Record failure reason — never retry
        reason = f"{type(e).__name__}: {e}"
        _unavailable_backends[model_id] = reason
        log.error(
            "Failed to import backend '%s' for model '%s': %s",
            module_path,
            model_id,
            reason,
        )
        raise RuntimeError(
            f"Backend for '{model_id}' is unavailable: {reason}"
        ) from e

    # Cache the successfully imported module
    _backend_cache[model_id] = module
    log.info("Backend '%s' loaded successfully", module_path)
    return module


def _reset_registry_state() -> None:
    """Reset internal caches. For testing only."""
    _backend_cache.clear()
    _unavailable_backends.clear()
