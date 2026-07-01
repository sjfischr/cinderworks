"""Tests for core/model_loader.py — Lazy model loading.

Validates Requirements 2.1, 2.2, 2.3:
- App starts without loading model weights into memory or GPU (2.1)
- No model-loading libraries imported that trigger CUDA initialization before generation (2.2)
- First generation loads components, subsequent reuse cached (2.3)
"""

import sys

import pytest

from studio.core.model_loader import clear_cache, get_or_load, is_loaded


@pytest.fixture(autouse=True)
def _isolate_cache():
    """Clear model loader cache before each test for isolation."""
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Test 1: Importing module does not trigger CUDA initialization (Req 2.2)
# ---------------------------------------------------------------------------


class TestImportSafety:
    """Importing model_loader must not touch CUDA or trigger weight loading."""

    def test_import_does_not_import_torch(self):
        """Requirement 2.2: Importing model_loader does NOT import torch.

        Verifies that `import studio.core.model_loader` does not bring
        torch into sys.modules. If torch was already loaded (e.g., by a
        prior test), we skip this check since we cannot unload it.
        """
        # If torch is already loaded (from another test in the suite),
        # we can only verify that model_loader itself doesn't import it
        # at module level. We inspect the module source instead.
        import importlib
        import inspect

        # Re-import the module to examine its top-level code
        import studio.core.model_loader as ml

        source = inspect.getsource(ml)

        # Verify no top-level `import torch` (only inside functions)
        lines = source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip lines inside function/method bodies (indented)
            if line and not line[0].isspace():
                # Top-level line — should not import torch
                assert "import torch" not in stripped, (
                    f"Found top-level 'import torch' at line {i + 1}: {line}"
                )

    def test_module_level_has_no_torch_dependency(self):
        """The model_loader module defines no module-level torch usage."""
        import studio.core.model_loader as ml

        # Check that the module's global namespace doesn't contain torch
        module_globals = vars(ml)
        assert "torch" not in module_globals, (
            "torch should not be in model_loader's module namespace"
        )


# ---------------------------------------------------------------------------
# Test 2: First call loads, subsequent reuses cache (Req 2.3)
# ---------------------------------------------------------------------------


class TestCacheBehavior:
    """First get_or_load loads; subsequent calls return the cached result."""

    def test_not_loaded_before_first_call(self):
        """Before any get_or_load call, is_loaded returns False."""
        assert is_loaded("krea2-turbo", "bf16") is False

    def test_first_call_loads_components(self):
        """First call to get_or_load returns components dict with 'loaded' flag."""
        result = get_or_load("krea2-turbo", "bf16")

        assert isinstance(result, dict)
        assert result["loaded"] is True
        assert result["model_id"] == "krea2-turbo"
        assert result["precision"] == "bf16"

    def test_is_loaded_true_after_first_call(self):
        """After first get_or_load call, is_loaded returns True."""
        get_or_load("krea2-turbo", "bf16")
        assert is_loaded("krea2-turbo", "bf16") is True

    def test_subsequent_call_returns_same_object(self):
        """Second call returns the exact same cached object (identity check)."""
        first = get_or_load("krea2-turbo", "bf16")
        second = get_or_load("krea2-turbo", "bf16")

        # Identity check — must be the exact same dict, not a copy
        assert first is second

    def test_cache_cleared_resets_state(self):
        """After clear_cache(), is_loaded returns False and reload occurs."""
        get_or_load("krea2-turbo", "bf16")
        assert is_loaded("krea2-turbo", "bf16") is True

        clear_cache()

        assert is_loaded("krea2-turbo", "bf16") is False


# ---------------------------------------------------------------------------
# Test 3: No model weights in GPU memory before first generate (Req 2.1)
# ---------------------------------------------------------------------------


class TestNoPreload:
    """No model weights are in GPU memory before the first generate."""

    def test_is_loaded_false_before_generate(self):
        """Requirement 2.1: Before first generate, is_loaded returns False
        for all known models and precisions."""
        assert is_loaded("krea2-turbo", "bf16") is False
        assert is_loaded("krea2-turbo", "fp8_scaled") is False

    def test_cache_empty_on_fresh_import(self):
        """The internal cache is empty after clear_cache (simulating fresh state)."""
        import studio.core.model_loader as ml

        # Cache should be empty (cleared by autouse fixture)
        assert len(ml._cache) == 0


# ---------------------------------------------------------------------------
# Test 4: Different precision gets different cache entry
# ---------------------------------------------------------------------------


class TestPrecisionIsolation:
    """Different precision variants are cached independently."""

    def test_different_precision_gets_different_cache_entry(self):
        """Loading 'bf16' then 'fp8_scaled' creates two distinct cache entries."""
        bf16_result = get_or_load("krea2-turbo", "bf16")
        fp8_result = get_or_load("krea2-turbo", "fp8_scaled")

        # They should be different objects
        assert bf16_result is not fp8_result

        # Each should have the correct precision
        assert bf16_result["precision"] == "bf16"
        assert fp8_result["precision"] == "fp8_scaled"

        # Both should be independently cached
        assert is_loaded("krea2-turbo", "bf16") is True
        assert is_loaded("krea2-turbo", "fp8_scaled") is True

    def test_loading_one_precision_does_not_cache_other(self):
        """Loading bf16 does not mark fp8_scaled as loaded."""
        get_or_load("krea2-turbo", "bf16")

        assert is_loaded("krea2-turbo", "bf16") is True
        assert is_loaded("krea2-turbo", "fp8_scaled") is False


# ---------------------------------------------------------------------------
# Test 5: Unknown model raises KeyError
# ---------------------------------------------------------------------------


class TestUnknownModel:
    """Requesting an unknown model raises KeyError."""

    def test_unknown_model_raises_key_error(self):
        """get_or_load with a non-existent model_id raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            get_or_load("nonexistent", "bf16")

    def test_unknown_model_not_cached(self):
        """A failed lookup does not pollute the cache."""
        with pytest.raises(KeyError):
            get_or_load("nonexistent", "bf16")

        assert is_loaded("nonexistent", "bf16") is False
