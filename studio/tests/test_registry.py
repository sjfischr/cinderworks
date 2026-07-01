"""Tests for studio.models.registry — unit tests and property-based tests.

Validates Requirements 9.1, 9.3, 9.4, 9.6:
- Shell never imports backend directly (all access through registry) (9.1)
- Failing backend is marked unavailable and app still constructs (9.3)
- Subsequent generation against unavailable backend returns recorded reason without
  re-attempting import (9.4)
- Exactly 1 entry in Phase 1 (9.6)
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from studio.models.registry import (
    _reset_registry_state,
    _unavailable_backends,
    get_meta,
    list_models,
    run_generation,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Reset registry caches before each test for isolation."""
    _reset_registry_state()
    yield
    _reset_registry_state()


# ---------------------------------------------------------------------------
# Requirement 9.6 — Exactly one Registry entry in Phase 1
# ---------------------------------------------------------------------------


class TestExactlyOneEntry:
    """Phase 1: registry ships exactly one entry (Krea 2 Turbo), no stubs."""

    def test_exactly_one_entry(self):
        """list_models() returns exactly 1 entry and it's 'krea2-turbo'."""
        models = list_models()
        assert len(models) == 1
        assert models[0].model_id == "krea2-turbo"

    def test_get_meta_returns_entry(self):
        """get_meta('krea2-turbo') returns the Krea 2 entry with correct fields."""
        entry = get_meta("krea2-turbo")
        assert entry.model_id == "krea2-turbo"
        assert entry.display_name == "Krea 2 Turbo"
        assert entry.backend_module == "studio.models.backends.krea2"
        assert len(entry.checkpoints) > 0
        assert entry.vae != ""
        assert entry.text_encoder != ""
        assert entry.sampler_defaults == {"steps": 8, "cfg": 1.0, "mu_shift": 1.15}
        assert "bf16" in entry.precision_options
        assert "fp8_scaled" in entry.precision_options

    def test_get_meta_unknown_raises(self):
        """get_meta('nonexistent') raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            get_meta("nonexistent")


# ---------------------------------------------------------------------------
# Requirement 9.3 — Failing backend marked unavailable, app still works
# ---------------------------------------------------------------------------


class TestFailingBackendMarkedUnavailable:
    """A backend that fails to import is marked unavailable; app keeps running."""

    def test_failing_backend_marked_unavailable(self):
        """Patch importlib.import_module to raise; verify RuntimeError with reason;
        verify app can still call list_models() and get_meta() (not crashed)."""
        with patch(
            "studio.models.registry.importlib.import_module",
            side_effect=ImportError("CUDA driver not found"),
        ):
            # Attempt generation — should fail with RuntimeError
            gen = run_generation("krea2-turbo", {"prompt": "test"})
            with pytest.raises(RuntimeError, match="unavailable"):
                # Exhaust the generator to trigger the lazy import
                list(gen)

        # App is still functional — list_models and get_meta work fine
        models = list_models()
        assert len(models) == 1
        assert models[0].model_id == "krea2-turbo"

        meta = get_meta("krea2-turbo")
        assert meta.display_name == "Krea 2 Turbo"

    def test_failing_backend_records_reason(self):
        """The recorded unavailability reason contains the original error info."""
        with patch(
            "studio.models.registry.importlib.import_module",
            side_effect=ImportError("libcuda.so not found"),
        ):
            gen = run_generation("krea2-turbo", {"prompt": "test"})
            with pytest.raises(RuntimeError, match="libcuda.so not found"):
                list(gen)

        # Second attempt should also fail with the recorded reason, no re-import
        gen2 = run_generation("krea2-turbo", {"prompt": "test2"})
        with pytest.raises(RuntimeError, match="libcuda.so not found"):
            list(gen2)


# ---------------------------------------------------------------------------
# Requirement 9.1 — Shell never imports backend directly
# ---------------------------------------------------------------------------


class TestBackendNotImportedAtModuleLevel:
    """Importing registry does NOT trigger import of the backend module."""

    def test_backend_not_imported_at_module_level(self):
        """Importing studio.models.registry does NOT trigger import of
        studio.models.backends.krea2."""
        # Remove the registry module from sys.modules so we can re-import it fresh
        modules_to_remove = [
            key for key in sys.modules if key.startswith("studio.models.registry")
        ]
        for mod in modules_to_remove:
            del sys.modules[mod]

        # Also ensure the backend is not already loaded
        backend_key = "studio.models.backends.krea2"
        if backend_key in sys.modules:
            del sys.modules[backend_key]

        # Patch importlib.import_module to track what gets imported
        original_import = importlib.import_module
        imported_modules: list[str] = []

        def tracking_import(name, *args, **kwargs):
            imported_modules.append(name)
            return original_import(name, *args, **kwargs)

        with patch("importlib.import_module", side_effect=tracking_import):
            # Re-import the registry module
            importlib.import_module("studio.models.registry")

        # The backend should NOT have been imported at module level
        assert backend_key not in imported_modules
        assert backend_key not in sys.modules


# ---------------------------------------------------------------------------
# Caching behavior — successful backend import is cached
# ---------------------------------------------------------------------------


class TestSuccessfulBackendCached:
    """After first successful import, second call reuses cached module."""

    def test_successful_backend_cached(self):
        """After first successful import, second call reuses the cached module."""
        import types

        fake_backend = types.ModuleType("fake_krea2_backend")
        fake_backend.generate = lambda params: iter(["done"])

        with patch(
            "studio.models.registry.importlib.import_module",
            return_value=fake_backend,
        ) as mock_import:
            # First generation — triggers import
            gen1 = run_generation("krea2-turbo", {"prompt": "hello"})
            result1 = list(gen1)
            assert result1 == ["done"]
            assert mock_import.call_count == 1

            # Second generation — should reuse cache, no new import
            gen2 = run_generation("krea2-turbo", {"prompt": "world"})
            result2 = list(gen2)
            assert result2 == ["done"]
            assert mock_import.call_count == 1  # Still 1, not 2


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------


# Feature: cinderworks, Property 19: Backend unavailability reason round-trip
class TestPropertyBackendUnavailabilityRoundTrip:
    """Property 19: For any exception that occurs during backend import, the
    registry SHALL store the exception's plain-language description as the
    unavailability reason, and for any subsequent generation request against
    that backend, the system SHALL return that exact recorded reason without
    re-attempting the import.

    Validates: Requirements 9.3, 9.4
    """

    @given(
        error_message=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=100, deadline=None)
    def test_unavailability_reason_stored_and_returned_without_reimport(
        self, error_message: str
    ):
        """For any error message, the registry stores it on failed import and
        returns it on subsequent requests without re-attempting the import."""
        # 1. Reset registry state for isolation
        _reset_registry_state()

        # 2. Patch importlib.import_module to raise ImportError with the generated message
        with patch(
            "studio.models.registry.importlib.import_module",
            side_effect=ImportError(error_message),
        ) as mock_import:
            # 3. First call to run_generation should raise RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                # Exhaust the generator to trigger the import
                list(run_generation("krea2-turbo", {}))

            # 4. Verify the RuntimeError message contains the original error message
            assert error_message in str(exc_info.value)

            # 5. Second call — should raise same RuntimeError with same reason
            with pytest.raises(RuntimeError) as exc_info2:
                list(run_generation("krea2-turbo", {}))

            # 6. Verify same RuntimeError with same reason (no re-import attempted)
            assert error_message in str(exc_info2.value)

            # 7. Verify importlib.import_module was called only ONCE (no re-attempt)
            assert mock_import.call_count == 1

        # Cleanup
        _reset_registry_state()


# ---------------------------------------------------------------------------
# Unit test: _unavailable_backends dict stores the reason
# ---------------------------------------------------------------------------


class TestUnavailableBackendsDict:
    """Unit test verifying _unavailable_backends dict stores the reason."""

    def test_unavailable_backends_stores_reason_on_import_failure(self):
        """When a backend import fails, the reason is stored in _unavailable_backends."""
        error_msg = "No module named 'studio.models.backends.krea2'"

        with patch(
            "studio.models.registry.importlib.import_module",
            side_effect=ImportError(error_msg),
        ):
            with pytest.raises(RuntimeError):
                list(run_generation("krea2-turbo", {}))

        # Verify the _unavailable_backends dict has the entry
        assert "krea2-turbo" in _unavailable_backends
        reason = _unavailable_backends["krea2-turbo"]
        assert "ImportError" in reason
        assert error_msg in reason
