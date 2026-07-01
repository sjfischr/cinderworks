"""Tests for core/system_check.py.

Covers Pattern B test obligations:
- Every not-ready reason produces a specific, human sentence
- Ready state only true when CUDA + weights + VAE + encoder are all present
- Property 6: all unmet conditions reported (no omission, no false report)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from studio.core.system_check import (
    _MODEL_FILES,
    _readiness_state,
    check_cuda_status,
    check_model_status,
    get_readiness_banner,
    get_system_status_text,
    is_ready_to_generate,
    startup_cuda_check,
)


# ---------------------------------------------------------------------------
# Test-friendly model file sizes (tiny, so tests don't need GB of disk)
# ---------------------------------------------------------------------------

_TEST_MODEL_FILES: dict[str, tuple[str, int]] = {
    "diffusion_fp8": ("krea2_turbo_fp8_scaled.safetensors", 1000),
    "diffusion_bf16": ("krea2_turbo_bf16.safetensors", 2000),
    "text_encoder": ("qwen3vl_4b_fp8_scaled.safetensors", 500),
    "vae": ("qwen_image_vae.safetensors", 100),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_readiness_state():
    """Reset module-level readiness state between tests."""
    _readiness_state["cuda_available"] = None
    yield
    _readiness_state["cuda_available"] = None


@pytest.fixture(autouse=True)
def patch_model_files():
    """Use test-friendly file sizes instead of real GB thresholds."""
    with patch.dict(
        "studio.core.system_check._MODEL_FILES",
        _TEST_MODEL_FILES,
    ):
        yield


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """Provide a temporary MODEL_DIR and return it for test use."""
    from studio.config import Config
    monkeypatch.setattr(Config, "MODEL_DIR", tmp_path)
    return tmp_path


def _create_valid_model_file(model_dir: Path, component: str) -> Path:
    """Create a model file that passes size validation."""
    filename, min_size = _TEST_MODEL_FILES[component]
    filepath = model_dir / filename
    filepath.write_bytes(b"\x00" * min_size)
    return filepath


def _create_undersized_model_file(model_dir: Path, component: str) -> Path:
    """Create a model file that is too small (partial download)."""
    filename, min_size = _TEST_MODEL_FILES[component]
    filepath = model_dir / filename
    filepath.write_bytes(b"\x00" * (min_size - 1))
    return filepath


# ---------------------------------------------------------------------------
# Unit Tests: check_cuda_status
# ---------------------------------------------------------------------------


class TestCheckCudaStatus:
    """Tests for check_cuda_status()."""

    def test_returns_true_when_torch_reports_cuda(self):
        """check_cuda_status returns True when torch.cuda.is_available() is True."""
        import types
        mock_torch = types.ModuleType("torch")
        mock_cuda = types.SimpleNamespace(is_available=lambda: True)
        mock_torch.cuda = mock_cuda

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = check_cuda_status()
            assert result is True
            assert _readiness_state["cuda_available"] is True

    def test_returns_false_when_torch_reports_no_cuda(self):
        """check_cuda_status returns False when torch.cuda.is_available() is False."""
        import types
        mock_torch = types.ModuleType("torch")
        mock_cuda = types.SimpleNamespace(is_available=lambda: False)
        mock_torch.cuda = mock_cuda

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = check_cuda_status()
            assert result is False
            assert _readiness_state["cuda_available"] is False

    def test_returns_false_when_torch_not_installed(self):
        """check_cuda_status returns False if torch cannot be imported."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = check_cuda_status()
            assert result is False
            assert _readiness_state["cuda_available"] is False

    def test_stores_result_in_readiness_state(self):
        """The CUDA check result persists in module state."""
        assert _readiness_state["cuda_available"] is None
        check_cuda_status()
        assert _readiness_state["cuda_available"] is not None


# ---------------------------------------------------------------------------
# Unit Tests: check_model_status
# ---------------------------------------------------------------------------


class TestCheckModelStatus:
    """Tests for check_model_status()."""

    def test_all_missing_when_dir_empty(self, model_dir):
        """All components report False when no files exist."""
        status = check_model_status()
        assert all(v is False for v in status.values())

    def test_valid_file_reports_present(self, model_dir):
        """A file meeting size threshold reports True."""
        _create_valid_model_file(model_dir, "text_encoder")
        status = check_model_status()
        assert status["text_encoder"] is True

    def test_undersized_file_reports_not_present(self, model_dir):
        """A file below size threshold is treated as partial/not present (R3.6)."""
        _create_undersized_model_file(model_dir, "text_encoder")
        status = check_model_status()
        assert status["text_encoder"] is False

    def test_all_files_present_and_valid(self, model_dir):
        """All components True when all files exist and pass size check."""
        for component in _TEST_MODEL_FILES:
            _create_valid_model_file(model_dir, component)
        status = check_model_status()
        assert all(v is True for v in status.values())

    def test_returns_all_four_components(self, model_dir):
        """Status dict contains entries for all expected components."""
        status = check_model_status()
        assert set(status.keys()) == {"diffusion_fp8", "diffusion_bf16", "text_encoder", "vae"}

    def test_zero_byte_file_is_not_present(self, model_dir):
        """An empty file (0 bytes) is not considered present."""
        filename, _ = _TEST_MODEL_FILES["vae"]
        (model_dir / filename).write_bytes(b"")
        status = check_model_status()
        assert status["vae"] is False


# ---------------------------------------------------------------------------
# Unit Tests: is_ready_to_generate
# ---------------------------------------------------------------------------


class TestIsReadyToGenerate:
    """Tests for is_ready_to_generate()."""

    def test_not_ready_when_no_cuda(self, model_dir):
        """Returns False when CUDA is unavailable, even with all files."""
        _readiness_state["cuda_available"] = False
        for component in _TEST_MODEL_FILES:
            _create_valid_model_file(model_dir, component)
        assert is_ready_to_generate() is False

    def test_not_ready_when_no_files(self, model_dir):
        """Returns False when CUDA is available but no model files."""
        _readiness_state["cuda_available"] = True
        assert is_ready_to_generate() is False

    def test_ready_with_cuda_and_fp8_encoder_vae(self, model_dir):
        """Ready when CUDA + fp8 diffusion + encoder + VAE present."""
        _readiness_state["cuda_available"] = True
        _create_valid_model_file(model_dir, "diffusion_fp8")
        _create_valid_model_file(model_dir, "text_encoder")
        _create_valid_model_file(model_dir, "vae")
        assert is_ready_to_generate() is True

    def test_ready_with_cuda_and_bf16_encoder_vae(self, model_dir):
        """Ready when CUDA + bf16 diffusion + encoder + VAE present."""
        _readiness_state["cuda_available"] = True
        _create_valid_model_file(model_dir, "diffusion_bf16")
        _create_valid_model_file(model_dir, "text_encoder")
        _create_valid_model_file(model_dir, "vae")
        assert is_ready_to_generate() is True

    def test_not_ready_without_encoder(self, model_dir):
        """Not ready if text encoder is missing."""
        _readiness_state["cuda_available"] = True
        _create_valid_model_file(model_dir, "diffusion_fp8")
        _create_valid_model_file(model_dir, "vae")
        assert is_ready_to_generate() is False

    def test_not_ready_without_vae(self, model_dir):
        """Not ready if VAE is missing."""
        _readiness_state["cuda_available"] = True
        _create_valid_model_file(model_dir, "diffusion_fp8")
        _create_valid_model_file(model_dir, "text_encoder")
        assert is_ready_to_generate() is False

    def test_not_ready_without_any_diffusion(self, model_dir):
        """Not ready if no diffusion checkpoint is present."""
        _readiness_state["cuda_available"] = True
        _create_valid_model_file(model_dir, "text_encoder")
        _create_valid_model_file(model_dir, "vae")
        assert is_ready_to_generate() is False


# ---------------------------------------------------------------------------
# Unit Tests: get_system_status_text
# ---------------------------------------------------------------------------


class TestGetSystemStatusText:
    """Tests for get_system_status_text()."""

    def test_reports_no_cuda(self, model_dir):
        """Produces a specific human sentence when CUDA is missing."""
        _readiness_state["cuda_available"] = False
        text = get_system_status_text()
        assert "No CUDA GPU detected" in text

    def test_reports_cuda_available(self, model_dir):
        """Reports CUDA detected when available."""
        _readiness_state["cuda_available"] = True
        text = get_system_status_text()
        assert "CUDA GPU detected" in text
        assert "No CUDA" not in text

    def test_reports_missing_diffusion(self, model_dir):
        """Reports diffusion model missing."""
        _readiness_state["cuda_available"] = True
        text = get_system_status_text()
        assert "Krea 2 Turbo model not downloaded yet" in text

    def test_reports_missing_encoder(self, model_dir):
        """Reports text encoder missing."""
        _readiness_state["cuda_available"] = True
        text = get_system_status_text()
        assert "Text encoder not downloaded yet" in text

    def test_reports_missing_vae(self, model_dir):
        """Reports VAE missing."""
        _readiness_state["cuda_available"] = True
        text = get_system_status_text()
        assert "VAE not downloaded yet" in text

    def test_all_ready_shows_checkmarks(self, model_dir):
        """When all conditions met, shows checkmarks and no error messages."""
        _readiness_state["cuda_available"] = True
        for component in _TEST_MODEL_FILES:
            _create_valid_model_file(model_dir, component)
        text = get_system_status_text()
        assert "❌" not in text
        assert "✅" in text

    def test_every_not_ready_reason_is_specific_sentence(self, model_dir):
        """Each unmet condition appears as a distinct, plain-language sentence."""
        _readiness_state["cuda_available"] = False
        text = get_system_status_text()
        lines = text.strip().split("\n")
        # Should have 4 lines (CUDA + diffusion + encoder + VAE)
        assert len(lines) == 4
        # Each line should be human-readable (not a code artifact)
        for line in lines:
            assert len(line) > 5  # Not just an emoji


# ---------------------------------------------------------------------------
# Unit Tests: get_readiness_banner
# ---------------------------------------------------------------------------


class TestGetReadinessBanner:
    """Tests for get_readiness_banner()."""

    def test_hidden_when_ready(self, model_dir):
        """Banner is invisible when system is ready."""
        _readiness_state["cuda_available"] = True
        _create_valid_model_file(model_dir, "diffusion_fp8")
        _create_valid_model_file(model_dir, "text_encoder")
        _create_valid_model_file(model_dir, "vae")
        banner = get_readiness_banner()
        assert banner["visible"] is False
        assert banner["value"] == ""

    def test_visible_when_not_ready(self, model_dir):
        """Banner is visible when any condition is unmet."""
        _readiness_state["cuda_available"] = False
        banner = get_readiness_banner()
        assert banner["visible"] is True
        assert banner["value"] != ""

    def test_lists_all_unmet_conditions(self, model_dir):
        """Banner text includes every unmet condition."""
        _readiness_state["cuda_available"] = False
        banner = get_readiness_banner()
        text = banner["value"]
        assert "No CUDA GPU detected" in text
        assert "Krea 2 Turbo model not downloaded yet" in text
        assert "Text encoder not downloaded yet" in text
        assert "VAE not downloaded yet" in text

    def test_does_not_list_met_conditions(self, model_dir):
        """Banner only lists unmet conditions, not met ones."""
        _readiness_state["cuda_available"] = True
        _create_valid_model_file(model_dir, "diffusion_fp8")
        _create_valid_model_file(model_dir, "text_encoder")
        # VAE missing
        banner = get_readiness_banner()
        assert banner["visible"] is True
        assert "No CUDA GPU detected" not in banner["value"]
        assert "Krea 2 Turbo model not downloaded yet" not in banner["value"]
        assert "Text encoder not downloaded yet" not in banner["value"]
        assert "VAE not downloaded yet" in banner["value"]

    def test_returns_dict_suitable_for_gradio(self, model_dir):
        """Banner returns a dict with 'visible' and 'value' keys."""
        _readiness_state["cuda_available"] = True
        banner = get_readiness_banner()
        assert "visible" in banner
        assert "value" in banner
        assert isinstance(banner["visible"], bool)
        assert isinstance(banner["value"], str)


# ---------------------------------------------------------------------------
# Unit Tests: startup_cuda_check
# ---------------------------------------------------------------------------


class TestStartupCudaCheck:
    """Tests for startup_cuda_check()."""

    def test_stores_result_regardless_of_outcome(self):
        """startup_cuda_check stores the result and does not raise."""
        assert _readiness_state["cuda_available"] is None
        startup_cuda_check()
        # Should have stored a bool (True or False depending on system)
        assert isinstance(_readiness_state["cuda_available"], bool)

    def test_continues_when_cuda_absent(self):
        """Startup continues (no exception) when CUDA is not available."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            # Should not raise
            startup_cuda_check()
            assert _readiness_state["cuda_available"] is False


# ---------------------------------------------------------------------------
# Property-Based Tests
# ---------------------------------------------------------------------------

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


# Feature: cinderworks, Property 3: Size mismatch means not-present
class TestProperty3SizeMismatchMeansNotPresent:
    """Property 3: For any model file whose size is below the minimum threshold,
    check_model_status reports that component as not-present (False).
    For any file whose size is >= the minimum threshold, check_model_status
    reports that component as present (True).

    Validates: Requirements 3.6
    """

    @given(
        component=st.sampled_from(
            ["diffusion_fp8", "diffusion_bf16", "text_encoder", "vae"]
        ),
        file_size=st.integers(min_value=0, max_value=99),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_undersized_file_reported_not_present(
        self, component: str, file_size: int, model_dir, patch_model_files
    ):
        """**Validates: Requirements 3.6**

        For any component file whose size is below its minimum threshold,
        check_model_status SHALL report that component as False (not-present).
        """
        filename, min_size = _TEST_MODEL_FILES[component]
        # Ensure file_size is strictly below the threshold
        actual_size = file_size % min_size if min_size > 0 else 0
        filepath = model_dir / filename
        filepath.write_bytes(b"\x00" * actual_size)

        status = check_model_status()
        assert status[component] is False, (
            f"Expected {component} to be not-present with size {actual_size} "
            f"(threshold: {min_size})"
        )

    @given(
        component=st.sampled_from(
            ["diffusion_fp8", "diffusion_bf16", "text_encoder", "vae"]
        ),
        extra_bytes=st.integers(min_value=0, max_value=500),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_adequately_sized_file_reported_present(
        self, component: str, extra_bytes: int, model_dir, patch_model_files
    ):
        """**Validates: Requirements 3.6**

        For any component file whose size is >= its minimum threshold,
        check_model_status SHALL report that component as True (present).
        """
        filename, min_size = _TEST_MODEL_FILES[component]
        actual_size = min_size + extra_bytes
        filepath = model_dir / filename
        filepath.write_bytes(b"\x00" * actual_size)

        status = check_model_status()
        assert status[component] is True, (
            f"Expected {component} to be present with size {actual_size} "
            f"(threshold: {min_size})"
        )


# ---------------------------------------------------------------------------
# Property-Based Test: Property 6 — Readiness reports all unmet conditions
# Feature: cinderworks, Property 6: Readiness reports all unmet conditions
# ---------------------------------------------------------------------------


class TestReadinessReportsAllUnmetConditions:
    """Property 6: For any combination of system conditions, the readiness
    banner SHALL report every unmet condition and never include a met one.

    **Validates: Requirements 4.1, 4.4**
    """

    @given(
        cuda_available=st.booleans(),
        diffusion_fp8_present=st.booleans(),
        diffusion_bf16_present=st.booleans(),
        text_encoder_present=st.booleans(),
        vae_present=st.booleans(),
    )
    @settings(max_examples=100)
    def test_banner_reports_exactly_unmet_conditions(
        self,
        cuda_available: bool,
        diffusion_fp8_present: bool,
        diffusion_bf16_present: bool,
        text_encoder_present: bool,
        vae_present: bool,
    ):
        """Every unmet condition appears in the banner; no met condition does."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # -- Arrange: set CUDA state --
            _readiness_state["cuda_available"] = cuda_available

            # -- Arrange: create model files based on drawn booleans --
            with patch.dict(
                "studio.core.system_check._MODEL_FILES", _TEST_MODEL_FILES
            ):
                from studio.config import Config

                with patch.object(Config, "MODEL_DIR", tmp_path):
                    components_to_create = []
                    if diffusion_fp8_present:
                        components_to_create.append("diffusion_fp8")
                    if diffusion_bf16_present:
                        components_to_create.append("diffusion_bf16")
                    if text_encoder_present:
                        components_to_create.append("text_encoder")
                    if vae_present:
                        components_to_create.append("vae")

                    for component in components_to_create:
                        filename, min_size = _TEST_MODEL_FILES[component]
                        filepath = tmp_path / filename
                        filepath.write_bytes(b"\x00" * min_size)

                    # -- Act --
                    banner = get_readiness_banner()
                    banner_text = banner["value"]

                    # -- Determine expected conditions --
                    has_diffusion = diffusion_fp8_present or diffusion_bf16_present
                    all_met = (
                        cuda_available
                        and has_diffusion
                        and text_encoder_present
                        and vae_present
                    )

                    # -- Assert: when all met, banner is hidden --
                    if all_met:
                        assert banner["visible"] is False
                        assert banner_text == ""
                        return

                    # -- Assert: banner is visible when anything is unmet --
                    assert banner["visible"] is True

                    # -- Assert: every UNMET condition appears in the banner --
                    if not cuda_available:
                        assert "No CUDA GPU detected" in banner_text, (
                            "CUDA is unmet but not reported in banner"
                        )

                    if not has_diffusion:
                        assert "Krea 2 Turbo model not downloaded yet" in banner_text, (
                            "Diffusion is unmet but not reported in banner"
                        )

                    if not text_encoder_present:
                        assert "Text encoder not downloaded yet" in banner_text, (
                            "Text encoder is unmet but not reported in banner"
                        )

                    if not vae_present:
                        assert "VAE not downloaded yet" in banner_text, (
                            "VAE is unmet but not reported in banner"
                        )

                    # -- Assert: no MET condition appears as unmet in the banner --
                    if cuda_available:
                        assert "No CUDA GPU detected" not in banner_text, (
                            "CUDA is met but reported as unmet in banner"
                        )

                    if has_diffusion:
                        assert "Krea 2 Turbo model not downloaded yet" not in banner_text, (
                            "Diffusion is met but reported as unmet in banner"
                        )

                    if text_encoder_present:
                        assert "Text encoder not downloaded yet" not in banner_text, (
                            "Text encoder is met but reported as unmet in banner"
                        )

                    if vae_present:
                        assert "VAE not downloaded yet" not in banner_text, (
                            "VAE is met but reported as unmet in banner"
                        )
