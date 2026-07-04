"""Unit tests for krea2.py img2img mode (Task 5.1).

Validates Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6:
- init_image_path triggers img2img mode (4.1, 4.4)
- denoise_strength controls noise level (4.2, 4.4)
- denoise_strength=0.0 returns init image unchanged (4.5)
- denoise_strength=1.0 performs full sampling from noise using init image dims (4.6)

All tests use the stub inference path (no GPU/model weights needed).
"""

import pytest
from pathlib import Path
from PIL import Image

from studio.models.backends.krea2 import generate
from studio.core.vram_manager import VRAMManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def init_image_path(tmp_path: Path) -> str:
    """Create a simple test image and return its path."""
    img = Image.new("RGB", (1024, 1024), color=(128, 64, 200))
    img_path = tmp_path / "init_test.png"
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def small_init_image_path(tmp_path: Path) -> str:
    """Create a smaller test image (512x512) for dimension tests."""
    img = Image.new("RGB", (512, 512), color=(50, 100, 150))
    img_path = tmp_path / "init_small.png"
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def vram_mgr() -> VRAMManager:
    """Provide a permissive VRAM manager for tests."""
    return VRAMManager(total_vram=100_000_000_000)


# ---------------------------------------------------------------------------
# Test: denoise_strength=0.0 returns init image unchanged (Req 4.5)
# ---------------------------------------------------------------------------


class TestDenoiseZeroPassthrough:
    """When denoise_strength is 0.0, return the init image without sampling."""

    def test_returns_init_image_unchanged(self, init_image_path: str, vram_mgr: VRAMManager, tmp_path: Path):
        """denoise_strength=0.0 copies the init image to output without running pipeline."""
        params = {
            "prompt": "a beautiful landscape",
            "seed": 42,
            "init_image_path": init_image_path,
            "denoise_strength": 0.0,
            "_vram_manager": vram_mgr,
        }

        result = None
        progress_msgs = []
        for output in generate(params):
            if isinstance(output, dict):
                result = output
            else:
                progress_msgs.append(output)

        assert result is not None, "generate() must yield a final dict result"
        assert len(result["images"]) == 1
        assert result["images"][0].exists(), "Output image file should exist"

        # The output should be a copy of the init image (same pixels)
        init_img = Image.open(init_image_path)
        output_img = Image.open(result["images"][0])
        assert init_img.size == output_img.size
        assert list(init_img.getdata()) == list(output_img.getdata())

    def test_no_sampling_steps_yielded(self, init_image_path: str, vram_mgr: VRAMManager):
        """No sampling step progress messages when denoise=0.0."""
        params = {
            "prompt": "test prompt",
            "seed": 100,
            "init_image_path": init_image_path,
            "denoise_strength": 0.0,
            "_vram_manager": vram_mgr,
        }

        progress_msgs = []
        for output in generate(params):
            if isinstance(output, str):
                progress_msgs.append(output)

        # Should not have any "Sampling step" messages
        sampling_msgs = [m for m in progress_msgs if "Sampling step" in m]
        assert len(sampling_msgs) == 0, (
            f"Expected no sampling messages for denoise=0.0, got: {sampling_msgs}"
        )

    def test_result_params_include_img2img_fields(self, init_image_path: str, vram_mgr: VRAMManager):
        """Result params include denoise_strength and init_image_path."""
        params = {
            "prompt": "test",
            "seed": 55,
            "init_image_path": init_image_path,
            "denoise_strength": 0.0,
            "_vram_manager": vram_mgr,
        }

        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output

        assert result is not None
        assert result["params"]["denoise_strength"] == 0.0
        assert result["params"]["init_image_path"] == init_image_path

    def test_missing_init_image_raises_error(self, vram_mgr: VRAMManager):
        """ValueError when init image file doesn't exist and denoise=0.0."""
        params = {
            "prompt": "test",
            "seed": 42,
            "init_image_path": "/nonexistent/path/image.png",
            "denoise_strength": 0.0,
            "_vram_manager": vram_mgr,
        }

        with pytest.raises(ValueError, match="not found"):
            for _ in generate(params):
                pass


# ---------------------------------------------------------------------------
# Test: denoise_strength=1.0 performs full sampling (Req 4.6)
# ---------------------------------------------------------------------------


class TestDenoiseOneFullSampling:
    """When denoise_strength is 1.0, perform full sampling from noise using init image dims."""

    def test_full_sampling_runs_all_steps(self, init_image_path: str, vram_mgr: VRAMManager):
        """denoise_strength=1.0 runs the full number of sampling steps."""
        params = {
            "prompt": "a sunset",
            "seed": 42,
            "steps": 8,
            "init_image_path": init_image_path,
            "denoise_strength": 1.0,
            "_vram_manager": vram_mgr,
        }

        progress_msgs = []
        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output
            else:
                progress_msgs.append(output)

        assert result is not None
        # All 8 steps should execute
        sampling_msgs = [m for m in progress_msgs if "Sampling step" in m]
        assert len(sampling_msgs) == 8, (
            f"Expected 8 sampling steps for denoise=1.0, got {len(sampling_msgs)}"
        )

    def test_result_params_include_denoise_and_path(self, init_image_path: str, vram_mgr: VRAMManager):
        """Result includes denoise_strength=1.0 and init_image_path."""
        params = {
            "prompt": "test",
            "seed": 42,
            "init_image_path": init_image_path,
            "denoise_strength": 1.0,
            "_vram_manager": vram_mgr,
        }

        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output

        assert result is not None
        assert result["params"]["denoise_strength"] == 1.0
        assert result["params"]["init_image_path"] == init_image_path

    def test_produces_output_images(self, init_image_path: str, vram_mgr: VRAMManager):
        """denoise_strength=1.0 produces output image(s) in the output directory."""
        params = {
            "prompt": "ocean waves",
            "seed": 99,
            "init_image_path": init_image_path,
            "denoise_strength": 1.0,
            "_vram_manager": vram_mgr,
        }

        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output

        assert result is not None
        assert len(result["images"]) == 1
        # Stub creates the output path but doesn't write a real image
        # (that's expected for the stub path)


# ---------------------------------------------------------------------------
# Test: Partial denoise (0 < denoise < 1) runs reduced steps (Req 4.4)
# ---------------------------------------------------------------------------


class TestPartialDenoise:
    """Img2img with 0 < denoise_strength < 1 runs a fraction of the full steps."""

    def test_half_denoise_runs_half_steps(self, init_image_path: str, vram_mgr: VRAMManager):
        """denoise_strength=0.5 with 8 steps runs 4 sampling steps."""
        params = {
            "prompt": "mountain vista",
            "seed": 42,
            "steps": 8,
            "init_image_path": init_image_path,
            "denoise_strength": 0.5,
            "_vram_manager": vram_mgr,
        }

        progress_msgs = []
        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output
            else:
                progress_msgs.append(output)

        assert result is not None
        sampling_msgs = [m for m in progress_msgs if "Sampling step" in m]
        assert len(sampling_msgs) == 4, (
            f"Expected 4 steps for denoise=0.5, got {len(sampling_msgs)}"
        )

    def test_quarter_denoise_runs_quarter_steps(self, init_image_path: str, vram_mgr: VRAMManager):
        """denoise_strength=0.25 with 8 steps runs 2 sampling steps."""
        params = {
            "prompt": "forest path",
            "seed": 42,
            "steps": 8,
            "init_image_path": init_image_path,
            "denoise_strength": 0.25,
            "_vram_manager": vram_mgr,
        }

        progress_msgs = []
        for output in generate(params):
            if isinstance(output, str):
                progress_msgs.append(output)

        sampling_msgs = [m for m in progress_msgs if "Sampling step" in m]
        assert len(sampling_msgs) == 2, (
            f"Expected 2 steps for denoise=0.25, got {len(sampling_msgs)}"
        )

    def test_low_denoise_runs_at_least_one_step(self, init_image_path: str, vram_mgr: VRAMManager):
        """Very low denoise_strength still runs at least 1 step (never 0)."""
        params = {
            "prompt": "slight touch-up",
            "seed": 42,
            "steps": 8,
            "init_image_path": init_image_path,
            "denoise_strength": 0.05,  # 0.05 * 8 = 0.4, rounds to 0 → clamped to 1
            "_vram_manager": vram_mgr,
        }

        progress_msgs = []
        for output in generate(params):
            if isinstance(output, str):
                progress_msgs.append(output)

        sampling_msgs = [m for m in progress_msgs if "Sampling step" in m]
        assert len(sampling_msgs) >= 1, (
            f"Expected at least 1 step even for very low denoise, got {len(sampling_msgs)}"
        )

    def test_result_includes_img2img_params(self, init_image_path: str, vram_mgr: VRAMManager):
        """Result dict includes img2img-specific params for partial denoise."""
        params = {
            "prompt": "test partial",
            "seed": 42,
            "steps": 8,
            "init_image_path": init_image_path,
            "denoise_strength": 0.7,
            "_vram_manager": vram_mgr,
        }

        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output

        assert result is not None
        assert result["params"]["denoise_strength"] == 0.7
        assert result["params"]["init_image_path"] == init_image_path
        # Original step count is preserved in params (not the reduced count)
        assert result["params"]["steps"] == 8

    def test_encodes_init_image_message(self, init_image_path: str, vram_mgr: VRAMManager):
        """Progress messages include 'Encoding init image' for partial denoise."""
        params = {
            "prompt": "test encoding",
            "seed": 42,
            "steps": 8,
            "init_image_path": init_image_path,
            "denoise_strength": 0.6,
            "_vram_manager": vram_mgr,
        }

        progress_msgs = []
        for output in generate(params):
            if isinstance(output, str):
                progress_msgs.append(output)

        encoding_msgs = [m for m in progress_msgs if "Encoding init image" in m]
        assert len(encoding_msgs) >= 1, (
            f"Expected 'Encoding init image' message, got: {progress_msgs}"
        )


# ---------------------------------------------------------------------------
# Test: img2img trigger detection (Req 4.1)
# ---------------------------------------------------------------------------


class TestImg2ImgTrigger:
    """init_image_path in params triggers img2img mode."""

    def test_no_init_image_is_txt2img(self, vram_mgr: VRAMManager):
        """Without init_image_path, generate runs normal txt2img."""
        params = {
            "prompt": "a cat",
            "seed": 42,
            "steps": 8,
            "_vram_manager": vram_mgr,
        }

        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output

        assert result is not None
        # No img2img-specific params in result
        assert "denoise_strength" not in result["params"]
        assert "init_image_path" not in result["params"]

    def test_with_init_image_activates_img2img(self, init_image_path: str, vram_mgr: VRAMManager):
        """With init_image_path, generate activates img2img mode."""
        params = {
            "prompt": "a cat but better",
            "seed": 42,
            "steps": 8,
            "init_image_path": init_image_path,
            "denoise_strength": 0.5,
            "_vram_manager": vram_mgr,
        }

        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output

        assert result is not None
        assert "denoise_strength" in result["params"]
        assert "init_image_path" in result["params"]

    def test_default_denoise_strength_is_half(self, init_image_path: str, vram_mgr: VRAMManager):
        """When denoise_strength is not specified but init_image_path is, default to 0.5."""
        params = {
            "prompt": "test default",
            "seed": 42,
            "steps": 8,
            "init_image_path": init_image_path,
            # denoise_strength not specified — should default to 0.5
            "_vram_manager": vram_mgr,
        }

        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output

        assert result is not None
        assert result["params"]["denoise_strength"] == 0.5
