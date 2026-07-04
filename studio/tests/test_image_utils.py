"""Tests for studio.core.image_utils — mask composite for inpainting.

Tests the standalone mask composite utility that preserves unmasked pixels
from the init image pixel-for-pixel after generation.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from studio.core.image_utils import (
    apply_mask_composite,
    composite_inpainting,
    load_mask,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def init_image(tmp_path: Path) -> tuple[Path, Image.Image]:
    """Create a 64x64 init image with known pixel values (all red)."""
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    path = tmp_path / "init.png"
    img.save(path)
    return path, img


@pytest.fixture
def output_image(tmp_path: Path) -> tuple[Path, Image.Image]:
    """Create a 64x64 output image with known pixel values (all blue)."""
    img = Image.new("RGB", (64, 64), color=(0, 0, 255))
    path = tmp_path / "output.png"
    img.save(path)
    return path, img


@pytest.fixture
def half_mask(tmp_path: Path) -> Path:
    """Create a 64x64 mask: top half masked (white=255), bottom half unmasked (black=0)."""
    mask_arr = np.zeros((64, 64), dtype=np.uint8)
    mask_arr[:32, :] = 255  # Top half masked
    mask_img = Image.fromarray(mask_arr, mode="L")
    path = tmp_path / "mask.png"
    mask_img.save(path)
    return path


@pytest.fixture
def full_mask(tmp_path: Path) -> Path:
    """Create a fully masked image (all 255 — entire image regenerated)."""
    mask_arr = np.full((64, 64), 255, dtype=np.uint8)
    mask_img = Image.fromarray(mask_arr, mode="L")
    path = tmp_path / "full_mask.png"
    mask_img.save(path)
    return path


@pytest.fixture
def empty_mask(tmp_path: Path) -> Path:
    """Create an empty mask (all 0 — entire image preserved from init)."""
    mask_arr = np.zeros((64, 64), dtype=np.uint8)
    mask_img = Image.fromarray(mask_arr, mode="L")
    path = tmp_path / "empty_mask.png"
    mask_img.save(path)
    return path


# ---------------------------------------------------------------------------
# Tests: load_mask
# ---------------------------------------------------------------------------


class TestLoadMask:
    def test_loads_mask_matching_dimensions(self, half_mask: Path):
        mask = load_mask(half_mask, 64, 64)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8

    def test_binarizes_mask_values(self, half_mask: Path):
        mask = load_mask(half_mask, 64, 64)
        unique_values = set(np.unique(mask))
        assert unique_values <= {0, 255}

    def test_resizes_mask_when_dimensions_differ(self, tmp_path: Path):
        """When mask is smaller than init image, it should be resized (Requirement 5.8)."""
        # Create a 32x32 mask
        mask_arr = np.zeros((32, 32), dtype=np.uint8)
        mask_arr[:16, :] = 255  # Top half masked
        mask_img = Image.fromarray(mask_arr, mode="L")
        mask_path = tmp_path / "small_mask.png"
        mask_img.save(mask_path)

        # Load with target size 64x64
        mask = load_mask(mask_path, 64, 64)
        assert mask.shape == (64, 64)

    def test_resizes_larger_mask_down(self, tmp_path: Path):
        """When mask is larger than init image, it should be resized down."""
        # Create a 128x128 mask
        mask_arr = np.full((128, 128), 255, dtype=np.uint8)
        mask_img = Image.fromarray(mask_arr, mode="L")
        mask_path = tmp_path / "large_mask.png"
        mask_img.save(mask_path)

        # Load with target size 64x64
        mask = load_mask(mask_path, 64, 64)
        assert mask.shape == (64, 64)

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_mask("/nonexistent/mask.png", 64, 64)

    def test_grayscale_threshold(self, tmp_path: Path):
        """Any non-zero pixel in mask becomes 255 (masked)."""
        # Create mask with intermediate values (50, 100, 200)
        mask_arr = np.array([[0, 50], [100, 200]], dtype=np.uint8)
        mask_img = Image.fromarray(mask_arr, mode="L")
        mask_path = tmp_path / "gradient_mask.png"
        mask_img.save(mask_path)

        mask = load_mask(mask_path, 2, 2)
        expected = np.array([[0, 255], [255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(mask, expected)


# ---------------------------------------------------------------------------
# Tests: composite_inpainting
# ---------------------------------------------------------------------------


class TestCompositeInpainting:
    def test_unmasked_pixels_from_init(self):
        """Pixels where mask=0 should be identical to init image (Requirement 5.5)."""
        init = Image.new("RGB", (64, 64), color=(255, 0, 0))  # Red
        output = Image.new("RGB", (64, 64), color=(0, 0, 255))  # Blue

        # Bottom half unmasked
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:32, :] = 255  # Top half masked

        result = composite_inpainting(output, init, mask)
        result_arr = np.array(result)

        # Bottom half (unmasked) should be red (from init)
        bottom_half = result_arr[32:, :, :]
        expected_red = np.full((32, 64, 3), [255, 0, 0], dtype=np.uint8)
        np.testing.assert_array_equal(bottom_half, expected_red)

    def test_masked_pixels_from_output(self):
        """Pixels where mask=255 should be from the generated output."""
        init = Image.new("RGB", (64, 64), color=(255, 0, 0))  # Red
        output = Image.new("RGB", (64, 64), color=(0, 0, 255))  # Blue

        # Top half masked
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:32, :] = 255

        result = composite_inpainting(output, init, mask)
        result_arr = np.array(result)

        # Top half (masked) should be blue (from output)
        top_half = result_arr[:32, :, :]
        expected_blue = np.full((32, 64, 3), [0, 0, 255], dtype=np.uint8)
        np.testing.assert_array_equal(top_half, expected_blue)

    def test_empty_mask_preserves_all_init(self):
        """Empty mask (all 0) means entire image preserved from init."""
        init = Image.new("RGB", (64, 64), color=(255, 0, 0))
        output = Image.new("RGB", (64, 64), color=(0, 0, 255))
        mask = np.zeros((64, 64), dtype=np.uint8)

        result = composite_inpainting(output, init, mask)
        result_arr = np.array(result)
        init_arr = np.array(init)
        np.testing.assert_array_equal(result_arr, init_arr)

    def test_full_mask_uses_all_output(self):
        """Full mask (all 255) means entire image from output."""
        init = Image.new("RGB", (64, 64), color=(255, 0, 0))
        output = Image.new("RGB", (64, 64), color=(0, 0, 255))
        mask = np.full((64, 64), 255, dtype=np.uint8)

        result = composite_inpainting(output, init, mask)
        result_arr = np.array(result)
        output_arr = np.array(output)
        np.testing.assert_array_equal(result_arr, output_arr)

    def test_dimension_mismatch_raises_error(self):
        """Mismatched dimensions between images and mask should raise ValueError."""
        init = Image.new("RGB", (64, 64), color=(255, 0, 0))
        output = Image.new("RGB", (64, 64), color=(0, 0, 255))
        wrong_mask = np.zeros((32, 32), dtype=np.uint8)

        with pytest.raises(ValueError, match="dimensions"):
            composite_inpainting(output, init, wrong_mask)

    def test_pixel_level_accuracy(self):
        """Test that compositing works at individual pixel level."""
        # Create images with unique per-pixel colors
        init_arr = np.zeros((4, 4, 3), dtype=np.uint8)
        init_arr[:, :, 0] = 100  # All red=100
        output_arr = np.zeros((4, 4, 3), dtype=np.uint8)
        output_arr[:, :, 2] = 200  # All blue=200

        init = Image.fromarray(init_arr)
        output = Image.fromarray(output_arr)

        # Checkerboard mask: alternating pixels
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, 0] = 255
        mask[0, 2] = 255
        mask[1, 1] = 255
        mask[1, 3] = 255

        result = composite_inpainting(output, init, mask)
        result_arr = np.array(result)

        # Masked pixels should be from output (blue=200)
        assert result_arr[0, 0, 2] == 200
        assert result_arr[0, 2, 2] == 200
        assert result_arr[1, 1, 2] == 200
        assert result_arr[1, 3, 2] == 200

        # Unmasked pixels should be from init (red=100)
        assert result_arr[0, 1, 0] == 100
        assert result_arr[0, 3, 0] == 100
        assert result_arr[1, 0, 0] == 100
        assert result_arr[1, 2, 0] == 100


# ---------------------------------------------------------------------------
# Tests: apply_mask_composite (high-level in-place function)
# ---------------------------------------------------------------------------


class TestApplyMaskComposite:
    def test_applies_composite_in_place(
        self, init_image: tuple[Path, Image.Image], output_image: tuple[Path, Image.Image], half_mask: Path
    ):
        """apply_mask_composite overwrites the output file with composited result."""
        init_path, _ = init_image
        output_path, _ = output_image

        apply_mask_composite(output_path, init_path, half_mask)

        # Re-read the output image
        result = Image.open(output_path)
        result_arr = np.array(result)

        # Top half (masked) should be blue (from original output)
        top_half = result_arr[:32, :, :]
        assert np.all(top_half[:, :, 2] == 255)  # Blue channel

        # Bottom half (unmasked) should be red (from init)
        bottom_half = result_arr[32:, :, :]
        assert np.all(bottom_half[:, :, 0] == 255)  # Red channel

    def test_full_mask_preserves_output(
        self, init_image: tuple[Path, Image.Image], output_image: tuple[Path, Image.Image], full_mask: Path
    ):
        """Full mask: entire output image preserved (all masked = use output)."""
        init_path, _ = init_image
        output_path, _ = output_image

        apply_mask_composite(output_path, init_path, full_mask)

        result = Image.open(output_path)
        result_arr = np.array(result)
        # All blue
        assert np.all(result_arr[:, :, 2] == 255)
        assert np.all(result_arr[:, :, 0] == 0)

    def test_empty_mask_replaces_with_init(
        self, init_image: tuple[Path, Image.Image], output_image: tuple[Path, Image.Image], empty_mask: Path
    ):
        """Empty mask: entire output replaced with init (all unmasked = use init)."""
        init_path, _ = init_image
        output_path, _ = output_image

        apply_mask_composite(output_path, init_path, empty_mask)

        result = Image.open(output_path)
        result_arr = np.array(result)
        # All red
        assert np.all(result_arr[:, :, 0] == 255)
        assert np.all(result_arr[:, :, 2] == 0)

    def test_mask_resize_during_composite(self, tmp_path: Path):
        """Mask is resized to match init image when dimensions differ (Requirement 5.8)."""
        # Init image: 64x64 red
        init_img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        init_path = tmp_path / "init.png"
        init_img.save(init_path)

        # Output image: 64x64 blue
        output_img = Image.new("RGB", (64, 64), color=(0, 0, 255))
        output_path = tmp_path / "output.png"
        output_img.save(output_path)

        # Mask: 32x32 (smaller — should be resized)
        mask_arr = np.zeros((32, 32), dtype=np.uint8)
        mask_arr[:16, :] = 255  # Top half masked
        mask_img = Image.fromarray(mask_arr, mode="L")
        mask_path = tmp_path / "small_mask.png"
        mask_img.save(mask_path)

        # Should not raise — mask gets resized to 64x64
        apply_mask_composite(output_path, init_path, mask_path)

        result = Image.open(output_path)
        assert result.size == (64, 64)


# ---------------------------------------------------------------------------
# Tests: Integration with generate() stub (inpainting via mask_path param)
# ---------------------------------------------------------------------------


class TestGenerateStubInpainting:
    """Test the krea2 generate() stub path with mask_path for inpainting."""

    def test_no_mask_is_standard_img2img(self, tmp_path: Path):
        """When no mask_path is provided, standard img2img (no compositing)."""
        from studio.core.vram_manager import VRAMManager

        vram_mgr = VRAMManager(total_vram=30_000_000_000)

        # Create init image
        init_img = Image.new("RGB", (512, 512), color=(255, 0, 0))
        init_path = tmp_path / "init.png"
        init_img.save(init_path)

        params = {
            "prompt": "test prompt",
            "width": 512,
            "height": 512,
            "init_image_path": str(init_path),
            "denoise_strength": 0.5,
            "_real_inference": False,
            "_vram_manager": vram_mgr,
        }

        from studio.models.backends.krea2 import generate

        results = list(generate(params))
        final = results[-1]
        assert isinstance(final, dict)
        assert "mask_path" not in final["params"]

    def test_mask_path_triggers_composite(self, tmp_path: Path, monkeypatch):
        """When mask_path is provided, inpainting composite is applied."""
        from studio.core.vram_manager import VRAMManager

        vram_mgr = VRAMManager(total_vram=30_000_000_000)

        # Redirect output dir to tmp
        monkeypatch.setattr("studio.config.Config.OUTPUT_DIR", tmp_path / "outputs")

        # Create init image (red)
        init_img = Image.new("RGB", (512, 512), color=(255, 0, 0))
        init_path = tmp_path / "init.png"
        init_img.save(init_path)

        # Create mask (bottom half masked)
        mask_arr = np.zeros((512, 512), dtype=np.uint8)
        mask_arr[256:, :] = 255
        mask_img = Image.fromarray(mask_arr, mode="L")
        mask_path = tmp_path / "mask.png"
        mask_img.save(mask_path)

        params = {
            "prompt": "test inpainting",
            "width": 512,
            "height": 512,
            "init_image_path": str(init_path),
            "denoise_strength": 0.7,
            "mask_path": str(mask_path),
            "_real_inference": False,
            "_vram_manager": vram_mgr,
        }

        from studio.models.backends.krea2 import generate

        results = list(generate(params))
        final = results[-1]
        assert isinstance(final, dict)
        assert final["params"]["mask_path"] == str(mask_path)

        # Verify the output image was composited
        output_path = final["images"][0]
        result_img = Image.open(output_path)
        result_arr = np.array(result_img)

        # Top half (unmasked) should be red (from init image)
        top_half = result_arr[:256, :, :]
        assert np.all(top_half[:, :, 0] == 255), "Unmasked top half should be red from init"
        assert np.all(top_half[:, :, 1] == 0)
        assert np.all(top_half[:, :, 2] == 0)

        # Bottom half (masked) should be from generated output (gray stub = 128,128,128)
        bottom_half = result_arr[256:, :, :]
        assert np.all(bottom_half[:, :, 0] == 128), "Masked bottom half should be from output"

    def test_mask_path_included_in_result_params(self, tmp_path: Path, monkeypatch):
        """mask_path should be persisted in result params for reproducibility."""
        from studio.core.vram_manager import VRAMManager

        vram_mgr = VRAMManager(total_vram=30_000_000_000)
        monkeypatch.setattr("studio.config.Config.OUTPUT_DIR", tmp_path / "outputs")

        init_img = Image.new("RGB", (512, 512), color=(255, 0, 0))
        init_path = tmp_path / "init.png"
        init_img.save(init_path)

        mask_arr = np.full((512, 512), 255, dtype=np.uint8)
        mask_img = Image.fromarray(mask_arr, mode="L")
        mask_path = tmp_path / "mask.png"
        mask_img.save(mask_path)

        params = {
            "prompt": "test",
            "width": 512,
            "height": 512,
            "init_image_path": str(init_path),
            "denoise_strength": 0.5,
            "mask_path": str(mask_path),
            "_real_inference": False,
            "_vram_manager": vram_mgr,
        }

        from studio.models.backends.krea2 import generate

        results = list(generate(params))
        final = results[-1]
        assert final["params"]["mask_path"] == str(mask_path)
        assert final["params"]["init_image_path"] == str(init_path)
        assert final["params"]["denoise_strength"] == 0.5
