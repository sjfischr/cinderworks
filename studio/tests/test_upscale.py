"""Tests for studio/models/upscale.py — Lanczos path and validation.

Model-based upscaling (spandrel/Real-ESRGAN) requires GPU + weights and
is exercised manually; these tests cover the always-available paths.
"""

from __future__ import annotations

import pytest
from PIL import Image

from studio.models import upscale


@pytest.fixture()
def sample_image(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (64, 48), color=(120, 30, 200)).save(path)
    return path


class TestListMethods:
    def test_lanczos_always_first(self):
        methods = upscale.list_methods()
        assert methods[0] == upscale.METHOD_LANCZOS

    def test_realesrgan_listed(self):
        assert upscale.METHOD_REALESRGAN in upscale.list_methods()


class TestLanczosUpscale:
    def test_doubles_dimensions(self, sample_image):
        out = upscale.upscale(sample_image, upscale.METHOD_LANCZOS, 2.0)
        with Image.open(out) as img:
            assert img.size == (128, 96)

    def test_fractional_scale(self, sample_image):
        out = upscale.upscale(sample_image, upscale.METHOD_LANCZOS, 1.5)
        with Image.open(out) as img:
            assert img.size == (96, 72)

    def test_output_in_upscaled_dir(self, sample_image):
        out = upscale.upscale(sample_image, upscale.METHOD_LANCZOS, 2.0)
        assert out.parent.name == "upscaled"
        assert out.suffix == ".png"


class TestValidation:
    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            upscale.upscale(tmp_path / "nope.png", upscale.METHOD_LANCZOS, 2.0)

    def test_scale_out_of_bounds_rejected(self, sample_image):
        with pytest.raises(ValueError, match="Scale"):
            upscale.upscale(sample_image, upscale.METHOD_LANCZOS, 5.0)

    def test_unknown_method_rejected(self, sample_image):
        with pytest.raises(ValueError, match="Unknown"):
            upscale.upscale(sample_image, "Bicubic", 2.0)

    def test_realesrgan_without_model_gives_plain_error(self, sample_image):
        if upscale.model_available():
            pytest.skip("upscaler model present on this machine")
        with pytest.raises(RuntimeError, match="not downloaded"):
            upscale.upscale(sample_image, upscale.METHOD_REALESRGAN, 2.0)
