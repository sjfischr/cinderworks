"""Unit tests for ui/controls.py — Parameter controls and validation.

Validates Requirements 5.5, 5.6, 6.3:
- Sampler params have correct bounds and defaults
- Batch controls are distinct with correct tooltips
- Validation rejects out-of-bounds values with specific messages
"""

import pytest

from studio.ui.controls import (
    STEPS_MIN,
    STEPS_MAX,
    STEPS_DEFAULT,
    SEED_MIN,
    SEED_MAX,
    SEED_DEFAULT,
    WIDTH_MIN,
    WIDTH_MAX,
    WIDTH_DEFAULT,
    HEIGHT_MIN,
    HEIGHT_MAX,
    HEIGHT_DEFAULT,
    SIZE_MULTIPLE,
    PRECISION_OPTIONS,
    PRECISION_DEFAULT,
    BATCH_SIZE_MIN,
    BATCH_SIZE_MAX,
    BATCH_SIZE_DEFAULT,
    BATCH_COUNT_MIN,
    BATCH_COUNT_MAX,
    BATCH_COUNT_DEFAULT,
    validate_ui_params,
)


# ---------------------------------------------------------------------------
# validate_ui_params — valid inputs
# ---------------------------------------------------------------------------


class TestValidateUIParamsValid:
    """Tests that valid parameters pass validation correctly."""

    def test_valid_defaults(self):
        """Default values produce a valid param dict."""
        result = validate_ui_params(
            prompt="A beautiful sunset",
            steps=STEPS_DEFAULT,
            seed=SEED_DEFAULT,
            width=WIDTH_DEFAULT,
            height=HEIGHT_DEFAULT,
            precision=PRECISION_DEFAULT,
            batch_size=BATCH_SIZE_DEFAULT,
            batch_count=BATCH_COUNT_DEFAULT,
        )
        assert result["prompt"] == "A beautiful sunset"
        assert result["steps"] == 8
        assert result["seed"] is None  # -1 → None (random)
        assert result["width"] == 1024
        assert result["height"] == 1024
        assert result["precision"] == "bf16"
        assert result["batch_size"] == 1
        assert result["batch_count"] == 1

    def test_explicit_seed_preserved(self):
        """Explicit seed (not -1) is preserved as-is."""
        result = validate_ui_params(
            prompt="test",
            steps=8,
            seed=42,
            width=1024,
            height=1024,
            precision="bf16",
            batch_size=1,
            batch_count=1,
        )
        assert result["seed"] == 42

    def test_seed_zero_is_valid(self):
        """Seed 0 is a valid explicit seed."""
        result = validate_ui_params(
            prompt="test",
            steps=8,
            seed=0,
            width=1024,
            height=1024,
            precision="bf16",
            batch_size=1,
            batch_count=1,
        )
        assert result["seed"] == 0

    def test_seed_max_is_valid(self):
        """Maximum seed value (2^32-1) is valid."""
        result = validate_ui_params(
            prompt="test",
            steps=8,
            seed=SEED_MAX,
            width=1024,
            height=1024,
            precision="bf16",
            batch_size=1,
            batch_count=1,
        )
        assert result["seed"] == SEED_MAX

    def test_fp8_precision(self):
        """fp8_scaled precision is accepted."""
        result = validate_ui_params(
            prompt="test",
            steps=8,
            seed=-1,
            width=1024,
            height=1024,
            precision="fp8_scaled",
            batch_size=1,
            batch_count=1,
        )
        assert result["precision"] == "fp8_scaled"

    def test_min_bounds(self):
        """All minimum bounds are accepted."""
        result = validate_ui_params(
            prompt="test",
            steps=STEPS_MIN,
            seed=-1,
            width=WIDTH_MIN,
            height=HEIGHT_MIN,
            precision="bf16",
            batch_size=BATCH_SIZE_MIN,
            batch_count=BATCH_COUNT_MIN,
        )
        assert result["steps"] == 1
        assert result["width"] == 512
        assert result["height"] == 512
        assert result["batch_size"] == 1
        assert result["batch_count"] == 1

    def test_max_bounds(self):
        """All maximum bounds are accepted."""
        result = validate_ui_params(
            prompt="test",
            steps=STEPS_MAX,
            seed=-1,
            width=WIDTH_MAX,
            height=HEIGHT_MAX,
            precision="bf16",
            batch_size=BATCH_SIZE_MAX,
            batch_count=BATCH_COUNT_MAX,
        )
        assert result["steps"] == 100
        assert result["width"] == 2048
        assert result["height"] == 2048
        assert result["batch_size"] == 16
        assert result["batch_count"] == 100

    def test_float_coercion_from_gradio(self):
        """Gradio sliders may pass float values; they should be coerced to int."""
        result = validate_ui_params(
            prompt="test",
            steps=8.0,
            seed=-1.0,
            width=1024.0,
            height=1024.0,
            precision="bf16",
            batch_size=1.0,
            batch_count=1.0,
        )
        assert result["steps"] == 8
        assert result["width"] == 1024

    def test_prompt_whitespace_stripped(self):
        """Prompt is stripped of leading/trailing whitespace."""
        result = validate_ui_params(
            prompt="  hello world  ",
            steps=8,
            seed=-1,
            width=1024,
            height=1024,
            precision="bf16",
            batch_size=1,
            batch_count=1,
        )
        assert result["prompt"] == "hello world"


# ---------------------------------------------------------------------------
# validate_ui_params — invalid inputs
# ---------------------------------------------------------------------------


class TestValidateUIParamsInvalid:
    """Tests that invalid parameters raise ValueError with specific messages."""

    def test_empty_prompt_rejected(self):
        """Empty prompt raises ValueError."""
        with pytest.raises(ValueError, match="prompt is required"):
            validate_ui_params(
                prompt="",
                steps=8,
                seed=-1,
                width=1024,
                height=1024,
                precision="bf16",
                batch_size=1,
                batch_count=1,
            )

    def test_whitespace_only_prompt_rejected(self):
        """Whitespace-only prompt raises ValueError."""
        with pytest.raises(ValueError, match="prompt is required"):
            validate_ui_params(
                prompt="   ",
                steps=8,
                seed=-1,
                width=1024,
                height=1024,
                precision="bf16",
                batch_size=1,
                batch_count=1,
            )

    def test_steps_below_min(self):
        """Steps below minimum raises ValueError."""
        with pytest.raises(ValueError, match="Steps must be between"):
            validate_ui_params(
                prompt="test",
                steps=0,
                seed=-1,
                width=1024,
                height=1024,
                precision="bf16",
                batch_size=1,
                batch_count=1,
            )

    def test_steps_above_max(self):
        """Steps above maximum raises ValueError."""
        with pytest.raises(ValueError, match="Steps must be between"):
            validate_ui_params(
                prompt="test",
                steps=101,
                seed=-1,
                width=1024,
                height=1024,
                precision="bf16",
                batch_size=1,
                batch_count=1,
            )

    def test_seed_below_min(self):
        """Seed below 0 (and not -1) raises ValueError."""
        with pytest.raises(ValueError, match="Seed must be"):
            validate_ui_params(
                prompt="test",
                steps=8,
                seed=-2,
                width=1024,
                height=1024,
                precision="bf16",
                batch_size=1,
                batch_count=1,
            )

    def test_seed_above_max(self):
        """Seed above 2^32-1 raises ValueError."""
        with pytest.raises(ValueError, match="Seed must be"):
            validate_ui_params(
                prompt="test",
                steps=8,
                seed=SEED_MAX + 1,
                width=1024,
                height=1024,
                precision="bf16",
                batch_size=1,
                batch_count=1,
            )

    def test_width_below_min(self):
        """Width below minimum raises ValueError."""
        with pytest.raises(ValueError, match="Width must be between"):
            validate_ui_params(
                prompt="test",
                steps=8,
                seed=-1,
                width=256,
                height=1024,
                precision="bf16",
                batch_size=1,
                batch_count=1,
            )

    def test_width_not_multiple_of_64(self):
        """Width not a multiple of 64 raises ValueError."""
        with pytest.raises(ValueError, match="Width must be a multiple of"):
            validate_ui_params(
                prompt="test",
                steps=8,
                seed=-1,
                width=600,
                height=1024,
                precision="bf16",
                batch_size=1,
                batch_count=1,
            )

    def test_height_below_min(self):
        """Height below minimum raises ValueError."""
        with pytest.raises(ValueError, match="Height must be between"):
            validate_ui_params(
                prompt="test",
                steps=8,
                seed=-1,
                width=1024,
                height=256,
                precision="bf16",
                batch_size=1,
                batch_count=1,
            )

    def test_height_not_multiple_of_64(self):
        """Height not a multiple of 64 raises ValueError."""
        with pytest.raises(ValueError, match="Height must be a multiple of"):
            validate_ui_params(
                prompt="test",
                steps=8,
                seed=-1,
                width=1024,
                height=1000,
                precision="bf16",
                batch_size=1,
                batch_count=1,
            )

    def test_invalid_precision(self):
        """Invalid precision raises ValueError."""
        with pytest.raises(ValueError, match="Precision must be one of"):
            validate_ui_params(
                prompt="test",
                steps=8,
                seed=-1,
                width=1024,
                height=1024,
                precision="fp32",
                batch_size=1,
                batch_count=1,
            )

    def test_batch_size_below_min(self):
        """Batch size below minimum raises ValueError."""
        with pytest.raises(ValueError, match="Batch size must be between"):
            validate_ui_params(
                prompt="test",
                steps=8,
                seed=-1,
                width=1024,
                height=1024,
                precision="bf16",
                batch_size=0,
                batch_count=1,
            )

    def test_batch_size_above_max(self):
        """Batch size above maximum raises ValueError."""
        with pytest.raises(ValueError, match="Batch size must be between"):
            validate_ui_params(
                prompt="test",
                steps=8,
                seed=-1,
                width=1024,
                height=1024,
                precision="bf16",
                batch_size=17,
                batch_count=1,
            )

    def test_batch_count_below_min(self):
        """Batch count below minimum raises ValueError."""
        with pytest.raises(ValueError, match="Batch count must be between"):
            validate_ui_params(
                prompt="test",
                steps=8,
                seed=-1,
                width=1024,
                height=1024,
                precision="bf16",
                batch_size=1,
                batch_count=0,
            )

    def test_batch_count_above_max(self):
        """Batch count above maximum raises ValueError."""
        with pytest.raises(ValueError, match="Batch count must be between"):
            validate_ui_params(
                prompt="test",
                steps=8,
                seed=-1,
                width=1024,
                height=1024,
                precision="bf16",
                batch_size=1,
                batch_count=101,
            )


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify constants match the spec."""

    def test_steps_bounds(self):
        assert STEPS_MIN == 1
        assert STEPS_MAX == 100
        assert STEPS_DEFAULT == 8

    def test_seed_bounds(self):
        assert SEED_MIN == 0
        assert SEED_MAX == 4294967295  # 2^32 - 1
        assert SEED_DEFAULT == -1

    def test_size_bounds(self):
        assert WIDTH_MIN == 512
        assert WIDTH_MAX == 2048
        assert WIDTH_DEFAULT == 1024
        assert HEIGHT_MIN == 512
        assert HEIGHT_MAX == 2048
        assert HEIGHT_DEFAULT == 1024
        assert SIZE_MULTIPLE == 64

    def test_precision_options(self):
        assert PRECISION_OPTIONS == ["bf16", "fp8_scaled"]
        assert PRECISION_DEFAULT == "bf16"

    def test_batch_bounds(self):
        assert BATCH_SIZE_MIN == 1
        assert BATCH_SIZE_MAX == 16
        assert BATCH_SIZE_DEFAULT == 1
        assert BATCH_COUNT_MIN == 1
        assert BATCH_COUNT_MAX == 100
        assert BATCH_COUNT_DEFAULT == 1
