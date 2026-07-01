"""UI Parameter Controls — Gradio component builders and validation.

Provides reusable control groups for the Generate tab:
- Prompt textbox (no internal template exposed)
- Sampler params: steps, seed, width, height
- Precision picker: bf16 / fp8_scaled
- Batch controls: batch_size (simultaneous/VRAM) and batch_count (sequential/queue)
- Parameter validation before passing to generation

Implements: Requirements 5.5, 5.6, 6.3
"""

from __future__ import annotations

from typing import Any

import gradio as gr


# ---------------------------------------------------------------------------
# Parameter Bounds (mirrored from backend for UI-layer validation)
# ---------------------------------------------------------------------------

STEPS_MIN, STEPS_MAX = 1, 100
STEPS_DEFAULT = 8

SEED_MIN, SEED_MAX = 0, 2**32 - 1
SEED_DEFAULT = -1  # -1 means random

WIDTH_MIN, WIDTH_MAX = 512, 2048
WIDTH_DEFAULT = 1024
HEIGHT_MIN, HEIGHT_MAX = 512, 2048
HEIGHT_DEFAULT = 1024
SIZE_MULTIPLE = 64

PRECISION_OPTIONS = ["bf16", "fp8_scaled"]
PRECISION_DEFAULT = "bf16"

BATCH_SIZE_MIN, BATCH_SIZE_MAX = 1, 16
BATCH_SIZE_DEFAULT = 1
BATCH_COUNT_MIN, BATCH_COUNT_MAX = 1, 100
BATCH_COUNT_DEFAULT = 1


# ---------------------------------------------------------------------------
# Component Builders
# ---------------------------------------------------------------------------


def create_prompt_controls() -> gr.Textbox:
    """Create the prompt textbox.

    The internal prompt template is NOT exposed — only a plain text input.
    """
    return gr.Textbox(
        label="Prompt",
        placeholder="Describe the image you want to generate...",
        lines=3,
        max_lines=8,
        info="Enter a text description. The model's internal template is applied automatically.",
    )


def create_sampler_controls() -> tuple[gr.Slider, gr.Number, gr.Slider, gr.Slider]:
    """Create sampler parameter controls.

    Returns:
        Tuple of (steps_slider, seed_input, width_slider, height_slider).
    """
    steps = gr.Slider(
        minimum=STEPS_MIN,
        maximum=STEPS_MAX,
        value=STEPS_DEFAULT,
        step=1,
        label="Steps",
        info="Number of sampling steps. Turbo default is 8. More steps can improve detail but takes longer.",
    )

    seed = gr.Number(
        value=SEED_DEFAULT,
        label="Seed",
        precision=0,
        minimum=-(1),  # Allow -1 for random
        maximum=SEED_MAX,
        info="Seed for reproducibility. Use -1 for a random seed each generation.",
    )

    width = gr.Slider(
        minimum=WIDTH_MIN,
        maximum=WIDTH_MAX,
        value=WIDTH_DEFAULT,
        step=SIZE_MULTIPLE,
        label="Width",
        info="Image width in pixels. Must be a multiple of 64.",
    )

    height = gr.Slider(
        minimum=HEIGHT_MIN,
        maximum=HEIGHT_MAX,
        value=HEIGHT_DEFAULT,
        step=SIZE_MULTIPLE,
        label="Height",
        info="Image height in pixels. Must be a multiple of 64.",
    )

    return steps, seed, width, height


def create_precision_picker() -> gr.Radio:
    """Create the precision picker control.

    Options: bf16 (full quality, more VRAM) or fp8_scaled (lower VRAM, slight quality trade-off).
    """
    return gr.Radio(
        choices=PRECISION_OPTIONS,
        value=PRECISION_DEFAULT,
        label="Precision",
        info="bf16 = full quality (~25 GB VRAM). fp8_scaled = reduced VRAM (~13 GB) with minimal quality loss.",
    )


def create_batch_controls() -> tuple[gr.Slider, gr.Slider]:
    """Create batch size and batch count controls.

    Returns:
        Tuple of (batch_size_slider, batch_count_slider).
    """
    batch_size = gr.Slider(
        minimum=BATCH_SIZE_MIN,
        maximum=BATCH_SIZE_MAX,
        value=BATCH_SIZE_DEFAULT,
        step=1,
        label="Batch Size",
        info="Number of images generated simultaneously (VRAM-bound). Higher values use more GPU memory.",
    )

    batch_count = gr.Slider(
        minimum=BATCH_COUNT_MIN,
        maximum=BATCH_COUNT_MAX,
        value=BATCH_COUNT_DEFAULT,
        step=1,
        label="Batch Count",
        info="Number of sequential batches to run (queue-bound). Does not increase VRAM usage.",
    )

    return batch_size, batch_count


# ---------------------------------------------------------------------------
# Parameter Validation
# ---------------------------------------------------------------------------


def validate_ui_params(
    prompt: str,
    steps: int | float,
    seed: int | float,
    width: int | float,
    height: int | float,
    precision: str,
    batch_size: int | float,
    batch_count: int | float,
) -> dict[str, Any]:
    """Validate all UI parameter bounds before passing to generation.

    Args:
        prompt: The text prompt.
        steps: Number of sampling steps.
        seed: Seed value (-1 for random, 0–2^32-1 for explicit).
        width: Image width in pixels.
        height: Image height in pixels.
        precision: Model precision ('bf16' or 'fp8_scaled').
        batch_size: Number of simultaneous images per batch.
        batch_count: Number of sequential batches.

    Returns:
        A validated parameter dict ready for generation. Seed of -1 is
        converted to None (signals random seed to the backend).

    Raises:
        ValueError: If any parameter is out of bounds, with a specific message.
    """
    # Coerce floats from Gradio sliders to int
    steps = int(steps)
    seed = int(seed)
    width = int(width)
    height = int(height)
    batch_size = int(batch_size)
    batch_count = int(batch_count)

    # Prompt validation
    if not prompt or not prompt.strip():
        raise ValueError(
            "A prompt is required — please enter a text description of the image you want to generate."
        )

    # Steps validation
    if steps < STEPS_MIN or steps > STEPS_MAX:
        raise ValueError(
            f"Steps must be between {STEPS_MIN} and {STEPS_MAX}, got {steps}."
        )

    # Seed validation: -1 means random, otherwise must be in [0, 2^32-1]
    if seed != -1 and (seed < SEED_MIN or seed > SEED_MAX):
        raise ValueError(
            f"Seed must be -1 (random) or between {SEED_MIN} and {SEED_MAX}, got {seed}."
        )

    # Width validation
    if width < WIDTH_MIN or width > WIDTH_MAX:
        raise ValueError(
            f"Width must be between {WIDTH_MIN} and {WIDTH_MAX}, got {width}."
        )
    if width % SIZE_MULTIPLE != 0:
        raise ValueError(
            f"Width must be a multiple of {SIZE_MULTIPLE}, got {width}."
        )

    # Height validation
    if height < HEIGHT_MIN or height > HEIGHT_MAX:
        raise ValueError(
            f"Height must be between {HEIGHT_MIN} and {HEIGHT_MAX}, got {height}."
        )
    if height % SIZE_MULTIPLE != 0:
        raise ValueError(
            f"Height must be a multiple of {SIZE_MULTIPLE}, got {height}."
        )

    # Precision validation
    if precision not in PRECISION_OPTIONS:
        raise ValueError(
            f"Precision must be one of {PRECISION_OPTIONS}, got '{precision}'."
        )

    # Batch size validation
    if batch_size < BATCH_SIZE_MIN or batch_size > BATCH_SIZE_MAX:
        raise ValueError(
            f"Batch size must be between {BATCH_SIZE_MIN} and {BATCH_SIZE_MAX}, got {batch_size}."
        )

    # Batch count validation
    if batch_count < BATCH_COUNT_MIN or batch_count > BATCH_COUNT_MAX:
        raise ValueError(
            f"Batch count must be between {BATCH_COUNT_MIN} and {BATCH_COUNT_MAX}, got {batch_count}."
        )

    # Build validated dict; convert seed -1 → None for backend
    return {
        "prompt": prompt.strip(),
        "steps": steps,
        "seed": None if seed == -1 else seed,
        "width": width,
        "height": height,
        "precision": precision,
        "batch_size": batch_size,
        "batch_count": batch_count,
    }
