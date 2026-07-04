"""UI Parameter Controls — Gradio component builders and validation.

Provides reusable control groups for the Generate tab:
- Prompt textbox (no internal template exposed)
- Sampler params: steps, seed, width, height
- Precision picker: bf16 / fp8_scaled
- Batch controls: batch_size (simultaneous/VRAM) and batch_count (sequential/queue)
- Checkpoint selector: dropdown populated from the model registry
- LoRA panel: dropdown, weight slider, add/refresh buttons, stack display
- Img2img controls: init image, denoise slider, mask editor, standard generation controls
- Parameter validation before passing to generation

Implements: Requirements 2.1, 2.2, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.3
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

PRECISION_OPTIONS = ["fp8_scaled", "bf16"]
PRECISION_DEFAULT = "fp8_scaled"

BATCH_SIZE_MIN, BATCH_SIZE_MAX = 1, 16
BATCH_SIZE_DEFAULT = 1
BATCH_COUNT_MIN, BATCH_COUNT_MAX = 1, 100
BATCH_COUNT_DEFAULT = 1

# Img2img denoise strength bounds
DENOISE_MIN = 0.0
DENOISE_MAX = 1.0
DENOISE_DEFAULT = 0.5
DENOISE_STEP = 0.05


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

    Options: fp8_scaled (default — fits 24 GB cards) or bf16 (full quality,
    needs a card with more than 25 GB of usable VRAM).
    """
    return gr.Radio(
        choices=PRECISION_OPTIONS,
        value=PRECISION_DEFAULT,
        label="Precision",
        info="fp8_scaled = ~13 GB VRAM, minimal quality loss (default). "
        "bf16 = full quality but ~25 GB VRAM — does not fit 24 GB cards.",
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


def create_checkpoint_selector() -> gr.Dropdown:
    """Create the checkpoint selector dropdown.

    Populates choices dynamically from the model registry — no hardcoded
    values. Each choice uses "{model_id}:{precision}" as the value
    (machine-readable identifier encoding both the model and its precision
    variant) and the registry's display_label as the user-facing label.

    Selection change does NOT trigger an immediate model reload. The new
    checkpoint is applied lazily on the next generation request.

    Returns:
        A Gradio Dropdown component with all available checkpoint options.
    """
    from studio.models.registry import list_checkpoint_options

    options = list_checkpoint_options()

    # Build choices as (label, value) tuples for Gradio
    choices: list[tuple[str, str]] = [
        (opt["display_label"], f"{opt['model_id']}:{opt['precision']}")
        for opt in options
    ]

    # Default to the first option if available
    default_value = choices[0][1] if choices else None

    return gr.Dropdown(
        choices=choices,
        value=default_value,
        label="Checkpoint",
        info="Select the model checkpoint for generation. Switching does not reload the model until the next generation.",
    )


# ---------------------------------------------------------------------------
# Img2Img Controls
# ---------------------------------------------------------------------------


def create_img2img_controls() -> tuple[
    gr.Image,
    gr.Slider,
    gr.ImageEditor,
    gr.Textbox,
    gr.Slider,
    gr.Number,
    gr.Slider,
    gr.Slider,
    gr.Radio,
]:
    """Create controls for the img2img workflow with inpainting support.

    Builds Gradio components for image-to-image generation:
    - An Image component for displaying/accepting the init image
    - A Slider for denoise_strength (0.0–1.0, default 0.5, step 0.05)
    - A Gradio ImageEditor for mask painting (brush tool with adjustable size,
      clear button, semi-transparent overlay on the init image)
    - Standard generation controls matching txt2img: prompt, seed, steps,
      width, height, precision

    The mask editor uses Gradio's built-in brush tool which provides:
    - Adjustable brush size
    - Clear button to reset mask
    - Semi-transparent overlay rendering on the canvas

    Returns:
        Tuple of (init_image, denoise_slider, mask_editor, prompt, steps,
        seed, width, height, precision).
    """
    # Init image component — accepts upload or receives from "Send to img2img"
    init_image = gr.Image(
        label="Init Image",
        type="filepath",
        height=512,
        sources=["upload", "clipboard"],
        interactive=True,
        show_download_button=False,
    )

    # Denoise strength slider (Requirement 4.2)
    denoise_slider = gr.Slider(
        minimum=DENOISE_MIN,
        maximum=DENOISE_MAX,
        value=DENOISE_DEFAULT,
        step=DENOISE_STEP,
        label="Denoise Strength",
        info="Controls how much of the init image is preserved (0.0 = unchanged, 1.0 = full regeneration).",
    )

    # Mask editor with brush tool (Requirements 5.1, 5.2, 5.3, 5.4)
    # Gradio's ImageEditor provides built-in brush with adjustable size,
    # clear button, and semi-transparent overlay rendering.
    mask_editor = gr.ImageEditor(
        label="Mask Editor",
        type="filepath",
        height=512,
        brush=gr.Brush(
            default_size=30,
            colors=["#FFFFFF"],
            color_mode="fixed",
        ),
        eraser=gr.Eraser(default_size=30),
        interactive=True,
        sources=[],
    )

    # Standard generation controls (same as txt2img — Requirement 4.3)
    prompt = gr.Textbox(
        label="Prompt",
        placeholder="Describe the image you want to generate...",
        lines=3,
        max_lines=8,
        info="Enter a text description. The model's internal template is applied automatically.",
    )

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

    precision = gr.Radio(
        choices=PRECISION_OPTIONS,
        value=PRECISION_DEFAULT,
        label="Precision",
        info="fp8_scaled = ~13 GB VRAM, minimal quality loss (default). "
        "bf16 = full quality but ~25 GB VRAM — does not fit 24 GB cards.",
    )

    return (
        init_image,
        denoise_slider,
        mask_editor,
        prompt,
        steps,
        seed,
        width,
        height,
        precision,
    )


# ---------------------------------------------------------------------------
# LoRA Panel
# ---------------------------------------------------------------------------

# LoRA weight slider bounds
LORA_WEIGHT_MIN = 0.0
LORA_WEIGHT_MAX = 2.0
LORA_WEIGHT_DEFAULT = 1.0
LORA_WEIGHT_STEP = 0.05


def create_lora_panel() -> tuple[gr.Dropdown, gr.Slider, gr.Button, gr.Button, gr.Dataframe, gr.Textbox]:
    """Create the LoRA panel for managing multi-LoRA stacking.

    The panel contains:
    - A Dropdown populated with available LoRA files (from scan_loras)
    - A weight Slider (0.0–2.0, default 1.0, step 0.05)
    - An "Add LoRA" button to add the selected LoRA to the stack
    - A "Refresh" button to rescan the loras directory
    - A Dataframe displaying the current LoRA stack (filename + weight)
    - A hidden Textbox holding the LoRA stack JSON state

    The refresh button is wired to rescan the loras directory and update
    the dropdown choices. The add button and stack management handlers
    are wired in a later integration task (9.3 / 10.1).

    Returns:
        Tuple of (lora_dropdown, weight_slider, add_button, refresh_button,
        stack_display, stack_state).
    """
    from studio.core.lora_manager import get_loras_dir, scan_loras

    # Scan for available LoRAs on initial panel creation
    loras_dir = get_loras_dir()
    available_loras = scan_loras(loras_dir)

    # Build guidance message for empty state
    no_loras_info = (
        f"No LoRAs found. Place .safetensors files in: {loras_dir}"
        if not available_loras
        else "Select a LoRA to add to the generation stack."
    )

    # Dropdown for selecting available LoRAs
    lora_dropdown = gr.Dropdown(
        choices=available_loras,
        value=None,
        label="Available LoRAs",
        info=no_loras_info,
        allow_custom_value=False,
    )

    # Weight slider for the LoRA being added
    weight_slider = gr.Slider(
        minimum=LORA_WEIGHT_MIN,
        maximum=LORA_WEIGHT_MAX,
        value=LORA_WEIGHT_DEFAULT,
        step=LORA_WEIGHT_STEP,
        label="LoRA Weight",
        info="Strength multiplier for the selected LoRA (0.0 = no effect, 2.0 = maximum).",
    )

    # Add button to add the selected LoRA + weight to the stack
    add_button = gr.Button(
        value="Add LoRA",
        variant="secondary",
        size="sm",
    )

    # Refresh button to rescan the loras directory
    refresh_button = gr.Button(
        value="🔄 Refresh",
        variant="secondary",
        size="sm",
    )

    # Dataframe displaying the current LoRA stack entries
    stack_display = gr.Dataframe(
        headers=["Filename", "Weight"],
        datatype=["str", "number"],
        value=[],
        label="LoRA Stack",
        interactive=False,
        row_count=(0, "dynamic"),
        col_count=(2, "fixed"),
    )

    # Hidden state component holding the LoRA stack JSON
    # (used for passing stack state between handlers)
    stack_state = gr.Textbox(
        value="[]",
        visible=False,
        label="LoRA Stack State",
    )

    # Wire the refresh button to rescan loras and update the dropdown
    def _on_refresh_loras() -> dict:
        """Rescan the loras directory and return updated dropdown choices."""
        refreshed = scan_loras(get_loras_dir())
        info_text = (
            f"No LoRAs found. Place .safetensors files in: {get_loras_dir()}"
            if not refreshed
            else "Select a LoRA to add to the generation stack."
        )
        return gr.Dropdown(choices=refreshed, value=None, info=info_text)

    refresh_button.click(
        fn=_on_refresh_loras,
        inputs=[],
        outputs=[lora_dropdown],
    )

    return lora_dropdown, weight_slider, add_button, refresh_button, stack_display, stack_state


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
