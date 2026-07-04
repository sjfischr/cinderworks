"""Image utility functions for Cinderworks Studio.

Standalone utilities for image manipulation used by generation backends.
These are intentionally decoupled from pipeline logic so they can be tested
independently.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def load_mask(mask_path: str | Path, target_width: int, target_height: int) -> np.ndarray:
    """Load a mask image and resize to match target dimensions if needed.

    The mask is treated as binary: any non-zero pixel is considered "masked"
    (region to regenerate). Zero pixels are "unmasked" (preserve from init image).

    Args:
        mask_path: Path to the mask image file.
        target_width: Width of the init image (target dimensions).
        target_height: Height of the init image (target dimensions).

    Returns:
        Binary numpy array of shape (target_height, target_width) with dtype uint8.
        Values are 0 (unmasked/preserve) or 255 (masked/regenerate).

    Raises:
        FileNotFoundError: If mask_path does not exist.
        ValueError: If mask_path cannot be opened as an image.
    """
    mask_path = Path(mask_path)
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    mask_img = Image.open(mask_path).convert("L")  # Grayscale

    # Resize mask to match init image dimensions if they differ (Requirement 5.8)
    if mask_img.size != (target_width, target_height):
        log.info(
            "Resizing mask from %dx%d to %dx%d to match init image",
            mask_img.width, mask_img.height, target_width, target_height,
        )
        mask_img = mask_img.resize((target_width, target_height), Image.NEAREST)

    # Convert to binary: any non-zero value becomes 255 (masked)
    mask_array = np.array(mask_img, dtype=np.uint8)
    mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)

    return mask_array


def composite_inpainting(
    output_image: Image.Image,
    init_image: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    """Composite inpainting result: preserve unmasked pixels from init image.

    After generation, pixels where the mask is 0 (unmasked) should come from
    the original init_image pixel-for-pixel; pixels where the mask is non-zero
    (masked) should come from the generated output.

    Args:
        output_image: The generated output image (PIL Image, RGB).
        init_image: The original init image (PIL Image, RGB).
        mask: Binary mask array of shape (H, W) with dtype uint8.
            0 = unmasked (preserve init), 255 = masked (use output).

    Returns:
        Composited PIL Image with unmasked regions from init_image and
        masked regions from output_image.

    Raises:
        ValueError: If dimensions don't match between images and mask.
    """
    # Ensure both images are RGB
    output_image = output_image.convert("RGB")
    init_image = init_image.convert("RGB")

    out_arr = np.array(output_image)
    init_arr = np.array(init_image)

    # Validate dimensions match
    if out_arr.shape[:2] != mask.shape:
        raise ValueError(
            f"Output image dimensions {out_arr.shape[:2]} don't match "
            f"mask dimensions {mask.shape}"
        )
    if init_arr.shape[:2] != mask.shape:
        raise ValueError(
            f"Init image dimensions {init_arr.shape[:2]} don't match "
            f"mask dimensions {mask.shape}"
        )

    # Expand mask to 3 channels for RGB compositing
    mask_3ch = np.stack([mask, mask, mask], axis=-1)

    # Composite: where mask is 0 (unmasked), use init_image; where non-zero, use output
    result_arr = np.where(mask_3ch > 0, out_arr, init_arr)

    return Image.fromarray(result_arr.astype(np.uint8), mode="RGB")


def apply_mask_composite(
    output_image_path: Path,
    init_image_path: str | Path,
    mask_path: str | Path,
) -> None:
    """Apply inpainting mask composite to a saved output image in-place.

    Loads the output image, init image, and mask, performs compositing,
    and saves the result back to the output image path.

    This is the high-level convenience function called by the generation
    backend after each image is saved.

    Args:
        output_image_path: Path to the generated output image (will be overwritten).
        init_image_path: Path to the original init image.
        mask_path: Path to the mask image.
    """
    init_img = Image.open(init_image_path).convert("RGB")
    output_img = Image.open(output_image_path).convert("RGB")

    # Load and resize mask to match init image dimensions
    mask = load_mask(mask_path, init_img.width, init_img.height)

    # Composite
    result = composite_inpainting(output_img, init_img, mask)

    # Save back in-place
    result.save(output_image_path)
    log.info(
        "Applied inpainting mask composite to %s (mask from %s)",
        output_image_path, mask_path,
    )
