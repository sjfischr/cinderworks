"""UI Event Handlers — Error boundary for all user-facing actions.

Every handler is wrapped in try/except. Exceptions are caught, logged
to the log file, and translated to plain-language ❌ strings. Tracebacks
NEVER reach the UI — they go only to the log file.

Implements: Requirements 4.4, 4.5, 5.8, 8.3, 8.4, 8.5, 11.1, 11.3
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any, Generator

from studio.config import Config

# ---------------------------------------------------------------------------
# Logging setup — tracebacks go to file, never to the user
# ---------------------------------------------------------------------------

LOG_PATH: Path = Config.BASE_DIR / "cinderworks.log"

# Configure file handler for the studio logger hierarchy
_file_handler = logging.FileHandler(str(LOG_PATH), encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
)

# Attach to the root 'studio' logger so all submodules inherit it
_studio_logger = logging.getLogger("studio")
_studio_logger.addHandler(_file_handler)
_studio_logger.setLevel(logging.DEBUG)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Friendly error mapper
# ---------------------------------------------------------------------------

import re

# Patterns that must never appear in user-facing output
_TRACEBACK_PATTERN = re.compile(r'File ".*", line \d+')
_TRACEBACK_HEADER_PATTERN = re.compile(r"Traceback \(most recent call last\)")
_EXCEPTION_CLASS_PATTERN = re.compile(
    r"\b(MemoryError|RuntimeError|TypeError|KeyError|IndexError|OSError|IOError|"
    r"AttributeError|ImportError|PermissionError|NotImplementedError|"
    r"StopIteration|ArithmeticError|ZeroDivisionError|UnicodeError|"
    r"BufferError|LookupError|ConnectionError|TimeoutError|"
    r"FileNotFoundError)\s*:"
)


def _contains_traceback_or_class_name(text: str) -> bool:
    """Return True if text contains traceback frames or raw exception class names."""
    if _TRACEBACK_PATTERN.search(text):
        return True
    if _TRACEBACK_HEADER_PATTERN.search(text):
        return True
    if _EXCEPTION_CLASS_PATTERN.search(text):
        return True
    return False


def friendly(exc: Exception) -> str:
    """Map an exception to a plain-language message for the user.

    Known exception classes get specific, actionable messages.
    Unknown exceptions get a generic message. ALL messages include
    the log file path. NEVER includes exception class names or
    traceback text.

    Args:
        exc: The caught exception.

    Returns:
        A user-facing string (without the ❌ prefix — caller adds that).
    """
    log_ref = f"Details logged to: {LOG_PATH}"

    if isinstance(exc, MemoryError):
        return f"Not enough VRAM — try lowering batch size or switching to fp8_scaled. {log_ref}"

    if isinstance(exc, (ConnectionError, TimeoutError)):
        return f"Cannot reach Hugging Face — check your internet connection. {log_ref}"

    if isinstance(exc, FileNotFoundError):
        return f"Model file not found — try re-downloading. {log_ref}"

    if isinstance(exc, ValueError):
        # ValueError messages from validation are already user-friendly,
        # but we must sanitize to ensure no traceback/class-name leakage.
        msg = str(exc)
        if _contains_traceback_or_class_name(msg):
            return f"Invalid input — check your parameters. {log_ref}"
        return f"{msg}. {log_ref}"

    # Check for OOM keywords in RuntimeError (torch OOM, etc.)
    if isinstance(exc, RuntimeError):
        msg_lower = str(exc).lower()
        if "out of memory" in msg_lower or "oom" in msg_lower:
            return f"Not enough VRAM — try lowering batch size or switching to fp8_scaled. {log_ref}"

    # Unknown error — generic message
    return f"Something went wrong. {log_ref}"


# ---------------------------------------------------------------------------
# Spinner helpers (Gradio update dicts)
# ---------------------------------------------------------------------------


def spinner_on() -> dict[str, Any]:
    """Return a Gradio update dict to show a loading/spinner state."""
    return {"visible": True, "value": "⏳ Working..."}


def spinner_off() -> dict[str, Any]:
    """Return a Gradio update dict to hide the spinner/loading state."""
    return {"visible": False, "value": ""}


# ---------------------------------------------------------------------------
# Event Handlers
# ---------------------------------------------------------------------------


def on_generate(
    prompt: str,
    steps: int | float,
    seed: int | float,
    width: int | float,
    height: int | float,
    precision: str,
    batch_size: int | float,
    batch_count: int | float,
    lora_stack_json: str = "[]",
    checkpoint_id: str = "",
) -> Generator[tuple[str, list[str]], None, None]:
    """Generate images: check readiness → validate → generate → persist → display.

    Yields (progress_text, gallery_images) tuples. During progress, gallery
    stays unchanged (empty list). On completion, gallery receives image paths.
    If not ready or validation fails, yields an ❌ error message with empty gallery.
    Never raises — all exceptions are caught and returned as plain text.

    Args:
        prompt: Text prompt for generation.
        steps: Number of sampling steps.
        seed: Seed value (-1 for random).
        width: Image width in pixels.
        height: Image height in pixels.
        precision: Model precision ('bf16' or 'fp8_scaled').
        batch_size: Images per parallel batch.
        batch_count: Number of sequential batches.
        lora_stack_json: JSON string representing the current LoRA stack
            (list of {path, weight} dicts). Defaults to empty stack "[]".
        checkpoint_id: Checkpoint selector value formatted as
            "{model_id}:{precision}" (e.g. "krea2-turbo:fp8_scaled").
            If empty, defaults to "krea2-turbo" with the precision from
            the precision picker.

    Yields:
        Tuples of (progress_string, gallery_image_paths).
    """
    try:
        # Import dependencies inside handler to keep module import lightweight
        from studio.core import system_check
        from studio.models import registry
        from studio.db import db
        from studio.ui.controls import validate_ui_params

        # 1. Readiness check
        if not system_check.is_ready_to_generate():
            banner = system_check.get_readiness_banner()
            reasons = banner.get("value", "System is not ready to generate.")
            yield f"❌ {reasons}", []
            return

        # 2. Validate parameters
        validated = validate_ui_params(
            prompt=prompt,
            steps=steps,
            seed=seed,
            width=width,
            height=height,
            precision=precision,
            batch_size=batch_size,
            batch_count=batch_count,
        )

        # 3. Resolve model_id and precision from checkpoint selector
        #    Format: "{model_id}:{precision}" e.g. "krea2-turbo:fp8_scaled"
        if checkpoint_id and ":" in str(checkpoint_id):
            model_id, resolved_precision = str(checkpoint_id).split(":", 1)
        else:
            model_id = "krea2-turbo"
            resolved_precision = validated["precision"]

        # Override precision with the checkpoint-selected precision
        validated["precision"] = resolved_precision
        validated["model_id"] = model_id

        # 4. Parse LoRA stack JSON and add to params if non-empty
        lora_stack: list[dict[str, Any]] = []
        if lora_stack_json and lora_stack_json.strip() not in ("", "[]"):
            try:
                lora_stack = json.loads(lora_stack_json)
            except (json.JSONDecodeError, TypeError):
                log.warning("Invalid LoRA stack JSON: %s", lora_stack_json)
                lora_stack = []

        if lora_stack:
            validated["lora_stack"] = lora_stack

        # 5. Run generation via registry, yielding progress updates
        yield "Encoding prompt...", []
        images: list[Any] = []
        resolved_params: dict[str, Any] = {}

        for update in registry.run_generation(model_id, validated):
            if isinstance(update, dict):
                # Final result dict from backend
                images = update.get("images", [])
                resolved_params = update.get("params", validated)
            else:
                # Progress string from backend
                yield str(update), []

        # 6. Persist job to database
        actual_seed = resolved_params.get("seed", validated.get("seed", 0))
        duration_ms = resolved_params.get("duration_ms")

        # Build image path list for gallery display
        image_paths: list[str] = []
        artifacts = []
        for img in images:
            img_path = str(img.get("path", "")) if isinstance(img, dict) else str(img)
            image_paths.append(img_path)
            artifacts.append({
                "path": img_path,
                "seed": img.get("seed", actual_seed) if isinstance(img, dict) else actual_seed,
                "width": validated["width"],
                "height": validated["height"],
            })

        try:
            db.init_db()
            # Use resolved_params from the backend for params_json — it includes
            # all reproducibility fields: model_id, precision, denoise_strength,
            # init_image_path, mask_path, lora_stack (when applicable).
            # Fall back to validated (UI input) if backend didn't return params.
            persist_params = resolved_params if resolved_params else validated
            # Ensure model_id + precision are stored in the job record (Req 3.8)
            persist_params.setdefault("model_id", model_id)
            persist_params.setdefault("precision", resolved_precision)
            job_id = db.create_job(
                prompt=validated["prompt"],
                params_json=json.dumps(persist_params),
                seed=actual_seed,
                model_id=model_id,
                duration_ms=duration_ms,
                status="complete",
                artifacts=artifacts,
            )
            yield f"✅ Generation complete — Job #{job_id} saved.", image_paths
        except Exception as db_exc:
            # DB failure: report but don't discard generated images
            log.exception("Database write failed after generation")
            yield f"⚠️ Images generated but save failed: {friendly(db_exc)}", image_paths

    except Exception as e:
        log.exception("on_generate failed")
        yield f"❌ {friendly(e)}", []


def on_generate_img2img(
    prompt: str,
    steps: int | float,
    seed: int | float,
    width: int | float,
    height: int | float,
    precision: str,
    init_image: str | None,
    denoise_strength: float,
    mask_data: Any | None,
    lora_stack_json: str,
    checkpoint_id: str,
) -> Generator[tuple[str, list[str]], None, None]:
    """Generate images using img2img mode with optional inpainting mask.

    Validates the init image is present, extracts model_id and precision
    from the checkpoint selector value, parses the LoRA stack, and
    delegates to registry.run_generation() in img2img mode.

    Follows the Phase 1 error boundary pattern: exceptions are caught,
    logged with full tracebacks, and translated to plain-language messages.

    Args:
        prompt: Text prompt for generation.
        steps: Number of sampling steps.
        seed: Seed value (-1 for random).
        width: Image width in pixels.
        height: Image height in pixels.
        precision: Model precision ('bf16' or 'fp8_scaled').
        init_image: File path to the init image (from Gradio Image component).
        denoise_strength: How much of the init image is preserved (0.0–1.0).
        mask_data: Mask data from the Gradio ImageEditor (dict with 'composite'
            key, or None if no mask painted).
        lora_stack_json: JSON string representing the current LoRA stack
            (list of {path, weight} dicts).
        checkpoint_id: Checkpoint selector value formatted as
            "{model_id}:{precision}" (e.g. "krea2-turbo:fp8_scaled").

    Yields:
        Tuples of (progress_string, gallery_image_paths).

    Implements: Requirements 4.1, 4.4, 4.8, 5.5, 5.6
    """
    try:
        from studio.models import registry
        from studio.db import db

        # 1. Validate init image is present (Requirement 4.8)
        if not init_image:
            yield "❌ Choose an image for img2img first.", []
            return

        # Verify the init image file exists on disk
        init_image_path = Path(init_image)
        if not init_image_path.is_file():
            yield "❌ The source image is no longer available — it may have been deleted.", []
            return

        # 2. Extract model_id and precision from checkpoint_id
        #    Format: "{model_id}:{precision}" e.g. "krea2-turbo:fp8_scaled"
        if checkpoint_id and ":" in checkpoint_id:
            model_id, resolved_precision = checkpoint_id.split(":", 1)
        else:
            # Fallback to default if checkpoint_id is malformed
            model_id = "krea2-turbo"
            resolved_precision = precision

        # 3. Parse LoRA stack JSON
        lora_stack: list[dict[str, Any]] = []
        if lora_stack_json and lora_stack_json.strip() not in ("", "[]"):
            try:
                lora_stack = json.loads(lora_stack_json)
            except (json.JSONDecodeError, TypeError):
                log.warning("Invalid LoRA stack JSON: %s", lora_stack_json)
                lora_stack = []

        # 4. Extract mask path from Gradio ImageEditor data
        #    Gradio ImageEditor returns a dict with 'composite' key pointing
        #    to the composited mask image filepath, or None if no mask painted.
        mask_path: str | None = None
        if mask_data is not None:
            if isinstance(mask_data, dict):
                # Gradio ImageEditor dict format
                composite = mask_data.get("composite")
                if composite and Path(str(composite)).is_file():
                    mask_path = str(composite)
            elif isinstance(mask_data, str) and Path(mask_data).is_file():
                # Direct file path (alternative format)
                mask_path = mask_data

        # 5. Coerce and validate parameters
        steps_int = int(steps)
        seed_int = int(seed)
        width_int = int(width)
        height_int = int(height)
        denoise_float = float(denoise_strength)

        # 6. Build params dict for registry.run_generation()
        params: dict[str, Any] = {
            "prompt": prompt.strip() if prompt else "",
            "steps": steps_int,
            "seed": None if seed_int == -1 else seed_int,
            "width": width_int,
            "height": height_int,
            "precision": resolved_precision,
            "model_id": model_id,
            "init_image_path": str(init_image_path),
            "denoise_strength": denoise_float,
        }

        # Add mask path if present (inpainting mode)
        if mask_path:
            params["mask_path"] = mask_path

        # Add LoRA stack if non-empty
        if lora_stack:
            params["lora_stack"] = lora_stack

        # 7. Run generation via registry, yielding progress updates
        yield "Encoding prompt...", []
        images: list[Any] = []
        resolved_params: dict[str, Any] = {}

        for update in registry.run_generation(model_id, params):
            if isinstance(update, dict):
                # Final result dict from backend
                images = update.get("images", [])
                resolved_params = update.get("params", params)
            else:
                # Progress string from backend
                yield str(update), []

        # 8. Persist job to database
        actual_seed = resolved_params.get("seed", params.get("seed", 0))
        duration_ms = resolved_params.get("duration_ms")

        # Build image path list for gallery display
        image_paths: list[str] = []
        artifacts = []
        for img in images:
            img_path = str(img.get("path", "")) if isinstance(img, dict) else str(img)
            image_paths.append(img_path)
            artifacts.append({
                "path": img_path,
                "seed": img.get("seed", actual_seed) if isinstance(img, dict) else actual_seed,
                "width": width_int,
                "height": height_int,
            })

        try:
            db.init_db()
            persist_params = resolved_params if resolved_params else params
            job_id = db.create_job(
                prompt=params["prompt"],
                params_json=json.dumps(persist_params),
                seed=actual_seed,
                model_id=model_id,
                duration_ms=duration_ms,
                status="complete",
                artifacts=artifacts,
            )
            yield f"✅ Img2img generation complete — Job #{job_id} saved.", image_paths
        except Exception as db_exc:
            # DB failure: report but don't discard generated images
            log.exception("Database write failed after img2img generation")
            yield f"⚠️ Images generated but save failed: {friendly(db_exc)}", image_paths

    except Exception as e:
        log.exception("on_generate_img2img failed")
        yield f"❌ {friendly(e)}", []


# ---------------------------------------------------------------------------
# LoRA Management Handlers
# ---------------------------------------------------------------------------


def on_refresh_loras() -> dict:
    """Rescan the loras directory and return updated dropdown choices.

    Calls scan_loras() to re-discover all valid .safetensors files in
    the configured loras directory. Returns a Gradio update dict for
    the LoRA dropdown component.

    Returns:
        A gr.Dropdown update dict with refreshed choices and info text.

    Implements: Requirements 1.1, 1.5
    """
    try:
        from studio.core.lora_manager import get_loras_dir, scan_loras
        import gradio as gr

        loras_dir = get_loras_dir()
        available = scan_loras(loras_dir)

        info_text = (
            f"No LoRAs found. Place .safetensors files in: {loras_dir}"
            if not available
            else "Select a LoRA to add to the generation stack."
        )

        return gr.Dropdown(choices=available, value=None, info=info_text)

    except Exception as e:
        log.exception("on_refresh_loras failed")
        import gradio as gr

        return gr.Dropdown(choices=[], value=None, info=f"❌ {friendly(e)}")


def _stack_json_to_dataframe(stack_json: str) -> list[list[Any]]:
    """Convert a LoRA stack JSON string to dataframe rows.

    Each row is [filename, weight] where filename is the stem extracted
    from the path.

    Args:
        stack_json: JSON string of the LoRA stack (list of {path, weight} dicts).

    Returns:
        List of [filename, weight] rows for the Gradio Dataframe.
    """
    if not stack_json or stack_json.strip() in ("", "null", "[]"):
        return []

    try:
        entries = json.loads(stack_json)
        if not isinstance(entries, list):
            return []
    except (json.JSONDecodeError, TypeError):
        return []

    rows: list[list[Any]] = []
    for entry in entries:
        path_str = entry.get("path", "")
        weight = entry.get("weight", 1.0)
        # Extract filename stem from path for display
        filename = Path(path_str).stem if path_str else "unknown"
        rows.append([filename, weight])

    return rows


def on_add_lora(lora_name: str, weight: float, current_stack_json: str) -> tuple[str, str, list[list[Any]]]:
    """Add a LoRA to the stack, rejecting duplicates.

    Thin handler that calls through to lora_manager.add_lora_to_stack()
    and formats the return values for Gradio components.

    Args:
        lora_name: Display name (stem) of the LoRA file to add.
        weight: Weight multiplier (0.0–2.0) for the LoRA.
        current_stack_json: JSON string of the current LoRA stack state.

    Returns:
        Tuple of (updated_stack_json, message, dataframe_rows).
        - updated_stack_json: The new LoRA stack JSON for the hidden state.
        - message: Informational message (empty on success, duplicate notice otherwise).
        - dataframe_rows: Updated rows [[filename, weight], ...] for the stack display.

    Implements: Requirements 2.1, 2.3, 2.8
    """
    try:
        from studio.core.lora_manager import add_lora_to_stack

        if not lora_name:
            return current_stack_json, "Select a LoRA from the dropdown first.", _stack_json_to_dataframe(current_stack_json)

        updated_json, message = add_lora_to_stack(
            lora_name=lora_name,
            weight=weight,
            current_stack_json=current_stack_json,
        )

        dataframe_rows = _stack_json_to_dataframe(updated_json)

        return updated_json, message, dataframe_rows

    except Exception as e:
        log.exception("on_add_lora failed")
        return current_stack_json, f"❌ {friendly(e)}", _stack_json_to_dataframe(current_stack_json)


def on_remove_lora(lora_name: str, current_stack_json: str) -> tuple[str, str, list[list[Any]]]:
    """Remove a LoRA from the stack.

    Thin handler that calls through to lora_manager.remove_lora_from_stack()
    and formats the return values for Gradio components.

    Args:
        lora_name: Display name (stem) of the LoRA file to remove.
        current_stack_json: JSON string of the current LoRA stack state.

    Returns:
        Tuple of (updated_stack_json, message, dataframe_rows).
        - updated_stack_json: The new LoRA stack JSON for the hidden state.
        - message: Informational message (empty on success, not-found notice otherwise).
        - dataframe_rows: Updated rows [[filename, weight], ...] for the stack display.

    Implements: Requirements 2.1, 2.3
    """
    try:
        from studio.core.lora_manager import remove_lora_from_stack

        if not lora_name:
            return current_stack_json, "Select a LoRA to remove.", _stack_json_to_dataframe(current_stack_json)

        updated_json, message = remove_lora_from_stack(
            lora_name=lora_name,
            current_stack_json=current_stack_json,
        )

        dataframe_rows = _stack_json_to_dataframe(updated_json)

        return updated_json, message, dataframe_rows

    except Exception as e:
        log.exception("on_remove_lora failed")
        return current_stack_json, f"❌ {friendly(e)}", _stack_json_to_dataframe(current_stack_json)


def on_send_to_img2img(gallery_selection: str | None) -> tuple[str | None, str]:
    """Extract selected gallery image path and return it for the img2img init image component.

    This handler is wired to the "Send to img2img" button. It validates
    that the selected image file still exists on disk and returns the path
    for populating the img2img section's init image component.

    Args:
        gallery_selection: The file path of the currently selected gallery
            image (from the gallery's selected_image_state).

    Returns:
        Tuple of (image_path_or_None, status_message). The first element
        populates the init image component; the second is a user-facing
        status message (empty on success, error string on failure).

    Implements: Requirements 9.1, 9.2, 9.5
    """
    try:
        if not gallery_selection:
            return None, "❌ No image selected — click an image in the gallery first."

        image_path = Path(gallery_selection)

        # Handle missing file case with plain-language error (Requirement 9.5)
        if not image_path.is_file():
            return None, "❌ Image unavailable — it may have been deleted externally."

        return str(image_path), ""

    except Exception as e:
        log.exception("on_send_to_img2img failed")
        return None, f"❌ {friendly(e)}"


def on_send_to_upscale(gallery_selection: str | None) -> Generator[tuple[str | None, str], None, None]:
    """Submit the selected gallery image to the upscaler pipeline immediately.

    This handler is wired to the "Send to Upscale" button. It coordinates
    GPU memory through the VRAM_Manager, runs the Real-ESRGAN 4x upscaler,
    saves the result to the outputs directory, and persists an artifact with
    type='upscaled' linked to the source artifact via source_artifact_id.

    Yields progress tuples during the operation for progress display.

    Args:
        gallery_selection: The file path of the currently selected gallery
            image (from the gallery's selected_image_state).

    Yields:
        Tuples of (output_image_path_or_None, status_message). During
        progress, output_image_path is None. On completion, it contains
        the path to the upscaled image.

    Implements: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 9.3, 9.5
    """
    try:
        from studio.models import upscale
        from studio.db import db

        # 1. Validate the selected image
        if not gallery_selection:
            yield None, "❌ No image selected — click an image in the gallery first."
            return

        image_path = Path(gallery_selection)

        # Handle missing file case with plain-language error (Requirement 9.5)
        if not image_path.is_file():
            yield None, "❌ Image unavailable — it may have been deleted externally."
            return

        # 2. Check upscaler model availability (Requirement 6.5)
        if not upscale.model_available():
            yield None, (
                "❌ The Real-ESRGAN upscaler model is not downloaded yet — "
                "go to the Models tab and click 'Download Upscaler'."
            )
            return

        # 3. Display progress indicator (Requirement 6.4)
        yield None, "⏳ Upscaling image..."

        # 4. Run the upscaler (coordinates GPU memory through VRAM_Manager
        #    internally — see upscale._upscale_realesrgan which acquires/
        #    releases a Tenant via get_vram_manager). Requirements 6.1, 6.6.
        out_path = upscale.upscale(
            str(image_path),
            upscale.METHOD_REALESRGAN,
            4.0,  # Default 4x scale
        )

        # 5. Persist artifact with type='upscaled' and source_artifact_id
        #    (Requirements 6.3, Property 15)
        db.init_db()

        # Look up the source artifact by path to get its id and job_id
        source_artifact = db.find_artifact_by_path(str(image_path))

        if source_artifact is not None:
            # Link the upscaled artifact to the source's job and artifact
            db.create_artifact(
                job_id=source_artifact.job_id,
                path=str(out_path),
                seed=source_artifact.seed,
                width=None,  # Will be filled by the upscaled image dimensions
                height=None,
                artifact_type="upscaled",
                source_artifact_id=source_artifact.id,
            )
        else:
            # Source artifact not found in DB (e.g. image from external source).
            # Still persist the upscaled artifact, just without a source link.
            # Create a minimal job record to host the artifact.
            job_id = db.create_job(
                prompt="(upscale)",
                params_json=json.dumps({"source_path": str(image_path), "scale": 4.0}),
                seed=0,
                model_id="upscaler",
                duration_ms=None,
                status="complete",
                artifacts=[{
                    "path": str(out_path),
                    "seed": 0,
                    "width": None,
                    "height": None,
                    "artifact_type": "upscaled",
                    "source_artifact_id": None,
                }],
            )

        yield str(out_path), f"✅ Upscaled to {out_path.name}"

    except Exception as e:
        log.exception("on_send_to_upscale failed")
        yield None, f"❌ {friendly(e)}"


def on_download(model_id: str = "krea2-turbo") -> Generator[str, None, None]:
    """Download model files, streaming progress to the UI.

    On completion, re-evaluates readiness so the banner updates.
    Never raises — all exceptions are caught and returned as plain text.

    Accumulates all messages into a running log so the UI shows the full
    history of what happened (not just the last message).

    Args:
        model_id: The registry model ID to download.

    Yields:
        Accumulated progress log (all messages joined by newlines).
    """
    lines: list[str] = []

    try:
        from studio.models import downloader
        from studio.core import system_check

        lines.append("Starting model download...")
        yield "\n".join(lines)

        for progress in downloader.download_all_models_generator(model_id):
            lines.append(progress)
            yield "\n".join(lines)

        # Re-evaluate readiness after download
        system_check.check_cuda_status()
        banner = system_check.get_readiness_banner()
        if not banner.get("visible", True):
            lines.append("✅ System is now ready to generate!")
            yield "\n".join(lines)

    except Exception as e:
        log.exception("on_download failed")
        lines.append(f"❌ {friendly(e)}")
        yield "\n".join(lines)


def on_load_history(page: int = 0) -> list[dict[str, Any]]:
    """Load paginated job history (20 per page).

    Returns a list of job summary dicts for display in the History tab.
    Never raises — returns an error message string on failure.

    Args:
        page: Zero-indexed page number.

    Returns:
        List of job summary dicts, or a single-element list with an error string.
    """
    try:
        from studio.db import db

        db.init_db()
        jobs = db.get_recent_jobs(limit=20, offset=page * 20)

        return [
            {
                "id": job.id,
                "created_at": job.created_at,
                "model_id": job.model_id,
                "prompt": job.prompt,
                "seed": job.seed,
                "status": job.status,
                "params": json.loads(job.params_json) if job.params_json else {},
            }
            for job in jobs
        ]

    except Exception as e:
        log.exception("on_load_history failed")
        return [{"error": f"❌ {friendly(e)}"}]


def on_load_params(job_id: int) -> dict[str, Any]:
    """Load parameters from a past job for populating the Generate tab.

    Returns a dict with all generation fields. Does NOT trigger generation.
    Never raises — returns an error dict on failure.

    Args:
        job_id: The database job ID to load params from.

    Returns:
        Dict with keys: prompt, seed, steps, width, height, precision,
        batch_size, batch_count. Or a dict with 'error' key on failure.
    """
    try:
        from studio.db import db

        db.init_db()
        job = db.get_job(job_id)

        if job is None:
            return {"error": f"❌ Job #{job_id} not found. Details logged to: {LOG_PATH}"}

        # Parse params_json to extract generation parameters
        params = json.loads(job.params_json) if job.params_json else {}

        return {
            "prompt": job.prompt,
            "seed": params.get("seed", job.seed),
            "steps": params.get("steps", 8),
            "width": params.get("width", 1024),
            "height": params.get("height", 1024),
            "precision": params.get("precision", "fp8_scaled"),
            "batch_size": params.get("batch_size", 1),
            "batch_count": params.get("batch_count", 1),
        }

    except Exception as e:
        log.exception("on_load_params failed")
        return {"error": f"❌ {friendly(e)}"}


def on_precision_change(precision: str) -> str:
    """Return an advisory warning when the selected precision won't fit VRAM.

    Called on precision picker change. Compares the registry's VRAM tier
    estimate for the selected precision against the VRAMManager's usable
    budget. Returns a plain-language warning string, or an empty string
    when the selection fits. Purely advisory — the hard refusal still
    happens at generate time (R6.4/R7.5); this just tells the owner
    BEFORE they click that the click will be refused.

    Args:
        precision: The newly selected precision ('bf16' or 'fp8_scaled').

    Returns:
        Warning markdown string, or "" if the selection fits.
    """
    try:
        from studio.core.vram_manager import get_vram_manager
        from studio.models import registry

        meta = registry.get_meta("krea2-turbo")
        tier_bytes = meta.vram_tiers.get(precision)
        if tier_bytes is None:
            return ""

        vram_mgr = get_vram_manager()
        if vram_mgr.can_fit(tier_bytes):
            return ""

        usable_gb = vram_mgr.estimate_available() / 1e9
        needed_gb = tier_bytes / 1e9
        fitting = [
            p for p, b in meta.vram_tiers.items() if vram_mgr.can_fit(b)
        ]
        suggestion = (
            f" Choose {' or '.join(fitting)} instead."
            if fitting
            else " No precision option fits this card."
        )
        return (
            f"⚠️ {precision} needs about {needed_gb:.0f} GB of VRAM but only "
            f"~{usable_gb:.0f} GB is usable on this card. Generation will be "
            f"refused.{suggestion}"
        )
    except Exception:
        log.exception("precision change check failed")
        # Advisory only — never block the UI over a failed check
        return ""


def on_upscale(image_path: str | None, method: str, scale: float) -> tuple[str | None, str]:
    """Upscale an image with the selected method.

    Args:
        image_path: Path to the source image (from the image input).
        method: Display name of the upscale method.
        scale: Requested scale factor.

    Returns:
        (output_image_path_or_None, status_message)
    """
    try:
        from studio.models import upscale

        if not image_path:
            return None, "Choose an image to upscale first."

        out_path = upscale.upscale(image_path, method, float(scale))
        return str(out_path), f"✅ Upscaled to {out_path.name}"

    except Exception as e:
        log.exception("on_upscale failed")
        return None, f"❌ {friendly(e)}"


def on_download_upscaler() -> Generator[str, None, None]:
    """Download the upscaler model, streaming accumulated progress."""
    lines: list[str] = []
    try:
        from studio.models import downloader

        lines.append("Starting upscaler download...")
        yield "\n".join(lines)

        for progress in downloader.download_upscaler_generator():
            lines.append(progress)
            yield "\n".join(lines)

    except Exception as e:
        log.exception("on_download_upscaler failed")
        lines.append(f"❌ {friendly(e)}")
        yield "\n".join(lines)


def _delete_job_files(job_id: int) -> int:
    """Delete a job's artifact image files from disk.

    Deleting only the DB row leaves the PNGs in outputs/ forever — the
    job "data" the owner wants gone is mostly the images. Removes each
    artifact file and any output directory left empty. Returns the
    number of files removed. Never raises.
    """
    from studio.db import db

    removed = 0
    parents: set[Path] = set()
    for artifact in db.get_job_artifacts(job_id):
        path = Path(artifact.path)
        try:
            if path.is_file():
                path.unlink()
                removed += 1
            parents.add(path.parent)
        except OSError:
            log.warning("Could not delete artifact file %s", path)
    for parent in parents:
        try:
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass
    return removed


def on_delete_job(job_id: int) -> str:
    """Delete a single job from history (DB rows AND image files).

    Args:
        job_id: The database job ID to delete.

    Returns:
        A status message string.
    """
    if not job_id or job_id <= 0:
        return "No job selected."
    return on_delete_jobs([int(job_id)])


def on_delete_jobs(job_ids: list[int]) -> str:
    """Delete multiple jobs from history, including image files on disk.

    Args:
        job_ids: Database job IDs to delete.

    Returns:
        A status message string summarizing what was removed.
    """
    try:
        from studio.db import db

        db.init_db()

        if not job_ids:
            return "No jobs selected."

        deleted = 0
        files_removed = 0
        for job_id in job_ids:
            files_removed += _delete_job_files(int(job_id))
            if db.delete_job(int(job_id)):
                deleted += 1

        if deleted == 0:
            return "No matching jobs found."
        return (
            f"✅ Deleted {deleted} job(s) and removed "
            f"{files_removed} image file(s) from disk."
        )

    except Exception as e:
        log.exception("on_delete_jobs failed")
        return f"❌ {friendly(e)}"
