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

        # 3. Run generation via registry, yielding progress updates
        yield "Encoding prompt...", []
        images: list[Any] = []
        resolved_params: dict[str, Any] = {}

        for update in registry.run_generation("krea2-turbo", validated):
            if isinstance(update, dict):
                # Final result dict from backend
                images = update.get("images", [])
                resolved_params = update.get("params", validated)
            else:
                # Progress string from backend
                yield str(update), []

        # 4. Persist job to database
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
            job_id = db.create_job(
                prompt=validated["prompt"],
                params_json=json.dumps(validated),
                seed=actual_seed,
                model_id="krea2-turbo",
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


def on_download(model_id: str = "krea2-turbo") -> Generator[str, None, None]:
    """Download model files, streaming progress to the UI.

    On completion, re-evaluates readiness so the banner updates.
    Never raises — all exceptions are caught and returned as plain text.

    Args:
        model_id: The registry model ID to download.

    Yields:
        Progress strings per chunk and final status.
    """
    try:
        from studio.models import downloader
        from studio.core import system_check

        yield "Starting model download..."

        for progress in downloader.download_all_models_generator(model_id):
            yield progress

        # Re-evaluate readiness after download
        system_check.check_cuda_status()
        banner = system_check.get_readiness_banner()
        if not banner.get("visible", True):
            yield "✅ System is now ready to generate!"

    except Exception as e:
        log.exception("on_download failed")
        yield f"❌ {friendly(e)}"


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
            "precision": params.get("precision", "bf16"),
            "batch_size": params.get("batch_size", 1),
            "batch_count": params.get("batch_count", 1),
        }

    except Exception as e:
        log.exception("on_load_params failed")
        return {"error": f"❌ {friendly(e)}"}
