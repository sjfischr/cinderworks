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
