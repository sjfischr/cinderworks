"""Streaming, resumable model downloader via huggingface_hub.

Mirrors BeatBunny worker/model_downloader.py pattern:
- download_all_models_generator(model_id) → Generator[str]
- get_model_info_text(model_id) → str
- get_download_state(model_id) → dict[str, str]
- check_huggingface_hub() → bool

Key behaviors:
- Uses huggingface_hub.hf_hub_download with resume_download=True
- Auto-places files into Config.MODEL_DIR (flat, no subfolders)
- Checks file presence FIRST: if file exists and size matches, reports
  "already downloaded" and skips
- Progress callback yields per-chunk progress strings
- Hub unreachable: catches connection errors, reports in plain language
- Partial failure: tracks which files failed, retains successful downloads
- Size validation uses 90% threshold (same as system_check)
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Generator

from studio.config import Config
from studio.models.registry import get_meta

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The Hugging Face repository holding Krea 2 model files
_HF_REPO_ID = "Comfy-Org/Krea-2"

# Mapping from local filename → path within the HF repo.
# Files live in subdirectories on HuggingFace but are stored flat locally.
_HF_FILE_PATHS: dict[str, str] = {
    "krea2_turbo_fp8_scaled.safetensors": "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
    "krea2_turbo_bf16.safetensors": "diffusion_models/krea2_turbo_bf16.safetensors",
    "qwen3vl_4b_fp8_scaled.safetensors": "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
    "qwen_image_vae.safetensors": "vae/qwen_image_vae.safetensors",
    "RealESRGAN_x4.pth": "RealESRGAN_x4.pth",
}

# Per-file HF repo overrides. Files not listed here come from _HF_REPO_ID.
# Upscaler models live in their own repos.
_HF_FILE_REPOS: dict[str, str] = {
    "RealESRGAN_x4.pth": "ai-forever/Real-ESRGAN",
}

# Expected file sizes (bytes) — used for presence validation.
# A file is considered "present" if its on-disk size >= 90% of expected.
_EXPECTED_SIZES: dict[str, int] = {
    "krea2_turbo_fp8_scaled.safetensors": int(13.1e9),
    "krea2_turbo_bf16.safetensors": int(26.3e9),
    "qwen3vl_4b_fp8_scaled.safetensors": int(5.24e9),
    "qwen_image_vae.safetensors": int(254e6),
    "RealESRGAN_x4.pth": int(67e6),
}

# Upscaler model files (not part of any generation model's registry
# entry — downloaded on demand from the Models tab).
UPSCALER_FILES: list[str] = ["RealESRGAN_x4.pth"]

_SIZE_THRESHOLD = 0.9  # 90% of expected = considered present

# Sentinel value to signal download thread completion
_DOWNLOAD_DONE = object()
_DOWNLOAD_ERROR = object()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_all_models_generator(model_id: str) -> Generator[str, None, None]:
    """Download all model files for the given model_id, yielding progress.

    Yields progress strings in the format:
        "Downloading {filename}: {percentage}% ({bytes_downloaded}/{total_bytes})"

    Files already present (passing size check) are reported as
    "already downloaded" and skipped. On partial failure, reports which
    files failed and retains successful downloads.

    Args:
        model_id: The model identifier from the registry (e.g. 'krea2-turbo').

    Yields:
        Progress strings per chunk, status messages, and error reports.
    """
    # Get model metadata from registry
    try:
        meta = get_meta(model_id)
    except KeyError:
        yield f"❌ Unknown model: '{model_id}'"
        return

    # Gather all files to download for this model
    files_to_download = _get_model_files(meta)

    if not files_to_download:
        yield "No files to download for this model."
        return

    # Ensure MODEL_DIR exists
    Config.ensure_dirs()

    # Check hub reachability first
    if not check_huggingface_hub():
        yield "❌ Cannot reach Hugging Face — check your internet connection and try again."
        return

    # Track results
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []  # (filename, reason)

    for filename in files_to_download:
        # Check if file is already present and valid
        # Files from HF are placed in subdirectories matching repo structure
        hf_path = _HF_FILE_PATHS.get(filename, filename)
        filepath = Config.MODEL_DIR / hf_path
        expected_size = _EXPECTED_SIZES.get(filename)

        if _file_is_present(filepath, expected_size):
            yield f"{filename}: already downloaded"
            succeeded.append(filename)
            continue

        # Download the file, yielding progress per chunk
        yield f"Starting download: {filename}"

        try:
            yield from _download_file_with_progress(filename)
            succeeded.append(filename)
            yield f"{filename}: download complete"
        except Exception as e:
            reason = str(e)
            failed.append((filename, reason))
            log.error("Failed to download %s: %s", filename, reason, exc_info=True)
            yield f"❌ {filename}: download failed — {reason}"

    # Summary
    if failed:
        failed_names = ", ".join(f[0] for f in failed)
        yield (
            f"⚠️ Partial download: {len(succeeded)} succeeded, "
            f"{len(failed)} failed ({failed_names}). "
            f"Re-trigger to retry failed files."
        )
    elif succeeded:
        yield f"✅ All {len(succeeded)} files downloaded successfully."


def download_upscaler_generator() -> Generator[str, None, None]:
    """Download the upscaler model file(s), yielding progress strings.

    Same behavior as download_all_models_generator but for the standalone
    upscaler files (which live outside any generation model's registry
    entry, in their own HF repos).
    """
    Config.ensure_dirs()

    if not check_huggingface_hub():
        yield "❌ Cannot reach Hugging Face — check your internet connection and try again."
        return

    for filename in UPSCALER_FILES:
        hf_path = _HF_FILE_PATHS.get(filename, filename)
        filepath = Config.MODEL_DIR / hf_path
        expected_size = _EXPECTED_SIZES.get(filename)

        if _file_is_present(filepath, expected_size):
            yield f"{filename}: already downloaded"
            continue

        yield f"Starting download: {filename}"
        try:
            yield from _download_file_with_progress(filename)
            yield f"✅ {filename}: download complete"
        except Exception as e:
            log.error("Failed to download %s: %s", filename, e, exc_info=True)
            yield f"❌ {filename}: download failed — {e}"


def get_upscaler_state() -> dict[str, str]:
    """Per-file download status for the upscaler files.

    Same status values as get_download_state: present/partial/missing.
    """
    state: dict[str, str] = {}
    for filename in UPSCALER_FILES:
        hf_path = _HF_FILE_PATHS.get(filename, filename)
        filepath = Config.MODEL_DIR / hf_path
        expected_size = _EXPECTED_SIZES.get(filename)
        if not filepath.is_file():
            state[filename] = "missing"
        elif expected_size and filepath.stat().st_size < int(expected_size * _SIZE_THRESHOLD):
            state[filename] = "partial"
        else:
            state[filename] = "present"
    return state


def get_model_info_text(model_id: str) -> str:
    """Human-readable summary of model files and their download status.

    Args:
        model_id: The model identifier from the registry.

    Returns:
        Multi-line string summarizing each file's status.
    """
    try:
        meta = get_meta(model_id)
    except KeyError:
        return f"Unknown model: '{model_id}'"

    files = _get_model_files(meta)
    lines: list[str] = [f"Model: {meta.display_name}", f"Repository: {_HF_REPO_ID}", ""]

    state = get_download_state(model_id)

    for filename in files:
        status = state.get(filename, "unknown")
        hf_path = _HF_FILE_PATHS.get(filename, filename)
        filepath = Config.MODEL_DIR / hf_path
        size_str = ""

        if filepath.is_file():
            actual_size = filepath.stat().st_size
            size_str = f" ({_format_bytes(actual_size)})"

        expected = _EXPECTED_SIZES.get(filename)
        expected_str = f" / expected ~{_format_bytes(expected)}" if expected else ""

        if status == "present":
            lines.append(f"  ✅ {filename}{size_str}{expected_str}")
        elif status == "partial":
            lines.append(f"  ⚠️ {filename}{size_str}{expected_str} (incomplete)")
        else:
            lines.append(f"  ❌ {filename} (not downloaded){expected_str}")

    return "\n".join(lines)


def get_download_state(model_id: str) -> dict[str, str]:
    """Per-file download status for the given model.

    Args:
        model_id: The model identifier from the registry.

    Returns:
        Dict mapping filename → status string:
        - 'present': file exists and passes size check
        - 'partial': file exists but is too small (incomplete download)
        - 'missing': file does not exist on disk
    """
    try:
        meta = get_meta(model_id)
    except KeyError:
        return {}

    files = _get_model_files(meta)
    state: dict[str, str] = {}

    for filename in files:
        hf_path = _HF_FILE_PATHS.get(filename, filename)
        filepath = Config.MODEL_DIR / hf_path
        expected_size = _EXPECTED_SIZES.get(filename)

        if not filepath.is_file():
            state[filename] = "missing"
        elif expected_size and filepath.stat().st_size < int(expected_size * _SIZE_THRESHOLD):
            state[filename] = "partial"
        else:
            state[filename] = "present"

    return state


def check_huggingface_hub() -> bool:
    """Check if the Hugging Face Hub is reachable.

    Uses a lightweight HTTP HEAD request with a 30-second timeout.
    Returns True if reachable, False otherwise. Never raises.
    """
    try:
        import urllib.request

        req = urllib.request.Request(
            "https://huggingface.co/api/models",
            method="HEAD",
        )
        with urllib.request.urlopen(req, timeout=30):
            return True
    except Exception as e:
        log.warning("Hugging Face Hub unreachable: %s", e)
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_model_files(meta) -> list[str]:
    """Collect all filenames for a model from its registry entry.

    Returns the checkpoints + text_encoder + vae filenames.
    """
    files: list[str] = []

    # Add checkpoints
    files.extend(meta.checkpoints)

    # Add text encoder
    if meta.text_encoder:
        files.append(meta.text_encoder)

    # Add VAE
    if meta.vae:
        files.append(meta.vae)

    return files


def _file_is_present(filepath: Path, expected_size: int | None) -> bool:
    """Check if a file is present and passes the size threshold.

    A file is considered present if:
    - It exists on disk AND
    - Its size is >= 90% of the expected size (or expected_size is None)
    """
    if not filepath.is_file():
        return False

    if expected_size is None:
        # No expected size known — existence alone counts
        return True

    actual_size = filepath.stat().st_size
    return actual_size >= int(expected_size * _SIZE_THRESHOLD)


class _ProgressTqdm:
    """Custom tqdm-compatible class that posts progress updates to a queue.

    Mimics the tqdm interface that huggingface_hub expects, capturing
    progress updates and forwarding them to a queue for consumption by
    the generator.
    """

    def __init__(
        self,
        *args,
        total: int | None = None,
        initial: int = 0,
        desc: str | None = None,
        **kwargs,
    ):
        self.total = total or 0
        self.n = initial
        self.desc = desc or ""
        self._queue: queue.Queue | None = None
        self._filename: str = ""

    def set_progress_queue(self, q: queue.Queue, filename: str) -> None:
        """Attach the progress queue and filename after construction."""
        self._queue = q
        self._filename = filename

    def update(self, n: int = 1) -> None:
        """Called by huggingface_hub on each chunk received."""
        self.n += n
        if self._queue and self.total > 0:
            percentage = min(int((self.n / self.total) * 100), 100)
            msg = (
                f"Downloading {self._filename}: {percentage}% "
                f"({_format_bytes(self.n)}/{_format_bytes(self.total)})"
            )
            self._queue.put(msg)

    def close(self) -> None:
        """Called when download completes."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # Additional tqdm methods that huggingface_hub might call
    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        if desc:
            self.desc = desc

    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        pass

    def refresh(self) -> None:
        pass

    def clear(self) -> None:
        pass

    def display(self, *args, **kwargs) -> None:
        pass

    @property
    def format_dict(self) -> dict:
        return {"n": self.n, "total": self.total}


def _make_tqdm_factory(progress_queue: queue.Queue, filename: str):
    """Create a tqdm class factory that produces queue-connected progress bars.

    Returns a class (not an instance) that huggingface_hub will instantiate.
    """

    class _ConnectedProgressTqdm(_ProgressTqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_progress_queue(progress_queue, filename)

    return _ConnectedProgressTqdm


def _download_file_with_progress(filename: str) -> Generator[str, None, None]:
    """Download a single file from HF, yielding per-chunk progress strings.

    Runs hf_hub_download in a background thread with a custom tqdm class
    that posts progress messages to a queue. The generator yields from
    the queue until the download completes or fails.

    Uses resume_download=True for resumability — if a partial file exists,
    the download resumes from the last byte.
    """
    progress_queue: queue.Queue = queue.Queue()

    error_holder: list[Exception] = []

    # Resolve the HF repo path for this file
    hf_path = _HF_FILE_PATHS.get(filename, filename)

    repo_id = _HF_FILE_REPOS.get(filename, _HF_REPO_ID)

    def _do_download():
        try:
            from huggingface_hub import hf_hub_download

            progress_queue.put(f"[DEBUG] Calling hf_hub_download(repo_id='{repo_id}', filename='{hf_path}', local_dir='{Config.MODEL_DIR}')")

            hf_hub_download(
                repo_id=repo_id,
                filename=hf_path,
                local_dir=str(Config.MODEL_DIR),
                resume_download=True,
                local_dir_use_symlinks=False,
            )
            progress_queue.put(_DOWNLOAD_DONE)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log.error("Download thread error for %s: %s\n%s", filename, e, tb)
            progress_queue.put(f"[ERROR] {filename}: {type(e).__name__}: {e}")
            error_holder.append(e)
            progress_queue.put(_DOWNLOAD_ERROR)

    # Start download in background thread
    thread = threading.Thread(target=_do_download, daemon=True)
    thread.start()

    # Resolve expected size for progress reporting
    expected_size = _EXPECTED_SIZES.get(filename)
    hf_path_resolved = _HF_FILE_PATHS.get(filename, filename)
    local_file = Config.MODEL_DIR / hf_path_resolved

    # Yield progress messages from the queue until done
    while True:
        try:
            msg = progress_queue.get(timeout=2.0)
        except queue.Empty:
            # No message yet — report file size progress if file exists
            if not thread.is_alive():
                if error_holder:
                    raise RuntimeError(str(error_holder[0]))
                break
            # Poll file size for progress
            if local_file.exists() and expected_size:
                current_size = local_file.stat().st_size
                pct = min(int((current_size / expected_size) * 100), 99)
                progress_queue.put(
                    f"Downloading {filename}: {pct}% "
                    f"({_format_bytes(current_size)}/{_format_bytes(expected_size)})"
                )
            continue

        if msg is _DOWNLOAD_DONE:
            break
        elif msg is _DOWNLOAD_ERROR:
            if error_holder:
                raise RuntimeError(str(error_holder[0]))
            raise RuntimeError("Download failed with unknown error")
        else:
            yield msg

    # Ensure thread is cleaned up
    thread.join(timeout=5.0)

    # Check for any errors that occurred
    if error_holder:
        raise RuntimeError(str(error_holder[0]))


def _format_bytes(size: int | None) -> str:
    """Format a byte count as a human-readable string."""
    if size is None:
        return "unknown"
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"
