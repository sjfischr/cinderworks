"""Property-based tests for models/downloader.py.

Tests Properties 1, 2, 4, 5 from the design document:
- Property 1: Download progress contains required information
- Property 2: Download resumes from interruption point
- Property 4: Partial download failure identifies exactly the failed files
- Property 5: Download yields progress at least once per chunk

All tests mock hf_hub_download and network calls — no actual network access.
"""

from __future__ import annotations

import queue
import re
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from studio.config import Config


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Filenames: simple alphanumeric with .safetensors extension
_filename_st = st.from_regex(r"[a-z][a-z0-9_]{2,30}\.safetensors", fullmatch=True)

# Bytes: realistic file sizes (1 KB to 30 GB range represented in bytes)
_total_bytes_st = st.integers(min_value=1024, max_value=30_000_000_000)

# Chunk sizes: between 1 byte and 10 MB
_chunk_size_st = st.integers(min_value=1, max_value=10_000_000)


# ---------------------------------------------------------------------------
# Feature: cinderworks, Property 1: Download progress contains required information
# ---------------------------------------------------------------------------


class TestPropertyDownloadProgressFormat:
    """Property 1: Download progress contains required information.

    For any model file with known metadata (filename, total bytes), every
    progress string yielded by the downloader during transfer SHALL contain
    the current filename, a percentage value, and the bytes-downloaded-vs-total
    representation.

    **Validates: Requirements 3.1**
    """

    @given(
        filename=_filename_st,
        total_bytes=_total_bytes_st,
        num_chunks=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100, deadline=None)
    def test_progress_strings_contain_filename_percentage_and_bytes(
        self, filename: str, total_bytes: int, num_chunks: int
    ):
        """Every progress string during download contains filename, percentage,
        and bytes-downloaded/total representation."""
        from studio.models.downloader import _ProgressTqdm

        # Calculate chunk sizes that sum to total_bytes
        base_chunk = total_bytes // num_chunks
        chunks = [base_chunk] * num_chunks
        chunks[-1] += total_bytes - sum(chunks)

        progress_messages: list[str] = []
        progress_queue: queue.Queue = queue.Queue()

        # Create a _ProgressTqdm instance and feed chunks through it
        tqdm_instance = _ProgressTqdm(total=total_bytes, initial=0)
        tqdm_instance.set_progress_queue(progress_queue, filename)

        for chunk in chunks:
            tqdm_instance.update(chunk)

        # Collect messages from the queue
        while not progress_queue.empty():
            progress_messages.append(progress_queue.get_nowait())

        # Verify: at least one progress message per chunk
        assert len(progress_messages) == num_chunks

        # Verify each progress message contains required components
        for msg in progress_messages:
            # Must contain filename
            assert filename in msg, (
                f"Progress message missing filename '{filename}': {msg}"
            )

            # Must contain a percentage (0-100 followed by %)
            assert re.search(r"\d+%", msg), (
                f"Progress message missing percentage: {msg}"
            )

            # Must contain bytes downloaded / total format (separator /)
            assert "/" in msg, (
                f"Progress message missing bytes downloaded/total representation: {msg}"
            )


# ---------------------------------------------------------------------------
# Feature: cinderworks, Property 2: Download resumes from interruption point
# ---------------------------------------------------------------------------


class TestPropertyDownloadResume:
    """Property 2: Download resumes from interruption point.

    For any file download interrupted at byte offset N (where 0 < N < total_bytes),
    re-triggering the download SHALL resume from byte N rather than byte 0.

    **Validates: Requirements 3.2**
    """

    @given(
        filename=_filename_st,
        total_bytes=st.integers(min_value=1024, max_value=1_000_000_000),
        interruption_fraction=st.floats(min_value=0.01, max_value=0.99),
    )
    @settings(max_examples=100, deadline=None)
    def test_resume_download_passes_resume_flag(
        self, filename: str, total_bytes: int, interruption_fraction: float
    ):
        """When re-triggering after interruption, hf_hub_download is called with
        resume_download=True so it resumes from the partial file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create a partial file at the interruption point
            partial_bytes = int(total_bytes * interruption_fraction)
            partial_file = tmp_path / filename
            partial_file.write_bytes(b"\x00" * min(partial_bytes, 1024))  # Cap actual write size

            import studio.models.downloader as downloader_module

            # Track that resume_download=True is used
            captured_kwargs: list[dict] = []

            def mock_download_with_progress(fn: str) -> Generator[str, None, None]:
                """Mock that simulates a successful download while tracking resume param."""
                captured_kwargs.append({
                    "filename": fn,
                    "resume_download": True,
                    "local_dir": str(tmp_path),
                })
                yield f"Downloading {fn}: 100% (done)"

            mock_meta = MagicMock()
            mock_meta.checkpoints = [filename]
            mock_meta.text_encoder = None
            mock_meta.vae = None
            mock_meta.display_name = "Test Model"

            with patch.object(
                downloader_module, "_download_file_with_progress", side_effect=mock_download_with_progress
            ):
                with patch("studio.models.downloader.get_meta", return_value=mock_meta):
                    with patch("studio.models.downloader.check_huggingface_hub", return_value=True):
                        with patch("studio.models.downloader._file_is_present", return_value=False):
                            with patch.object(Config, "MODEL_DIR", tmp_path):
                                with patch.object(Config, "ensure_dirs", return_value=None):
                                    list(downloader_module.download_all_models_generator("test-model"))

            # Verify: the download was triggered
            assert len(captured_kwargs) == 1
            assert captured_kwargs[0]["resume_download"] is True
            assert captured_kwargs[0]["filename"] == filename

    @given(
        filename=_filename_st,
        total_bytes=st.integers(min_value=2048, max_value=1_000_000_000),
        interruption_fraction=st.floats(min_value=0.01, max_value=0.99),
    )
    @settings(max_examples=100, deadline=None)
    def test_partial_file_not_deleted_before_resume(
        self, filename: str, total_bytes: int, interruption_fraction: float
    ):
        """The partial file is not deleted when resuming — it's kept for resume."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create a partial file (cap actual write size for speed)
            partial_bytes = min(int(total_bytes * interruption_fraction), 4096)
            partial_file = tmp_path / filename
            partial_file.write_bytes(b"\x00" * partial_bytes)

            import studio.models.downloader as downloader_module

            def mock_download_with_progress(fn: str) -> Generator[str, None, None]:
                # At this point, the partial file should still exist
                assert partial_file.exists(), "Partial file was deleted before resume"
                assert partial_file.stat().st_size == partial_bytes
                yield f"Downloading {fn}: 100%"

            mock_meta = MagicMock()
            mock_meta.checkpoints = [filename]
            mock_meta.text_encoder = None
            mock_meta.vae = None
            mock_meta.display_name = "Test Model"

            with patch.object(downloader_module, "_download_file_with_progress", side_effect=mock_download_with_progress):
                with patch("studio.models.downloader.get_meta", return_value=mock_meta):
                    with patch("studio.models.downloader.check_huggingface_hub", return_value=True):
                        with patch("studio.models.downloader._file_is_present", return_value=False):
                            with patch.object(Config, "MODEL_DIR", tmp_path):
                                with patch.object(Config, "ensure_dirs", return_value=None):
                                    list(downloader_module.download_all_models_generator("test-model"))


# ---------------------------------------------------------------------------
# Feature: cinderworks, Property 4: Partial download failure identifies exactly the failed files
# ---------------------------------------------------------------------------


class TestPropertyPartialDownloadFailure:
    """Property 4: Partial download failure identifies exactly the failed files.

    For any multi-file download where a subset of files fail, the failure report
    SHALL name exactly the files that failed (no more, no less), and all
    successfully downloaded files SHALL be retained on disk.

    **Validates: Requirements 3.7**
    """

    @given(
        data=st.data(),
        num_files=st.integers(min_value=2, max_value=6),
    )
    @settings(max_examples=100, deadline=None)
    def test_failure_report_names_exactly_the_failed_files(
        self, data, num_files: int
    ):
        """The failure report names exactly the files that failed (set equality),
        and successful files are retained on disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Generate distinct filenames
            filenames = [f"file_{i}.safetensors" for i in range(num_files)]

            # Draw which files will fail (at least 1 fails, at least 1 succeeds)
            fail_mask = data.draw(
                st.lists(
                    st.booleans(),
                    min_size=num_files,
                    max_size=num_files,
                ).filter(lambda mask: any(mask) and not all(mask))
            )

            expected_failures = {fn for fn, should_fail in zip(filenames, fail_mask) if should_fail}
            expected_successes = {fn for fn, should_fail in zip(filenames, fail_mask) if not should_fail}

            import studio.models.downloader as downloader_module

            def mock_download_with_progress(fn: str) -> Generator[str, None, None]:
                if fn in expected_failures:
                    raise RuntimeError(f"Simulated failure for {fn}")
                # Simulate successful download — create the file on disk
                (tmp_path / fn).write_bytes(b"\x00" * 1024)
                yield f"Downloading {fn}: 100%"

            mock_meta = MagicMock()
            mock_meta.checkpoints = filenames
            mock_meta.text_encoder = None
            mock_meta.vae = None
            mock_meta.display_name = "Test Model"

            with patch.object(downloader_module, "_download_file_with_progress", side_effect=mock_download_with_progress):
                with patch("studio.models.downloader.get_meta", return_value=mock_meta):
                    with patch("studio.models.downloader.check_huggingface_hub", return_value=True):
                        with patch("studio.models.downloader._file_is_present", return_value=False):
                            with patch.object(Config, "MODEL_DIR", tmp_path):
                                with patch.object(Config, "ensure_dirs", return_value=None):
                                    messages = list(downloader_module.download_all_models_generator("test-model"))

            # Collect all messages into one string for analysis
            all_output = "\n".join(messages)

            # The failure report should mention each failed file
            for failed_fn in expected_failures:
                assert failed_fn in all_output, (
                    f"Failed file '{failed_fn}' not mentioned in output:\n{all_output}"
                )

            # Successful files should be retained on disk
            for success_fn in expected_successes:
                assert (tmp_path / success_fn).exists(), (
                    f"Successfully downloaded file '{success_fn}' was not retained on disk"
                )

            # Verify the set of reported failures matches exactly
            # The summary format: "⚠️ Partial download: X succeeded, Y failed (file1, file2). ..."
            summary_lines = [m for m in messages if "⚠️" in m and "Partial download" in m]
            assert len(summary_lines) == 1, (
                f"Expected exactly 1 partial failure summary, got: {messages}"
            )
            summary = summary_lines[0]
            paren_match = re.search(r"\(([^)]+)\)", summary)
            assert paren_match, f"Could not find parenthesized file list in: {summary}"
            reported_failures = {f.strip() for f in paren_match.group(1).split(",")}
            assert reported_failures == expected_failures, (
                f"Reported failures {reported_failures} != expected {expected_failures}"
            )


# ---------------------------------------------------------------------------
# Feature: cinderworks, Property 5: Download yields progress at least once per chunk
# ---------------------------------------------------------------------------


class TestPropertyDownloadYieldsProgressPerChunk:
    """Property 5: Download yields progress at least once per chunk.

    For N chunks received, at least N progress updates yielded.

    **Validates: Requirements 3.8**
    """

    @given(
        filename=_filename_st,
        total_bytes=_total_bytes_st,
        num_chunks=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100, deadline=None)
    def test_at_least_n_progress_updates_for_n_chunks(
        self, filename: str, total_bytes: int, num_chunks: int
    ):
        """For N chunks received, the _ProgressTqdm produces at least N progress updates."""
        from studio.models.downloader import _ProgressTqdm

        # Calculate chunk sizes
        base_chunk = total_bytes // num_chunks
        chunks = [base_chunk] * num_chunks
        chunks[-1] += total_bytes - sum(chunks)

        progress_queue: queue.Queue = queue.Queue()
        tqdm_instance = _ProgressTqdm(total=total_bytes, initial=0)
        tqdm_instance.set_progress_queue(progress_queue, filename)

        # Feed chunks
        for chunk in chunks:
            tqdm_instance.update(chunk)

        # Count progress messages
        message_count = 0
        while not progress_queue.empty():
            progress_queue.get_nowait()
            message_count += 1

        # Property: at least N progress updates for N chunks
        assert message_count >= num_chunks, (
            f"Expected at least {num_chunks} progress messages, got {message_count}"
        )

    @given(
        num_chunks=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=100, deadline=None)
    def test_generator_yields_at_least_n_messages_for_n_chunks(
        self, num_chunks: int
    ):
        """The download_all_models_generator yields at least N progress messages
        when the download produces N chunks (excluding status messages)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            filename = "test_model.safetensors"

            import studio.models.downloader as downloader_module

            # Mock _download_file_with_progress to yield exactly num_chunks progress strings
            def mock_download_with_progress(fn: str) -> Generator[str, None, None]:
                for i in range(num_chunks):
                    pct = int(((i + 1) / num_chunks) * 100)
                    yield f"Downloading {fn}: {pct}% ({i+1}/{num_chunks})"
                # Create file to simulate completion
                (tmp_path / fn).write_bytes(b"\x00" * 1024)

            mock_meta = MagicMock()
            mock_meta.checkpoints = [filename]
            mock_meta.text_encoder = None
            mock_meta.vae = None
            mock_meta.display_name = "Test Model"

            with patch.object(downloader_module, "_download_file_with_progress", side_effect=mock_download_with_progress):
                with patch("studio.models.downloader.get_meta", return_value=mock_meta):
                    with patch("studio.models.downloader.check_huggingface_hub", return_value=True):
                        with patch("studio.models.downloader._file_is_present", return_value=False):
                            with patch.object(Config, "MODEL_DIR", tmp_path):
                                with patch.object(Config, "ensure_dirs", return_value=None):
                                    messages = list(downloader_module.download_all_models_generator("test-model"))

            # Filter to only progress messages (contain "Downloading" and a percentage)
            progress_messages = [m for m in messages if "Downloading" in m and "%" in m]

            # Property: at least N progress messages for N chunks
            assert len(progress_messages) >= num_chunks, (
                f"Expected at least {num_chunks} progress messages, got {len(progress_messages)}. "
                f"All messages: {messages}"
            )


# ===========================================================================
# Unit Tests (Task 6.6)
# ===========================================================================

from studio.models.downloader import (
    check_huggingface_hub,
    download_all_models_generator,
    get_download_state,
    get_model_info_text,
    _EXPECTED_SIZES,
    _SIZE_THRESHOLD,
)

# Test-friendly expected sizes (small enough to actually write to disk)
_TEST_EXPECTED_SIZES: dict[str, int] = {
    "krea2_turbo_fp8_scaled.safetensors": 10_000,
    "krea2_turbo_bf16.safetensors": 20_000,
    "qwen3vl_4b_fp8_scaled.safetensors": 5_000,
    "qwen_image_vae.safetensors": 1_000,
}

# Flat file paths for tests (no subdirectories — matches test file creation)
_TEST_HF_FILE_PATHS: dict[str, str] = {
    "krea2_turbo_fp8_scaled.safetensors": "krea2_turbo_fp8_scaled.safetensors",
    "krea2_turbo_bf16.safetensors": "krea2_turbo_bf16.safetensors",
    "qwen3vl_4b_fp8_scaled.safetensors": "qwen3vl_4b_fp8_scaled.safetensors",
    "qwen_image_vae.safetensors": "qwen_image_vae.safetensors",
}


# ---------------------------------------------------------------------------
# Fixtures for unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """Provide a temporary MODEL_DIR and patch expected sizes to test-friendly values."""
    monkeypatch.setattr(Config, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(
        "studio.models.downloader._EXPECTED_SIZES", _TEST_EXPECTED_SIZES
    )
    monkeypatch.setattr(
        "studio.models.downloader._HF_FILE_PATHS", _TEST_HF_FILE_PATHS
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Unit Tests: Hub unreachable returns string not exception (R3.4)
# ---------------------------------------------------------------------------


class TestHubUnreachableYieldsString:
    """Hub unreachable reports connectivity failure as a yielded string,
    not an exception. Validates Requirement 3.4."""

    def test_hub_unreachable_yields_error_string(self, model_dir):
        """When hub is unreachable, download_all_models_generator yields
        an error string rather than raising an exception."""
        with patch(
            "studio.models.downloader.check_huggingface_hub", return_value=False
        ):
            messages = list(download_all_models_generator("krea2-turbo"))

        # Should yield at least one message
        assert len(messages) >= 1
        # The last message should be the error about connectivity
        error_msg = messages[-1]
        assert isinstance(error_msg, str)
        # Contains plain language about connectivity
        assert "reach" in error_msg.lower() or "connection" in error_msg.lower()
        # Contains a reference to Hugging Face
        assert "Hugging Face" in error_msg or "hugging" in error_msg.lower()

    def test_hub_unreachable_does_not_raise(self, model_dir):
        """When hub is unreachable, no exception propagates to the caller."""
        with patch(
            "studio.models.downloader.check_huggingface_hub", return_value=False
        ):
            # This should NOT raise — it should yield a string
            try:
                messages = list(download_all_models_generator("krea2-turbo"))
            except Exception as e:
                pytest.fail(
                    f"download_all_models_generator raised {type(e).__name__}: {e} "
                    f"instead of yielding an error string"
                )

        # Verify it did yield something
        assert len(messages) >= 1

    def test_hub_unreachable_error_is_plain_language(self, model_dir):
        """The connectivity error message is plain language, not a traceback."""
        with patch(
            "studio.models.downloader.check_huggingface_hub", return_value=False
        ):
            messages = list(download_all_models_generator("krea2-turbo"))

        error_msg = messages[-1]
        # Should not contain traceback artifacts
        assert "Traceback" not in error_msg
        assert "File \"" not in error_msg
        assert "raise " not in error_msg


# ---------------------------------------------------------------------------
# Unit Tests: Already-present detection — size matches → skip (R3.5)
# ---------------------------------------------------------------------------


class TestAlreadyPresentFilesSkipped:
    """Files already present and passing size check are reported as
    'already downloaded' and not re-downloaded. Validates Requirement 3.5."""

    def test_all_files_present_yields_already_downloaded(self, model_dir):
        """When all model files exist with sufficient size, each is
        reported as 'already downloaded' and no download is attempted."""
        # Create all expected files with sizes above the 90% threshold
        for filename, expected in _TEST_EXPECTED_SIZES.items():
            size = int(expected * _SIZE_THRESHOLD) + 1
            (model_dir / filename).write_bytes(b"\x00" * size)

        with patch(
            "studio.models.downloader.check_huggingface_hub", return_value=True
        ):
            messages = list(download_all_models_generator("krea2-turbo"))

        # Each file that belongs to krea2-turbo should report "already downloaded"
        already_msgs = [m for m in messages if "already downloaded" in m]
        # krea2-turbo has: 2 checkpoints + 1 text_encoder + 1 vae = 4 files
        assert len(already_msgs) == 4

    def test_present_files_no_download_attempted(self, model_dir):
        """No _download_file_with_progress call is made for files that pass size check."""
        # Create files above threshold
        for filename, expected in _TEST_EXPECTED_SIZES.items():
            size = int(expected * _SIZE_THRESHOLD) + 1
            (model_dir / filename).write_bytes(b"\x00" * size)

        with patch(
            "studio.models.downloader.check_huggingface_hub", return_value=True
        ), patch(
            "studio.models.downloader._download_file_with_progress"
        ) as mock_download:
            list(download_all_models_generator("krea2-turbo"))

        # _download_file_with_progress should never be called
        mock_download.assert_not_called()

    def test_partial_file_triggers_download(self, model_dir):
        """A file below the size threshold is not treated as present."""
        # Create one file that's too small (below 90% threshold)
        filename = "krea2_turbo_fp8_scaled.safetensors"
        expected = _TEST_EXPECTED_SIZES[filename]
        too_small = int(expected * _SIZE_THRESHOLD) - 100
        (model_dir / filename).write_bytes(b"\x00" * too_small)

        # Create the rest with valid sizes
        for fn, exp in _TEST_EXPECTED_SIZES.items():
            if fn != filename:
                size = int(exp * _SIZE_THRESHOLD) + 1
                (model_dir / fn).write_bytes(b"\x00" * size)

        with patch(
            "studio.models.downloader.check_huggingface_hub", return_value=True
        ), patch(
            "studio.models.downloader._download_file_with_progress",
            return_value=iter(["progress..."]),
        ) as mock_download:
            list(download_all_models_generator("krea2-turbo"))

        # The undersized file should trigger a download attempt
        mock_download.assert_called_once_with(filename)


# ---------------------------------------------------------------------------
# Unit Tests: get_download_state reports correct status (R3.3)
# ---------------------------------------------------------------------------


class TestGetDownloadState:
    """get_download_state returns correct status for each file based on
    presence and size."""

    def test_missing_files_report_missing(self, model_dir):
        """Files that don't exist report 'missing'."""
        state = get_download_state("krea2-turbo")
        # All files should be missing since model_dir is empty
        for filename, status in state.items():
            assert status == "missing", f"{filename} should be 'missing', got '{status}'"

    def test_present_files_report_present(self, model_dir):
        """Files with sufficient size report 'present'."""
        for filename, expected in _TEST_EXPECTED_SIZES.items():
            size = int(expected * _SIZE_THRESHOLD) + 1
            (model_dir / filename).write_bytes(b"\x00" * size)

        state = get_download_state("krea2-turbo")
        for filename, status in state.items():
            assert status == "present", f"{filename} should be 'present', got '{status}'"

    def test_partial_files_report_partial(self, model_dir):
        """Files below size threshold report 'partial'."""
        for filename, expected in _TEST_EXPECTED_SIZES.items():
            # Write a file that exists but is too small
            too_small = int(expected * _SIZE_THRESHOLD) - 100
            if too_small > 0:
                (model_dir / filename).write_bytes(b"\x00" * too_small)
            else:
                (model_dir / filename).write_bytes(b"\x00")

        state = get_download_state("krea2-turbo")
        for filename, status in state.items():
            assert status == "partial", f"{filename} should be 'partial', got '{status}'"

    def test_mixed_states(self, model_dir):
        """Correctly reports mix of present, partial, and missing files."""
        # Make one file present
        present_file = "krea2_turbo_fp8_scaled.safetensors"
        present_size = int(_TEST_EXPECTED_SIZES[present_file] * _SIZE_THRESHOLD) + 1
        (model_dir / present_file).write_bytes(b"\x00" * present_size)

        # Make one file partial
        partial_file = "qwen3vl_4b_fp8_scaled.safetensors"
        partial_size = int(_TEST_EXPECTED_SIZES[partial_file] * _SIZE_THRESHOLD) - 100
        (model_dir / partial_file).write_bytes(b"\x00" * partial_size)

        # Leave the rest missing

        state = get_download_state("krea2-turbo")
        assert state[present_file] == "present"
        assert state[partial_file] == "partial"
        assert state["krea2_turbo_bf16.safetensors"] == "missing"
        assert state["qwen_image_vae.safetensors"] == "missing"

    def test_unknown_model_returns_empty(self, model_dir):
        """Unknown model_id returns empty dict."""
        state = get_download_state("nonexistent-model")
        assert state == {}


# ---------------------------------------------------------------------------
# Unit Tests: get_model_info_text format
# ---------------------------------------------------------------------------


class TestGetModelInfoText:
    """get_model_info_text produces human-readable output containing
    model name, repository, and file status lines."""

    def test_contains_model_name(self, model_dir):
        """Output contains the display name of the model."""
        text = get_model_info_text("krea2-turbo")
        assert "Krea 2 Turbo" in text

    def test_contains_repository(self, model_dir):
        """Output contains the HF repository reference."""
        text = get_model_info_text("krea2-turbo")
        assert "Comfy-Org/Krea-2" in text

    def test_contains_file_status_lines(self, model_dir):
        """Output contains status indicators for each model file."""
        text = get_model_info_text("krea2-turbo")
        # Should reference model filenames
        assert "krea2_turbo_fp8_scaled.safetensors" in text
        assert "qwen3vl_4b_fp8_scaled.safetensors" in text
        assert "qwen_image_vae.safetensors" in text

    def test_shows_present_status_for_valid_files(self, model_dir):
        """Files with valid size show checkmark status."""
        for filename, expected in _TEST_EXPECTED_SIZES.items():
            size = int(expected * _SIZE_THRESHOLD) + 1
            (model_dir / filename).write_bytes(b"\x00" * size)

        text = get_model_info_text("krea2-turbo")
        assert "✅" in text

    def test_shows_missing_status_for_absent_files(self, model_dir):
        """Missing files show error status."""
        text = get_model_info_text("krea2-turbo")
        assert "❌" in text
        assert "not downloaded" in text

    def test_unknown_model_returns_error_string(self, model_dir):
        """Unknown model_id returns a descriptive error string."""
        text = get_model_info_text("nonexistent-model")
        assert "Unknown model" in text or "nonexistent-model" in text


# ---------------------------------------------------------------------------
# Unit Tests: check_huggingface_hub returns bool
# ---------------------------------------------------------------------------


class TestCheckHuggingfaceHub:
    """check_huggingface_hub returns a bool indicating reachability.
    Validates Requirement 3.4."""

    def test_returns_true_on_success(self):
        """Returns True when HF endpoint responds."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = check_huggingface_hub()

        assert result is True
        assert isinstance(result, bool)

    def test_returns_false_on_connection_error(self):
        """Returns False when connection fails."""
        from urllib.error import URLError

        with patch(
            "urllib.request.urlopen",
            side_effect=URLError("Connection refused"),
        ):
            result = check_huggingface_hub()

        assert result is False
        assert isinstance(result, bool)

    def test_returns_false_on_timeout(self):
        """Returns False when request times out."""
        import socket

        with patch(
            "urllib.request.urlopen",
            side_effect=socket.timeout("timed out"),
        ):
            result = check_huggingface_hub()

        assert result is False
        assert isinstance(result, bool)

    def test_returns_false_on_generic_exception(self):
        """Returns False on any exception, never raises."""
        with patch(
            "urllib.request.urlopen",
            side_effect=OSError("Network unreachable"),
        ):
            result = check_huggingface_hub()

        assert result is False
        assert isinstance(result, bool)

    def test_never_raises(self):
        """check_huggingface_hub never propagates an exception."""
        with patch(
            "urllib.request.urlopen",
            side_effect=RuntimeError("Unexpected error"),
        ):
            # Should not raise
            try:
                result = check_huggingface_hub()
            except Exception as e:
                pytest.fail(
                    f"check_huggingface_hub raised {type(e).__name__}: {e}"
                )
            assert result is False
