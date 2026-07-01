"""Property-based tests for ui/handlers.py — error boundary and graceful degradation.

Covers:
- Property 7: Error handler produces plain-language output without tracebacks
- Property 20: Graceful degradation preserves app state on failure

Validates: Requirements 4.5, 11.1, 11.3
"""

from __future__ import annotations

import json
import re
import sys
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from studio.config import Config
from studio.ui.handlers import friendly, LOG_PATH, on_generate, on_download


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for exception messages (including adversarial ones with traceback-like text)
_exception_message_st = st.one_of(
    st.text(min_size=0, max_size=200),
    # Messages that look like tracebacks — friendly() must NOT leak them
    st.sampled_from([
        'File "/usr/lib/python3.11/site.py", line 73, in <module>',
        "Traceback (most recent call last):",
        '  File "studio/models/backends/krea2.py", line 42, in generate',
        "RuntimeError: CUDA out of memory",
        "MemoryError: not enough memory",
        "out of memory. Tried to allocate 2.00 GiB",
        "OOM killer invoked",
    ]),
)

# Strategy for generating exception instances across many types
_exception_st = st.one_of(
    st.builds(MemoryError),
    st.builds(ConnectionError, _exception_message_st),
    st.builds(TimeoutError, _exception_message_st),
    st.builds(FileNotFoundError, _exception_message_st),
    st.builds(ValueError, _exception_message_st),
    st.builds(RuntimeError, _exception_message_st),
    st.builds(TypeError, _exception_message_st),
    st.builds(KeyError, _exception_message_st),
    st.builds(IndexError, _exception_message_st),
    st.builds(OSError, _exception_message_st),
    st.builds(AttributeError, _exception_message_st),
    st.builds(ImportError, _exception_message_st),
    st.builds(PermissionError, _exception_message_st),
    st.builds(NotImplementedError, _exception_message_st),
    st.builds(ArithmeticError, _exception_message_st),
    st.builds(ZeroDivisionError, _exception_message_st),
)

# Traceback-like patterns that must never appear in user-facing output
_TRACEBACK_PATTERNS = [
    re.compile(r'File ".*", line \d+'),
    re.compile(r"Traceback \(most recent call last\)"),
]

# Raw exception class name patterns (e.g., "MemoryError:", "RuntimeError:")
_EXCEPTION_CLASS_NAME_PATTERN = re.compile(
    r"\b(MemoryError|RuntimeError|TypeError|KeyError|IndexError|OSError|IOError|"
    r"AttributeError|ImportError|PermissionError|NotImplementedError|"
    r"StopIteration|ArithmeticError|ZeroDivisionError|UnicodeError|"
    r"BufferError|LookupError|ConnectionError|TimeoutError|"
    r"FileNotFoundError)\s*:"
)


# ---------------------------------------------------------------------------
# Property 7: Error handler produces plain-language output without tracebacks
# ---------------------------------------------------------------------------

# Feature: cinderworks, Property 7: Error handler produces plain-language output without tracebacks
class TestPropertyErrorHandlerPlainLanguage:
    """Property 7: For any exception raised within a handler, the friendly()
    error mapping produces a string that:
    - Contains a plain-language description of the failure category
    - Includes the log file path
    - Does NOT contain Python traceback frames
    - Does NOT contain raw exception class names visible to the user

    **Validates: Requirements 4.5, 11.1, 11.3**
    """

    @given(exc=_exception_st)
    @settings(max_examples=100)
    def test_friendly_contains_log_path(self, exc):
        """friendly() output always includes the log file path."""
        result = friendly(exc)
        assert str(LOG_PATH) in result, (
            f"friendly() output must include log path '{LOG_PATH}', got: {result!r}"
        )

    @given(exc=_exception_st)
    @settings(max_examples=100)
    def test_friendly_no_traceback_frames(self, exc):
        """friendly() output never contains Python traceback frame lines."""
        result = friendly(exc)
        for pattern in _TRACEBACK_PATTERNS:
            assert not pattern.search(result), (
                f"friendly() output must not contain traceback patterns. "
                f"Found match for {pattern.pattern!r} in: {result!r}"
            )

    @given(exc=_exception_st)
    @settings(max_examples=100)
    def test_friendly_no_raw_exception_class_names(self, exc):
        """friendly() output never contains raw exception class names (e.g. 'MemoryError:')."""
        result = friendly(exc)
        assert not _EXCEPTION_CLASS_NAME_PATTERN.search(result), (
            f"friendly() output must not contain raw exception class names. "
            f"Found match in: {result!r}"
        )

    @given(exc=_exception_st)
    @settings(max_examples=100)
    def test_friendly_returns_nonempty_string(self, exc):
        """friendly() always returns a non-empty string (plain-language description)."""
        result = friendly(exc)
        assert isinstance(result, str)
        assert len(result.strip()) > 0


# ---------------------------------------------------------------------------
# Property 20: Graceful degradation preserves app state on failure
# ---------------------------------------------------------------------------

# Feature: cinderworks, Property 20: Graceful degradation preserves app state on failure
class TestPropertyGracefulDegradation:
    """Property 20: For any failure occurring in the downloader, a model backend,
    or during a generation:
    - The application remains running (handler returns a string, doesn't raise)
    - The failure is surfaced as a plain-language string prefixed with ❌
    - Previously persisted jobs remain intact and accessible

    **Validates: Requirements 11.1, 11.3**
    """

    @pytest.fixture(autouse=True)
    def _setup_tmp_db(self, tmp_path, monkeypatch):
        """Set up a temp DB with pre-existing jobs for each test."""
        from studio.db.db import init_db, create_job

        db_path = tmp_path / "test.db"
        monkeypatch.setattr(Config, "DB_PATH", db_path)
        init_db()

        # Pre-persist some jobs so we can verify they survive failures
        self._pre_existing_job_ids = []
        for i in range(3):
            job_id = create_job(
                prompt=f"pre-existing job {i}",
                params_json=json.dumps({"steps": 8, "seed": i}),
                seed=i,
                model_id="krea2-turbo",
                duration_ms=100 * (i + 1),
                status="complete",
                artifacts=[
                    {"path": f"outputs/job_{i}/0.png", "seed": i, "width": 1024, "height": 1024}
                ],
            )
            self._pre_existing_job_ids.append(job_id)

    @given(exc=_exception_st)
    @settings(max_examples=100)
    def test_on_generate_catches_all_exceptions(self, exc):
        """on_generate never raises — always yields a tuple with ❌ in the text."""
        # Create mock modules that simulate the imports inside on_generate
        mock_system_check = MagicMock()
        mock_system_check.is_ready_to_generate.return_value = True
        mock_system_check.get_readiness_banner.return_value = {"value": "ready"}

        mock_registry = MagicMock()
        mock_registry.run_generation.side_effect = exc

        mock_db = MagicMock()

        mock_controls = MagicMock()
        mock_controls.validate_ui_params.return_value = {
            "prompt": "test", "steps": 8, "seed": 42,
            "width": 1024, "height": 1024,
            "precision": "bf16", "batch_size": 1, "batch_count": 1,
        }

        with patch("studio.core.system_check.is_ready_to_generate", mock_system_check.is_ready_to_generate), \
             patch("studio.core.system_check.get_readiness_banner", mock_system_check.get_readiness_banner), \
             patch("studio.models.registry.run_generation", mock_registry.run_generation), \
             patch("studio.ui.controls.validate_ui_params", mock_controls.validate_ui_params), \
             patch("studio.db.db.init_db", mock_db.init_db), \
             patch("studio.db.db.create_job", mock_db.create_job):

            # on_generate is a generator yielding (text, gallery) tuples — consume it
            results = list(on_generate(
                prompt="test prompt",
                steps=8,
                seed=42,
                width=1024,
                height=1024,
                precision="bf16",
                batch_size=1,
                batch_count=1,
            ))

            # Handler must not raise — we got results
            assert len(results) > 0
            # Each result is a (text, gallery_list) tuple
            final = results[-1]
            assert isinstance(final, tuple)
            text, gallery = final
            assert isinstance(text, str)
            assert "❌" in text

    @given(exc=_exception_st)
    @settings(max_examples=100)
    def test_on_download_catches_all_exceptions(self, exc):
        """on_download never raises — always yields a string including ❌."""
        mock_downloader = MagicMock()
        mock_downloader.download_all_models_generator.side_effect = exc

        with patch("studio.models.downloader.download_all_models_generator", mock_downloader.download_all_models_generator), \
             patch("studio.core.system_check.check_cuda_status", MagicMock()), \
             patch("studio.core.system_check.get_readiness_banner", MagicMock(return_value={"visible": False})):

            # on_download is a generator — consume it
            results = list(on_download("krea2-turbo"))

            # Handler must not raise — we got results
            assert len(results) > 0
            # The last yielded value must contain ❌
            final = results[-1]
            assert isinstance(final, str)
            assert "❌" in final

    @given(exc=_exception_st)
    @settings(max_examples=100)
    def test_pre_existing_jobs_survive_generation_failure(self, exc):
        """After a generation failure, all previously persisted jobs remain accessible."""
        from studio.db.db import get_job, get_recent_jobs

        mock_system_check = MagicMock()
        mock_system_check.is_ready_to_generate.return_value = True
        mock_system_check.get_readiness_banner.return_value = {"value": "ready"}

        mock_registry = MagicMock()
        mock_registry.run_generation.side_effect = exc

        mock_controls = MagicMock()
        mock_controls.validate_ui_params.return_value = {
            "prompt": "test", "steps": 8, "seed": 42,
            "width": 1024, "height": 1024,
            "precision": "bf16", "batch_size": 1, "batch_count": 1,
        }

        with patch("studio.core.system_check.is_ready_to_generate", mock_system_check.is_ready_to_generate), \
             patch("studio.core.system_check.get_readiness_banner", mock_system_check.get_readiness_banner), \
             patch("studio.models.registry.run_generation", mock_registry.run_generation), \
             patch("studio.ui.controls.validate_ui_params", mock_controls.validate_ui_params):

            # Trigger a failing generation
            list(on_generate(
                prompt="failing prompt",
                steps=8,
                seed=42,
                width=1024,
                height=1024,
                precision="bf16",
                batch_size=1,
                batch_count=1,
            ))

        # Verify pre-existing jobs are still intact (use real DB)
        for job_id in self._pre_existing_job_ids:
            job = get_job(job_id)
            assert job is not None, f"Pre-existing job {job_id} should still be accessible"
            assert job.status == "complete"

        # Verify all jobs are listed in history
        jobs = get_recent_jobs(limit=20, offset=0)
        assert len(jobs) >= 3
        listed_ids = {j.id for j in jobs}
        for job_id in self._pre_existing_job_ids:
            assert job_id in listed_ids

    @given(exc=_exception_st)
    @settings(max_examples=100)
    def test_error_output_is_plain_language(self, exc):
        """Failure messages from handlers are plain-language (no tracebacks)."""
        mock_downloader = MagicMock()
        mock_downloader.download_all_models_generator.side_effect = exc

        with patch("studio.models.downloader.download_all_models_generator", mock_downloader.download_all_models_generator), \
             patch("studio.core.system_check.check_cuda_status", MagicMock()), \
             patch("studio.core.system_check.get_readiness_banner", MagicMock(return_value={"visible": False})):

            results = list(on_download("krea2-turbo"))
            final = results[-1]

            # No traceback patterns in the error output
            for pattern in _TRACEBACK_PATTERNS:
                assert not pattern.search(final), (
                    f"Handler error output must not contain traceback patterns. "
                    f"Found match for {pattern.pattern!r} in: {final!r}"
                )

            # No raw exception class names
            assert not _EXCEPTION_CLASS_NAME_PATTERN.search(final), (
                f"Handler error output must not contain raw exception class names. "
                f"Found in: {final!r}"
            )


# ---------------------------------------------------------------------------
# Unit Tests for Task 10.6
# ---------------------------------------------------------------------------
# Requirements: 5.7, 4.4, 11.1


class TestSpinnerLifecycle:
    """spinner_on returns visible=True with loading message; spinner_off returns visible=False."""

    def test_spinner_on_returns_visible_true(self):
        from studio.ui.handlers import spinner_on
        result = spinner_on()
        assert isinstance(result, dict)
        assert result["visible"] is True

    def test_spinner_on_contains_loading_message(self):
        from studio.ui.handlers import spinner_on
        result = spinner_on()
        assert "value" in result
        assert isinstance(result["value"], str)
        assert len(result["value"]) > 0

    def test_spinner_off_returns_visible_false(self):
        from studio.ui.handlers import spinner_off
        result = spinner_off()
        assert isinstance(result, dict)
        assert result["visible"] is False


class TestEmptyPromptRefused:
    """on_generate('') yields a ❌ string containing 'prompt' — refuses before generation."""

    def test_empty_string_refused(self):
        with patch(
            "studio.core.system_check.is_ready_to_generate", return_value=True
        ), patch(
            "studio.core.system_check.get_readiness_banner",
            return_value={"visible": False, "value": ""},
        ):
            results = list(on_generate("", 8, -1, 1024, 1024, "bf16", 1, 1))
            assert len(results) >= 1
            text, gallery = results[-1]
            assert "❌" in text
            assert "prompt" in text.lower()

    def test_whitespace_only_refused(self):
        with patch(
            "studio.core.system_check.is_ready_to_generate", return_value=True
        ), patch(
            "studio.core.system_check.get_readiness_banner",
            return_value={"visible": False, "value": ""},
        ):
            results = list(on_generate("   ", 8, -1, 1024, 1024, "bf16", 1, 1))
            assert len(results) >= 1
            text, gallery = results[-1]
            assert "❌" in text
            assert "prompt" in text.lower()


class TestHandlerNeverReRaises:
    """When any dependency raises, the handler catches it and yields a string."""

    def test_runtime_error_caught(self):
        """RuntimeError during generation is caught and surfaced as string."""
        with patch(
            "studio.core.system_check.is_ready_to_generate", return_value=True
        ), patch(
            "studio.core.system_check.get_readiness_banner",
            return_value={"visible": False, "value": ""},
        ), patch(
            "studio.ui.controls.validate_ui_params",
            side_effect=RuntimeError("Something exploded"),
        ):
            results = list(
                on_generate("a nice cat", 8, 42, 1024, 1024, "bf16", 1, 1)
            )
            assert len(results) >= 1
            assert all(isinstance(r, tuple) and isinstance(r[0], str) for r in results)
            assert any("❌" in r[0] for r in results)

    def test_memory_error_caught(self):
        """MemoryError during generation is caught and surfaced as plain text."""
        with patch(
            "studio.core.system_check.is_ready_to_generate", return_value=True
        ), patch(
            "studio.core.system_check.get_readiness_banner",
            return_value={"visible": False, "value": ""},
        ), patch(
            "studio.ui.controls.validate_ui_params",
            side_effect=MemoryError("CUDA OOM"),
        ):
            results = list(
                on_generate("a nice cat", 8, 42, 1024, 1024, "bf16", 1, 1)
            )
            assert len(results) >= 1
            assert all(isinstance(r, tuple) and isinstance(r[0], str) for r in results)
            assert any("❌" in r[0] for r in results)
            assert any("VRAM" in r[0] for r in results)

    def test_connection_error_caught(self):
        """ConnectionError during generation is caught and surfaced as plain text."""
        with patch(
            "studio.core.system_check.is_ready_to_generate", return_value=True
        ), patch(
            "studio.core.system_check.get_readiness_banner",
            return_value={"visible": False, "value": ""},
        ), patch(
            "studio.ui.controls.validate_ui_params",
            side_effect=ConnectionError("Network down"),
        ):
            results = list(
                on_generate("a nice cat", 8, 42, 1024, 1024, "bf16", 1, 1)
            )
            assert len(results) >= 1
            assert all(isinstance(r, tuple) and isinstance(r[0], str) for r in results)
            assert any("❌" in r[0] for r in results)

    def test_unexpected_exception_caught(self):
        """Totally unexpected exception is also caught — never propagates."""
        with patch(
            "studio.core.system_check.is_ready_to_generate", return_value=True
        ), patch(
            "studio.core.system_check.get_readiness_banner",
            return_value={"visible": False, "value": ""},
        ), patch(
            "studio.ui.controls.validate_ui_params",
            side_effect=ZeroDivisionError("oops"),
        ):
            results = list(
                on_generate("a nice cat", 8, 42, 1024, 1024, "bf16", 1, 1)
            )
            assert len(results) >= 1
            assert all(isinstance(r, tuple) and isinstance(r[0], str) for r in results)
            assert any("❌" in r[0] for r in results)


class TestGenerationRefusedWhenNotReady:
    """When system_check.is_ready_to_generate() returns False, on_generate yields ❌ with reasons."""

    def test_not_ready_shows_error_with_reasons(self):
        """Refusing generation shows specific unmet conditions."""
        banner_text = "CUDA not detected. Krea 2 Turbo model not downloaded yet."
        with patch(
            "studio.core.system_check.is_ready_to_generate", return_value=False
        ), patch(
            "studio.core.system_check.get_readiness_banner",
            return_value={"visible": True, "value": banner_text},
        ):
            results = list(
                on_generate("a nice cat", 8, 42, 1024, 1024, "bf16", 1, 1)
            )
            assert len(results) == 1
            text, gallery = results[0]
            assert "❌" in text
            assert "CUDA not detected" in text
            assert "model not downloaded" in text

    def test_not_ready_does_not_start_generation(self):
        """When not ready, registry.run_generation is never called."""
        with patch(
            "studio.core.system_check.is_ready_to_generate", return_value=False
        ), patch(
            "studio.core.system_check.get_readiness_banner",
            return_value={"visible": True, "value": "Model not downloaded."},
        ), patch(
            "studio.models.registry.run_generation"
        ) as mock_gen:
            list(on_generate("a nice cat", 8, 42, 1024, 1024, "bf16", 1, 1))
            mock_gen.assert_not_called()

    def test_not_ready_single_reason(self):
        """A single unmet condition is included in the error."""
        with patch(
            "studio.core.system_check.is_ready_to_generate", return_value=False
        ), patch(
            "studio.core.system_check.get_readiness_banner",
            return_value={"visible": True, "value": "CUDA not detected."},
        ):
            results = list(
                on_generate("test prompt", 8, 42, 1024, 1024, "bf16", 1, 1)
            )
            assert len(results) == 1
            text, gallery = results[0]
            assert "CUDA not detected" in text
