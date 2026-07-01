"""Tests for db/db.py — SQLite persistence module.

Covers schema creation, CRUD operations, ordering, paging, and round-trip
integrity per Requirements 8.1, 8.3, 8.5.
"""

import json
import sqlite3

import pytest

from studio.config import Config
from studio.db.db import (
    Artifact,
    Job,
    JobSummary,
    create_job,
    get_job,
    get_job_artifacts,
    get_recent_jobs,
    init_db,
)


@pytest.fixture(autouse=True)
def _use_tmp_db(tmp_path, monkeypatch):
    """Point Config.DB_PATH to a temp file for each test."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(Config, "DB_PATH", db_path)
    init_db()


# ---------------------------------------------------------------------------
# Schema / init_db
# ---------------------------------------------------------------------------


class TestInitDb:
    def test_creates_job_table(self):
        """init_db creates the job table."""
        conn = sqlite3.connect(str(Config.DB_PATH))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='job'"
        ).fetchall()
        conn.close()
        assert len(tables) == 1

    def test_creates_artifact_table(self):
        """init_db creates the artifact table."""
        conn = sqlite3.connect(str(Config.DB_PATH))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact'"
        ).fetchall()
        conn.close()
        assert len(tables) == 1

    def test_idempotent(self):
        """Calling init_db twice does not raise."""
        init_db()
        init_db()


# ---------------------------------------------------------------------------
# create_job
# ---------------------------------------------------------------------------


class TestCreateJob:
    def test_returns_integer_id(self):
        """create_job returns the new job's integer id."""
        job_id = create_job(
            prompt="a cat",
            params_json=json.dumps({"steps": 8, "cfg": 1.0}),
            seed=42,
            model_id="krea2-turbo",
            duration_ms=1500,
            status="complete",
            artifacts=[],
        )
        assert isinstance(job_id, int)
        assert job_id >= 1

    def test_persists_all_fields(self):
        """All job fields are stored and retrievable."""
        params = {"steps": 8, "cfg": 1.0, "width": 1024, "height": 1024}
        job_id = create_job(
            prompt="a dog in the park",
            params_json=json.dumps(params),
            seed=12345,
            model_id="krea2-turbo",
            duration_ms=2000,
            status="complete",
            artifacts=[],
        )
        job = get_job(job_id)
        assert job is not None
        assert job.prompt == "a dog in the park"
        assert json.loads(job.params_json) == params
        assert job.seed == 12345
        assert job.model_id == "krea2-turbo"
        assert job.duration_ms == 2000
        assert job.status == "complete"

    def test_persists_artifacts(self):
        """Artifacts are stored with correct metadata."""
        artifacts = [
            {"path": "outputs/job_1/0.png", "seed": 42, "width": 1024, "height": 1024},
            {"path": "outputs/job_1/1.png", "seed": 43, "width": 1024, "height": 1024},
        ]
        job_id = create_job(
            prompt="test",
            params_json=json.dumps({"steps": 8}),
            seed=42,
            model_id="krea2-turbo",
            duration_ms=1000,
            status="complete",
            artifacts=artifacts,
        )
        result = get_job_artifacts(job_id)
        assert len(result) == 2
        assert result[0].path == "outputs/job_1/0.png"
        assert result[0].seed == 42
        assert result[0].width == 1024
        assert result[0].height == 1024
        assert result[1].seed == 43

    def test_artifacts_with_none_dimensions(self):
        """Artifacts can have None for width/height."""
        artifacts = [
            {"path": "outputs/job_1/0.png", "seed": 42},
        ]
        job_id = create_job(
            prompt="test",
            params_json=json.dumps({"steps": 8}),
            seed=42,
            model_id="krea2-turbo",
            duration_ms=None,
            status="failed",
            artifacts=artifacts,
        )
        result = get_job_artifacts(job_id)
        assert len(result) == 1
        assert result[0].width is None
        assert result[0].height is None

    def test_duration_ms_nullable(self):
        """duration_ms can be None (failed job)."""
        job_id = create_job(
            prompt="test",
            params_json=json.dumps({}),
            seed=1,
            model_id="krea2-turbo",
            duration_ms=None,
            status="failed",
            artifacts=[],
        )
        job = get_job(job_id)
        assert job.duration_ms is None

    def test_stores_params_as_json_blob(self):
        """params_json is stored as a JSON string and round-trips correctly."""
        params = {
            "steps": 8,
            "cfg": 1.0,
            "mu_shift": 1.15,
            "width": 1024,
            "height": 1024,
            "precision": "bf16",
            "batch_size": 2,
            "batch_count": 3,
        }
        job_id = create_job(
            prompt="test",
            params_json=json.dumps(params),
            seed=99,
            model_id="krea2-turbo",
            duration_ms=500,
            status="complete",
            artifacts=[],
        )
        job = get_job(job_id)
        assert json.loads(job.params_json) == params


# ---------------------------------------------------------------------------
# get_recent_jobs
# ---------------------------------------------------------------------------


class TestGetRecentJobs:
    def _create_jobs(self, count: int) -> list[int]:
        """Helper to create N jobs with sequential data."""
        ids = []
        for i in range(count):
            job_id = create_job(
                prompt=f"prompt number {i}",
                params_json=json.dumps({"steps": 8}),
                seed=i,
                model_id="krea2-turbo",
                duration_ms=100 * i,
                status="complete",
                artifacts=[],
            )
            ids.append(job_id)
        return ids

    def test_returns_list_of_job_summaries(self):
        """get_recent_jobs returns JobSummary instances."""
        self._create_jobs(3)
        jobs = get_recent_jobs()
        assert all(isinstance(j, JobSummary) for j in jobs)

    def test_ordered_by_created_at_desc(self):
        """Jobs are returned newest-first."""
        self._create_jobs(5)
        jobs = get_recent_jobs()
        # created_at should be in descending order
        timestamps = [j.created_at for j in jobs]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_default_limit_is_20(self):
        """Default limit returns at most 20 jobs."""
        self._create_jobs(25)
        jobs = get_recent_jobs()
        assert len(jobs) == 20

    def test_custom_limit(self):
        """Custom limit is respected."""
        self._create_jobs(10)
        jobs = get_recent_jobs(limit=5)
        assert len(jobs) == 5

    def test_offset_skips_rows(self):
        """Offset skips the correct number of rows."""
        self._create_jobs(10)
        all_jobs = get_recent_jobs(limit=10, offset=0)
        page_2 = get_recent_jobs(limit=5, offset=5)
        assert page_2 == all_jobs[5:]

    def test_prompt_truncated_to_120_chars(self):
        """Long prompts are truncated to 120 characters in summaries."""
        long_prompt = "x" * 200
        create_job(
            prompt=long_prompt,
            params_json=json.dumps({}),
            seed=1,
            model_id="krea2-turbo",
            duration_ms=100,
            status="complete",
            artifacts=[],
        )
        jobs = get_recent_jobs()
        assert len(jobs[0].prompt) == 120

    def test_short_prompt_not_truncated(self):
        """Prompts shorter than 120 chars are not modified."""
        short_prompt = "a beautiful sunset"
        create_job(
            prompt=short_prompt,
            params_json=json.dumps({}),
            seed=1,
            model_id="krea2-turbo",
            duration_ms=100,
            status="complete",
            artifacts=[],
        )
        jobs = get_recent_jobs()
        assert jobs[0].prompt == short_prompt

    def test_empty_db_returns_empty_list(self):
        """No jobs → empty list, not an error."""
        jobs = get_recent_jobs()
        assert jobs == []

    def test_offset_beyond_total_returns_empty(self):
        """Offset past total rows returns empty list."""
        self._create_jobs(5)
        jobs = get_recent_jobs(limit=20, offset=100)
        assert jobs == []


# ---------------------------------------------------------------------------
# get_job
# ---------------------------------------------------------------------------


class TestGetJob:
    def test_returns_full_job(self):
        """get_job returns a complete Job dataclass."""
        job_id = create_job(
            prompt="full job test",
            params_json=json.dumps({"steps": 12}),
            seed=777,
            model_id="krea2-turbo",
            duration_ms=3000,
            status="complete",
            artifacts=[],
        )
        job = get_job(job_id)
        assert isinstance(job, Job)
        assert job.id == job_id
        assert job.prompt == "full job test"

    def test_nonexistent_id_returns_none(self):
        """get_job returns None for an id that doesn't exist."""
        result = get_job(99999)
        assert result is None

    def test_prompt_not_truncated(self):
        """get_job returns the full prompt (unlike get_recent_jobs)."""
        long_prompt = "y" * 300
        job_id = create_job(
            prompt=long_prompt,
            params_json=json.dumps({}),
            seed=1,
            model_id="krea2-turbo",
            duration_ms=100,
            status="complete",
            artifacts=[],
        )
        job = get_job(job_id)
        assert job.prompt == long_prompt
        assert len(job.prompt) == 300


# ---------------------------------------------------------------------------
# get_job_artifacts
# ---------------------------------------------------------------------------


class TestGetJobArtifacts:
    def test_returns_artifacts_for_job(self):
        """Retrieves correct artifacts for a given job_id."""
        artifacts_data = [
            {"path": "outputs/job_1/0.png", "seed": 10, "width": 512, "height": 512},
            {"path": "outputs/job_1/1.png", "seed": 11, "width": 512, "height": 768},
        ]
        job_id = create_job(
            prompt="test",
            params_json=json.dumps({}),
            seed=10,
            model_id="krea2-turbo",
            duration_ms=500,
            status="complete",
            artifacts=artifacts_data,
        )
        result = get_job_artifacts(job_id)
        assert len(result) == 2
        assert all(isinstance(a, Artifact) for a in result)
        assert result[0].job_id == job_id
        assert result[1].job_id == job_id

    def test_empty_artifacts(self):
        """Job with no artifacts returns empty list."""
        job_id = create_job(
            prompt="no images",
            params_json=json.dumps({}),
            seed=1,
            model_id="krea2-turbo",
            duration_ms=None,
            status="failed",
            artifacts=[],
        )
        result = get_job_artifacts(job_id)
        assert result == []

    def test_does_not_return_other_jobs_artifacts(self):
        """Artifacts from different jobs are not mixed."""
        job_1 = create_job(
            prompt="job 1",
            params_json=json.dumps({}),
            seed=1,
            model_id="krea2-turbo",
            duration_ms=100,
            status="complete",
            artifacts=[{"path": "a.png", "seed": 1, "width": 512, "height": 512}],
        )
        job_2 = create_job(
            prompt="job 2",
            params_json=json.dumps({}),
            seed=2,
            model_id="krea2-turbo",
            duration_ms=200,
            status="complete",
            artifacts=[{"path": "b.png", "seed": 2, "width": 1024, "height": 1024}],
        )
        arts_1 = get_job_artifacts(job_1)
        arts_2 = get_job_artifacts(job_2)
        assert len(arts_1) == 1
        assert arts_1[0].path == "a.png"
        assert len(arts_2) == 1
        assert arts_2[0].path == "b.png"


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

from datetime import datetime, timezone, timedelta
from hypothesis import given, settings
from hypothesis import strategies as st


# Feature: cinderworks, Property 16: History listing ordered by creation time descending
class TestPropertyHistoryListingOrder:
    """Property 16: History listing ordered by creation time descending.

    For any set of N jobs with distinct creation timestamps, get_recent_jobs
    SHALL return them in strictly descending creation-time order, with prompts
    truncated to 120 characters.

    **Validates: Requirements 8.3**
    """

    @given(
        n=st.integers(min_value=2, max_value=30),
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_history_ordered_descending_with_truncation(self, n, data):
        """get_recent_jobs returns jobs in strictly descending created_at order
        and all prompts are truncated to at most 120 characters."""
        # Generate N distinct prompts, some > 120 chars to test truncation
        prompts = data.draw(
            st.lists(
                st.text(
                    alphabet=st.characters(categories=("L", "N", "P", "S", "Z")),
                    min_size=1,
                    max_size=250,
                ),
                min_size=n,
                max_size=n,
            )
        )

        # Generate N distinct timestamps by picking a base and adding unique offsets
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        offsets = data.draw(
            st.lists(
                st.integers(min_value=0, max_value=1_000_000),
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
        timestamps = [
            (base_time + timedelta(seconds=off)).isoformat() for off in offsets
        ]

        # Insert jobs directly into the database with controlled timestamps
        conn = sqlite3.connect(str(Config.DB_PATH))
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            for prompt, ts in zip(prompts, timestamps):
                conn.execute(
                    """
                    INSERT INTO job (created_at, model_id, prompt, params_json, seed, duration_ms, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (ts, "krea2-turbo", prompt, "{}", 42, 100, "complete"),
                )
            conn.commit()
        finally:
            conn.close()

        # Retrieve all jobs (limit high enough to get all N)
        jobs = get_recent_jobs(limit=n, offset=0)

        # Property 1: Results are in strictly descending created_at order
        assert len(jobs) == n
        for i in range(len(jobs) - 1):
            assert jobs[i].created_at > jobs[i + 1].created_at, (
                f"Job at index {i} ({jobs[i].created_at}) should be strictly after "
                f"job at index {i+1} ({jobs[i+1].created_at})"
            )

        # Property 2: All prompts are truncated to at most 120 characters
        for job in jobs:
            assert len(job.prompt) <= 120, (
                f"Prompt should be at most 120 chars, got {len(job.prompt)}"
            )

        # Clean up for hypothesis reuse (DB is per-test via fixture, but
        # hypothesis re-runs within a single test invocation)
        conn = sqlite3.connect(str(Config.DB_PATH))
        conn.execute("DELETE FROM job")
        conn.commit()
        conn.close()


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings
from hypothesis import strategies as st


# Feature: cinderworks, Property 15: Job persistence round-trip
# Validates: Requirements 8.1
class TestJobPersistenceRoundTrip:
    """Property 15: For any valid job data, persisting to the database and reading
    it back SHALL return data identical to what was written — no field loss, no
    truncation, no type coercion that changes values."""

    # Strategies for generating valid job data
    _prompt_st = st.text(min_size=1, max_size=500)
    _params_json_st = st.fixed_dictionaries(
        {
            "steps": st.integers(min_value=1, max_value=100),
            "cfg": st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False),
            "width": st.integers(min_value=512, max_value=2048).filter(lambda x: x % 64 == 0),
            "height": st.integers(min_value=512, max_value=2048).filter(lambda x: x % 64 == 0),
            "precision": st.sampled_from(["bf16", "fp8_scaled"]),
            "batch_size": st.integers(min_value=1, max_value=16),
            "batch_count": st.integers(min_value=1, max_value=100),
        }
    ).map(json.dumps)
    _seed_st = st.integers(min_value=0, max_value=2**32 - 1)
    _model_id_st = st.text(min_size=1, max_size=100)
    _duration_ms_st = st.one_of(st.none(), st.integers(min_value=0, max_value=10_000_000))
    _status_st = st.sampled_from(["complete", "failed"])
    _artifact_st = st.lists(
        st.fixed_dictionaries(
            {
                "path": st.text(min_size=1, max_size=200),
                "seed": st.integers(min_value=0, max_value=2**32 - 1),
            },
            optional={
                "width": st.integers(min_value=1, max_value=4096),
                "height": st.integers(min_value=1, max_value=4096),
            },
        ),
        min_size=0,
        max_size=5,
    )

    @given(data=st.data())
    @settings(max_examples=100)
    def test_job_fields_round_trip(self, data):
        """Persisting a job and reading it back returns identical field values."""
        prompt = data.draw(self._prompt_st)
        params_json = data.draw(self._params_json_st)
        seed = data.draw(self._seed_st)
        model_id = data.draw(self._model_id_st)
        duration_ms = data.draw(self._duration_ms_st)
        status = data.draw(self._status_st)

        job_id = create_job(
            prompt=prompt,
            params_json=params_json,
            seed=seed,
            model_id=model_id,
            duration_ms=duration_ms,
            status=status,
            artifacts=[],
        )

        job = get_job(job_id)
        assert job is not None
        assert job.id == job_id
        assert job.prompt == prompt
        assert job.params_json == params_json
        assert json.loads(job.params_json) == json.loads(params_json)
        assert job.seed == seed
        assert job.model_id == model_id
        assert job.duration_ms == duration_ms
        assert job.status == status

    @given(data=st.data())
    @settings(max_examples=100)
    def test_artifact_fields_round_trip(self, data):
        """Persisting artifacts and reading them back returns identical field values."""
        prompt = data.draw(self._prompt_st)
        params_json = data.draw(self._params_json_st)
        seed = data.draw(self._seed_st)
        model_id = data.draw(self._model_id_st)
        duration_ms = data.draw(self._duration_ms_st)
        status = data.draw(self._status_st)
        artifacts = data.draw(self._artifact_st)

        job_id = create_job(
            prompt=prompt,
            params_json=params_json,
            seed=seed,
            model_id=model_id,
            duration_ms=duration_ms,
            status=status,
            artifacts=artifacts,
        )

        result = get_job_artifacts(job_id)
        assert len(result) == len(artifacts)

        for stored, original in zip(result, artifacts):
            assert stored.job_id == job_id
            assert stored.path == original["path"]
            assert stored.seed == original["seed"]
            assert stored.width == original.get("width")
            assert stored.height == original.get("height")


# Feature: cinderworks, Property 18: History paging returns correct page sizes
# Validates: Requirements 8.5
class TestHistoryPagingPageSizes:
    """Property 18: For any total job count N, requesting page P of size 20
    SHALL return min(20, N - P×20) jobs (or 0 if P×20 ≥ N), and SHALL not
    load all rows into memory."""

    def _reset_and_bulk_insert(self, n: int):
        """Clear DB and bulk-insert N jobs directly via SQL for speed."""
        conn = sqlite3.connect(str(Config.DB_PATH))
        conn.execute("DELETE FROM artifact")
        conn.execute("DELETE FROM job")
        if n > 0:
            from datetime import datetime, timezone, timedelta

            base = datetime(2024, 1, 1, tzinfo=timezone.utc)
            rows = [
                (
                    (base + timedelta(seconds=i)).isoformat(),
                    "krea2-turbo",
                    f"job {i}",
                    '{"steps": 8}',
                    i,
                    100,
                    "complete",
                )
                for i in range(n)
            ]
            conn.executemany(
                """INSERT INTO job (created_at, model_id, prompt, params_json, seed, duration_ms, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        conn.commit()
        conn.close()

    @given(
        n=st.integers(min_value=0, max_value=80),
        page=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=100, deadline=None)
    def test_page_returns_correct_count(self, n, page):
        """Requesting page P of size 20 returns the expected number of results."""
        self._reset_and_bulk_insert(n)

        offset = page * 20
        result = get_recent_jobs(limit=20, offset=offset)
        expected = min(20, max(0, n - offset))
        assert len(result) == expected

    @given(
        n=st.integers(min_value=21, max_value=80),
        page=st.integers(min_value=0, max_value=3),
    )
    @settings(max_examples=100, deadline=None)
    def test_consecutive_pages_no_overlap(self, n, page):
        """Results from consecutive pages don't overlap."""
        self._reset_and_bulk_insert(n)

        page_a = get_recent_jobs(limit=20, offset=page * 20)
        page_b = get_recent_jobs(limit=20, offset=(page + 1) * 20)

        ids_a = {j.id for j in page_a}
        ids_b = {j.id for j in page_b}
        assert ids_a.isdisjoint(ids_b)


# ---------------------------------------------------------------------------
# DB write failure handling (Requirement 8.2)
# ---------------------------------------------------------------------------


class TestDbWriteFailure:
    """Verify that DB write failures propagate as exceptions.

    Requirement 8.2: If the database write fails, the error must be surfaceable
    to the UI layer. The DB module lets exceptions propagate; the handler layer
    catches them.
    """

    def test_create_job_raises_on_read_only_db(self, tmp_path, monkeypatch):
        """create_job raises an OperationalError when the DB is read-only."""
        import os

        db_path = tmp_path / "readonly.db"
        monkeypatch.setattr(Config, "DB_PATH", db_path)
        init_db()

        # Make the DB file read-only
        db_path.chmod(0o444)

        with pytest.raises(sqlite3.OperationalError):
            create_job(
                prompt="should fail",
                params_json=json.dumps({"steps": 8}),
                seed=1,
                model_id="krea2-turbo",
                duration_ms=100,
                status="complete",
                artifacts=[],
            )

        # Restore permissions for cleanup
        db_path.chmod(0o644)

    def test_create_job_raises_on_invalid_db_path(self, tmp_path, monkeypatch):
        """create_job raises when the DB path is in a non-existent, non-creatable location."""
        # Point to a path that cannot be created (nested under a file, not a dir)
        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file, not a directory")
        impossible_path = blocker / "subdir" / "test.db"
        monkeypatch.setattr(Config, "DB_PATH", impossible_path)

        with pytest.raises((sqlite3.OperationalError, OSError, NotADirectoryError)):
            init_db()


# ---------------------------------------------------------------------------
# Missing image file — artifact path behavior (Requirement 8.6)
# ---------------------------------------------------------------------------


class TestArtifactPathWithMissingFile:
    """Verify that get_job_artifacts returns stored paths regardless of file existence.

    Requirement 8.6: The DB module stores paths. If a referenced image file is
    missing from disk, the path is still returned. Placeholder display is a UI
    concern handled by handlers.py, not the DB layer.
    """

    def test_get_job_artifacts_returns_path_for_nonexistent_file(self):
        """Artifact path is returned even when no file exists at that path."""
        nonexistent_path = "outputs/job_999/does_not_exist.png"
        job_id = create_job(
            prompt="test missing file",
            params_json=json.dumps({"steps": 8}),
            seed=42,
            model_id="krea2-turbo",
            duration_ms=500,
            status="complete",
            artifacts=[
                {"path": nonexistent_path, "seed": 42, "width": 1024, "height": 1024},
            ],
        )
        artifacts = get_job_artifacts(job_id)
        assert len(artifacts) == 1
        assert artifacts[0].path == nonexistent_path

    def test_get_job_artifacts_returns_multiple_paths_regardless_of_existence(self):
        """Multiple artifact paths are returned even if none of the files exist."""
        paths = [
            "outputs/job_100/0.png",
            "outputs/job_100/1.png",
            "outputs/job_100/2.png",
        ]
        artifacts_data = [
            {"path": p, "seed": 42 + i, "width": 512, "height": 512}
            for i, p in enumerate(paths)
        ]
        job_id = create_job(
            prompt="multi missing",
            params_json=json.dumps({"steps": 8}),
            seed=42,
            model_id="krea2-turbo",
            duration_ms=300,
            status="complete",
            artifacts=artifacts_data,
        )
        result = get_job_artifacts(job_id)
        assert len(result) == 3
        assert [a.path for a in result] == paths
