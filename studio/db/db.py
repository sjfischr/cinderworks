"""SQLite persistence for Cinderworks Studio.

Plain sqlite3, no ORM. Mirrors BeatBunny db/db.py pattern.
Only this module touches SQLite — handlers call these functions,
they don't write SQL inline.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from studio.config import Config


# ---------------------------------------------------------------------------
# Data classes for return types
# ---------------------------------------------------------------------------


@dataclass
class JobSummary:
    """Lightweight job representation for history listing."""

    id: int
    created_at: str
    model_id: str
    prompt: str  # truncated to 120 chars
    seed: int
    status: str
    params_json: str  # JSON blob for display (steps, resolution, precision, etc.)


@dataclass
class Job:
    """Full job record."""

    id: int
    created_at: str
    model_id: str
    prompt: str
    params_json: str
    seed: int
    duration_ms: int | None
    status: str


@dataclass
class Artifact:
    """Image artifact belonging to a job."""

    id: int
    job_id: int
    path: str
    seed: int
    width: int | None
    height: int | None
    artifact_type: str = "generated"
    source_artifact_id: int | None = None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS job (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    model_id      TEXT NOT NULL,
    prompt        TEXT NOT NULL,
    params_json   TEXT NOT NULL,
    seed          INTEGER NOT NULL,
    duration_ms   INTEGER,
    status        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        INTEGER NOT NULL REFERENCES job(id),
    path          TEXT NOT NULL,
    seed          INTEGER NOT NULL,
    width         INTEGER,
    height        INTEGER,
    artifact_type TEXT NOT NULL DEFAULT 'generated',
    source_artifact_id INTEGER REFERENCES artifact(id)
);
"""

# Migrations applied after schema creation to handle existing databases
_MIGRATIONS = [
    # M1: Add artifact_type and source_artifact_id columns for upscale tracking
    (
        "m1_artifact_type",
        [
            "ALTER TABLE artifact ADD COLUMN artifact_type TEXT NOT NULL DEFAULT 'generated'",
            "ALTER TABLE artifact ADD COLUMN source_artifact_id INTEGER REFERENCES artifact(id)",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------


def _get_connection() -> sqlite3.Connection:
    """Open a connection to the configured DB path."""
    conn = sqlite3.connect(str(Config.DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Create tables if they don't exist, then run any pending migrations."""
    Config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _get_connection()
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        _run_migrations(conn)
    finally:
        conn.close()


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Apply pending migrations that haven't been run yet.

    Migrations are idempotent — if a column already exists, the ALTER TABLE
    is skipped gracefully.
    """
    for name, statements in _MIGRATIONS:
        for sql in statements:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError as e:
                # "duplicate column name" means migration already applied — skip
                if "duplicate column" in str(e).lower():
                    continue
                raise
    conn.commit()


def create_job(
    prompt: str,
    params_json: str,
    seed: int,
    model_id: str,
    duration_ms: int | None,
    status: str,
    artifacts: list[dict],
) -> int:
    """Persist a completed (or failed) generation job with its artifacts.

    Args:
        prompt: The user's prompt text.
        params_json: JSON string of generation parameters.
        seed: The base seed used for generation.
        model_id: Registry model id (e.g. 'krea2-turbo').
        duration_ms: Generation duration in milliseconds, or None if failed.
        status: 'complete' or 'failed'.
        artifacts: List of dicts with keys: path, seed, width, height,
                   and optionally artifact_type ('generated'|'upscaled')
                   and source_artifact_id (int reference to parent artifact).

    Returns:
        The new job's integer id.
    """
    created_at = datetime.now(timezone.utc).isoformat()
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO job (created_at, model_id, prompt, params_json, seed, duration_ms, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (created_at, model_id, prompt, params_json, seed, duration_ms, status),
        )
        job_id = cursor.lastrowid

        for artifact in artifacts:
            conn.execute(
                """
                INSERT INTO artifact (job_id, path, seed, width, height, artifact_type, source_artifact_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    artifact["path"],
                    artifact["seed"],
                    artifact.get("width"),
                    artifact.get("height"),
                    artifact.get("artifact_type", "generated"),
                    artifact.get("source_artifact_id"),
                ),
            )

        conn.commit()
        return job_id
    finally:
        conn.close()


def get_recent_jobs(limit: int = 20, offset: int = 0) -> list[JobSummary]:
    """Retrieve jobs ordered by creation time descending, with paging.

    Uses LIMIT/OFFSET at the SQL level — does NOT load all rows into memory.
    Prompt is truncated to 120 characters for display.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, created_at, model_id, prompt, seed, status, params_json
            FROM job
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

        return [
            JobSummary(
                id=row["id"],
                created_at=row["created_at"],
                model_id=row["model_id"],
                prompt=row["prompt"][:120],
                seed=row["seed"],
                status=row["status"],
                params_json=row["params_json"],
            )
            for row in rows
        ]
    finally:
        conn.close()


def get_job(job_id: int) -> Job | None:
    """Retrieve a single job by id, or None if not found."""
    conn = _get_connection()
    try:
        row = conn.execute(
            """
            SELECT id, created_at, model_id, prompt, params_json, seed, duration_ms, status
            FROM job
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()

        if row is None:
            return None

        return Job(
            id=row["id"],
            created_at=row["created_at"],
            model_id=row["model_id"],
            prompt=row["prompt"],
            params_json=row["params_json"],
            seed=row["seed"],
            duration_ms=row["duration_ms"],
            status=row["status"],
        )
    finally:
        conn.close()


def get_job_artifacts(job_id: int) -> list[Artifact]:
    """Retrieve all artifacts for a given job."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, job_id, path, seed, width, height, artifact_type, source_artifact_id
            FROM artifact
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchall()

        return [
            Artifact(
                id=row["id"],
                job_id=row["job_id"],
                path=row["path"],
                seed=row["seed"],
                width=row["width"],
                height=row["height"],
                artifact_type=row["artifact_type"],
                source_artifact_id=row["source_artifact_id"],
            )
            for row in rows
        ]
    finally:
        conn.close()


def create_artifact(
    job_id: int,
    path: str,
    seed: int,
    width: int | None = None,
    height: int | None = None,
    artifact_type: str = "generated",
    source_artifact_id: int | None = None,
) -> int:
    """Persist a single artifact linked to an existing job.

    Used for adding upscaled images or other derived artifacts to a job
    after initial creation.

    Args:
        job_id: The parent job's id.
        path: File path to the artifact image.
        seed: The seed used to produce this artifact.
        width: Image width in pixels, or None.
        height: Image height in pixels, or None.
        artifact_type: 'generated' or 'upscaled'.
        source_artifact_id: If this is a derived artifact (e.g. upscaled),
                            the id of the source artifact it was created from.

    Returns:
        The new artifact's integer id.
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO artifact (job_id, path, seed, width, height, artifact_type, source_artifact_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, path, seed, width, height, artifact_type, source_artifact_id),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def find_artifact_by_path(path: str) -> Artifact | None:
    """Find an artifact record by its file path.

    Used to look up the source artifact when creating upscaled derivatives.

    Args:
        path: The file path stored in the artifact record.

    Returns:
        The matching Artifact, or None if not found.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            """
            SELECT id, job_id, path, seed, width, height, artifact_type, source_artifact_id
            FROM artifact
            WHERE path = ?
            LIMIT 1
            """,
            (path,),
        ).fetchone()

        if row is None:
            return None

        return Artifact(
            id=row["id"],
            job_id=row["job_id"],
            path=row["path"],
            seed=row["seed"],
            width=row["width"],
            height=row["height"],
            artifact_type=row["artifact_type"],
            source_artifact_id=row["source_artifact_id"],
        )
    finally:
        conn.close()


def delete_job(job_id: int) -> bool:
    """Delete a job and its artifacts from the database.

    Args:
        job_id: The job ID to delete.

    Returns:
        True if the job existed and was deleted, False if not found.
    """
    conn = _get_connection()
    try:
        # Delete artifacts first (FK constraint)
        conn.execute("DELETE FROM artifact WHERE job_id = ?", (job_id,))
        cursor = conn.execute("DELETE FROM job WHERE id = ?", (job_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()
