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
    height        INTEGER
);
"""


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
    """Create tables if they don't exist."""
    Config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _get_connection()
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


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
        artifacts: List of dicts with keys: path, seed, width, height.

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
                INSERT INTO artifact (job_id, path, seed, width, height)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    artifact["path"],
                    artifact["seed"],
                    artifact.get("width"),
                    artifact.get("height"),
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
            SELECT id, job_id, path, seed, width, height
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
            )
            for row in rows
        ]
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
