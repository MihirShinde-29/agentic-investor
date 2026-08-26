"""SQLite-backed persistence for paper portfolio recommendations.

One table, one JSON blob per recommendation. Stdlib sqlite3 keeps the dep list
short; when we outgrow it (multi-user, richer queries) it swaps to Postgres
via SQLAlchemy without touching callers.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from agentic_investor.config import get_settings
from agentic_investor.orchestrator.state import Recommendation


def _db_path(url: str) -> Path:
    if not url.startswith("sqlite:///"):
        raise ValueError(f"only sqlite:/// URLs supported for now (got {url!r})")
    return Path(url.removeprefix("sqlite:///"))


def _connect(url: str) -> sqlite3.Connection:
    path = _db_path(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    return conn


def _resolve_url(url: str | None) -> str:
    return url if url is not None else get_settings().database_url


def save_recommendation(rec: Recommendation, *, url: str | None = None) -> int:
    """Persist a recommendation and return its new row id."""
    with _connect(_resolve_url(url)) as conn:
        cur = conn.execute(
            "INSERT INTO recommendations (created_at, payload_json) VALUES (?, ?)",
            (datetime.now(UTC).isoformat(), rec.model_dump_json()),
        )
        conn.commit()
        return int(cur.lastrowid)


def load_recommendation(rec_id: int, *, url: str | None = None) -> Recommendation | None:
    with _connect(_resolve_url(url)) as conn:
        row = conn.execute(
            "SELECT payload_json FROM recommendations WHERE id = ?", (rec_id,)
        ).fetchone()
    if row is None:
        return None
    return Recommendation.model_validate_json(row[0])


def list_recommendations(
    *, url: str | None = None, limit: int = 10
) -> list[tuple[int, str]]:
    """Return (id, created_at) for the most recent recommendations, newest first."""
    with _connect(_resolve_url(url)) as conn:
        return conn.execute(
            "SELECT id, created_at FROM recommendations ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
