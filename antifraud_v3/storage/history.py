"""Call history persistence — SQLite, single file. See REWRITE_PLAN.md §9 (History screen)
and §6 (the same per-call log doubles as eval material once real usage exists).
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from ..pipeline.call_state import CallState

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.sqlite3"


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at REAL NOT NULL,
                duration_seconds REAL,
                final_risk_level TEXT,
                transcript_json TEXT,
                risk_trajectory_json TEXT,
                ended_reason TEXT
            )
            """
        )


def create_call() -> int:
    """Call at connection start. Returns the row id to pass to finish_call()."""
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO calls (started_at, ended_reason) VALUES (?, ?)",
            (time.time(), "in_progress"),
        )
        return cursor.lastrowid


def finish_call(call_id: int, call_state: CallState, ended_reason: str = "stopped") -> None:
    """Call when a live call ends — persists the transcript + risk trajectory + a summary
    (highest risk level reached) rather than every intermediate LLM justification text, to
    keep rows small; the full per-chunk justification is still in risk_trajectory_json.
    """
    levels_by_severity = {"low": 0, "medium": 1, "high": 2}
    final_level = "low"
    for point in call_state.risk_trajectory:
        if levels_by_severity.get(point.risk_level, 0) > levels_by_severity.get(final_level, 0):
            final_level = point.risk_level

    with _connect() as conn:
        conn.execute(
            """
            UPDATE calls
            SET duration_seconds = ?, final_risk_level = ?, transcript_json = ?,
                risk_trajectory_json = ?, ended_reason = ?
            WHERE id = ?
            """,
            (
                call_state.elapsed_seconds(),
                final_level,
                json.dumps([asdict(t) for t in call_state.transcript], ensure_ascii=False),
                json.dumps([asdict(r) for r in call_state.risk_trajectory], ensure_ascii=False),
                ended_reason,
                call_id,
            ),
        )


def list_calls(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, started_at, duration_seconds, final_risk_level, ended_reason
            FROM calls
            WHERE ended_reason != 'in_progress'
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_call_detail(call_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM calls WHERE id = ?", (call_id,)).fetchone()
        if row is None:
            return None
        detail = dict(row)
        detail["transcript"] = json.loads(detail.pop("transcript_json") or "[]")
        detail["risk_trajectory"] = json.loads(detail.pop("risk_trajectory_json") or "[]")
        return detail
