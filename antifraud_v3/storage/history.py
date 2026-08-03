"""Call history persistence — SQLite, single file. See docs/DESIGN.md §9 (History screen)
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
                ended_reason TEXT,
                source TEXT NOT NULL DEFAULT 'live'
            )
            """
        )
        # Migration for DBs created before the upload-analysis feature (server/upload.py)
        # existed — CREATE TABLE IF NOT EXISTS is a no-op against an already-existing table,
        # so an older on-disk DB wouldn't otherwise get the new column. "live" vs "upload" is
        # how the History screen tells apart a call that was actually monitored in real time
        # from a pre-recorded file that was analyzed after the fact (e.g. so the UI doesn't
        # imply a live "duration ticking" indicator for something that was never live).
        try:
            conn.execute("ALTER TABLE calls ADD COLUMN source TEXT NOT NULL DEFAULT 'live'")
        except sqlite3.OperationalError:
            pass  # column already exists (fresh DB created by the CREATE TABLE above)


def create_call(source: str = "live") -> int:
    """Call at connection start (live) or before batch-processing an upload. Returns the row
    id to pass to finish_call(). source is "live" or "upload" — see init_db()'s migration note.
    """
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO calls (started_at, ended_reason, source) VALUES (?, ?, ?)",
            (time.time(), "in_progress", source),
        )
        return cursor.lastrowid


def finish_call(
    call_id: int,
    call_state: CallState,
    ended_reason: str = "stopped",
    duration_seconds: float | None = None,
) -> None:
    """Call when a live call ends — persists the transcript + risk trajectory + a summary
    (highest risk level reached) rather than every intermediate LLM justification text, to
    keep rows small; the full per-chunk justification is still in risk_trajectory_json.

    duration_seconds overrides call_state.elapsed_seconds() (wall-clock time since the
    CallState was constructed). For a live call that's the right measure of call length; for
    a batch-uploaded recording (server/upload.py) it isn't — elapsed_seconds() there measures
    how long ASR/LLM processing took, not the recording's actual length, so the caller passes
    the real audio duration explicitly instead.
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
                duration_seconds if duration_seconds is not None else call_state.elapsed_seconds(),
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
            SELECT id, started_at, duration_seconds, final_risk_level, ended_reason, source
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
