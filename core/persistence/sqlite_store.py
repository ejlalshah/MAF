"""
FRIDAY-MAF SQLite Persistence Layer
Stores tasks, subtasks, and events durably.
On restart, unfinished tasks are loaded and resumed automatically.
"""
from __future__ import annotations
import json
import sqlite3
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.models import Task, Event, TaskStatus
from core.observability.logger import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id     TEXT PRIMARY KEY,
    data        TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id    TEXT PRIMARY KEY,
    task_id     TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    timestamp   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_events_task  ON events(task_id);
CREATE INDEX IF NOT EXISTS idx_events_ts    ON events(timestamp);
"""


class SQLiteStore:
    """
    Thread-safe SQLite store.  All async methods run blocking I/O in an executor
    so they don't block the event loop.
    """

    def __init__(self, db_path: str = "friday_maf.db"):
        self._db_path = db_path
        self._lock    = asyncio.Lock()
        self._init_db()
        logger.info("sqlite_store_init", extra={"db_path": db_path})

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ Tasks

    async def save_task(self, task: Task) -> None:
        data = task.model_dump_json()
        async with self._lock:
            await asyncio.get_event_loop().run_in_executor(
                None, self._save_task_sync, task.task_id, data,
                task.status.value, task.created_at.isoformat(), task.updated_at.isoformat()
            )

    def _save_task_sync(self, task_id, data, status, created_at, updated_at) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO tasks (task_id, data, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(task_id) DO UPDATE SET
                       data=excluded.data, status=excluded.status,
                       updated_at=excluded.updated_at""",
                (task_id, data, status, created_at, updated_at),
            )

    async def load_task(self, task_id: str) -> Optional[Task]:
        row = await asyncio.get_event_loop().run_in_executor(
            None, self._load_task_sync, task_id
        )
        if row is None:
            return None
        return Task.model_validate_json(row["data"])

    def _load_task_sync(self, task_id: str):
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()

    async def load_all_tasks(self) -> List[Task]:
        rows = await asyncio.get_event_loop().run_in_executor(
            None, self._load_all_tasks_sync
        )
        tasks = []
        for row in rows:
            try:
                tasks.append(Task.model_validate_json(row["data"]))
            except Exception as exc:
                logger.error("task_deserialize_error", extra={"task_id": row["task_id"], "error": str(exc)})
        return tasks

    def _load_all_tasks_sync(self):
        with self._connect() as conn:
            return conn.execute("SELECT * FROM tasks ORDER BY created_at").fetchall()

    async def load_unfinished_tasks(self) -> List[Task]:
        """Load tasks that are not COMPLETED or FAILED — needs resumption."""
        rows = await asyncio.get_event_loop().run_in_executor(
            None, self._load_unfinished_sync
        )
        tasks = []
        for row in rows:
            try:
                tasks.append(Task.model_validate_json(row["data"]))
            except Exception as exc:
                logger.error("task_deserialize_error", extra={"task_id": row["task_id"], "error": str(exc)})
        return tasks

    def _load_unfinished_sync(self):
        terminal = (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value)
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM tasks WHERE status NOT IN (?, ?) ORDER BY created_at",
                terminal,
            ).fetchall()

    async def get_task_stats(self) -> Dict[str, int]:
        return await asyncio.get_event_loop().run_in_executor(None, self._stats_sync)

    def _stats_sync(self) -> Dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM tasks GROUP BY status"
            ).fetchall()
        return {row["status"]: row["cnt"] for row in rows}

    # ------------------------------------------------------------------ Events

    async def save_event(self, event: Event) -> None:
        await asyncio.get_event_loop().run_in_executor(
            None, self._save_event_sync,
            event.event_id, event.task_id, event.event_type.value,
            json.dumps(event.payload), event.timestamp.isoformat()
        )

    def _save_event_sync(self, event_id, task_id, event_type, payload, timestamp) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO events
                   (event_id, task_id, event_type, payload, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (event_id, task_id, event_type, payload, timestamp),
            )

    async def load_events(
        self,
        task_id:  Optional[str] = None,
        limit:    int = 200,
        after_ts: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._load_events_sync, task_id, limit, after_ts
        )

    def _load_events_sync(self, task_id, limit, after_ts) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            if task_id:
                rows = conn.execute(
                    "SELECT * FROM events WHERE task_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (task_id, limit),
                ).fetchall()
            elif after_ts:
                rows = conn.execute(
                    "SELECT * FROM events WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?",
                    (after_ts, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
        return [dict(r) for r in rows]

    async def get_event_count(self) -> int:
        return await asyncio.get_event_loop().run_in_executor(None, self._event_count_sync)

    def _event_count_sync(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
