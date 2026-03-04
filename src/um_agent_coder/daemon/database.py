"""SQLite persistence for daemon tasks via aiosqlite."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiosqlite

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    spec TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    source TEXT NOT NULL DEFAULT 'api',
    source_meta TEXT,
    result TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS task_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL REFERENCES tasks(id),
    level TEXT NOT NULL DEFAULT 'info',
    message TEXT NOT NULL,
    data TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_task_logs_task_id ON task_logs(task_id);
"""


class Database:
    """Async SQLite database wrapper for daemon tasks."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None

    # -- Task CRUD --

    async def create_task(
        self,
        task_id: str,
        prompt: str,
        source: str = "api",
        spec: Optional[Dict[str, Any]] = None,
        source_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO tasks (id, prompt, spec, status, source, source_meta, created_at)
               VALUES (?, ?, ?, 'pending', ?, ?, ?)""",
            (
                task_id,
                prompt,
                json.dumps(spec) if spec else None,
                source,
                json.dumps(source_meta) if source_meta else None,
                now,
            ),
        )
        await self._db.commit()
        return await self.get_task(task_id)

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        cursor = await self._db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
        return self._row_to_dict(row) if row else None

    async def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        if status:
            cursor = await self._db.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def update_task(self, task_id: str, **fields) -> Optional[Dict[str, Any]]:
        allowed = {"status", "result", "error", "started_at", "completed_at"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return await self.get_task(task_id)
        # Serialize dict/list values to JSON strings
        for k, v in updates.items():
            if isinstance(v, (dict, list)):
                updates[k] = json.dumps(v)
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [task_id]
        await self._db.execute(
            f"UPDATE tasks SET {set_clause} WHERE id = ?", values  # noqa: S608
        )
        await self._db.commit()
        return await self.get_task(task_id)

    async def count_tasks(self, status: Optional[str] = None) -> int:
        if status:
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = ?", (status,)
            )
        else:
            cursor = await self._db.execute("SELECT COUNT(*) FROM tasks")
        row = await cursor.fetchone()
        return row[0]

    # -- Task Logs --

    async def add_log(
        self,
        task_id: str,
        message: str,
        level: str = "info",
        data: Optional[Dict[str, Any]] = None,
    ):
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT INTO task_logs (task_id, level, message, data, created_at) VALUES (?, ?, ?, ?, ?)",
            (task_id, level, message, json.dumps(data) if data else None, now),
        )
        await self._db.commit()

    async def get_logs(
        self, task_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM task_logs WHERE task_id = ? ORDER BY created_at ASC LIMIT ?",
            (task_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -- Helpers --

    @staticmethod
    def _row_to_dict(row: aiosqlite.Row) -> Dict[str, Any]:
        d = dict(row)
        # Parse JSON fields back
        for key in ("spec", "source_meta", "result"):
            if d.get(key) and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d
