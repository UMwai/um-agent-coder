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

CREATE TABLE IF NOT EXISTS gemini_sessions (
    id TEXT PRIMARY KEY,
    system_prompt TEXT,
    model TEXT NOT NULL,
    temperature REAL NOT NULL DEFAULT 0.7,
    max_tokens INTEGER NOT NULL DEFAULT 8192,
    metadata TEXT,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    turn_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS gemini_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES gemini_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    enhancement_applied INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gemini_batch_jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    total_queries INTEGER NOT NULL DEFAULT 0,
    completed_queries INTEGER NOT NULL DEFAULT 0,
    failed_queries INTEGER NOT NULL DEFAULT 0,
    model TEXT,
    config TEXT,
    results TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_gemini_messages_session ON gemini_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_gemini_sessions_expires ON gemini_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_gemini_batch_status ON gemini_batch_jobs(status);

CREATE TABLE IF NOT EXISTS gemini_iterations (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'running',
    original_prompt TEXT NOT NULL,
    system_prompt TEXT,
    eval_context TEXT,
    config TEXT,
    best_response TEXT,
    best_score REAL NOT NULL DEFAULT 0.0,
    best_iteration INTEGER NOT NULL DEFAULT 0,
    total_iterations INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS gemini_iteration_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration_id TEXT NOT NULL REFERENCES gemini_iterations(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    prompt_sent TEXT,
    response TEXT,
    generation_model TEXT,
    generation_duration_ms INTEGER NOT NULL DEFAULT 0,
    generation_tokens INTEGER NOT NULL DEFAULT 0,
    eval_scores TEXT,
    eval_models TEXT,
    eval_duration_ms INTEGER NOT NULL DEFAULT 0,
    strategies_applied TEXT,
    finish_reason TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_gemini_iterations_status ON gemini_iterations(status);
CREATE INDEX IF NOT EXISTS idx_gemini_iteration_steps_iter ON gemini_iteration_steps(iteration_id);
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

    # -- Gemini Sessions --

    async def create_gemini_session(
        self,
        session_id: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO gemini_sessions
               (id, system_prompt, model, temperature, max_tokens, metadata,
                total_tokens, turn_count, created_at, updated_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)""",
            (
                session_id, system_prompt, model, temperature, max_tokens,
                json.dumps(metadata) if metadata else None,
                now, now, expires_at,
            ),
        )
        await self._db.commit()
        return await self.get_gemini_session(session_id)

    async def get_gemini_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM gemini_sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        return self._row_to_dict(row) if row else None

    async def list_gemini_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM gemini_sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def update_gemini_session(self, session_id: str, **fields) -> Optional[Dict[str, Any]]:
        allowed = {"total_tokens", "turn_count", "updated_at", "system_prompt"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return await self.get_gemini_session(session_id)
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [session_id]
        await self._db.execute(
            f"UPDATE gemini_sessions SET {set_clause} WHERE id = ?", values  # noqa: S608
        )
        await self._db.commit()
        return await self.get_gemini_session(session_id)

    async def delete_gemini_session(self, session_id: str) -> bool:
        await self._db.execute("DELETE FROM gemini_messages WHERE session_id = ?", (session_id,))
        cursor = await self._db.execute("DELETE FROM gemini_sessions WHERE id = ?", (session_id,))
        await self._db.commit()
        return cursor.rowcount > 0

    # -- Gemini Messages --

    async def add_gemini_message(
        self,
        message_id: str,
        session_id: str,
        role: str,
        content: str,
        token_count: int = 0,
        enhancement_applied: bool = False,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO gemini_messages
               (id, session_id, role, content, token_count, enhancement_applied, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (message_id, session_id, role, content, token_count,
             1 if enhancement_applied else 0, now),
        )
        # Update session counters
        await self._db.execute(
            """UPDATE gemini_sessions
               SET turn_count = turn_count + 1,
                   total_tokens = total_tokens + ?,
                   updated_at = ?
               WHERE id = ?""",
            (token_count, now, session_id),
        )
        await self._db.commit()
        return await self.get_gemini_message(message_id)

    async def get_gemini_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM gemini_messages WHERE id = ?", (message_id,)
        )
        row = await cursor.fetchone()
        return self._row_to_dict(row) if row else None

    async def get_session_messages(
        self, session_id: str, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM gemini_messages WHERE session_id = ? ORDER BY created_at ASC LIMIT ?",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -- Gemini Batch Jobs --

    async def create_gemini_batch(
        self,
        batch_id: str,
        total_queries: int,
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO gemini_batch_jobs
               (id, status, total_queries, model, config, created_at)
               VALUES (?, 'pending', ?, ?, ?, ?)""",
            (batch_id, total_queries, model,
             json.dumps(config) if config else None, now),
        )
        await self._db.commit()
        return await self.get_gemini_batch(batch_id)

    async def get_gemini_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM gemini_batch_jobs WHERE id = ?", (batch_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        d = self._row_to_dict(row)
        # Parse results JSON
        if d.get("results") and isinstance(d["results"], str):
            try:
                d["results"] = json.loads(d["results"])
            except (json.JSONDecodeError, TypeError):
                pass
        if d.get("config") and isinstance(d["config"], str):
            try:
                d["config"] = json.loads(d["config"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    async def update_gemini_batch(self, batch_id: str, **fields) -> Optional[Dict[str, Any]]:
        allowed = {
            "status", "completed_queries", "failed_queries",
            "results", "error", "started_at", "completed_at",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return await self.get_gemini_batch(batch_id)
        for k, v in updates.items():
            if isinstance(v, (dict, list)):
                updates[k] = json.dumps(v)
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [batch_id]
        await self._db.execute(
            f"UPDATE gemini_batch_jobs SET {set_clause} WHERE id = ?", values  # noqa: S608
        )
        await self._db.commit()
        return await self.get_gemini_batch(batch_id)

    # -- Gemini Iterations --

    async def create_gemini_iteration(
        self,
        iteration_id: str,
        original_prompt: str,
        system_prompt: Optional[str] = None,
        eval_context: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO gemini_iterations
               (id, status, original_prompt, system_prompt, eval_context,
                config, best_score, best_iteration, total_iterations,
                total_tokens, created_at, started_at)
               VALUES (?, 'running', ?, ?, ?, ?, 0.0, 0, 0, 0, ?, ?)""",
            (
                iteration_id, original_prompt, system_prompt, eval_context,
                json.dumps(config) if config else None,
                now, now,
            ),
        )
        await self._db.commit()
        return await self.get_gemini_iteration(iteration_id)

    async def get_gemini_iteration(self, iteration_id: str) -> Optional[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM gemini_iterations WHERE id = ?", (iteration_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        d = self._row_to_dict(row)
        for key in ("config",):
            if d.get(key) and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    async def list_gemini_iterations(
        self, limit: int = 50, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if status:
            cursor = await self._db.execute(
                "SELECT * FROM gemini_iterations WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM gemini_iterations ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            d = self._row_to_dict(row)
            for key in ("config",):
                if d.get(key) and isinstance(d[key], str):
                    try:
                        d[key] = json.loads(d[key])
                    except (json.JSONDecodeError, TypeError):
                        pass
            result.append(d)
        return result

    async def update_gemini_iteration(self, iteration_id: str, **fields) -> Optional[Dict[str, Any]]:
        allowed = {
            "status", "best_response", "best_score", "best_iteration",
            "total_iterations", "total_tokens", "error", "completed_at",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return await self.get_gemini_iteration(iteration_id)
        for k, v in updates.items():
            if isinstance(v, (dict, list)):
                updates[k] = json.dumps(v)
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [iteration_id]
        await self._db.execute(
            f"UPDATE gemini_iterations SET {set_clause} WHERE id = ?", values  # noqa: S608
        )
        await self._db.commit()
        return await self.get_gemini_iteration(iteration_id)

    async def add_gemini_iteration_step(
        self,
        iteration_id: str,
        step_number: int,
        prompt_sent: str = "",
        response: str = "",
        generation_model: str = "",
        generation_duration_ms: int = 0,
        generation_tokens: int = 0,
        eval_scores: Optional[Dict[str, Any]] = None,
        eval_models: Optional[List[str]] = None,
        eval_duration_ms: int = 0,
        strategies_applied: Optional[List[str]] = None,
        finish_reason: str = "",
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO gemini_iteration_steps
               (iteration_id, step_number, prompt_sent, response,
                generation_model, generation_duration_ms, generation_tokens,
                eval_scores, eval_models, eval_duration_ms,
                strategies_applied, finish_reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                iteration_id, step_number, prompt_sent, response,
                generation_model, generation_duration_ms, generation_tokens,
                json.dumps(eval_scores) if eval_scores else None,
                json.dumps(eval_models) if eval_models else None,
                eval_duration_ms,
                json.dumps(strategies_applied) if strategies_applied else None,
                finish_reason, now,
            ),
        )
        await self._db.commit()
        # Return the inserted row
        cursor = await self._db.execute(
            "SELECT * FROM gemini_iteration_steps WHERE iteration_id = ? AND step_number = ?",
            (iteration_id, step_number),
        )
        row = await cursor.fetchone()
        return self._row_to_dict(row) if row else {}

    async def get_gemini_iteration_steps(
        self, iteration_id: str
    ) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM gemini_iteration_steps WHERE iteration_id = ? ORDER BY step_number ASC",
            (iteration_id,),
        )
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            d = self._row_to_dict(row)
            for key in ("eval_scores", "eval_models", "strategies_applied"):
                if d.get(key) and isinstance(d[key], str):
                    try:
                        d[key] = json.loads(d[key])
                    except (json.JSONDecodeError, TypeError):
                        pass
            result.append(d)
        return result

    # -- Helpers --

    @staticmethod
    def _row_to_dict(row: aiosqlite.Row) -> Dict[str, Any]:
        d = dict(row)
        # Parse JSON fields back
        for key in ("spec", "source_meta", "result", "metadata"):
            if d.get(key) and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d
