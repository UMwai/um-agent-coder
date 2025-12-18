"""
State persistence for the 24/7 Codex Harness.

Uses SQLite for durable state across restarts.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from .models import Task, TaskStatus, HarnessState, ExecutionResult

logger = logging.getLogger(__name__)


class StateManager:
    """Manage persistent state in SQLite."""

    def __init__(self, db_path: str = ".harness/state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS harness_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    roadmap_path TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    tasks_completed INTEGER DEFAULT 0,
                    tasks_failed INTEGER DEFAULT 0,
                    total_execution_time REAL DEFAULT 0.0,
                    in_growth_mode INTEGER DEFAULT 0,
                    growth_iterations INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    depends TEXT DEFAULT '[]',
                    timeout_minutes INTEGER DEFAULT 30,
                    success_criteria TEXT DEFAULT '',
                    cwd TEXT DEFAULT './',
                    status TEXT DEFAULT 'pending',
                    attempts INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    output TEXT DEFAULT '',
                    error TEXT DEFAULT '',
                    conversation_id TEXT,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS execution_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    output TEXT,
                    error TEXT,
                    duration_seconds REAL,
                    FOREIGN KEY (task_id) REFERENCES tasks(id)
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_execution_log_task ON execution_log(task_id);
            """)

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # Harness State Operations

    def init_harness(self, roadmap_path: str) -> HarnessState:
        """Initialize or load harness state."""
        now = datetime.utcnow().isoformat()

        with self._connection() as conn:
            # Check if state exists
            row = conn.execute("SELECT * FROM harness_state WHERE id = 1").fetchone()

            if row:
                # Resume existing state
                state = HarnessState(
                    roadmap_path=row["roadmap_path"],
                    started_at=datetime.fromisoformat(row["started_at"]),
                    last_activity=datetime.fromisoformat(row["last_activity"]),
                    tasks_completed=row["tasks_completed"],
                    tasks_failed=row["tasks_failed"],
                    total_execution_time=row["total_execution_time"],
                    in_growth_mode=bool(row["in_growth_mode"]),
                    growth_iterations=row["growth_iterations"],
                )
                logger.info(f"Resuming harness state from {state.started_at}")
            else:
                # Create new state
                conn.execute("""
                    INSERT INTO harness_state (id, roadmap_path, started_at, last_activity)
                    VALUES (1, ?, ?, ?)
                """, (roadmap_path, now, now))

                state = HarnessState(
                    roadmap_path=roadmap_path,
                    started_at=datetime.fromisoformat(now),
                    last_activity=datetime.fromisoformat(now),
                )
                logger.info("Created new harness state")

        return state

    def update_harness_state(
        self,
        tasks_completed: Optional[int] = None,
        tasks_failed: Optional[int] = None,
        execution_time: Optional[float] = None,
        in_growth_mode: Optional[bool] = None,
        growth_iterations: Optional[int] = None,
    ) -> None:
        """Update harness state."""
        updates = ["last_activity = ?"]
        values = [datetime.utcnow().isoformat()]

        if tasks_completed is not None:
            updates.append("tasks_completed = ?")
            values.append(tasks_completed)
        if tasks_failed is not None:
            updates.append("tasks_failed = ?")
            values.append(tasks_failed)
        if execution_time is not None:
            updates.append("total_execution_time = total_execution_time + ?")
            values.append(execution_time)
        if in_growth_mode is not None:
            updates.append("in_growth_mode = ?")
            values.append(int(in_growth_mode))
        if growth_iterations is not None:
            updates.append("growth_iterations = ?")
            values.append(growth_iterations)

        sql = f"UPDATE harness_state SET {', '.join(updates)} WHERE id = 1"

        with self._connection() as conn:
            conn.execute(sql, values)

    def get_harness_state(self) -> Optional[HarnessState]:
        """Get current harness state."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM harness_state WHERE id = 1").fetchone()
            if not row:
                return None

            return HarnessState(
                roadmap_path=row["roadmap_path"],
                started_at=datetime.fromisoformat(row["started_at"]),
                last_activity=datetime.fromisoformat(row["last_activity"]),
                tasks_completed=row["tasks_completed"],
                tasks_failed=row["tasks_failed"],
                total_execution_time=row["total_execution_time"],
                in_growth_mode=bool(row["in_growth_mode"]),
                growth_iterations=row["growth_iterations"],
            )

    # Task Operations

    def save_task(self, task: Task) -> None:
        """Save or update a task."""
        with self._connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tasks (
                    id, description, phase, depends, timeout_minutes,
                    success_criteria, cwd, status, attempts, max_retries,
                    output, error, conversation_id, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.id,
                task.description,
                task.phase,
                json.dumps(task.depends),
                task.timeout_minutes,
                task.success_criteria,
                task.cwd,
                task.status.value,
                task.attempts,
                task.max_retries,
                task.output,
                task.error,
                task.conversation_id,
                task.started_at.isoformat() if task.started_at else None,
                task.completed_at.isoformat() if task.completed_at else None,
            ))

    def load_task(self, task_id: str) -> Optional[Task]:
        """Load a task by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()

            if not row:
                return None

            return self._row_to_task(row)

    def load_all_tasks(self) -> list[Task]:
        """Load all tasks."""
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM tasks ORDER BY phase, id").fetchall()
            return [self._row_to_task(row) for row in rows]

    def get_pending_tasks(self) -> list[Task]:
        """Get all pending tasks."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status = 'pending' ORDER BY phase, id"
            ).fetchall()
            return [self._row_to_task(row) for row in rows]

    def get_completed_task_ids(self) -> set[str]:
        """Get IDs of all completed tasks."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id FROM tasks WHERE status = 'completed'"
            ).fetchall()
            return {row["id"] for row in rows}

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        """Convert a database row to a Task object."""
        return Task(
            id=row["id"],
            description=row["description"],
            phase=row["phase"],
            depends=json.loads(row["depends"]),
            timeout_minutes=row["timeout_minutes"],
            success_criteria=row["success_criteria"],
            cwd=row["cwd"],
            status=TaskStatus(row["status"]),
            attempts=row["attempts"],
            max_retries=row["max_retries"],
            output=row["output"],
            error=row["error"],
            conversation_id=row["conversation_id"],
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        )

    # Execution Log Operations

    def log_execution(
        self,
        task_id: str,
        attempt: int,
        result: ExecutionResult,
    ) -> None:
        """Log a task execution attempt."""
        with self._connection() as conn:
            conn.execute("""
                INSERT INTO execution_log (
                    task_id, timestamp, attempt, success, output, error, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                datetime.utcnow().isoformat(),
                attempt,
                int(result.success),
                result.output[:10000],  # Truncate large outputs
                result.error,
                result.duration_seconds,
            ))

    def get_execution_history(self, task_id: str) -> list[dict]:
        """Get execution history for a task."""
        with self._connection() as conn:
            rows = conn.execute("""
                SELECT * FROM execution_log WHERE task_id = ?
                ORDER BY timestamp DESC
            """, (task_id,)).fetchall()

            return [dict(row) for row in rows]

    # Statistics

    def get_statistics(self) -> dict:
        """Get harness statistics."""
        with self._connection() as conn:
            state = conn.execute("SELECT * FROM harness_state WHERE id = 1").fetchone()
            task_counts = conn.execute("""
                SELECT status, COUNT(*) as count FROM tasks GROUP BY status
            """).fetchall()
            total_executions = conn.execute(
                "SELECT COUNT(*) as count FROM execution_log"
            ).fetchone()

            return {
                "started_at": state["started_at"] if state else None,
                "last_activity": state["last_activity"] if state else None,
                "tasks_completed": state["tasks_completed"] if state else 0,
                "tasks_failed": state["tasks_failed"] if state else 0,
                "total_execution_time": state["total_execution_time"] if state else 0,
                "in_growth_mode": bool(state["in_growth_mode"]) if state else False,
                "growth_iterations": state["growth_iterations"] if state else 0,
                "task_counts": {row["status"]: row["count"] for row in task_counts},
                "total_executions": total_executions["count"],
            }

    def reset(self) -> None:
        """Reset all state (for testing or starting fresh)."""
        with self._connection() as conn:
            conn.execute("DELETE FROM execution_log")
            conn.execute("DELETE FROM tasks")
            conn.execute("DELETE FROM harness_state")
        logger.info("State reset complete")
