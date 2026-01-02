"""
SQLite persistence for ralph loop state.

Enables resumption of ralph loops after interruption by persisting
iteration state to disk.
"""

import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from .iteration_tracker import IterationRecord, IterationTracker

logger = logging.getLogger(__name__)


class RalphPersistence:
    """Manage persistent state for ralph loops in SQLite.

    Example:
        persistence = RalphPersistence()
        tracker = IterationTracker(task_id="task-001", max_iterations=10)

        # Save after each iteration
        tracker.start_iteration()
        # ... do work ...
        tracker.end_iteration(output="result", promise_found=False)
        persistence.save_tracker(tracker)

        # Resume after interruption
        tracker = persistence.load_tracker("task-001")
        if tracker and tracker.can_continue():
            # Continue from where we left off
            ...
    """

    def __init__(self, db_path: str = ".harness/ralph_state.db"):
        """Initialize the persistence layer.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS ralph_trackers (
                    task_id TEXT PRIMARY KEY,
                    max_iterations INTEGER NOT NULL,
                    current_iteration INTEGER NOT NULL DEFAULT 0,
                    start_time TEXT NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 0,
                    completion_reason TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ralph_iterations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    iteration_num INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    output_snippet TEXT,
                    promise_found INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    FOREIGN KEY (task_id) REFERENCES ralph_trackers(task_id),
                    UNIQUE (task_id, iteration_num)
                );

                CREATE INDEX IF NOT EXISTS idx_iterations_task
                    ON ralph_iterations(task_id);
                CREATE INDEX IF NOT EXISTS idx_trackers_completed
                    ON ralph_trackers(completed);
            """
            )

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

    def save_tracker(self, tracker: IterationTracker) -> None:
        """Save an iteration tracker to the database.

        Args:
            tracker: The IterationTracker to save
        """
        now = datetime.utcnow().isoformat()

        with self._connection() as conn:
            # Upsert tracker state
            conn.execute(
                """
                INSERT INTO ralph_trackers (
                    task_id, max_iterations, current_iteration,
                    start_time, completed, completion_reason, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    max_iterations = excluded.max_iterations,
                    current_iteration = excluded.current_iteration,
                    completed = excluded.completed,
                    completion_reason = excluded.completion_reason,
                    updated_at = excluded.updated_at
            """,
                (
                    tracker.task_id,
                    tracker.max_iterations,
                    tracker.current_iteration,
                    tracker.start_time.isoformat(),
                    int(tracker.is_complete),
                    tracker.completion_reason,
                    now,
                ),
            )

            # Save iteration history (only new records)
            for record in tracker.iteration_history:
                conn.execute(
                    """
                    INSERT INTO ralph_iterations (
                        task_id, iteration_num, started_at, ended_at,
                        output_snippet, promise_found, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(task_id, iteration_num) DO UPDATE SET
                        ended_at = excluded.ended_at,
                        output_snippet = excluded.output_snippet,
                        promise_found = excluded.promise_found,
                        error = excluded.error
                """,
                    (
                        tracker.task_id,
                        record.iteration_num,
                        record.started_at.isoformat(),
                        record.ended_at.isoformat() if record.ended_at else None,
                        record.output_snippet,
                        int(record.promise_found),
                        record.error,
                    ),
                )

        logger.debug(
            f"Saved tracker for {tracker.task_id}: "
            f"iteration {tracker.current_iteration}/{tracker.max_iterations}"
        )

    def load_tracker(self, task_id: str) -> Optional[IterationTracker]:
        """Load an iteration tracker from the database.

        Args:
            task_id: The task ID to load

        Returns:
            IterationTracker if found, None otherwise
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM ralph_trackers WHERE task_id = ?", (task_id,)
            ).fetchone()

            if not row:
                return None

            # Load iteration history
            iter_rows = conn.execute(
                """SELECT * FROM ralph_iterations
                   WHERE task_id = ?
                   ORDER BY iteration_num""",
                (task_id,),
            ).fetchall()

            history = [
                IterationRecord(
                    iteration_num=ir["iteration_num"],
                    started_at=datetime.fromisoformat(ir["started_at"]),
                    ended_at=datetime.fromisoformat(ir["ended_at"]) if ir["ended_at"] else None,
                    output_snippet=ir["output_snippet"] or "",
                    promise_found=bool(ir["promise_found"]),
                    error=ir["error"],
                )
                for ir in iter_rows
            ]

            tracker = IterationTracker(
                task_id=row["task_id"],
                max_iterations=row["max_iterations"],
                current_iteration=row["current_iteration"],
                start_time=datetime.fromisoformat(row["start_time"]),
                iteration_history=history,
            )
            tracker._completed = bool(row["completed"])
            tracker._completion_reason = row["completion_reason"]

            logger.debug(
                f"Loaded tracker for {task_id}: "
                f"iteration {tracker.current_iteration}/{tracker.max_iterations}"
            )

            return tracker

    def delete_tracker(self, task_id: str) -> bool:
        """Delete a tracker and its history from the database.

        Args:
            task_id: The task ID to delete

        Returns:
            True if tracker was deleted, False if not found
        """
        with self._connection() as conn:
            # Delete iteration history first (foreign key)
            conn.execute("DELETE FROM ralph_iterations WHERE task_id = ?", (task_id,))

            # Delete tracker
            result = conn.execute("DELETE FROM ralph_trackers WHERE task_id = ?", (task_id,))

            return result.rowcount > 0

    def list_active_trackers(self) -> list[str]:
        """List all active (non-completed) tracker task IDs.

        Returns:
            List of task IDs with active ralph loops
        """
        with self._connection() as conn:
            rows = conn.execute("SELECT task_id FROM ralph_trackers WHERE completed = 0").fetchall()
            return [row["task_id"] for row in rows]

    def list_all_trackers(self) -> list[dict]:
        """List all trackers with summary info.

        Returns:
            List of tracker summaries
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    t.task_id,
                    t.max_iterations,
                    t.current_iteration,
                    t.start_time,
                    t.completed,
                    t.completion_reason,
                    t.updated_at,
                    COUNT(i.id) as iteration_count
                FROM ralph_trackers t
                LEFT JOIN ralph_iterations i ON t.task_id = i.task_id
                GROUP BY t.task_id
                ORDER BY t.updated_at DESC
            """
            ).fetchall()

            return [
                {
                    "task_id": row["task_id"],
                    "max_iterations": row["max_iterations"],
                    "current_iteration": row["current_iteration"],
                    "start_time": row["start_time"],
                    "completed": bool(row["completed"]),
                    "completion_reason": row["completion_reason"],
                    "updated_at": row["updated_at"],
                    "iteration_count": row["iteration_count"],
                }
                for row in rows
            ]

    def get_iteration_history(self, task_id: str) -> list[IterationRecord]:
        """Get iteration history for a task.

        Args:
            task_id: The task ID to get history for

        Returns:
            List of IterationRecords
        """
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT * FROM ralph_iterations
                   WHERE task_id = ?
                   ORDER BY iteration_num""",
                (task_id,),
            ).fetchall()

            return [
                IterationRecord(
                    iteration_num=row["iteration_num"],
                    started_at=datetime.fromisoformat(row["started_at"]),
                    ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
                    output_snippet=row["output_snippet"] or "",
                    promise_found=bool(row["promise_found"]),
                    error=row["error"],
                )
                for row in rows
            ]

    def reset(self) -> None:
        """Delete all ralph state (for testing or fresh start)."""
        with self._connection() as conn:
            conn.execute("DELETE FROM ralph_iterations")
            conn.execute("DELETE FROM ralph_trackers")
        logger.info("Ralph state reset complete")
