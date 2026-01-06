"""
MetaStateManager for tracking multiple sub-harnesses.

Provides persistent state for the meta-harness layer, tracking all spawned
sub-harnesses, their progress, and coordination state.
"""

import json
import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .result import HarnessStatus

logger = logging.getLogger(__name__)


class MetaStateManager:
    """
    Manage persistent state for meta-harness in SQLite.

    Tracks:
    - Registered sub-harnesses and their status
    - Progress updates from sub-harnesses
    - Coordination metadata (strategy, dependencies)
    - Aggregated results

    Example:
        meta_state = MetaStateManager()

        # Register new sub-harness
        meta_state.register_sub_harness(
            harness_id="auth-harness",
            roadmap_path="/path/to/auth/roadmap.md",
            cli="codex"
        )

        # Update progress
        meta_state.update_progress(
            harness_id="auth-harness",
            progress=0.5,
            current_task="Implement JWT"
        )

        # Get all running
        running = meta_state.get_running_harnesses()
    """

    def __init__(self, db_path: str = ".harness/meta_state.db"):
        """Initialize the meta state manager.

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
                CREATE TABLE IF NOT EXISTS meta_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    started_at TEXT NOT NULL,
                    strategy TEXT DEFAULT 'parallel',
                    total_harnesses INTEGER DEFAULT 0,
                    completed_harnesses INTEGER DEFAULT 0,
                    failed_harnesses INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS sub_harnesses (
                    harness_id TEXT PRIMARY KEY,
                    pid INTEGER,
                    roadmap_path TEXT NOT NULL,
                    working_dir TEXT NOT NULL,
                    cli TEXT DEFAULT 'auto',
                    model TEXT DEFAULT '',
                    status TEXT DEFAULT 'pending',
                    progress REAL DEFAULT 0.0,
                    current_task TEXT DEFAULT '',
                    current_iteration INTEGER DEFAULT 0,
                    tasks_completed INTEGER DEFAULT 0,
                    tasks_failed INTEGER DEFAULT 0,
                    error TEXT DEFAULT '',
                    registered_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    parent_context TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS progress_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    harness_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    progress REAL NOT NULL,
                    current_task TEXT,
                    message TEXT,
                    FOREIGN KEY (harness_id) REFERENCES sub_harnesses(harness_id)
                );

                CREATE TABLE IF NOT EXISTS coordination_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    harness_id TEXT,
                    details TEXT DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_harnesses_status
                    ON sub_harnesses(status);
                CREATE INDEX IF NOT EXISTS idx_progress_harness
                    ON progress_log(harness_id);
                CREATE INDEX IF NOT EXISTS idx_events_type
                    ON coordination_events(event_type);
                """
            )

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Create a database connection context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_meta_harness(self, strategy: str = "parallel") -> None:
        """Initialize meta-harness state.

        Args:
            strategy: Coordination strategy (parallel, pipeline, race, voting)
        """
        now = datetime.utcnow().isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO meta_state
                (id, started_at, strategy, total_harnesses, completed_harnesses, failed_harnesses)
                VALUES (1, ?, ?, 0, 0, 0)
                """,
                (now, strategy),
            )
            logger.info(f"Initialized meta-harness with strategy: {strategy}")

    def register_sub_harness(
        self,
        harness_id: str,
        roadmap_path: str,
        working_dir: str,
        cli: str = "auto",
        model: str = "",
        pid: Optional[int] = None,
        parent_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a new sub-harness.

        Args:
            harness_id: Unique identifier for the sub-harness
            roadmap_path: Path to the roadmap file
            working_dir: Working directory for execution
            cli: CLI backend to use
            model: Model override
            pid: Process ID if already spawned
            parent_context: Context passed from parent
        """
        now = datetime.utcnow().isoformat()
        context_json = json.dumps(parent_context or {})

        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sub_harnesses
                (harness_id, pid, roadmap_path, working_dir, cli, model,
                 status, registered_at, parent_context)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    harness_id,
                    pid,
                    roadmap_path,
                    working_dir,
                    cli,
                    model,
                    now,
                    context_json,
                ),
            )

            # Update total count
            conn.execute(
                "UPDATE meta_state SET total_harnesses = total_harnesses + 1 WHERE id = 1"
            )

            # Log event
            self._log_event(conn, "harness_registered", harness_id, {"cli": cli})

        logger.info(f"Registered sub-harness: {harness_id}")

    def update_harness_started(self, harness_id: str, pid: int) -> None:
        """Mark a sub-harness as started.

        Args:
            harness_id: Sub-harness identifier
            pid: Process ID
        """
        now = datetime.utcnow().isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE sub_harnesses
                SET status = 'running', pid = ?, started_at = ?
                WHERE harness_id = ?
                """,
                (pid, now, harness_id),
            )
            self._log_event(conn, "harness_started", harness_id, {"pid": pid})

    def update_progress(
        self,
        harness_id: str,
        progress: float,
        current_task: Optional[str] = None,
        current_iteration: Optional[int] = None,
        tasks_completed: Optional[int] = None,
        tasks_failed: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        """Update progress for a sub-harness.

        Args:
            harness_id: Sub-harness identifier
            progress: Progress value (0.0 to 1.0)
            current_task: Current task description
            current_iteration: Current iteration number
            tasks_completed: Number of completed tasks
            tasks_failed: Number of failed tasks
            message: Optional progress message
        """
        now = datetime.utcnow().isoformat()

        with self._connection() as conn:
            # Build update query dynamically
            updates = ["progress = ?"]
            params: List[Any] = [progress]

            if current_task is not None:
                updates.append("current_task = ?")
                params.append(current_task)

            if current_iteration is not None:
                updates.append("current_iteration = ?")
                params.append(current_iteration)

            if tasks_completed is not None:
                updates.append("tasks_completed = ?")
                params.append(tasks_completed)

            if tasks_failed is not None:
                updates.append("tasks_failed = ?")
                params.append(tasks_failed)

            params.append(harness_id)

            conn.execute(
                f"UPDATE sub_harnesses SET {', '.join(updates)} WHERE harness_id = ?",
                params,
            )

            # Log progress
            conn.execute(
                """
                INSERT INTO progress_log (harness_id, timestamp, progress, current_task, message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (harness_id, now, progress, current_task, message),
            )

    def update_harness_completed(
        self,
        harness_id: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Mark a sub-harness as completed.

        Args:
            harness_id: Sub-harness identifier
            success: Whether it completed successfully
            error: Error message if failed
        """
        now = datetime.utcnow().isoformat()
        status = HarnessStatus.COMPLETED.value if success else HarnessStatus.FAILED.value

        with self._connection() as conn:
            conn.execute(
                """
                UPDATE sub_harnesses
                SET status = ?, completed_at = ?, error = ?
                WHERE harness_id = ?
                """,
                (status, now, error or "", harness_id),
            )

            # Update counters
            if success:
                conn.execute(
                    "UPDATE meta_state SET completed_harnesses = completed_harnesses + 1 WHERE id = 1"
                )
            else:
                conn.execute(
                    "UPDATE meta_state SET failed_harnesses = failed_harnesses + 1 WHERE id = 1"
                )

            self._log_event(
                conn,
                "harness_completed",
                harness_id,
                {"success": success, "error": error},
            )

        logger.info(f"Sub-harness {harness_id} completed: success={success}")

    def update_harness_stopped(self, harness_id: str) -> None:
        """Mark a sub-harness as stopped.

        Args:
            harness_id: Sub-harness identifier
        """
        now = datetime.utcnow().isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE sub_harnesses
                SET status = 'stopped', completed_at = ?
                WHERE harness_id = ?
                """,
                (now, harness_id),
            )
            self._log_event(conn, "harness_stopped", harness_id, {})

    def get_harness(self, harness_id: str) -> Optional[Dict[str, Any]]:
        """Get sub-harness by ID.

        Args:
            harness_id: Sub-harness identifier

        Returns:
            Harness data dict or None
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM sub_harnesses WHERE harness_id = ?", (harness_id,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_all_harnesses(self) -> List[Dict[str, Any]]:
        """Get all registered sub-harnesses.

        Returns:
            List of harness data dicts
        """
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM sub_harnesses ORDER BY registered_at")
            return [dict(row) for row in cursor.fetchall()]

    def get_running_harnesses(self) -> List[Dict[str, Any]]:
        """Get all running sub-harnesses.

        Returns:
            List of running harness data dicts
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM sub_harnesses WHERE status = 'running'"
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_pending_harnesses(self) -> List[Dict[str, Any]]:
        """Get all pending sub-harnesses.

        Returns:
            List of pending harness data dicts
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM sub_harnesses WHERE status = 'pending'"
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_completed_harnesses(self) -> List[Dict[str, Any]]:
        """Get all completed sub-harnesses.

        Returns:
            List of completed harness data dicts
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM sub_harnesses
                WHERE status IN ('completed', 'failed', 'stopped')
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_meta_state(self) -> Dict[str, Any]:
        """Get overall meta-harness state.

        Returns:
            Meta-state dict with totals and strategy
        """
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM meta_state WHERE id = 1")
            row = cursor.fetchone()
            if row:
                return dict(row)
            return {
                "started_at": None,
                "strategy": "parallel",
                "total_harnesses": 0,
                "completed_harnesses": 0,
                "failed_harnesses": 0,
            }

    def get_progress_history(
        self,
        harness_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get progress history for a sub-harness.

        Args:
            harness_id: Sub-harness identifier
            limit: Maximum records to return

        Returns:
            List of progress records
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM progress_log
                WHERE harness_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (harness_id, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent coordination events.

        Args:
            limit: Maximum events to return

        Returns:
            List of event records
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM coordination_events
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def _log_event(
        self,
        conn: sqlite3.Connection,
        event_type: str,
        harness_id: Optional[str],
        details: Dict[str, Any],
    ) -> None:
        """Log a coordination event.

        Args:
            conn: Database connection
            event_type: Type of event
            harness_id: Related harness ID (optional)
            details: Event details dict
        """
        now = datetime.utcnow().isoformat()
        conn.execute(
            """
            INSERT INTO coordination_events (timestamp, event_type, harness_id, details)
            VALUES (?, ?, ?, ?)
            """,
            (now, event_type, harness_id, json.dumps(details)),
        )

    def reset(self) -> None:
        """Reset all meta-harness state."""
        with self._connection() as conn:
            conn.execute("DELETE FROM sub_harnesses")
            conn.execute("DELETE FROM progress_log")
            conn.execute("DELETE FROM coordination_events")
            conn.execute("DELETE FROM meta_state")
            logger.info("Meta-harness state reset")
