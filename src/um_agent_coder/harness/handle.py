"""
HarnessHandle for controlling and monitoring sub-harnesses.

Provides async control interface for spawned sub-harness processes.
"""

import json
import logging
import os
import signal
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .result import HarnessResult, HarnessStatus, HarnessMetrics

logger = logging.getLogger(__name__)


@dataclass
class HarnessHandle:
    """Handle to control and monitor a sub-harness."""

    # Identification
    harness_id: str
    pid: int
    working_dir: Path
    roadmap_path: Path
    cli: str = "auto"
    model: str = ""

    # State paths
    state_dir: Path = field(default_factory=Path)
    state_db: Path = field(default_factory=Path)
    log_file: Path = field(default_factory=Path)
    inbox_dir: Path = field(default_factory=Path)

    # Observable state (updated by refresh())
    status: HarnessStatus = HarnessStatus.PENDING
    progress: float = 0.0
    current_task: Optional[str] = None
    current_iteration: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0

    # Results (populated on completion)
    result: Optional[HarnessResult] = None
    error: Optional[str] = None

    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Process reference
    _process: Any = None

    def __post_init__(self):
        """Initialize derived paths."""
        if not self.state_dir or self.state_dir == Path():
            self.state_dir = Path(".harness") / self.harness_id
        if not self.state_db or self.state_db == Path():
            self.state_db = self.state_dir / "state.db"
        if not self.log_file or self.log_file == Path():
            self.log_file = self.state_dir / "harness.log"
        if not self.inbox_dir or self.inbox_dir == Path():
            self.inbox_dir = self.state_dir / "inbox"

    def send_instruction(self, instruction: str) -> bool:
        """
        Send instruction to sub-harness.

        Writes to .harness/{id}/inbox/{timestamp}.txt
        """
        try:
            self.inbox_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            instruction_file = self.inbox_dir / f"{timestamp}.txt"
            instruction_file.write_text(instruction)
            logger.info(f"Sent instruction to {self.harness_id}: {instruction[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send instruction to {self.harness_id}: {e}")
            return False

    def request_pause(self) -> bool:
        """Request pause. Creates .harness/{id}/pause file."""
        try:
            pause_file = self.state_dir / "pause"
            pause_file.touch()
            logger.info(f"Requested pause for {self.harness_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to request pause for {self.harness_id}: {e}")
            return False

    def request_stop(self) -> bool:
        """Request graceful stop. Creates .harness/{id}/stop file."""
        try:
            stop_file = self.state_dir / "stop"
            stop_file.write_text("stop")
            logger.info(f"Requested stop for {self.harness_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to request stop for {self.harness_id}: {e}")
            return False

    def force_kill(self, timeout: int = 30) -> bool:
        """Force kill subprocess after timeout."""
        try:
            if self._process is not None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=timeout)
                except Exception:
                    self._process.kill()
                logger.info(f"Force killed {self.harness_id}")
                self.status = HarnessStatus.STOPPED
                return True
            elif self.pid > 0:
                os.kill(self.pid, signal.SIGTERM)
                logger.info(f"Sent SIGTERM to {self.harness_id} (pid={self.pid})")
                self.status = HarnessStatus.STOPPED
                return True
            return False
        except ProcessLookupError:
            logger.warning(f"Process {self.pid} for {self.harness_id} not found")
            self.status = HarnessStatus.STOPPED
            return True
        except Exception as e:
            logger.error(f"Failed to force kill {self.harness_id}: {e}")
            return False

    def get_logs(self, tail: int = 100) -> List[str]:
        """Get recent log lines from .harness/{id}/harness.log."""
        try:
            if self.log_file.exists():
                with open(self.log_file) as f:
                    lines = f.readlines()
                    return lines[-tail:]
            return []
        except Exception as e:
            logger.error(f"Failed to read logs for {self.harness_id}: {e}")
            return []

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get alerts from .harness/{id}/alerts.log."""
        alerts_file = self.state_dir / "alerts.log"
        try:
            if alerts_file.exists():
                alerts = []
                with open(alerts_file) as f:
                    for line in f:
                        try:
                            alerts.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                return alerts
            return []
        except Exception as e:
            logger.error(f"Failed to read alerts for {self.harness_id}: {e}")
            return []

    def refresh(self) -> None:
        """Refresh status from subprocess/state."""
        # Check if process is still running
        if self._process is not None:
            poll_result = self._process.poll()
            if poll_result is not None:
                # Process has exited
                if poll_result == 0:
                    self.status = HarnessStatus.COMPLETED
                else:
                    self.status = HarnessStatus.FAILED
                    self.error = f"Process exited with code {poll_result}"
                self.completed_at = datetime.now()
        elif self.pid > 0:
            # Check if PID is still running
            try:
                os.kill(self.pid, 0)  # Signal 0 just checks existence
            except ProcessLookupError:
                # Process is gone
                if self.status == HarnessStatus.RUNNING:
                    self.status = HarnessStatus.COMPLETED
                self.completed_at = datetime.now()

        # Read state from database if available
        self._refresh_from_db()

        # Check for pause/stop files
        if (self.state_dir / "pause").exists():
            self.status = HarnessStatus.PAUSED
        if (self.state_dir / "stop").exists() and self.status != HarnessStatus.STOPPED:
            self.status = HarnessStatus.STOPPED

    def _refresh_from_db(self) -> None:
        """Read state from sub-harness database."""
        if not self.state_db.exists():
            return

        try:
            conn = sqlite3.connect(self.state_db)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get harness state
            cursor.execute("SELECT * FROM harness_state WHERE id = 1")
            row = cursor.fetchone()
            if row:
                self.tasks_completed = row["tasks_completed"]
                self.tasks_failed = row["tasks_failed"]

            # Get current task
            cursor.execute(
                "SELECT id, description FROM tasks WHERE status = 'in_progress' LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                self.current_task = row["description"][:50]

            # Calculate progress
            cursor.execute("SELECT COUNT(*) as total FROM tasks")
            total = cursor.fetchone()["total"]
            if total > 0:
                self.progress = self.tasks_completed / total

            conn.close()
        except Exception as e:
            logger.debug(f"Could not read state from {self.state_db}: {e}")

    def is_running(self) -> bool:
        """Check if harness is still running."""
        self.refresh()
        return self.status == HarnessStatus.RUNNING

    def is_complete(self) -> bool:
        """Check if harness has completed (success or failure)."""
        self.refresh()
        return self.status in (
            HarnessStatus.COMPLETED,
            HarnessStatus.FAILED,
            HarnessStatus.STOPPED,
        )

    def get_result(self) -> HarnessResult:
        """Build HarnessResult from current state."""
        self.refresh()
        return HarnessResult(
            harness_id=self.harness_id,
            status=self.status,
            total_iterations=self.current_iteration,
            tasks_completed=self.tasks_completed,
            tasks_failed=self.tasks_failed,
            error=self.error,
            started_at=self.started_at,
            completed_at=self.completed_at,
            metrics=HarnessMetrics(
                tasks_completed=self.tasks_completed,
                tasks_failed=self.tasks_failed,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "harness_id": self.harness_id,
            "pid": self.pid,
            "working_dir": str(self.working_dir),
            "roadmap_path": str(self.roadmap_path),
            "cli": self.cli,
            "model": self.model,
            "status": self.status.value,
            "progress": self.progress,
            "current_task": self.current_task,
            "current_iteration": self.current_iteration,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarnessHandle":
        """Create from dictionary."""
        handle = cls(
            harness_id=data["harness_id"],
            pid=data["pid"],
            working_dir=Path(data["working_dir"]),
            roadmap_path=Path(data["roadmap_path"]),
            cli=data.get("cli", "auto"),
            model=data.get("model", ""),
        )
        handle.status = HarnessStatus(data.get("status", "pending"))
        handle.progress = data.get("progress", 0.0)
        handle.current_task = data.get("current_task")
        handle.current_iteration = data.get("current_iteration", 0)
        handle.tasks_completed = data.get("tasks_completed", 0)
        handle.tasks_failed = data.get("tasks_failed", 0)
        handle.error = data.get("error")
        if data.get("started_at"):
            handle.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            handle.completed_at = datetime.fromisoformat(data["completed_at"])
        return handle
