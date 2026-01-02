"""
Task checkpointing for long-running agent tasks.

Inspired by LangGraph's checkpoint architecture, this module provides
durable execution capabilities - allowing tasks to survive failures
and resume from exactly where they left off.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class TaskStatus(Enum):
    """Status of a long-running task."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StepState:
    """State of a single execution step."""

    step_index: int
    description: str
    action: str
    parameters: dict[str, Any]
    status: str  # pending, completed, failed
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class TaskState:
    """
    Complete state of a task, serializable for checkpointing.

    This captures everything needed to resume a task:
    - Task identification and metadata
    - Execution plan and step progress
    - Context and cost tracking state
    - Timing information
    """

    # Identity
    task_id: str
    prompt: str

    # Status
    status: TaskStatus
    current_step: int

    # Execution state
    steps: list[StepState] = field(default_factory=list)

    # Context state (serialized from ContextManager)
    context_items: list[dict[str, Any]] = field(default_factory=list)

    # Cost tracking state (serialized from CostTracker)
    cost_state: dict[str, Any] = field(default_factory=dict)

    # Task analysis (for resume without re-analyzing)
    task_analysis: Optional[dict[str, Any]] = None

    # Timing
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskState":
        """Create from dictionary (JSON deserialization)."""
        data = data.copy()
        data["status"] = TaskStatus(data["status"])

        # Convert steps back to StepState objects
        if "steps" in data:
            data["steps"] = [
                StepState(**step) if isinstance(step, dict) else step for step in data["steps"]
            ]

        return cls(**data)


class TaskCheckpointer:
    """
    Manages task checkpoints for durable execution.

    Features:
    - Save/load task state to filesystem (SQLite coming soon)
    - List resumable tasks
    - Automatic checkpoint versioning
    - Thread-safe operations

    Usage:
        checkpointer = TaskCheckpointer()

        # Save checkpoint
        checkpointer.save(task_state)

        # Load checkpoint
        state = checkpointer.load(task_id)

        # List all tasks
        tasks = checkpointer.list_tasks()

        # Resume from checkpoint
        if state := checkpointer.load(task_id):
            agent.resume(state)
    """

    def __init__(self, storage_path: str = ".task_checkpoints"):
        """
        Initialize the checkpointer.

        Args:
            storage_path: Directory to store checkpoint files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_task_path(self, task_id: str) -> Path:
        """Get the file path for a task checkpoint."""
        return self.storage_path / f"{task_id}.json"

    def _get_history_path(self, task_id: str) -> Path:
        """Get the directory for task history/versions."""
        return self.storage_path / "history" / task_id

    def save(self, state: TaskState, create_version: bool = True) -> bool:
        """
        Save a task checkpoint.

        Args:
            state: The task state to save
            create_version: If True, also save a versioned copy

        Returns:
            True if saved successfully
        """
        try:
            state.updated_at = datetime.now().isoformat()

            # Save current state
            task_path = self._get_task_path(state.task_id)
            with open(task_path, "w") as f:
                json.dump(state.to_dict(), f, indent=2)

            # Optionally save versioned copy
            if create_version:
                history_dir = self._get_history_path(state.task_id)
                history_dir.mkdir(parents=True, exist_ok=True)

                version_file = history_dir / f"{int(time.time() * 1000)}.json"
                with open(version_file, "w") as f:
                    json.dump(state.to_dict(), f, indent=2)

            return True

        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return False

    def load(self, task_id: str) -> Optional[TaskState]:
        """
        Load a task checkpoint.

        Args:
            task_id: The task ID to load

        Returns:
            TaskState if found, None otherwise
        """
        task_path = self._get_task_path(task_id)

        if not task_path.exists():
            return None

        try:
            with open(task_path) as f:
                data = json.load(f)
            return TaskState.from_dict(data)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None

    def load_version(self, task_id: str, version_ts: int) -> Optional[TaskState]:
        """
        Load a specific version of a task checkpoint.

        Args:
            task_id: The task ID
            version_ts: Timestamp of the version to load

        Returns:
            TaskState if found, None otherwise
        """
        version_file = self._get_history_path(task_id) / f"{version_ts}.json"

        if not version_file.exists():
            return None

        try:
            with open(version_file) as f:
                data = json.load(f)
            return TaskState.from_dict(data)
        except Exception as e:
            print(f"Error loading checkpoint version: {e}")
            return None

    def list_versions(self, task_id: str) -> list[dict[str, Any]]:
        """
        List all versions of a task checkpoint.

        Returns:
            List of version info dicts with timestamp and path
        """
        history_dir = self._get_history_path(task_id)

        if not history_dir.exists():
            return []

        versions = []
        for version_file in sorted(history_dir.glob("*.json"), reverse=True):
            ts = int(version_file.stem)
            versions.append(
                {
                    "timestamp": ts,
                    "datetime": datetime.fromtimestamp(ts / 1000).isoformat(),
                    "path": str(version_file),
                }
            )

        return versions

    def delete(self, task_id: str, include_history: bool = False) -> bool:
        """
        Delete a task checkpoint.

        Args:
            task_id: The task ID to delete
            include_history: Also delete version history

        Returns:
            True if deleted successfully
        """
        try:
            task_path = self._get_task_path(task_id)
            if task_path.exists():
                task_path.unlink()

            if include_history:
                history_dir = self._get_history_path(task_id)
                if history_dir.exists():
                    for f in history_dir.glob("*.json"):
                        f.unlink()
                    history_dir.rmdir()

            return True
        except Exception as e:
            print(f"Error deleting checkpoint: {e}")
            return False

    def list_tasks(self, status_filter: Optional[TaskStatus] = None) -> list[dict[str, Any]]:
        """
        List all tasks with their current status.

        Args:
            status_filter: Optional filter by status

        Returns:
            List of task summaries
        """
        tasks = []

        for task_file in self.storage_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    data = json.load(f)

                task_status = TaskStatus(data.get("status", "pending"))

                if status_filter and task_status != status_filter:
                    continue

                # Calculate progress
                steps = data.get("steps", [])
                completed_steps = sum(1 for s in steps if s.get("status") == "completed")
                total_steps = len(steps)

                tasks.append(
                    {
                        "task_id": data["task_id"],
                        "prompt": (
                            data["prompt"][:80] + "..."
                            if len(data["prompt"]) > 80
                            else data["prompt"]
                        ),
                        "status": task_status.value,
                        "progress": f"{completed_steps}/{total_steps}",
                        "progress_pct": (
                            (completed_steps / total_steps * 100) if total_steps > 0 else 0
                        ),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "current_step": data.get("current_step", 0),
                    }
                )

            except Exception as e:
                print(f"Error reading task file {task_file}: {e}")
                continue

        # Sort by updated_at descending
        tasks.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        return tasks

    def get_resumable_tasks(self) -> list[dict[str, Any]]:
        """
        Get all tasks that can be resumed (running or paused).

        Returns:
            List of resumable task summaries
        """
        all_tasks = self.list_tasks()
        return [
            t
            for t in all_tasks
            if t["status"] in [TaskStatus.RUNNING.value, TaskStatus.PAUSED.value]
        ]

    def cleanup_old_versions(self, task_id: str, keep_last: int = 10) -> int:
        """
        Clean up old versions, keeping only the most recent N.

        Args:
            task_id: Task to clean up
            keep_last: Number of versions to keep

        Returns:
            Number of versions deleted
        """
        history_dir = self._get_history_path(task_id)

        if not history_dir.exists():
            return 0

        versions = sorted(history_dir.glob("*.json"), reverse=True)
        to_delete = versions[keep_last:]

        deleted = 0
        for version_file in to_delete:
            try:
                version_file.unlink()
                deleted += 1
            except Exception:
                pass

        return deleted

    def cleanup_completed(self, older_than_days: int = 7) -> int:
        """
        Clean up completed tasks older than N days.

        Args:
            older_than_days: Delete completed tasks older than this

        Returns:
            Number of tasks deleted
        """
        cutoff = time.time() - (older_than_days * 24 * 60 * 60)
        deleted = 0

        for task in self.list_tasks(TaskStatus.COMPLETED):
            try:
                completed_at = task.get("completed_at")
                if completed_at:
                    completed_ts = datetime.fromisoformat(completed_at).timestamp()
                    if completed_ts < cutoff:
                        self.delete(task["task_id"], include_history=True)
                        deleted += 1
            except Exception:
                pass

        return deleted
