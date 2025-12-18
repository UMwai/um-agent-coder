"""
Data models for the 24/7 Codex Harness.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class TaskStatus(Enum):
    """Task lifecycle states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    GROWTH = "growth"  # Task complete, now in improvement mode


@dataclass
class Task:
    """A single executable task from the roadmap."""
    id: str
    description: str
    phase: str
    depends: list[str] = field(default_factory=list)
    timeout_minutes: int = 30
    success_criteria: str = ""
    cwd: str = "./"
    cli: str = ""  # CLI backend override (codex, gemini, claude) - empty uses default
    model: str = ""  # Model override - empty uses CLI default
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    max_retries: int = 3
    output: str = ""
    error: str = ""
    conversation_id: Optional[str] = None  # For conversation continuity
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def can_execute(self, completed_tasks: set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.depends)

    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.BLOCKED)


@dataclass
class Phase:
    """A phase containing multiple related tasks."""
    name: str
    tasks: list[Task] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks)


@dataclass
class Roadmap:
    """The full roadmap parsed from specs/roadmap.md."""
    name: str
    objective: str
    success_criteria: list[str] = field(default_factory=list)
    phases: list[Phase] = field(default_factory=list)
    growth_instructions: list[str] = field(default_factory=list)

    # Constraints
    max_time_per_task: int = 30  # minutes
    max_retries: int = 3
    working_directory: str = "./"

    @property
    def all_tasks(self) -> list[Task]:
        """Flatten all tasks from all phases."""
        return [task for phase in self.phases for task in phase.tasks]

    @property
    def is_complete(self) -> bool:
        """Check if all phases are complete."""
        return all(phase.is_complete for phase in self.phases)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Find a task by ID."""
        for task in self.all_tasks:
            if task.id == task_id:
                return task
        return None


@dataclass
class ExecutionResult:
    """Result from a Codex execution."""
    success: bool
    output: str
    error: str = ""
    conversation_id: Optional[str] = None
    duration_seconds: float = 0.0

    @property
    def summary(self) -> str:
        """Brief summary for logging."""
        status = "SUCCESS" if self.success else "FAILED"
        duration = f"{self.duration_seconds:.1f}s"
        preview = self.output[:100] + "..." if len(self.output) > 100 else self.output
        return f"[{status}] ({duration}) {preview}"


@dataclass
class HarnessState:
    """Overall harness state for persistence."""
    roadmap_path: str
    started_at: datetime
    last_activity: datetime
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0  # seconds
    in_growth_mode: bool = False
    growth_iterations: int = 0
