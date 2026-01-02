"""
Iteration tracking for ralph loop execution.

Tracks iteration state with configurable limits and provides
history tracking for debugging and analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class IterationRecord:
    """Record of a single iteration attempt."""

    iteration_num: int
    started_at: datetime
    ended_at: Optional[datetime] = None
    output_snippet: str = ""
    promise_found: bool = False
    error: Optional[str] = None

    @property
    def duration(self) -> Optional[timedelta]:
        """Get the duration of this iteration."""
        if self.ended_at is None:
            return None
        return self.ended_at - self.started_at

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "iteration_num": self.iteration_num,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "output_snippet": self.output_snippet,
            "promise_found": self.promise_found,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IterationRecord":
        """Create from dictionary."""
        return cls(
            iteration_num=data["iteration_num"],
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            output_snippet=data.get("output_snippet", ""),
            promise_found=data.get("promise_found", False),
            error=data.get("error"),
        )


@dataclass
class IterationTracker:
    """Tracks iteration state for a ralph loop task.

    Example:
        tracker = IterationTracker(task_id="task-001", max_iterations=10)
        while tracker.can_continue():
            # Execute task
            tracker.start_iteration()
            result = execute_task()
            tracker.end_iteration(result.output, promise_found=True)
            if promise_found:
                break
    """

    task_id: str
    max_iterations: int = 30
    current_iteration: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    iteration_history: list[IterationRecord] = field(default_factory=list)

    # Internal state
    _current_record: Optional[IterationRecord] = field(default=None, repr=False)
    _completed: bool = field(default=False, repr=False)
    _completion_reason: Optional[str] = field(default=None, repr=False)

    def can_continue(self) -> bool:
        """Check if another iteration can be executed.

        Returns:
            True if max_iterations not exceeded and not already completed
        """
        if self._completed:
            return False
        return self.current_iteration < self.max_iterations

    def start_iteration(self) -> IterationRecord:
        """Start a new iteration.

        Returns:
            The new IterationRecord for this iteration
        """
        self.current_iteration += 1
        self._current_record = IterationRecord(
            iteration_num=self.current_iteration,
            started_at=datetime.utcnow(),
        )
        return self._current_record

    def end_iteration(
        self,
        output: str = "",
        promise_found: bool = False,
        error: Optional[str] = None,
        output_snippet_length: int = 500,
    ) -> IterationRecord:
        """End the current iteration and record results.

        Args:
            output: Full output from the executor
            promise_found: Whether the completion promise was found
            error: Error message if iteration failed
            output_snippet_length: Max length of output to store in history

        Returns:
            The completed IterationRecord
        """
        if self._current_record is None:
            raise RuntimeError("end_iteration called without start_iteration")

        self._current_record.ended_at = datetime.utcnow()
        self._current_record.output_snippet = output[:output_snippet_length] if output else ""
        self._current_record.promise_found = promise_found
        self._current_record.error = error

        self.iteration_history.append(self._current_record)

        if promise_found:
            self._completed = True
            self._completion_reason = "promise_found"

        record = self._current_record
        self._current_record = None
        return record

    def increment(self) -> int:
        """Convenience method to increment iteration count without tracking details.

        Returns:
            The new iteration count
        """
        self.current_iteration += 1
        return self.current_iteration

    def mark_complete(self, reason: str = "manual") -> None:
        """Mark the task as complete without promise detection."""
        self._completed = True
        self._completion_reason = reason

    def mark_exceeded(self) -> None:
        """Mark that max iterations was exceeded."""
        self._completed = True
        self._completion_reason = "max_iterations_exceeded"

    @property
    def is_complete(self) -> bool:
        """Check if tracking is complete."""
        return self._completed

    @property
    def completion_reason(self) -> Optional[str]:
        """Get the reason for completion."""
        return self._completion_reason

    @property
    def total_duration(self) -> timedelta:
        """Get total time elapsed since start."""
        return datetime.utcnow() - self.start_time

    @property
    def iterations_remaining(self) -> int:
        """Get number of iterations remaining."""
        return max(0, self.max_iterations - self.current_iteration)

    def get_summary(self) -> dict:
        """Get a summary of the iteration state.

        Returns:
            Dictionary with iteration statistics
        """
        successful_iterations = sum(1 for r in self.iteration_history if r.promise_found)
        failed_iterations = sum(1 for r in self.iteration_history if r.error is not None)

        avg_duration = None
        if self.iteration_history:
            durations = [
                r.duration.total_seconds() for r in self.iteration_history if r.duration is not None
            ]
            if durations:
                avg_duration = sum(durations) / len(durations)

        return {
            "task_id": self.task_id,
            "max_iterations": self.max_iterations,
            "current_iteration": self.current_iteration,
            "iterations_remaining": self.iterations_remaining,
            "is_complete": self._completed,
            "completion_reason": self._completion_reason,
            "total_duration_seconds": self.total_duration.total_seconds(),
            "successful_iterations": successful_iterations,
            "failed_iterations": failed_iterations,
            "average_iteration_seconds": avg_duration,
            "start_time": self.start_time.isoformat(),
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "max_iterations": self.max_iterations,
            "current_iteration": self.current_iteration,
            "start_time": self.start_time.isoformat(),
            "iteration_history": [r.to_dict() for r in self.iteration_history],
            "completed": self._completed,
            "completion_reason": self._completion_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IterationTracker":
        """Create from dictionary."""
        tracker = cls(
            task_id=data["task_id"],
            max_iterations=data["max_iterations"],
            current_iteration=data["current_iteration"],
            start_time=datetime.fromisoformat(data["start_time"]),
            iteration_history=[
                IterationRecord.from_dict(r) for r in data.get("iteration_history", [])
            ],
        )
        tracker._completed = data.get("completed", False)
        tracker._completion_reason = data.get("completion_reason")
        return tracker
