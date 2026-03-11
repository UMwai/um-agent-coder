"""
Iteration tracking for ralph loop execution.

Tracks iteration state with configurable limits and provides
history tracking for debugging and analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional


@dataclass
class IterationRecord:
    """Record of a single iteration attempt."""

    iteration_num: int
    started_at: datetime
    ended_at: Optional[datetime] = None
    output_snippet: str = ""
    promise_found: bool = False
    error: Optional[str] = None
    # QA validation fields
    test_passed: Optional[bool] = None  # None if tests not run, bool otherwise
    test_summary: str = ""  # Brief test result summary
    # Goal validation fields
    goal_score: Optional[float] = None  # None if goal validation not run
    goal_passed: Optional[bool] = None
    # Eval score tracking
    eval_score: Optional[float] = None  # Per-iteration eval score (0.0-1.0)

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
            "test_passed": self.test_passed,
            "test_summary": self.test_summary,
            "goal_score": self.goal_score,
            "goal_passed": self.goal_passed,
            "eval_score": self.eval_score,
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
            test_passed=data.get("test_passed"),
            test_summary=data.get("test_summary", ""),
            goal_score=data.get("goal_score"),
            goal_passed=data.get("goal_passed"),
            eval_score=data.get("eval_score"),
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

    def detect_oscillation(self, window: int = 4, spread: float = 0.03) -> dict:
        """Detect if the loop is stuck oscillating around the same score.

        Analyzes the last `window` eval_scores from iteration_history to
        determine if progress has stalled.

        Args:
            window: Number of recent scores to analyze.
            spread: Maximum spread (max - min) to consider oscillating.

        Returns:
            Dictionary with oscillation analysis results.
        """
        scores: List[float] = [
            r.eval_score for r in self.iteration_history if r.eval_score is not None
        ]
        recent = scores[-window:]

        if len(recent) < window:
            return {
                "oscillating": False,
                "scores": recent,
                "spread": 0.0,
                "mean": 0.0,
                "suggestion": "continue",
            }

        score_spread = max(recent) - min(recent)
        score_mean = sum(recent) / len(recent)

        if score_spread <= spread:
            if score_mean < 0.5:
                suggestion = "escalate_model"
            elif score_mean < 0.8:
                suggestion = "mutate_prompt"
            else:
                suggestion = "continue"
            return {
                "oscillating": True,
                "scores": recent,
                "spread": score_spread,
                "mean": score_mean,
                "suggestion": suggestion,
            }

        return {
            "oscillating": False,
            "scores": recent,
            "spread": score_spread,
            "mean": score_mean,
            "suggestion": "continue",
        }

    def get_score_trajectory(self, window: int = 5) -> dict:
        """Return score trajectory information.

        Args:
            window: Number of recent scores to consider for trend analysis.

        Returns:
            Dictionary with trend, scores, best score info, and improvement rate.
        """
        scores: List[float] = [
            r.eval_score for r in self.iteration_history if r.eval_score is not None
        ]

        if len(scores) < 2:
            return {
                "trend": "insufficient_data",
                "scores": scores,
                "best_score": scores[0] if scores else 0.0,
                "best_iteration": 1 if scores else 0,
                "current_score": scores[-1] if scores else 0.0,
                "improvement_rate": 0.0,
            }

        recent = scores[-window:]
        improvement_rate = (recent[-1] - recent[0]) / len(recent)

        # Determine trend from last 3 scores (or fewer if not enough)
        tail = recent[-3:] if len(recent) >= 3 else recent
        if len(tail) >= 3 and tail[-1] > tail[-2] > tail[-3]:
            trend = "improving"
        elif len(tail) >= 3 and tail[-1] < tail[-2] < tail[-3]:
            trend = "declining"
        else:
            trend = "flat"

        # Find best score and its iteration number
        best_score = 0.0
        best_iteration = 0
        for record in self.iteration_history:
            if record.eval_score is not None and record.eval_score > best_score:
                best_score = record.eval_score
                best_iteration = record.iteration_num

        return {
            "trend": trend,
            "scores": scores,
            "best_score": best_score,
            "best_iteration": best_iteration,
            "current_score": scores[-1],
            "improvement_rate": improvement_rate,
        }

    def should_score_this_iteration(self, scoring_interval: int = 3) -> bool:
        """Determine if the current iteration should run a scoring eval.

        Args:
            scoring_interval: Run scoring every N iterations.

        Returns:
            True if scoring should run on this iteration.
        """
        # Always score on iteration 1
        if self.current_iteration == 1:
            return True

        # Always score if previous iteration found a promise (to validate)
        if self.iteration_history:
            last_record = self.iteration_history[-1]
            if last_record.promise_found:
                return True

        # Score every scoring_interval iterations
        return self.current_iteration % scoring_interval == 0

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
