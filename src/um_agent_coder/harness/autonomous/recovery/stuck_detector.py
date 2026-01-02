"""Stuck detection for autonomous loop.

Tracks consecutive iterations without progress and determines when
the loop is stuck and needs recovery intervention.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class StuckState(Enum):
    """Current stuck state of the loop."""

    PROGRESSING = "progressing"  # Making progress
    WARNING = "warning"  # Some iterations without progress
    STUCK = "stuck"  # Needs recovery intervention
    RECOVERING = "recovering"  # Recovery in progress


@dataclass
class ProgressRecord:
    """Record of progress for a single iteration."""

    iteration: int
    timestamp: datetime
    progress_score: float
    had_progress: bool
    output_snippet: str = ""


@dataclass
class StuckDetector:
    """Detect when the autonomous loop is stuck.

    Tracks consecutive iterations without progress and triggers
    recovery when threshold is exceeded.

    Args:
        stuck_threshold: Iterations without progress before marking stuck.
            Default is 3 for normal operation.
        warning_threshold: Iterations without progress before warning.
            Default is 2.
        recovery_budget: Behind-the-scenes recovery iterations.
            Default is 20.
    """

    stuck_threshold: int = 3
    warning_threshold: int = 2
    recovery_budget: int = 20

    # State tracking
    consecutive_no_progress: int = 0
    total_no_progress: int = 0
    history: list[ProgressRecord] = field(default_factory=list)
    current_state: StuckState = StuckState.PROGRESSING
    recovery_iterations_used: int = 0

    def record_iteration(
        self,
        iteration: int,
        progress_score: float,
        had_progress: bool,
        output_snippet: str = "",
    ) -> StuckState:
        """Record an iteration's progress and update state.

        Args:
            iteration: The iteration number.
            progress_score: The calculated progress score (0.0-1.0).
            had_progress: Whether this iteration made meaningful progress.
            output_snippet: Optional snippet of output for debugging.

        Returns:
            Current stuck state after this iteration.
        """
        record = ProgressRecord(
            iteration=iteration,
            timestamp=datetime.now(),
            progress_score=progress_score,
            had_progress=had_progress,
            output_snippet=output_snippet[:200] if output_snippet else "",
        )
        self.history.append(record)

        if had_progress:
            # Reset consecutive counter on progress
            self.consecutive_no_progress = 0
            self.current_state = StuckState.PROGRESSING
        else:
            # Increment counters
            self.consecutive_no_progress += 1
            self.total_no_progress += 1

            # Update state based on thresholds
            if self.consecutive_no_progress >= self.stuck_threshold:
                self.current_state = StuckState.STUCK
            elif self.consecutive_no_progress >= self.warning_threshold:
                self.current_state = StuckState.WARNING

        return self.current_state

    def is_stuck(self) -> bool:
        """Check if currently stuck."""
        return self.current_state == StuckState.STUCK

    def is_warning(self) -> bool:
        """Check if in warning state."""
        return self.current_state == StuckState.WARNING

    def needs_recovery(self) -> bool:
        """Check if recovery intervention is needed."""
        return self.current_state == StuckState.STUCK

    def start_recovery(self) -> None:
        """Mark recovery as started."""
        self.current_state = StuckState.RECOVERING

    def end_recovery(self, success: bool) -> None:
        """Mark recovery as ended.

        Args:
            success: Whether recovery was successful.
        """
        if success:
            self.consecutive_no_progress = 0
            self.current_state = StuckState.PROGRESSING
        else:
            # Recovery failed, back to stuck
            self.current_state = StuckState.STUCK

    def use_recovery_iteration(self) -> bool:
        """Use one recovery iteration from budget.

        Returns:
            True if iteration was available, False if budget exhausted.
        """
        if self.recovery_iterations_used >= self.recovery_budget:
            return False
        self.recovery_iterations_used += 1
        return True

    def recovery_budget_remaining(self) -> int:
        """Get remaining recovery budget."""
        return max(0, self.recovery_budget - self.recovery_iterations_used)

    def reset(self) -> None:
        """Reset all state."""
        self.consecutive_no_progress = 0
        self.total_no_progress = 0
        self.history.clear()
        self.current_state = StuckState.PROGRESSING
        self.recovery_iterations_used = 0

    def get_summary(self) -> dict:
        """Get summary of stuck detection state."""
        recent_scores = [r.progress_score for r in self.history[-5:]] if self.history else []

        return {
            "current_state": self.current_state.value,
            "consecutive_no_progress": self.consecutive_no_progress,
            "total_no_progress": self.total_no_progress,
            "total_iterations": len(self.history),
            "recovery_budget_remaining": self.recovery_budget_remaining(),
            "recent_scores": recent_scores,
            "stuck_threshold": self.stuck_threshold,
            "warning_threshold": self.warning_threshold,
        }

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "stuck_threshold": self.stuck_threshold,
            "warning_threshold": self.warning_threshold,
            "recovery_budget": self.recovery_budget,
            "consecutive_no_progress": self.consecutive_no_progress,
            "total_no_progress": self.total_no_progress,
            "current_state": self.current_state.value,
            "recovery_iterations_used": self.recovery_iterations_used,
            "history": [
                {
                    "iteration": r.iteration,
                    "timestamp": r.timestamp.isoformat(),
                    "progress_score": r.progress_score,
                    "had_progress": r.had_progress,
                    "output_snippet": r.output_snippet,
                }
                for r in self.history
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StuckDetector":
        """Deserialize from dictionary."""
        detector = cls(
            stuck_threshold=data.get("stuck_threshold", 3),
            warning_threshold=data.get("warning_threshold", 2),
            recovery_budget=data.get("recovery_budget", 20),
        )
        detector.consecutive_no_progress = data.get("consecutive_no_progress", 0)
        detector.total_no_progress = data.get("total_no_progress", 0)
        detector.current_state = StuckState(data.get("current_state", "progressing"))
        detector.recovery_iterations_used = data.get("recovery_iterations_used", 0)

        # Restore history
        for h in data.get("history", []):
            detector.history.append(
                ProgressRecord(
                    iteration=h["iteration"],
                    timestamp=datetime.fromisoformat(h["timestamp"]),
                    progress_score=h["progress_score"],
                    had_progress=h["had_progress"],
                    output_snippet=h.get("output_snippet", ""),
                )
            )

        return detector
