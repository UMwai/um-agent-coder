"""Status reporter for autonomous loop.

Generates periodic status summaries and supports status queries.

Reference: specs/autonomous-loop-spec.md Section 10.2, 10.3
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..context_manager import LoopContext


class StatusFormat(Enum):
    """Output format for status reports."""

    TEXT = "text"  # Human-readable text
    JSON = "json"  # JSON format
    BRIEF = "brief"  # One-line summary


@dataclass
class LoopStatus:
    """Current status of the autonomous loop."""

    task_id: str
    task_description: str
    status: str  # RUNNING, PAUSED, STOPPED, COMPLETE, FAILED
    current_iteration: int
    max_iterations: Optional[int]
    elapsed: timedelta
    current_cli: str
    current_model: str
    avg_progress: float
    recent_markers: list[str]
    stuck_state: str
    recovery_attempts: int
    alerts_issued: int

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "status": self.status,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "elapsed_seconds": self.elapsed.total_seconds(),
            "current_cli": self.current_cli,
            "current_model": self.current_model,
            "avg_progress": self.avg_progress,
            "recent_markers": self.recent_markers,
            "stuck_state": self.stuck_state,
            "recovery_attempts": self.recovery_attempts,
            "alerts_issued": self.alerts_issued,
        }


class StatusReporter:
    """Generate periodic status summaries.

    Produces formatted status reports at configurable intervals
    and supports on-demand status queries.

    Args:
        interval_iterations: Generate summary every N iterations.
        status_path: Path to write status file for external queries.
    """

    def __init__(
        self,
        interval_iterations: int = 10,
        status_path: Optional[Path] = None,
    ):
        """Initialize status reporter."""
        self.interval = interval_iterations
        self.status_path = status_path or Path(".harness/status.json")
        self.last_report_iteration = 0

    def should_report(self, iteration: int) -> bool:
        """Check if a status report should be generated.

        Args:
            iteration: Current iteration number.

        Returns:
            True if a report should be generated.
        """
        if iteration == 0:
            return False
        return iteration % self.interval == 0

    def maybe_report(
        self,
        context: LoopContext,
        format: StatusFormat = StatusFormat.TEXT,
    ) -> Optional[str]:
        """Generate a status report if interval reached.

        Args:
            context: The current loop context.
            format: Output format.

        Returns:
            Status report string if interval reached, None otherwise.
        """
        if not self.should_report(context.total_iterations):
            return None

        self.last_report_iteration = context.total_iterations
        return self.generate_summary(context, format)

    def generate_summary(
        self,
        context: LoopContext,
        format: StatusFormat = StatusFormat.TEXT,
    ) -> str:
        """Generate a status summary.

        Args:
            context: The current loop context.
            format: Output format.

        Returns:
            Formatted status summary.
        """
        status = self._build_status(context)

        if format == StatusFormat.JSON:
            import json

            return json.dumps(status.to_dict(), indent=2)
        elif format == StatusFormat.BRIEF:
            return self._format_brief(status)
        else:
            return self._format_text(status)

    def _build_status(self, context: LoopContext) -> LoopStatus:
        """Build status object from context."""
        # Calculate averages
        recent_progress = [it.progress_score for it in context.iterations[-10:]]
        avg_progress = sum(recent_progress) / len(recent_progress) if recent_progress else 0

        # Get recent markers
        recent_markers = []
        for it in context.iterations[-5:]:
            recent_markers.extend(it.progress_markers)

        # Determine status string
        status = "RUNNING"
        if context.iterations and context.iterations[-1].progress_score < 0:
            status = "FAILED"

        # Handle start_time being None
        if context.start_time:
            elapsed = datetime.now() - context.start_time
        else:
            elapsed = timedelta(seconds=0)

        return LoopStatus(
            task_id=context.task_id,
            task_description=context.goal[:100] if context.goal else "",
            status=status,
            current_iteration=context.total_iterations,
            max_iterations=None,  # Set by caller if known
            elapsed=elapsed,
            current_cli=context.iterations[-1].cli_used if context.iterations else "unknown",
            current_model=context.iterations[-1].model_used if context.iterations else "unknown",
            avg_progress=avg_progress,
            recent_markers=recent_markers[-5:],
            stuck_state="unknown",  # Set by caller if known
            recovery_attempts=0,  # Set by caller if known
            alerts_issued=0,  # Set by caller if known
        )

    def _format_text(self, status: LoopStatus) -> str:
        """Format status as human-readable text."""
        elapsed_str = str(status.elapsed).split(".")[0]  # Remove microseconds
        iter_str = f"{status.current_iteration}"
        if status.max_iterations:
            iter_str += f"/{status.max_iterations}"

        markers_str = "\n".join(f"  - {m}" for m in status.recent_markers) or "  (none)"

        return f"""
+{'='*62}+
|{'AUTONOMOUS LOOP STATUS':^62}|
+{'='*62}+
| Task: {status.task_id[:52]:<52}   |
| Goal: {status.task_description[:52]:<52}   |
+{'-'*62}+
| Iterations: {iter_str:<12} Elapsed: {elapsed_str:<18}   |
| Current CLI: {status.current_cli:<12} Model: {status.current_model:<18} |
| Avg Progress (recent): {status.avg_progress:.2f}                                |
| Status: {status.status:<15} Stuck: {status.stuck_state:<18} |
+{'-'*62}+
| Recent Progress Markers:
{markers_str}
+{'='*62}+
"""

    def _format_brief(self, status: LoopStatus) -> str:
        """Format status as one-line summary."""
        elapsed_str = str(status.elapsed).split(".")[0]
        return (
            f"{status.task_id}: {status.status} | "
            f"iter={status.current_iteration} | "
            f"progress={status.avg_progress:.2f} | "
            f"cli={status.current_cli} | "
            f"elapsed={elapsed_str}"
        )

    def write_status_file(
        self,
        context: LoopContext,
        extra_info: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write current status to file for external queries.

        Args:
            context: The current loop context.
            extra_info: Additional info to include.
        """
        import json

        status = self._build_status(context)
        data = status.to_dict()

        if extra_info:
            data.update(extra_info)

        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_path, "w") as f:
            json.dump(data, f, indent=2)

    def read_status_file(self) -> Optional[dict[str, Any]]:
        """Read status from file.

        Returns:
            Status dictionary if file exists, None otherwise.
        """
        import json

        if not self.status_path.exists():
            return None

        try:
            with open(self.status_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def format_status_output(status_data: dict[str, Any]) -> str:
        """Format status data for CLI output.

        Args:
            status_data: Status dictionary from file.

        Returns:
            Formatted string for terminal output.
        """
        if not status_data:
            return "No status available"

        elapsed = timedelta(seconds=status_data.get("elapsed_seconds", 0))
        elapsed_str = str(elapsed).split(".")[0]

        return f"""
Task: {status_data.get('task_id', 'unknown')} ({status_data.get('task_description', '')[:50]})
Status: {status_data.get('status', 'UNKNOWN')}
Iteration: {status_data.get('current_iteration', 0)}/{status_data.get('max_iterations', 'unlimited')}
Progress: {status_data.get('avg_progress', 0):.2f} (recent avg)
CLI: {status_data.get('current_cli', 'unknown')} ({status_data.get('current_model', 'unknown')})
Elapsed: {elapsed_str}
Last markers: {', '.join(status_data.get('recent_markers', [])[:3]) or 'none'}
"""


@dataclass
class MetricsCollector:
    """Collect and aggregate metrics from autonomous loop execution.

    Tracks iteration times, progress scores, CLI usage, and other
    metrics for analysis and optimization.
    """

    iteration_durations: list[float] = field(default_factory=list)
    progress_scores: list[float] = field(default_factory=list)
    cli_usage: dict[str, int] = field(default_factory=dict)
    model_usage: dict[str, int] = field(default_factory=dict)
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    total_tokens_used: int = 0
    start_time: Optional[datetime] = None

    def record_iteration(
        self,
        duration: float,
        progress: float,
        cli: str,
        model: str,
    ) -> None:
        """Record metrics for an iteration.

        Args:
            duration: Iteration duration in seconds.
            progress: Progress score (0.0-1.0).
            cli: CLI used.
            model: Model used.
        """
        self.iteration_durations.append(duration)
        self.progress_scores.append(progress)
        self.cli_usage[cli] = self.cli_usage.get(cli, 0) + 1
        self.model_usage[model] = self.model_usage.get(model, 0) + 1

    def record_recovery(self, success: bool) -> None:
        """Record a recovery attempt.

        Args:
            success: Whether recovery was successful.
        """
        self.recovery_attempts += 1
        if success:
            self.successful_recoveries += 1

    def record_tokens(self, tokens: int) -> None:
        """Record token usage.

        Args:
            tokens: Number of tokens used.
        """
        self.total_tokens_used += tokens

    def get_summary(self) -> dict[str, Any]:
        """Get summary of collected metrics.

        Returns:
            Dictionary with metric summaries.
        """
        total_iterations = len(self.iteration_durations)

        return {
            "total_iterations": total_iterations,
            "total_duration_seconds": sum(self.iteration_durations),
            "avg_iteration_duration": (
                sum(self.iteration_durations) / total_iterations if total_iterations > 0 else 0
            ),
            "avg_progress": (
                sum(self.progress_scores) / len(self.progress_scores) if self.progress_scores else 0
            ),
            "min_progress": min(self.progress_scores) if self.progress_scores else 0,
            "max_progress": max(self.progress_scores) if self.progress_scores else 0,
            "cli_usage": dict(self.cli_usage),
            "model_usage": dict(self.model_usage),
            "recovery_attempts": self.recovery_attempts,
            "successful_recoveries": self.successful_recoveries,
            "recovery_success_rate": (
                self.successful_recoveries / self.recovery_attempts
                if self.recovery_attempts > 0
                else 0
            ),
            "total_tokens_used": self.total_tokens_used,
        }

    def reset(self) -> None:
        """Reset all collected metrics."""
        self.iteration_durations.clear()
        self.progress_scores.clear()
        self.cli_usage.clear()
        self.model_usage.clear()
        self.recovery_attempts = 0
        self.successful_recoveries = 0
        self.total_tokens_used = 0
        self.start_time = None
