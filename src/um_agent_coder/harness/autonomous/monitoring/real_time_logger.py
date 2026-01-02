"""Real-time logging for autonomous loop.

Streams logs to terminal and file with color-coded output.

Reference: specs/autonomous-loop-spec.md Section 10.1
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class LogLevel(Enum):
    """Log levels for autonomous loop events."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SUCCESS = "SUCCESS"


@dataclass
class LogEntry:
    """A single log entry."""

    timestamp: datetime
    level: LogLevel
    message: str
    iteration: Optional[int] = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "iteration": self.iteration,
            "context": self.context,
        }


# ANSI color codes
COLORS = {
    LogLevel.DEBUG: "\033[90m",  # Gray
    LogLevel.INFO: "\033[94m",  # Blue
    LogLevel.WARNING: "\033[93m",  # Yellow
    LogLevel.ERROR: "\033[91m",  # Red
    LogLevel.CRITICAL: "\033[95m",  # Magenta
    LogLevel.SUCCESS: "\033[92m",  # Green
}
RESET = "\033[0m"
BOLD = "\033[1m"


class RealTimeLogger:
    """Stream logs to terminal and file.

    Provides color-coded console output and persistent file logging
    for autonomous loop events.

    Args:
        log_path: Path to log file. Defaults to .harness/harness.log
        console_output: Whether to print to console.
        file_output: Whether to write to file.
        min_level: Minimum log level to output.
        use_colors: Whether to use ANSI colors in console output.
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        console_output: bool = True,
        file_output: bool = True,
        min_level: LogLevel = LogLevel.INFO,
        use_colors: bool = True,
    ):
        """Initialize real-time logger."""
        self.log_path = log_path or Path(".harness/harness.log")
        self.console_output = console_output
        self.file_output = file_output
        self.min_level = min_level
        self.use_colors = use_colors
        self.entries: list[LogEntry] = []

        # Ensure directory exists
        if self.file_output:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Standard logging integration
        self.logger = logging.getLogger(__name__)

    def _should_log(self, level: LogLevel) -> bool:
        """Check if level should be logged."""
        levels = list(LogLevel)
        return levels.index(level) >= levels.index(self.min_level)

    def _format_console(self, entry: LogEntry, style: Optional[str] = None) -> str:
        """Format entry for console output."""
        timestamp = entry.timestamp.strftime("%H:%M:%S")
        iteration_str = f"[{entry.iteration}]" if entry.iteration is not None else ""

        if self.use_colors:
            color = COLORS.get(entry.level, "")
            level_str = f"{color}[{entry.level.value}]{RESET}"
            if style:
                msg = f"{style}{entry.message}{RESET}"
            else:
                msg = entry.message
            return f"{timestamp} {level_str} {iteration_str} {msg}"
        else:
            return f"{timestamp} [{entry.level.value}] {iteration_str} {entry.message}"

    def _format_file(self, entry: LogEntry) -> str:
        """Format entry for file output."""
        return (
            f"[{entry.timestamp.isoformat()}] [{entry.level.value}] "
            + (f"[iter={entry.iteration}] " if entry.iteration is not None else "")
            + entry.message
        )

    def _write(self, entry: LogEntry, style: Optional[str] = None) -> None:
        """Write log entry to outputs."""
        if not self._should_log(entry.level):
            return

        self.entries.append(entry)

        if self.console_output:
            print(self._format_console(entry, style))

        if self.file_output:
            try:
                with open(self.log_path, "a") as f:
                    f.write(self._format_file(entry) + "\n")
            except OSError as e:
                self.logger.warning(f"Failed to write to log file: {e}")

    def log(
        self,
        level: LogLevel,
        message: str,
        iteration: Optional[int] = None,
        context: Optional[dict[str, Any]] = None,
        style: Optional[str] = None,
    ) -> None:
        """Log a message.

        Args:
            level: Log level.
            message: The message to log.
            iteration: Optional iteration number.
            context: Optional context dictionary.
            style: Optional ANSI style code.
        """
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            iteration=iteration,
            context=context or {},
        )
        self._write(entry, style)

    def debug(self, message: str, iteration: Optional[int] = None, **context) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, iteration, context)

    def info(self, message: str, iteration: Optional[int] = None, **context) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, message, iteration, context)

    def warning(self, message: str, iteration: Optional[int] = None, **context) -> None:
        """Log warning message."""
        self.log(LogLevel.WARNING, message, iteration, context)

    def error(self, message: str, iteration: Optional[int] = None, **context) -> None:
        """Log error message."""
        self.log(LogLevel.ERROR, message, iteration, context)

    def critical(self, message: str, iteration: Optional[int] = None, **context) -> None:
        """Log critical message."""
        self.log(LogLevel.CRITICAL, message, iteration, context)

    def success(self, message: str, iteration: Optional[int] = None, **context) -> None:
        """Log success message."""
        self.log(LogLevel.SUCCESS, message, iteration, context)

    # Specific event logging methods

    def log_iteration_start(
        self,
        iteration: int,
        cli: str,
        model: str,
    ) -> None:
        """Log iteration start.

        Args:
            iteration: The iteration number.
            cli: The CLI being used.
            model: The model being used.
        """
        self.log(
            LogLevel.INFO,
            f"Iteration {iteration} starting (cli={cli}, model={model})",
            iteration=iteration,
            context={"cli": cli, "model": model},
            style=BOLD,
        )

    def log_iteration_complete(
        self,
        iteration: int,
        progress: float,
        duration: float,
    ) -> None:
        """Log iteration completion.

        Args:
            iteration: The iteration number.
            progress: The progress score (0.0-1.0).
            duration: Duration in seconds.
        """
        # Choose color based on progress
        if progress > 0.3:
            level = LogLevel.SUCCESS
        elif progress > 0.1:
            level = LogLevel.INFO
        else:
            level = LogLevel.WARNING

        self.log(
            level,
            f"Iteration {iteration} complete (progress={progress:.2f}, duration={duration:.1f}s)",
            iteration=iteration,
            context={"progress": progress, "duration": duration},
        )

    def log_progress_marker(self, iteration: int, marker: str) -> None:
        """Log a progress marker.

        Args:
            iteration: The iteration number.
            marker: The progress marker text.
        """
        self.log(
            LogLevel.INFO,
            f"Progress: {marker}",
            iteration=iteration,
            context={"marker": marker},
            style="\033[96m",  # Cyan
        )

    def log_stuck_detected(self, iteration: int, consecutive: int) -> None:
        """Log stuck detection.

        Args:
            iteration: The iteration number.
            consecutive: Consecutive no-progress iterations.
        """
        self.log(
            LogLevel.WARNING,
            f"STUCK DETECTED at iteration {iteration} ({consecutive} consecutive no-progress)",
            iteration=iteration,
            context={"consecutive_no_progress": consecutive},
            style=f"{BOLD}\033[91m",  # Bold red
        )

    def log_recovery_attempt(
        self,
        iteration: int,
        strategy: str,
        details: str,
    ) -> None:
        """Log a recovery attempt.

        Args:
            iteration: The iteration number.
            strategy: The recovery strategy name.
            details: Details about the recovery.
        """
        self.log(
            LogLevel.INFO,
            f"Recovery: {strategy} - {details}",
            iteration=iteration,
            context={"strategy": strategy, "details": details},
            style="\033[95m",  # Magenta
        )

    def log_recovery_result(
        self,
        iteration: int,
        success: bool,
        strategy: str,
    ) -> None:
        """Log recovery result.

        Args:
            iteration: The iteration number.
            success: Whether recovery was successful.
            strategy: The recovery strategy used.
        """
        if success:
            self.log(
                LogLevel.SUCCESS,
                f"Recovery successful ({strategy})",
                iteration=iteration,
                context={"strategy": strategy, "success": True},
            )
        else:
            self.log(
                LogLevel.WARNING,
                f"Recovery failed ({strategy})",
                iteration=iteration,
                context={"strategy": strategy, "success": False},
            )

    def log_goal_complete(
        self,
        iteration: int,
        promise: str,
        duration_total: float,
    ) -> None:
        """Log goal completion.

        Args:
            iteration: The final iteration number.
            promise: The completion promise text.
            duration_total: Total execution duration in seconds.
        """
        self.log(
            LogLevel.SUCCESS,
            f"GOAL COMPLETE! Promise detected: {promise} (total: {duration_total:.1f}s)",
            iteration=iteration,
            context={"promise": promise, "total_duration": duration_total},
            style=f"{BOLD}\033[92m",  # Bold green
        )

    def log_termination(
        self,
        iteration: int,
        reason: str,
        success: bool,
    ) -> None:
        """Log loop termination.

        Args:
            iteration: The final iteration number.
            reason: The termination reason.
            success: Whether termination was successful.
        """
        level = LogLevel.SUCCESS if success else LogLevel.WARNING
        self.log(
            level,
            f"Loop terminated: {reason} (iterations={iteration})",
            iteration=iteration,
            context={"reason": reason, "success": success},
        )

    def log_alert(
        self,
        iteration: int,
        alert_type: str,
        severity: str,
        message: str,
    ) -> None:
        """Log an alert.

        Args:
            iteration: The iteration number.
            alert_type: The type of alert.
            severity: The alert severity.
            message: The alert message.
        """
        # Map severity to log level
        level_map = {
            "INFO": LogLevel.INFO,
            "WARNING": LogLevel.WARNING,
            "ERROR": LogLevel.ERROR,
            "CRITICAL": LogLevel.CRITICAL,
            "SUCCESS": LogLevel.SUCCESS,
        }
        level = level_map.get(severity, LogLevel.INFO)

        self.log(
            level,
            f"Alert ({alert_type}): {message}",
            iteration=iteration,
            context={"alert_type": alert_type, "severity": severity},
        )

    def log_environment_change(
        self,
        iteration: int,
        change_type: str,
        details: str,
    ) -> None:
        """Log an environment change.

        Args:
            iteration: The iteration number.
            change_type: The type of change.
            details: Details about the change.
        """
        self.log(
            LogLevel.INFO,
            f"Environment change ({change_type}): {details}",
            iteration=iteration,
            context={"change_type": change_type, "details": details},
        )

    def get_recent_entries(self, count: int = 50) -> list[LogEntry]:
        """Get recent log entries.

        Args:
            count: Number of entries to return.

        Returns:
            List of recent log entries.
        """
        return self.entries[-count:]

    def get_entries_by_level(self, level: LogLevel) -> list[LogEntry]:
        """Get entries filtered by level.

        Args:
            level: The log level to filter by.

        Returns:
            List of matching log entries.
        """
        return [e for e in self.entries if e.level == level]

    def clear(self) -> None:
        """Clear in-memory log entries."""
        self.entries.clear()
