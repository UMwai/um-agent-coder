"""Alert manager for autonomous loop notifications.

Handles CLI notifications and file-based alerts for monitoring
loop execution status.

Reference: specs/autonomous-loop-spec.md Section 5.1-5.2
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


class AlertType(Enum):
    """Predefined alert types."""

    ITERATION_MILESTONE = "iteration_milestone"
    NO_PROGRESS = "no_progress"
    STUCK_RECOVERY = "stuck_recovery"
    APPROACHING_LIMIT = "approaching_limit"
    MODEL_ESCALATION = "model_escalation"
    RUNAWAY_DETECTED = "runaway_detected"
    GOAL_COMPLETE = "goal_complete"
    FATAL_ERROR = "fatal_error"
    CUSTOM = "custom"


@dataclass
class Alert:
    """Represents an alert event."""

    alert_type: str
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    iteration: int = 0
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "type": self.alert_type,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "iteration": self.iteration,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Alert":
        """Deserialize from dictionary."""
        return cls(
            alert_type=data["type"],
            severity=AlertSeverity(data["severity"]),
            message=data["message"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            iteration=data.get("iteration", 0),
            context=data.get("context", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())


class PauseRequestedError(Exception):
    """Exception raised when a pause is requested due to critical alert."""

    def __init__(self, alert: Alert):
        self.alert = alert
        super().__init__(f"Pause requested: {alert.message}")


# Backwards compatibility alias
PauseRequested = PauseRequestedError


@dataclass
class AlertConfig:
    """Configuration for alert manager."""

    # File output
    alert_log_path: Path = field(default_factory=lambda: Path(".harness/alerts.log"))
    write_to_file: bool = True

    # CLI output
    cli_notify: bool = True
    use_colors: bool = True

    # Behavior
    pause_on_critical: bool = True
    max_alerts_in_memory: int = 1000

    # Milestone settings
    milestone_interval: int = 10  # Alert every N iterations

    # Handlers
    custom_handlers: list[Callable[["Alert"], None]] = field(default_factory=list)


# ANSI color codes
SEVERITY_COLORS = {
    AlertSeverity.INFO: "\033[94m",  # Blue
    AlertSeverity.WARNING: "\033[93m",  # Yellow
    AlertSeverity.CRITICAL: "\033[91m",  # Red
    AlertSeverity.SUCCESS: "\033[92m",  # Green
    AlertSeverity.ERROR: "\033[91m",  # Red
}
RESET_COLOR = "\033[0m"


class AlertManager:
    """Manage alerts for autonomous loop.

    Handles CLI notifications, file logging, and custom handlers
    for monitoring loop execution.

    Args:
        config: Alert configuration.
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        """Initialize alert manager."""
        self.config = config or AlertConfig()
        self.alerts: list[Alert] = []
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Ensure alert log directory exists."""
        if self.config.write_to_file:
            self.config.alert_log_path.parent.mkdir(parents=True, exist_ok=True)

    def alert(
        self,
        alert_type: str,
        message: str,
        severity: AlertSeverity,
        iteration: int = 0,
        **context: Any,
    ) -> Alert:
        """Create and dispatch an alert.

        Args:
            alert_type: Type of alert (use AlertType enum values).
            message: Human-readable alert message.
            severity: Alert severity level.
            iteration: Current iteration number.
            **context: Additional context data.

        Returns:
            The created Alert object.

        Raises:
            PauseRequestedError: If severity is CRITICAL and pause_on_critical enabled.
        """
        alert_obj = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            iteration=iteration,
            context=context,
        )

        # Write to file
        if self.config.write_to_file:
            self._write_to_file(alert_obj)

        # CLI notification
        if self.config.cli_notify:
            self._cli_notify(alert_obj)

        # Custom handlers
        for handler in self.config.custom_handlers:
            try:
                handler(alert_obj)
            except Exception:
                pass  # Don't let handler errors break alerting

        # Track in memory
        self.alerts.append(alert_obj)
        self._trim_alerts()

        # Check for pause condition
        if severity == AlertSeverity.CRITICAL and self.config.pause_on_critical:
            raise PauseRequestedError(alert_obj)

        return alert_obj

    def _write_to_file(self, alert: Alert) -> None:
        """Append alert to log file."""
        try:
            with open(self.config.alert_log_path, "a") as f:
                f.write(alert.to_json() + "\n")
        except OSError:
            pass  # Silently fail if can't write

    def _cli_notify(self, alert: Alert) -> None:
        """Print alert to terminal."""
        if self.config.use_colors:
            color = SEVERITY_COLORS.get(alert.severity, "")
            print(
                f"{color}[{alert.severity.value}] {alert.alert_type}: {alert.message}{RESET_COLOR}",
                file=sys.stderr,
            )
        else:
            print(
                f"[{alert.severity.value}] {alert.alert_type}: {alert.message}",
                file=sys.stderr,
            )

    def _trim_alerts(self) -> None:
        """Trim in-memory alerts to max size."""
        if len(self.alerts) > self.config.max_alerts_in_memory:
            # Keep most recent
            self.alerts = self.alerts[-self.config.max_alerts_in_memory :]

    # Convenience methods for common alert types

    def info(self, message: str, alert_type: str = AlertType.CUSTOM.value, **context) -> Alert:
        """Send an INFO level alert."""
        return self.alert(alert_type, message, AlertSeverity.INFO, **context)

    def warning(self, message: str, alert_type: str = AlertType.CUSTOM.value, **context) -> Alert:
        """Send a WARNING level alert."""
        return self.alert(alert_type, message, AlertSeverity.WARNING, **context)

    def critical(self, message: str, alert_type: str = AlertType.CUSTOM.value, **context) -> Alert:
        """Send a CRITICAL level alert."""
        return self.alert(alert_type, message, AlertSeverity.CRITICAL, **context)

    def success(self, message: str, alert_type: str = AlertType.CUSTOM.value, **context) -> Alert:
        """Send a SUCCESS level alert."""
        return self.alert(alert_type, message, AlertSeverity.SUCCESS, **context)

    def error(self, message: str, alert_type: str = AlertType.CUSTOM.value, **context) -> Alert:
        """Send an ERROR level alert."""
        return self.alert(alert_type, message, AlertSeverity.ERROR, **context)

    # Predefined alert methods

    def iteration_milestone(self, iteration: int, **context) -> Optional[Alert]:
        """Alert on iteration milestone.

        Args:
            iteration: Current iteration number.

        Returns:
            Alert if milestone reached, None otherwise.
        """
        if iteration > 0 and iteration % self.config.milestone_interval == 0:
            return self.alert(
                AlertType.ITERATION_MILESTONE.value,
                f"Reached iteration {iteration}",
                AlertSeverity.INFO,
                iteration=iteration,
                **context,
            )
        return None

    def no_progress(self, iterations_without_progress: int, iteration: int, **context) -> Alert:
        """Alert on lack of progress."""
        return self.alert(
            AlertType.NO_PROGRESS.value,
            f"No progress for {iterations_without_progress} consecutive iterations",
            AlertSeverity.WARNING,
            iteration=iteration,
            iterations_without_progress=iterations_without_progress,
            **context,
        )

    def stuck_recovery(self, strategy: str, iteration: int, **context) -> Alert:
        """Alert on stuck recovery triggered."""
        return self.alert(
            AlertType.STUCK_RECOVERY.value,
            f"Stuck recovery triggered: {strategy}",
            AlertSeverity.WARNING,
            iteration=iteration,
            strategy=strategy,
            **context,
        )

    def approaching_limit(
        self,
        limit_type: str,
        current: int,
        limit: int,
        iteration: int,
        **context,
    ) -> Alert:
        """Alert when approaching a limit."""
        percentage = (current / limit) * 100 if limit > 0 else 0
        return self.alert(
            AlertType.APPROACHING_LIMIT.value,
            f"Approaching {limit_type} limit: {current}/{limit} ({percentage:.0f}%)",
            AlertSeverity.WARNING,
            iteration=iteration,
            limit_type=limit_type,
            current=current,
            limit=limit,
            percentage=percentage,
            **context,
        )

    def model_escalation(
        self,
        from_model: str,
        to_model: str,
        iteration: int,
        **context,
    ) -> Alert:
        """Alert on model escalation."""
        return self.alert(
            AlertType.MODEL_ESCALATION.value,
            f"Model escalated: {from_model} -> {to_model}",
            AlertSeverity.INFO,
            iteration=iteration,
            from_model=from_model,
            to_model=to_model,
            **context,
        )

    def goal_complete(self, iteration: int, promise: str, **context) -> Alert:
        """Alert on goal completion."""
        return self.alert(
            AlertType.GOAL_COMPLETE.value,
            f"Goal complete! Promise detected: {promise}",
            AlertSeverity.SUCCESS,
            iteration=iteration,
            promise=promise,
            **context,
        )

    def fatal_error(self, error: str, iteration: int = 0, **context) -> Alert:
        """Alert on fatal error."""
        return self.alert(
            AlertType.FATAL_ERROR.value,
            f"Fatal error: {error}",
            AlertSeverity.ERROR,
            iteration=iteration,
            error=error,
            **context,
        )

    def runaway_detected(self, reason: str, iteration: int, **context) -> Alert:
        """Alert on runaway detection."""
        return self.alert(
            AlertType.RUNAWAY_DETECTED.value,
            f"Runaway detected: {reason}",
            AlertSeverity.CRITICAL,
            iteration=iteration,
            reason=reason,
            **context,
        )

    # Query methods

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        alert_type: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> list[Alert]:
        """Get alerts matching filters.

        Args:
            severity: Filter by severity.
            alert_type: Filter by alert type.
            since: Filter by timestamp.

        Returns:
            List of matching alerts.
        """
        result = self.alerts

        if severity is not None:
            result = [a for a in result if a.severity == severity]

        if alert_type is not None:
            result = [a for a in result if a.alert_type == alert_type]

        if since is not None:
            result = [a for a in result if a.timestamp >= since]

        return result

    def get_recent_alerts(self, count: int = 10) -> list[Alert]:
        """Get most recent alerts."""
        return self.alerts[-count:]

    def count_by_severity(self) -> dict[AlertSeverity, int]:
        """Count alerts by severity."""
        counts = dict.fromkeys(AlertSeverity, 0)
        for alert in self.alerts:
            counts[alert.severity] += 1
        return counts

    def clear_alerts(self) -> None:
        """Clear in-memory alerts."""
        self.alerts.clear()

    def get_summary(self) -> dict[str, Any]:
        """Get alert summary statistics."""
        counts = self.count_by_severity()
        return {
            "total": len(self.alerts),
            "by_severity": {s.value: c for s, c in counts.items()},
            "recent": [a.to_dict() for a in self.get_recent_alerts(5)],
        }
