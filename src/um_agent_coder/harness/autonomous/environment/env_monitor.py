"""Environment variable monitor.

Monitors HARNESS_* environment variables for configuration
changes during loop execution.

Reference: specs/autonomous-loop-spec.md Section 4.4
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

# Environment variables to monitor with valid values
MONITORED_ENV_VARS = {
    "HARNESS_MODE": {"normal", "turbo", "conservative"},
    "HARNESS_PAUSE": {"true", "false", "1", "0", "yes", "no"},
    "HARNESS_CLI": {"codex", "gemini", "claude", "auto"},
    "HARNESS_PRIORITY": {"speed", "quality", "cost"},
    "HARNESS_LOG_LEVEL": {"debug", "info", "warning", "error"},
    "HARNESS_MAX_ITERATIONS": None,  # Any positive integer
    "HARNESS_TIMEOUT": None,  # Duration string
    "HARNESS_STOP": {"true", "false", "1", "0"},  # Signal to stop
}


@dataclass
class EnvChange:
    """Represents an environment variable change."""

    var: str
    old_value: Optional[str]
    new_value: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)
    is_valid: bool = True

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "var": self.var,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp.isoformat(),
            "is_valid": self.is_valid,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EnvChange":
        """Deserialize from dictionary."""
        return cls(
            var=data["var"],
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            is_valid=data.get("is_valid", True),
        )

    @property
    def is_set(self) -> bool:
        """Check if variable was set (new value exists)."""
        return self.new_value is not None

    @property
    def is_unset(self) -> bool:
        """Check if variable was unset (new value is None)."""
        return self.old_value is not None and self.new_value is None


class EnvMonitor:
    """Monitor environment variables for changes.

    Tracks HARNESS_* environment variables and detects changes
    between checks.

    Args:
        additional_vars: Additional variable names to monitor.
        validate: Whether to validate against known valid values.
    """

    def __init__(
        self,
        additional_vars: Optional[set[str]] = None,
        validate: bool = True,
    ):
        """Initialize environment monitor."""
        self.monitored_vars = set(MONITORED_ENV_VARS.keys())
        if additional_vars:
            self.monitored_vars.update(additional_vars)

        self.validate = validate
        self._snapshot: dict[str, Optional[str]] = {}
        self._take_snapshot()

    def _take_snapshot(self) -> None:
        """Take snapshot of current environment."""
        self._snapshot = {}
        for var in self.monitored_vars:
            self._snapshot[var] = os.environ.get(var)

    def check_changes(self) -> list[EnvChange]:
        """Check for environment variable changes.

        Returns:
            List of changes since last check.
        """
        changes = []

        for var in self.monitored_vars:
            current = os.environ.get(var)
            previous = self._snapshot.get(var)

            if current != previous:
                is_valid = self._validate_value(var, current) if self.validate else True
                changes.append(
                    EnvChange(
                        var=var,
                        old_value=previous,
                        new_value=current,
                        is_valid=is_valid,
                    )
                )
                self._snapshot[var] = current

        return changes

    def get_current(self, var: str) -> Optional[str]:
        """Get current value of a monitored variable.

        Args:
            var: Variable name.

        Returns:
            Current value or None if not set.
        """
        return os.environ.get(var)

    def get_all_current(self) -> dict[str, Optional[str]]:
        """Get current values of all monitored variables.

        Returns:
            Dictionary of variable names to values.
        """
        return {var: os.environ.get(var) for var in self.monitored_vars}

    def get_snapshot(self) -> dict[str, Optional[str]]:
        """Get the last snapshot.

        Returns:
            Dictionary of variable names to values from last check.
        """
        return dict(self._snapshot)

    def is_paused(self) -> bool:
        """Check if HARNESS_PAUSE is set to true."""
        value = os.environ.get("HARNESS_PAUSE", "").lower()
        return value in ("true", "1", "yes")

    def should_stop(self) -> bool:
        """Check if HARNESS_STOP is set to true."""
        value = os.environ.get("HARNESS_STOP", "").lower()
        return value in ("true", "1", "yes")

    def get_mode(self) -> str:
        """Get current HARNESS_MODE.

        Returns:
            Mode string or 'normal' if not set.
        """
        return os.environ.get("HARNESS_MODE", "normal").lower()

    def get_priority(self) -> str:
        """Get current HARNESS_PRIORITY.

        Returns:
            Priority string or 'quality' if not set.
        """
        return os.environ.get("HARNESS_PRIORITY", "quality").lower()

    def get_cli_override(self) -> Optional[str]:
        """Get CLI override if set.

        Returns:
            CLI name or None if not overridden.
        """
        cli = os.environ.get("HARNESS_CLI")
        if cli and cli.lower() in ("codex", "gemini", "claude"):
            return cli.lower()
        return None

    def get_max_iterations(self) -> Optional[int]:
        """Get max iterations override if set.

        Returns:
            Max iterations or None if not set.
        """
        value = os.environ.get("HARNESS_MAX_ITERATIONS")
        if value:
            try:
                return int(value)
            except ValueError:
                pass
        return None

    def _validate_value(self, var: str, value: Optional[str]) -> bool:
        """Validate a value against known valid values.

        Args:
            var: Variable name.
            value: Value to validate.

        Returns:
            True if valid, False otherwise.
        """
        if value is None:
            return True  # Unset is always valid

        valid_values = MONITORED_ENV_VARS.get(var)
        if valid_values is None:
            return True  # No validation for this var

        return value.lower() in valid_values

    def add_monitored_var(self, var: str) -> None:
        """Add a variable to monitor.

        Args:
            var: Variable name to add.
        """
        self.monitored_vars.add(var)
        self._snapshot[var] = os.environ.get(var)

    def remove_monitored_var(self, var: str) -> None:
        """Remove a variable from monitoring.

        Args:
            var: Variable name to remove.
        """
        self.monitored_vars.discard(var)
        self._snapshot.pop(var, None)


def get_harness_config_from_env() -> dict[str, Any]:
    """Get harness configuration from environment variables.

    Returns:
        Dictionary of configuration values.
    """
    config = {}

    # Mode
    mode = os.environ.get("HARNESS_MODE")
    if mode:
        config["mode"] = mode.lower()

    # CLI
    cli = os.environ.get("HARNESS_CLI")
    if cli:
        config["cli"] = cli.lower()

    # Priority
    priority = os.environ.get("HARNESS_PRIORITY")
    if priority:
        config["priority"] = priority.lower()

    # Max iterations
    max_iter = os.environ.get("HARNESS_MAX_ITERATIONS")
    if max_iter:
        try:
            config["max_iterations"] = int(max_iter)
        except ValueError:
            pass

    # Timeout
    timeout = os.environ.get("HARNESS_TIMEOUT")
    if timeout:
        config["timeout"] = timeout

    # Log level
    log_level = os.environ.get("HARNESS_LOG_LEVEL")
    if log_level:
        config["log_level"] = log_level.lower()

    return config
