"""Runaway detection for autonomous loop.

Detects potential infinite loops or runaway execution patterns
that indicate the agent is not making meaningful progress.

Reference: specs/autonomous-loop-spec.md Section 5.3
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

from .alert_manager import Alert, AlertSeverity, AlertType


@dataclass
class RunawayConfig:
    """Configuration for runaway detection."""

    # Max iterations warning
    max_iterations_warning: int = 500
    has_time_limit: bool = False

    # Output loop detection
    output_loop_window: int = 5  # Check last N outputs for repetition
    output_hash_history: int = 100  # How many hashes to keep

    # Speed detection
    min_iteration_seconds: float = 0.5  # Iterations faster than this are suspicious
    speedup_window: int = 10  # Window for speed pattern detection
    speedup_threshold: float = 0.3  # If avg time drops to 30% of initial, warn

    # Combined patterns
    enable_output_loop_detection: bool = True
    enable_speedup_detection: bool = True
    enable_iteration_warning: bool = True


@dataclass
class RunawayState:
    """State tracked by runaway detector."""

    iteration_times: list[float] = field(default_factory=list)
    output_hashes: list[str] = field(default_factory=list)
    last_check_iteration: int = 0
    warnings_issued: int = 0
    critical_issued: bool = False

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "iteration_times": self.iteration_times[-100:],  # Keep last 100
            "output_hashes": self.output_hashes[-100:],
            "last_check_iteration": self.last_check_iteration,
            "warnings_issued": self.warnings_issued,
            "critical_issued": self.critical_issued,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunawayState":
        """Deserialize from dictionary."""
        return cls(
            iteration_times=data.get("iteration_times", []),
            output_hashes=data.get("output_hashes", []),
            last_check_iteration=data.get("last_check_iteration", 0),
            warnings_issued=data.get("warnings_issued", 0),
            critical_issued=data.get("critical_issued", False),
        )


def hash_output(output: str) -> str:
    """Hash output for comparison.

    Uses first 1000 chars to avoid hashing huge outputs.
    """
    # Normalize: lowercase, strip whitespace
    normalized = output.lower().strip()[:1000]
    return hashlib.md5(normalized.encode()).hexdigest()


class RunawayDetector:
    """Detect potential infinite loops or runaway execution.

    Monitors iteration patterns for signs of:
    - Repeated identical outputs (infinite loop)
    - Suspiciously fast iterations (no real work)
    - Too many iterations without limits

    Args:
        config: Runaway detection configuration.
    """

    def __init__(self, config: Optional[RunawayConfig] = None):
        """Initialize runaway detector."""
        self.config = config or RunawayConfig()
        self.state = RunawayState()

    def check(
        self,
        iteration: int,
        duration: float,
        output: str,
    ) -> Optional[Alert]:
        """Check for runaway conditions.

        Args:
            iteration: Current iteration number.
            duration: Time taken for this iteration in seconds.
            output: Output from the iteration.

        Returns:
            Alert if runaway condition detected, None otherwise.
        """
        # Record data
        self.state.iteration_times.append(duration)
        self.state.output_hashes.append(hash_output(output))
        self.state.last_check_iteration = iteration

        # Trim to configured limits
        self._trim_history()

        # Run checks
        checks = []

        if self.config.enable_iteration_warning:
            checks.append(self._check_iteration_limit(iteration))

        if self.config.enable_output_loop_detection:
            checks.append(self._check_output_loop(iteration))

        if self.config.enable_speedup_detection:
            checks.append(self._check_speedup_pattern(iteration))

        # Return first non-None alert (prioritize critical)
        criticals = [c for c in checks if c and c.severity == AlertSeverity.CRITICAL]
        if criticals:
            self.state.critical_issued = True
            return criticals[0]

        warnings = [c for c in checks if c and c.severity == AlertSeverity.WARNING]
        if warnings:
            self.state.warnings_issued += 1
            return warnings[0]

        return None

    def _trim_history(self) -> None:
        """Trim history to configured limits."""
        max_history = self.config.output_hash_history
        if len(self.state.iteration_times) > max_history:
            self.state.iteration_times = self.state.iteration_times[-max_history:]
        if len(self.state.output_hashes) > max_history:
            self.state.output_hashes = self.state.output_hashes[-max_history:]

    def _check_iteration_limit(self, iteration: int) -> Optional[Alert]:
        """Check if too many iterations without time limit."""
        if iteration > self.config.max_iterations_warning and not self.config.has_time_limit:
            return Alert(
                alert_type=AlertType.RUNAWAY_DETECTED.value,
                severity=AlertSeverity.WARNING,
                message=f"Reached {iteration} iterations without time limit set",
                iteration=iteration,
                context={"check": "iteration_limit"},
            )
        return None

    def _check_output_loop(self, iteration: int) -> Optional[Alert]:
        """Check for repeated identical outputs."""
        window = self.config.output_loop_window
        if len(self.state.output_hashes) < window:
            return None

        recent = self.state.output_hashes[-window:]
        if len(set(recent)) == 1:
            return Alert(
                alert_type=AlertType.RUNAWAY_DETECTED.value,
                severity=AlertSeverity.CRITICAL,
                message=f"Detected {window} identical consecutive outputs - possible infinite loop",
                iteration=iteration,
                context={
                    "check": "output_loop",
                    "window": window,
                    "hash": recent[0],
                },
            )
        return None

    def _check_speedup_pattern(self, iteration: int) -> Optional[Alert]:
        """Check for suspicious speedup in iteration times."""
        window = self.config.speedup_window
        if len(self.state.iteration_times) < window * 2:
            return None

        # Compare early iterations to recent
        early = self.state.iteration_times[:window]
        recent = self.state.iteration_times[-window:]

        early_avg = sum(early) / len(early)
        recent_avg = sum(recent) / len(recent)

        # Check if recent is suspiciously faster
        if early_avg > 0 and recent_avg / early_avg < self.config.speedup_threshold:
            return Alert(
                alert_type=AlertType.RUNAWAY_DETECTED.value,
                severity=AlertSeverity.WARNING,
                message=f"Iterations completing {early_avg/recent_avg:.1f}x faster than initial - may not be doing work",
                iteration=iteration,
                context={
                    "check": "speedup_pattern",
                    "early_avg": early_avg,
                    "recent_avg": recent_avg,
                    "ratio": recent_avg / early_avg if early_avg > 0 else 0,
                },
            )

        # Also check for iterations that are just too fast
        if recent_avg < self.config.min_iteration_seconds:
            return Alert(
                alert_type=AlertType.RUNAWAY_DETECTED.value,
                severity=AlertSeverity.WARNING,
                message=f"Average iteration time ({recent_avg:.2f}s) below minimum threshold",
                iteration=iteration,
                context={
                    "check": "min_time",
                    "avg_time": recent_avg,
                    "threshold": self.config.min_iteration_seconds,
                },
            )

        return None

    def has_detected_runaway(self) -> bool:
        """Check if runaway has been detected."""
        return self.state.critical_issued

    def get_warning_count(self) -> int:
        """Get number of warnings issued."""
        return self.state.warnings_issued

    def get_state(self) -> RunawayState:
        """Get current detector state."""
        return self.state

    def reset(self) -> None:
        """Reset detector state."""
        self.state = RunawayState()

    def get_statistics(self) -> dict[str, Any]:
        """Get detector statistics."""
        times = self.state.iteration_times
        return {
            "total_iterations_tracked": len(times),
            "avg_iteration_time": sum(times) / len(times) if times else 0,
            "min_iteration_time": min(times) if times else 0,
            "max_iteration_time": max(times) if times else 0,
            "warnings_issued": self.state.warnings_issued,
            "critical_issued": self.state.critical_issued,
            "unique_outputs_recent": len(set(self.state.output_hashes[-20:])),
        }
