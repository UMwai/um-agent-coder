"""Alert system for autonomous loop.

Provides CLI notifications and file-based alerts for monitoring
autonomous loop execution.

Reference: specs/autonomous-loop-spec.md Section 5
"""

from .alert_manager import (
    Alert,
    AlertConfig,
    AlertManager,
    AlertSeverity,
    AlertType,
    PauseRequested,
    PauseRequestedError,
)
from .runaway_detector import (
    RunawayConfig,
    RunawayDetector,
)

__all__ = [
    # Alert manager
    "Alert",
    "AlertConfig",
    "AlertManager",
    "AlertSeverity",
    "AlertType",
    "PauseRequested",  # Backwards compat alias
    "PauseRequestedError",
    # Runaway detection
    "RunawayConfig",
    "RunawayDetector",
]
