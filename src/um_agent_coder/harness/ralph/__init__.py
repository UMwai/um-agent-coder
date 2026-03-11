"""
Ralph-Wiggum style autonomous loop capability for the harness.

This module enables iterative task execution where tasks cycle until
completion criteria are met.
"""

from .executor import RalphExecutor, RalphResult
from .goal_validator import GoalValidationResult, GoalValidator
from .iteration_tracker import IterationRecord, IterationTracker
from .persistence import RalphPersistence
from .promise_detector import DetectionResult, PromiseDetector

__all__ = [
    "DetectionResult",
    "GoalValidationResult",
    "GoalValidator",
    "IterationRecord",
    "IterationTracker",
    "PromiseDetector",
    "RalphExecutor",
    "RalphPersistence",
    "RalphResult",
]
