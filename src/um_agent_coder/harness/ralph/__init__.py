"""
Ralph-Wiggum style autonomous loop capability for the harness.

This module enables iterative task execution where tasks cycle until
completion criteria are met.
"""

from .executor import RalphExecutor, RalphResult
from .iteration_tracker import IterationRecord, IterationTracker
from .persistence import RalphPersistence
from .promise_detector import DetectionResult, PromiseDetector

__all__ = [
    "DetectionResult",
    "IterationRecord",
    "IterationTracker",
    "PromiseDetector",
    "RalphExecutor",
    "RalphPersistence",
    "RalphResult",
]
