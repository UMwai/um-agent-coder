"""
Result types for meta-harness execution.

Defines HarnessResult and HarnessMetrics for capturing sub-harness outcomes.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class HarnessStatus(Enum):
    """Status of a sub-harness."""

    PENDING = "pending"  # Spawned but not started
    RUNNING = "running"  # Actively executing
    PAUSED = "paused"  # Paused (waiting for resume)
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Failed with error
    STOPPED = "stopped"  # Stopped by request


@dataclass
class HarnessMetrics:
    """Metrics for comparing harness results."""

    tests_passed: int = 0
    tests_failed: int = 0
    test_coverage: float = 0.0
    performance_score: float = 0.0  # Benchmark results
    code_quality_score: float = 0.0  # Linting/complexity
    lines_of_code: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0


@dataclass
class Artifact:
    """Shared file artifact between harnesses."""

    name: str
    source_harness: str
    path: Path
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessResult:
    """Result from a completed sub-harness."""

    harness_id: str
    status: HarnessStatus

    # Execution stats
    total_iterations: int = 0
    total_duration: timedelta = field(default_factory=lambda: timedelta())
    tasks_completed: int = 0
    tasks_failed: int = 0

    # Output
    final_output: str = ""
    artifacts: List[Artifact] = field(default_factory=list)

    # Metrics (for voting/selection)
    metrics: HarnessMetrics = field(default_factory=HarnessMetrics)

    # Error info (if failed)
    error: Optional[str] = None
    traceback: Optional[str] = None

    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def success(self) -> bool:
        """Return True if harness completed successfully."""
        return self.status == HarnessStatus.COMPLETED

    @property
    def progress(self) -> float:
        """Calculate progress score for comparison."""
        if self.tasks_completed + self.tasks_failed == 0:
            return 0.0
        return self.tasks_completed / (self.tasks_completed + self.tasks_failed)


@dataclass
class AggregatedResult:
    """Result from coordinating multiple harnesses."""

    strategy: str  # parallel, pipeline, race, voting
    success: bool
    results: List[HarnessResult] = field(default_factory=list)
    winner: Optional[HarnessResult] = None  # For RACE/VOTING
    aggregated_output: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def total_duration(self) -> timedelta:
        """Calculate total duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return timedelta()

    @property
    def all_succeeded(self) -> bool:
        """Return True if all harnesses succeeded."""
        return all(r.success for r in self.results)

    @property
    def any_succeeded(self) -> bool:
        """Return True if any harness succeeded."""
        return any(r.success for r in self.results)
