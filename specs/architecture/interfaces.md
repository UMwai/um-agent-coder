# Architecture Interfaces

> **Status**: Current
> **Last Updated**: January 2026

This document defines the key interfaces (contracts) that components must implement. These interfaces enable extensibility and ensure consistent behavior across the system.

---

## CLI Executor Interface

All CLI backends implement this interface.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class ExecutionResult:
    """Result from executing a task via CLI."""
    success: bool
    output: str
    error: Optional[str] = None
    duration_seconds: float = 0.0
    conversation_id: Optional[str] = None  # For multi-turn


@dataclass
class Task:
    """Task to execute."""
    id: str
    description: str
    goal: str
    success_criteria: Optional[str] = None
    timeout_minutes: int = 30
    cwd: Optional[Path] = None
    cli: str = ""  # Override CLI for this task
    model: str = ""  # Override model for this task


class BaseCLIExecutor(ABC):
    """
    Abstract base class for CLI executors.

    Implementations: CodexExecutor, GeminiExecutor, ClaudeExecutor
    """

    @property
    @abstractmethod
    def cli_name(self) -> str:
        """Name of this CLI (codex, gemini, claude)."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model for this CLI."""
        pass

    @abstractmethod
    def execute(
        self,
        task: Task,
        context: Optional[str] = None,
        model: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a task using this CLI.

        Args:
            task: The task to execute
            context: Optional context to include in prompt
            model: Optional model override

        Returns:
            ExecutionResult with success/failure and output
        """
        pass

    @abstractmethod
    def verify_success(
        self,
        task: Task,
        result: ExecutionResult
    ) -> bool:
        """
        Verify if task was completed successfully.

        Uses success_criteria if defined.
        """
        pass

    def _build_prompt(self, task: Task, context: Optional[str]) -> str:
        """Build prompt from task and context."""
        sections = [f"## Goal\n{task.goal}"]
        if context:
            sections.append(f"## Context\n{context}")
        if task.success_criteria:
            sections.append(f"## Success Criteria\n{task.success_criteria}")
        return "\n\n".join(sections)

    def _detect_failure(self, output: str, error: str) -> bool:
        """Detect obvious failures in output."""
        failure_indicators = [
            "error:",
            "failed:",
            "exception:",
            "traceback",
            "fatal:",
        ]
        combined = (output + error).lower()
        return any(indicator in combined for indicator in failure_indicators)
```

### Implementation Requirements

1. **execute()** must:
   - Build CLI command with appropriate flags
   - Run command via subprocess
   - Capture stdout/stderr
   - Handle timeouts gracefully
   - Return structured result

2. **verify_success()** must:
   - Check success_criteria if defined
   - May re-run CLI to verify
   - Return boolean

### Example Implementation

```python
class CodexExecutor(BaseCLIExecutor):

    @property
    def cli_name(self) -> str:
        return "codex"

    @property
    def default_model(self) -> str:
        return "gpt-5.2"

    def execute(self, task: Task, context: Optional[str] = None, model: Optional[str] = None) -> ExecutionResult:
        prompt = self._build_prompt(task, context)
        model = model or self.default_model

        cmd = [
            "codex",
            "--model", model,
            "--ask-for-approval", "never",
            "--sandbox", "danger-full-access",
            "exec", prompt
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=task.cwd,
            timeout=task.timeout_minutes * 60
        )

        return ExecutionResult(
            success=result.returncode == 0 and not self._detect_failure(result.stdout, result.stderr),
            output=result.stdout,
            error=result.stderr if result.returncode != 0 else None,
            duration_seconds=...
        )
```

---

## Coordination Strategy Interface

All coordination strategies implement this interface.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class CoordinationStrategy(Enum):
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    RACE = "race"
    VOTING = "voting"


@dataclass
class AggregatedResult:
    """Result from coordinating multiple harnesses."""
    strategy: CoordinationStrategy
    success: bool
    results: List["HarnessResult"]
    winner: Optional["HarnessResult"] = None  # For RACE/VOTING
    aggregated_output: Optional[str] = None


class BaseStrategy(ABC):
    """
    Abstract base class for coordination strategies.

    Implementations: ParallelStrategy, PipelineStrategy, RaceStrategy, VotingStrategy
    """

    @property
    @abstractmethod
    def strategy_type(self) -> CoordinationStrategy:
        """Type of this strategy."""
        pass

    @abstractmethod
    def execute(
        self,
        manager: "HarnessManager",
        handles: List["HarnessHandle"],
        config: Optional[dict] = None
    ) -> AggregatedResult:
        """
        Execute coordination strategy on handles.

        Args:
            manager: HarnessManager for control operations
            handles: List of HarnessHandles to coordinate
            config: Strategy-specific configuration

        Returns:
            AggregatedResult with all results and winner (if applicable)
        """
        pass

    @abstractmethod
    def on_harness_complete(
        self,
        handle: "HarnessHandle",
        result: "HarnessResult"
    ):
        """Called when a harness completes."""
        pass

    @abstractmethod
    def should_terminate_others(
        self,
        completed: "HarnessHandle",
        remaining: List["HarnessHandle"]
    ) -> bool:
        """Return True if remaining harnesses should be terminated."""
        pass
```

### Strategy Contracts

| Strategy | execute() Behavior | should_terminate_others() |
|----------|-------------------|---------------------------|
| PARALLEL | Wait for all | Never (False) |
| PIPELINE | Execute sequentially | On failure (True) |
| RACE | Wait for first | Always (True) |
| VOTING | Wait for min_votes | When enough votes |

---

## Progress Detection Interface

```python
from dataclasses import dataclass
from typing import List


@dataclass
class ProgressSignal:
    """Signals used to calculate progress."""
    output_diff_score: float      # 0.0 = identical, 1.0 = completely different
    file_changes_score: float     # 0.0 = no changes, 1.0 = significant
    explicit_markers: List[str]   # <progress>...</progress> content
    checklist_progress: float     # 0.0 = none, 1.0 = all complete


class ProgressDetector:
    """
    Detects progress between iterations.

    Uses weighted combination of signals.
    """

    WEIGHTS = {
        'output_diff': 0.30,
        'file_changes': 0.30,
        'explicit_markers': 0.25,
        'checklist': 0.15
    }

    def detect(
        self,
        prev_output: str,
        curr_output: str,
        workspace: Path
    ) -> ProgressSignal:
        """Detect progress signals."""
        pass

    def calculate_score(self, signal: ProgressSignal) -> float:
        """Calculate weighted progress score (0.0-1.0)."""
        marker_score = min(1.0, len(signal.explicit_markers) * 0.5)

        return (
            self.WEIGHTS['output_diff'] * signal.output_diff_score +
            self.WEIGHTS['file_changes'] * signal.file_changes_score +
            self.WEIGHTS['explicit_markers'] * marker_score +
            self.WEIGHTS['checklist'] * signal.checklist_progress
        )
```

---

## Recovery Strategy Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    new_prompt: Optional[str] = None
    escalated_to: Optional[tuple] = None  # (cli, model)
    branch: Optional["ExplorationBranch"] = None
    needs_human: bool = False


class RecoveryStrategy(ABC):
    """
    Abstract base class for stuck recovery strategies.

    Implementations: PromptMutator, ModelEscalator, BranchExplorer
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this recovery strategy."""
        pass

    @abstractmethod
    def can_apply(self, context: "LoopContext") -> bool:
        """Return True if this strategy can be applied."""
        pass

    @abstractmethod
    def apply(
        self,
        task: Task,
        context: "LoopContext",
        executor: BaseCLIExecutor
    ) -> RecoveryResult:
        """
        Apply recovery strategy.

        Returns:
            RecoveryResult indicating success and what changed
        """
        pass
```

---

## HarnessHandle Interface

```python
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List


class HarnessStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class HarnessHandle:
    """
    Handle to control and monitor a sub-harness.

    Properties are updated by polling the subprocess.
    """

    # Identification
    harness_id: str
    pid: int
    working_dir: Path
    state_db: Path

    # Observable state
    status: HarnessStatus
    progress: float  # 0.0 - 1.0
    current_task: Optional[str]
    current_iteration: int

    # Results
    result: Optional["HarnessResult"] = None
    error: Optional[str] = None

    def send_instruction(self, instruction: str):
        """
        Send instruction to sub-harness.
        Writes to .harness/{id}/inbox/
        """
        pass

    def request_pause(self):
        """Request pause. Creates .harness/{id}/pause file."""
        pass

    def request_stop(self):
        """Request graceful stop. Creates .harness/{id}/stop file."""
        pass

    def force_kill(self, timeout: int = 30):
        """Force kill subprocess after timeout."""
        pass

    def get_logs(self, tail: int = 100) -> List[str]:
        """Get recent log lines from .harness/{id}/harness.log."""
        pass

    def get_alerts(self) -> List["Alert"]:
        """Get alerts from .harness/{id}/alerts.log."""
        pass

    def refresh(self):
        """Refresh status from subprocess/state."""
        pass
```

---

## State Manager Interface

```python
from abc import ABC, abstractmethod
from typing import Optional, List


class BaseStateManager(ABC):
    """
    Abstract base class for state management.

    Implementations: StateManager (SQLite), MetaStateManager
    """

    @abstractmethod
    def init_harness(self, roadmap_path: Path):
        """Initialize harness state."""
        pass

    @abstractmethod
    def save_task(self, task: Task):
        """Save task state."""
        pass

    @abstractmethod
    def load_task(self, task_id: str) -> Optional[Task]:
        """Load task state."""
        pass

    @abstractmethod
    def update_task_status(
        self,
        task_id: str,
        status: str,
        output: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Update task status."""
        pass

    @abstractmethod
    def get_incomplete_tasks(self) -> List[Task]:
        """Get tasks that haven't completed."""
        pass

    @abstractmethod
    def log_execution(
        self,
        task_id: str,
        attempt: int,
        success: bool,
        output: str,
        error: Optional[str],
        duration: float
    ):
        """Log execution attempt."""
        pass
```

---

## Shared Context Interface

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime


@dataclass
class Artifact:
    """Shared file artifact between harnesses."""
    name: str
    source_harness: str
    path: Path
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossHarnessEvent:
    """Event in cross-harness communication."""
    timestamp: datetime
    event_type: str  # spawn, progress, complete, fail, instruction
    harness_id: Optional[str]
    details: Dict[str, Any]


class SharedContext:
    """
    Cross-harness context sharing.

    Persisted to .harness/shared/context.json
    """

    def set(self, key: str, value: Any, harness_id: Optional[str] = None):
        """
        Set context value.

        Args:
            key: Context key
            value: Context value (must be JSON-serializable)
            harness_id: Optional scope to specific harness
        """
        pass

    def get(self, key: str, harness_id: Optional[str] = None) -> Any:
        """Get context value."""
        pass

    def publish_artifact(self, name: str, path: Path, harness_id: str):
        """
        Publish artifact for other harnesses.

        Copies file to .harness/shared/artifacts/{name}
        """
        pass

    def consume_artifact(self, name: str) -> Path:
        """
        Consume artifact published by another harness.

        Returns path to artifact in shared directory.
        """
        pass

    def emit_event(self, event_type: str, harness_id: str, details: Dict[str, Any]):
        """Emit cross-harness event."""
        pass

    def get_events(self, since: Optional[datetime] = None) -> List[CrossHarnessEvent]:
        """Get events, optionally since timestamp."""
        pass

    def sync_to_file(self):
        """Sync context to .harness/shared/context.json."""
        pass

    def load_from_file(self):
        """Load context from .harness/shared/context.json."""
        pass
```

---

## Alert Manager Interface

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Dict, Any, List


@dataclass
class Alert:
    """Alert from harness execution."""
    type: str
    severity: Literal["INFO", "WARNING", "CRITICAL", "SUCCESS", "ERROR"]
    message: str
    timestamp: datetime
    iteration: int
    context: Dict[str, Any]


class AlertManager:
    """
    Manages alerts from harness execution.

    Logs to .harness/alerts.log
    """

    def alert(
        self,
        alert_type: str,
        message: str,
        severity: str,
        **context
    ):
        """
        Emit an alert.

        Args:
            alert_type: Type of alert (e.g., "no_progress", "stuck_recovery")
            message: Human-readable message
            severity: INFO, WARNING, CRITICAL, SUCCESS, ERROR
            **context: Additional context
        """
        pass

    def get_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """Get alerts, optionally filtered by severity."""
        pass

    def get_recent(self, count: int = 10) -> List[Alert]:
        """Get recent alerts."""
        pass
```

---

## Interface Summary

| Interface | Purpose | Implementations |
|-----------|---------|-----------------|
| BaseCLIExecutor | CLI backend abstraction | Codex, Gemini, Claude |
| BaseStrategy | Coordination strategies | Parallel, Pipeline, Race, Voting |
| ProgressDetector | Progress signal detection | ProgressDetector |
| RecoveryStrategy | Stuck recovery | PromptMutator, ModelEscalator, BranchExplorer |
| HarnessHandle | Sub-harness control | HarnessHandle |
| BaseStateManager | State persistence | StateManager, MetaStateManager |
| SharedContext | Cross-harness sharing | SharedContext |
| AlertManager | Alert management | AlertManager |

---

*Last Updated: January 2026*
