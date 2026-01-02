"""Autonomous loop execution module.

This module provides enhanced autonomous execution capabilities on top of
the existing Ralph Loop, including:
- Multi-signal progress detection
- Stuck recovery system
- Context management with rolling window + summarization
- Multi-CLI routing
- Environmental awareness
- Alert system
"""

from .alerts import (
    Alert,
    AlertConfig,
    AlertManager,
    AlertSeverity,
    AlertType,
    PauseRequested,
    PauseRequestedError,
    RunawayConfig,
    RunawayDetector,
)
from .cli_router import (
    AutoRouter,
    CLIRouter,
    OpusGuard,
    TaskAnalysis,
    TaskAnalyzer,
    TaskType,
    parse_cli_list,
)
from .context_manager import (
    ContextManager,
    ContextSummarizer,
    IterationContext,
    LoopContext,
)
from .environment import (
    MONITORED_ENV_VARS,
    EnvChange,
    EnvironmentManager,
    EnvironmentState,
    EnvMonitor,
    FileEvent,
    FileEventType,
    Instruction,
    InstructionPriority,
    InstructionQueue,
    WorkspaceWatcher,
)
from .executor import (
    AutonomousConfig,
    AutonomousExecutor,
    AutonomousResult,
    TerminationReason,
)
from .monitoring import (
    LogLevel,
    RealTimeLogger,
    StatusFormat,
    StatusReporter,
)
from .progress_detector import (
    ProgressDetector,
    ProgressSignal,
    calculate_progress_score,
)
from .progress_markers import extract_progress_markers
from .recovery import (
    ESCALATION_ORDER,
    BranchExplorer,
    ExplorationBranch,
    ModelEscalator,
    MutationType,
    PromptMutator,
    RecoveryManager,
    RecoveryResult,
    RecoveryStrategy,
    StuckDetector,
    StuckState,
)

__all__ = [
    # Progress detection
    "ProgressDetector",
    "ProgressSignal",
    "calculate_progress_score",
    "extract_progress_markers",
    # Stuck recovery
    "StuckDetector",
    "StuckState",
    "PromptMutator",
    "MutationType",
    "ModelEscalator",
    "ESCALATION_ORDER",
    "BranchExplorer",
    "ExplorationBranch",
    "RecoveryManager",
    "RecoveryResult",
    "RecoveryStrategy",
    # Context management
    "ContextManager",
    "ContextSummarizer",
    "IterationContext",
    "LoopContext",
    # CLI routing
    "CLIRouter",
    "AutoRouter",
    "TaskAnalyzer",
    "TaskAnalysis",
    "TaskType",
    "OpusGuard",
    "parse_cli_list",
    # Environmental awareness
    "EnvironmentManager",
    "EnvironmentState",
    "FileEvent",
    "FileEventType",
    "WorkspaceWatcher",
    "Instruction",
    "InstructionPriority",
    "InstructionQueue",
    "EnvChange",
    "EnvMonitor",
    "MONITORED_ENV_VARS",
    # Alert system
    "Alert",
    "AlertConfig",
    "AlertManager",
    "AlertSeverity",
    "AlertType",
    "PauseRequested",  # Backwards compat alias
    "PauseRequestedError",
    "RunawayConfig",
    "RunawayDetector",
    # Executor
    "AutonomousConfig",
    "AutonomousExecutor",
    "AutonomousResult",
    "TerminationReason",
    # Monitoring
    "LogLevel",
    "RealTimeLogger",
    "StatusFormat",
    "StatusReporter",
]
