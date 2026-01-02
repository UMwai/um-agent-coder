"""Environmental awareness module for autonomous loop.

This module provides real-time environmental monitoring including:
- File system watching for workspace changes
- Instruction queue for mid-loop directives
- Environment variable monitoring
- Aggregated environment management

Reference: specs/autonomous-loop-spec.md Section 4
"""

from .env_monitor import (
    MONITORED_ENV_VARS,
    EnvChange,
    EnvMonitor,
)
from .environment_manager import (
    EnvironmentManager,
    EnvironmentState,
)
from .file_watcher import (
    FileEvent,
    FileEventType,
    WorkspaceWatcher,
)
from .instruction_queue import (
    Instruction,
    InstructionPriority,
    InstructionQueue,
)

__all__ = [
    # File watcher
    "FileEvent",
    "FileEventType",
    "WorkspaceWatcher",
    # Instruction queue
    "Instruction",
    "InstructionPriority",
    "InstructionQueue",
    # Environment monitor
    "EnvChange",
    "EnvMonitor",
    "MONITORED_ENV_VARS",
    # Environment manager
    "EnvironmentManager",
    "EnvironmentState",
]
