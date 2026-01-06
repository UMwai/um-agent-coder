"""Environment manager aggregating all environmental inputs.

Combines file watcher, instruction queue, and environment monitor
into a unified interface for the autonomous loop.

Reference: specs/autonomous-loop-spec.md Section 4.5
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .env_monitor import EnvChange, EnvMonitor
from .file_watcher import WATCHDOG_AVAILABLE, FileEvent, PollingWatcher, WorkspaceWatcher
from .instruction_queue import Instruction, InstructionQueue


@dataclass
class EnvironmentState:
    """Aggregated environmental state for an iteration."""

    file_events: list[FileEvent] = field(default_factory=list)
    instructions: list[Instruction] = field(default_factory=list)
    env_changes: list[EnvChange] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def has_changes(self) -> bool:
        """Check if there are any environmental changes."""
        return bool(self.file_events or self.instructions or self.env_changes)

    @property
    def has_urgent_instructions(self) -> bool:
        """Check if there are urgent instructions."""
        from .instruction_queue import InstructionPriority

        return any(i.priority == InstructionPriority.URGENT for i in self.instructions)

    @property
    def should_pause(self) -> bool:
        """Check if environment requests pause."""
        for change in self.env_changes:
            if change.var == "HARNESS_PAUSE" and change.new_value in ("true", "1", "yes"):
                return True
        return False

    @property
    def should_stop(self) -> bool:
        """Check if environment requests stop."""
        for change in self.env_changes:
            if change.var == "HARNESS_STOP" and change.new_value in ("true", "1", "yes"):
                return True
        return False

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "file_events": [e.to_dict() for e in self.file_events],
            "instructions": [i.to_dict() for i in self.instructions],
            "env_changes": [c.to_dict() for c in self.env_changes],
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EnvironmentState":
        """Deserialize from dictionary."""
        return cls(
            file_events=[FileEvent.from_dict(e) for e in data.get("file_events", [])],
            instructions=[Instruction.from_dict(i) for i in data.get("instructions", [])],
            env_changes=[EnvChange.from_dict(c) for c in data.get("env_changes", [])],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )

    def get_prompt_section(self) -> str:
        """Generate prompt section for environmental changes.

        Returns:
            Formatted string for including in iteration prompt.
        """
        sections = []

        if self.env_changes:
            lines = ["## Environment Changes"]
            for change in self.env_changes:
                lines.append(f"- {change.var}: {change.old_value} -> {change.new_value}")
            sections.append("\n".join(lines))

        if self.instructions:
            lines = ["## New Instructions (incorporate these)"]
            for inst in self.instructions:
                priority_marker = ""
                from .instruction_queue import InstructionPriority

                if inst.priority == InstructionPriority.URGENT:
                    priority_marker = "[URGENT] "
                elif inst.priority == InstructionPriority.HIGH:
                    priority_marker = "[HIGH] "
                lines.append(f"- {priority_marker}{inst.content}")
            sections.append("\n".join(lines))

        if self.file_events:
            lines = ["## File Changes Detected"]
            for event in self.file_events[:20]:  # Limit to 20 events
                lines.append(f"- {event.event_type.value}: {event.path}")
            if len(self.file_events) > 20:
                lines.append(f"  ... and {len(self.file_events) - 20} more")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else ""


class EnvironmentManager:
    """Aggregate all environmental inputs for the autonomous loop.

    Manages:
    - File watcher for workspace changes
    - Instruction queue for mid-loop directives
    - Environment variable monitor

    Args:
        workspace_path: Root workspace path.
        harness_path: Path to .harness directory.
        enable_file_watcher: Whether to enable file watching.
        use_polling: Force polling instead of watchdog.
    """

    def __init__(
        self,
        workspace_path: Path,
        harness_path: Optional[Path] = None,
        enable_file_watcher: bool = True,
        use_polling: bool = False,
    ):
        """Initialize environment manager."""
        self.workspace_path = Path(workspace_path).resolve()
        self.harness_path = Path(harness_path) if harness_path else self.workspace_path / ".harness"

        # Initialize components
        self.env_monitor = EnvMonitor()

        self.instruction_queue = InstructionQueue(
            inbox_path=self.harness_path / "inbox",
            auto_create=True,
        )

        # Initialize file watcher
        self._file_watcher: Optional[WorkspaceWatcher | PollingWatcher] = None
        self._enable_file_watcher = enable_file_watcher
        force_polling = os.environ.get("UM_AGENT_CODER_FORCE_POLLING", "").lower() in ("1", "true", "yes")
        under_pytest = "PYTEST_CURRENT_TEST" in os.environ
        self._use_polling = use_polling or force_polling or under_pytest

        if enable_file_watcher:
            self._init_file_watcher()

        self._started = False

    def _init_file_watcher(self) -> None:
        """Initialize file watcher."""
        if self._use_polling or not WATCHDOG_AVAILABLE:
            self._file_watcher = PollingWatcher(
                workspace_path=self.workspace_path,
            )
        else:
            self._file_watcher = WorkspaceWatcher(
                workspace_path=self.workspace_path,
            )

    def start(self) -> None:
        """Start environmental monitoring."""
        if self._started:
            return

        if self._file_watcher and isinstance(self._file_watcher, WorkspaceWatcher):
            self._file_watcher.start()

        self._started = True

    def stop(self) -> None:
        """Stop environmental monitoring."""
        if not self._started:
            return

        if self._file_watcher and isinstance(self._file_watcher, WorkspaceWatcher):
            self._file_watcher.stop()

        self._started = False

    def poll(self, mark_instructions_processed: bool = True) -> EnvironmentState:
        """Poll all environmental inputs.

        Args:
            mark_instructions_processed: Whether to mark instructions as processed.

        Returns:
            Aggregated environment state.
        """
        state = EnvironmentState()

        # Get file events
        if self._file_watcher:
            if isinstance(self._file_watcher, PollingWatcher):
                self._file_watcher.scan()
            state.file_events = self._file_watcher.get_events()

        # Get instructions
        state.instructions = self.instruction_queue.poll()
        if mark_instructions_processed:
            for inst in state.instructions:
                self.instruction_queue.mark_processed(inst)

        # Get environment changes
        state.env_changes = self.env_monitor.check_changes()

        return state

    def poll_urgent_only(self) -> EnvironmentState:
        """Poll only for urgent changes.

        Checks for urgent instructions and critical env changes
        without clearing other events.

        Returns:
            Environment state with only urgent items.
        """
        state = EnvironmentState()

        # Check for urgent instructions
        state.instructions = self.instruction_queue.poll_urgent()
        for inst in state.instructions:
            self.instruction_queue.mark_processed(inst)

        # Check for pause/stop signals
        state.env_changes = [
            c
            for c in self.env_monitor.check_changes()
            if c.var in ("HARNESS_PAUSE", "HARNESS_STOP")
        ]

        return state

    def check_stop_file(self) -> Optional[str]:
        """Check for stop file.

        Returns:
            Stop mode ('stop' or 'abort') if file exists, None otherwise.
        """
        stop_file = self.harness_path / "stop"
        if stop_file.exists():
            try:
                content = stop_file.read_text().strip().lower()
                stop_file.unlink()  # Remove after reading
                if content in ("abort", "stop"):
                    return content
                return "stop"  # Default to graceful stop
            except OSError:
                return None
        return None

    def is_paused(self) -> bool:
        """Check if loop should pause."""
        return self.env_monitor.is_paused()

    def should_stop(self) -> bool:
        """Check if loop should stop."""
        # Check env var
        if self.env_monitor.should_stop():
            return True
        # Check stop file
        return self.check_stop_file() is not None

    def get_config_overrides(self) -> dict[str, Any]:
        """Get configuration overrides from environment.

        Returns:
            Dictionary of config overrides.
        """
        from .env_monitor import get_harness_config_from_env

        return get_harness_config_from_env()

    def add_instruction(
        self,
        content: str,
        priority: Optional[int] = None,
    ) -> Instruction:
        """Programmatically add an instruction.

        Args:
            content: Instruction content.
            priority: Optional priority level.

        Returns:
            Created instruction.
        """
        from .instruction_queue import InstructionPriority

        prio = InstructionPriority(priority) if priority is not None else InstructionPriority.NORMAL
        return self.instruction_queue.add_instruction(content, prio)

    def get_status(self) -> dict[str, Any]:
        """Get environment manager status.

        Returns:
            Dictionary with status information.
        """
        status = {
            "started": self._started,
            "workspace_path": str(self.workspace_path),
            "harness_path": str(self.harness_path),
            "file_watcher_enabled": self._enable_file_watcher,
            "file_watcher_running": False,
            "watchdog_available": WATCHDOG_AVAILABLE,
            "instruction_queue": self.instruction_queue.get_queue_status(),
            "env_snapshot": self.env_monitor.get_snapshot(),
        }

        if self._file_watcher and isinstance(self._file_watcher, WorkspaceWatcher):
            status["file_watcher_running"] = self._file_watcher.is_running()

        return status

    def __enter__(self) -> "EnvironmentManager":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


def build_environment_prompt_section(
    env_state: EnvironmentState,
    include_file_events: bool = True,
    max_file_events: int = 20,
) -> str:
    """Build prompt section from environment state.

    Args:
        env_state: The environment state.
        include_file_events: Whether to include file events.
        max_file_events: Maximum file events to include.

    Returns:
        Formatted prompt section.
    """
    if not env_state.has_changes:
        return ""

    return env_state.get_prompt_section()
