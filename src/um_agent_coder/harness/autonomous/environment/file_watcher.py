"""File watcher for workspace changes.

Monitors the workspace for file changes that should influence
the autonomous loop's behavior.

Reference: specs/autonomous-loop-spec.md Section 4.2
"""

import fnmatch
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

# Optional watchdog import - graceful fallback if not installed
try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object
    FileSystemEvent = None


class FileEventType(Enum):
    """Types of file system events."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileEvent:
    """Represents a file system event."""

    event_type: FileEventType
    path: str
    timestamp: datetime = field(default_factory=datetime.now)
    is_directory: bool = False
    dest_path: Optional[str] = None  # For move events

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "event_type": self.event_type.value,
            "path": self.path,
            "timestamp": self.timestamp.isoformat(),
            "is_directory": self.is_directory,
            "dest_path": self.dest_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileEvent":
        """Deserialize from dictionary."""
        return cls(
            event_type=FileEventType(data["event_type"]),
            path=data["path"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            is_directory=data.get("is_directory", False),
            dest_path=data.get("dest_path"),
        )


# Default patterns to ignore
DEFAULT_IGNORE_PATTERNS = [
    ".harness/*",
    ".git/*",
    "__pycache__/*",
    "*.pyc",
    "*.pyo",
    ".pytest_cache/*",
    ".mypy_cache/*",
    "*.egg-info/*",
    ".venv/*",
    "venv/*",
    "node_modules/*",
    ".DS_Store",
    "*.swp",
    "*.swo",
    "*~",
]


class WorkspaceWatcher:
    """Watch workspace for file changes.

    Uses watchdog library for real-time file system monitoring.
    Falls back to polling if watchdog is not available.

    Args:
        workspace_path: Root path to watch.
        ignore_patterns: Glob patterns to ignore.
        callback: Optional callback for immediate event handling.
    """

    def __init__(
        self,
        workspace_path: Path,
        ignore_patterns: Optional[list[str]] = None,
        callback: Optional[Callable[[FileEvent], None]] = None,
    ):
        """Initialize workspace watcher."""
        self.workspace_path = Path(workspace_path).resolve()
        self.ignore_patterns = ignore_patterns or DEFAULT_IGNORE_PATTERNS
        self.callback = callback

        self._events: list[FileEvent] = []
        self._events_lock = threading.Lock()
        self._changed_event = threading.Event()

        self._observer: Optional[Observer] = None
        self._running = False

    def should_ignore(self, path: str) -> bool:
        """Check if path matches ignore patterns."""
        # Get relative path for pattern matching
        # Resolve the path to handle symlinks (e.g., /var -> /private/var on macOS)
        try:
            resolved_path = Path(path).resolve()
            rel_path = resolved_path.relative_to(self.workspace_path)
            rel_str = str(rel_path)
        except ValueError:
            # Path not under workspace, use as-is
            rel_str = path

        for pattern in self.ignore_patterns:
            # Direct match
            if fnmatch.fnmatch(rel_str, pattern):
                return True

            # Check if pattern ends with /* (directory contents pattern)
            if pattern.endswith("/*"):
                dir_pattern = pattern[:-2]  # Remove /*
                # Check if path starts with the directory
                parts = rel_str.split("/")
                for i, _part in enumerate(parts):
                    prefix = "/".join(parts[: i + 1])
                    if fnmatch.fnmatch(prefix, dir_pattern):
                        return True

            # Check if any parent directory matches the pattern
            parts = rel_str.split("/")
            for i in range(len(parts)):
                partial = "/".join(parts[: i + 1])
                # Match directory patterns like ".git/*"
                if fnmatch.fnmatch(partial + "/*", pattern):
                    return True
                # Match exact directory patterns
                if fnmatch.fnmatch(partial, pattern.rstrip("/*")):
                    return True

        return False

    def start(self) -> bool:
        """Start watching the workspace.

        Returns:
            True if started successfully, False if watchdog unavailable.
        """
        if not WATCHDOG_AVAILABLE:
            return False

        if self._running:
            return True

        handler = _WatchdogHandler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.workspace_path), recursive=True)
        self._observer.start()
        self._running = True
        return True

    def stop(self) -> None:
        """Stop watching the workspace."""
        if self._observer and self._running:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
            self._running = False

    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running

    def add_event(self, event: FileEvent) -> None:
        """Add an event to the queue."""
        with self._events_lock:
            self._events.append(event)
        self._changed_event.set()

        if self.callback:
            self.callback(event)

    def get_events(self, clear: bool = True) -> list[FileEvent]:
        """Get all pending events.

        Args:
            clear: Whether to clear events after getting.

        Returns:
            List of file events.
        """
        with self._events_lock:
            events = list(self._events)
            if clear:
                self._events.clear()
                self._changed_event.clear()
        return events

    def has_changes(self) -> bool:
        """Check if there are pending changes."""
        return self._changed_event.is_set()

    def wait_for_changes(self, timeout: Optional[float] = None) -> bool:
        """Wait for file changes.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if changes occurred, False if timed out.
        """
        return self._changed_event.wait(timeout=timeout)

    def get_modified_files(self) -> set[str]:
        """Get set of modified file paths since last check."""
        events = self.get_events()
        modified = set()
        for event in events:
            if event.event_type in (FileEventType.CREATED, FileEventType.MODIFIED):
                modified.add(event.path)
        return modified

    def __enter__(self) -> "WorkspaceWatcher":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class _WatchdogHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """Internal handler for watchdog events."""

    def __init__(self, watcher: WorkspaceWatcher):
        self.watcher = watcher
        if WATCHDOG_AVAILABLE:
            super().__init__()

    def on_created(self, event: "FileSystemEvent") -> None:
        if self.watcher.should_ignore(event.src_path):
            return
        self.watcher.add_event(
            FileEvent(
                event_type=FileEventType.CREATED,
                path=event.src_path,
                is_directory=event.is_directory,
            )
        )

    def on_modified(self, event: "FileSystemEvent") -> None:
        if self.watcher.should_ignore(event.src_path):
            return
        self.watcher.add_event(
            FileEvent(
                event_type=FileEventType.MODIFIED,
                path=event.src_path,
                is_directory=event.is_directory,
            )
        )

    def on_deleted(self, event: "FileSystemEvent") -> None:
        if self.watcher.should_ignore(event.src_path):
            return
        self.watcher.add_event(
            FileEvent(
                event_type=FileEventType.DELETED,
                path=event.src_path,
                is_directory=event.is_directory,
            )
        )

    def on_moved(self, event: "FileSystemEvent") -> None:
        if self.watcher.should_ignore(event.src_path):
            return
        self.watcher.add_event(
            FileEvent(
                event_type=FileEventType.MOVED,
                path=event.src_path,
                is_directory=event.is_directory,
                dest_path=getattr(event, "dest_path", None),
            )
        )


class PollingWatcher:
    """Fallback polling-based watcher when watchdog unavailable.

    Scans the workspace periodically to detect changes.
    """

    def __init__(
        self,
        workspace_path: Path,
        ignore_patterns: Optional[list[str]] = None,
        poll_interval: float = 1.0,
    ):
        """Initialize polling watcher."""
        self.workspace_path = Path(workspace_path).resolve()
        self.ignore_patterns = ignore_patterns or DEFAULT_IGNORE_PATTERNS
        self.poll_interval = poll_interval

        self._file_mtimes: dict[str, float] = {}
        self._events: list[FileEvent] = []
        self._events_lock = threading.Lock()

    def should_ignore(self, path: str) -> bool:
        """Check if path matches ignore patterns."""
        try:
            rel_path = Path(path).relative_to(self.workspace_path)
            rel_str = str(rel_path)
        except ValueError:
            rel_str = path

        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(rel_str, pattern):
                return True

        return False

    def scan(self) -> list[FileEvent]:
        """Scan workspace for changes.

        Returns:
            List of detected file events.
        """
        events = []
        current_files = {}

        # Scan all files
        for file_path in self.workspace_path.rglob("*"):
            if file_path.is_file():
                path_str = str(file_path)
                if self.should_ignore(path_str):
                    continue

                try:
                    mtime = file_path.stat().st_mtime
                    current_files[path_str] = mtime

                    if path_str not in self._file_mtimes:
                        # New file
                        events.append(
                            FileEvent(
                                event_type=FileEventType.CREATED,
                                path=path_str,
                            )
                        )
                    elif self._file_mtimes[path_str] != mtime:
                        # Modified file
                        events.append(
                            FileEvent(
                                event_type=FileEventType.MODIFIED,
                                path=path_str,
                            )
                        )
                except (OSError, FileNotFoundError):
                    continue

        # Check for deleted files
        for path_str in self._file_mtimes:
            if path_str not in current_files:
                events.append(
                    FileEvent(
                        event_type=FileEventType.DELETED,
                        path=path_str,
                    )
                )

        self._file_mtimes = current_files

        with self._events_lock:
            self._events.extend(events)

        return events

    def get_events(self, clear: bool = True) -> list[FileEvent]:
        """Get all pending events."""
        with self._events_lock:
            events = list(self._events)
            if clear:
                self._events.clear()
        return events
