"""
SharedContext for cross-harness context sharing.

Provides a mechanism for sub-harnesses to share context data and artifacts
with each other and with the parent meta-harness.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SharedContext:
    """
    Shared context for cross-harness communication.

    Stores context data in a JSON file that can be read/written by
    multiple sub-harnesses. Provides thread-safe access to shared data.

    Example:
        context = SharedContext()

        # Parent sets initial context
        context.set("project_name", "my-app")
        context.set("config", {"debug": True})

        # Sub-harness reads context
        name = context.get("project_name")

        # Sub-harness publishes result
        context.set("auth_harness.result", {"token_format": "JWT"})

        # Get all context for a namespace
        auth_data = context.get_namespace("auth_harness")
    """

    def __init__(self, context_dir: Path = Path(".harness/shared")):
        """Initialize shared context.

        Args:
            context_dir: Directory for shared context files
        """
        self.context_dir = Path(context_dir)
        self.context_dir.mkdir(parents=True, exist_ok=True)
        self.context_file = self.context_dir / "context.json"
        self._lock = threading.Lock()

        # Initialize context file if it doesn't exist
        if not self.context_file.exists():
            self._write_context({
                "_meta": {
                    "created_at": datetime.utcnow().isoformat(),
                    "version": 1,
                }
            })

    def _read_context(self) -> Dict[str, Any]:
        """Read context from file.

        Returns:
            Context data dict
        """
        try:
            if self.context_file.exists():
                return json.loads(self.context_file.read_text())
            return {}
        except json.JSONDecodeError:
            logger.warning("Context file corrupted, returning empty context")
            return {}

    def _write_context(self, context: Dict[str, Any]) -> None:
        """Write context to file.

        Args:
            context: Context data to write
        """
        # Update metadata
        context["_meta"] = context.get("_meta", {})
        context["_meta"]["updated_at"] = datetime.utcnow().isoformat()

        self.context_file.write_text(json.dumps(context, indent=2))

    def set(self, key: str, value: Any) -> None:
        """Set a context value.

        Args:
            key: Context key (can be dotted path like "harness.result")
            value: Value to store
        """
        with self._lock:
            context = self._read_context()

            # Support dotted paths
            parts = key.split(".")
            current = context
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value
            self._write_context(context)

        logger.debug(f"Set context: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value.

        Args:
            key: Context key (can be dotted path)
            default: Default value if key not found

        Returns:
            Context value or default
        """
        with self._lock:
            context = self._read_context()

        # Support dotted paths
        parts = key.split(".")
        current = context
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]

        return current

    def delete(self, key: str) -> bool:
        """Delete a context key.

        Args:
            key: Context key to delete

        Returns:
            True if key was deleted
        """
        with self._lock:
            context = self._read_context()

            # Support dotted paths
            parts = key.split(".")
            current = context
            for part in parts[:-1]:
                if part not in current:
                    return False
                current = current[part]

            if parts[-1] in current:
                del current[parts[-1]]
                self._write_context(context)
                return True

        return False

    def get_namespace(self, namespace: str) -> Dict[str, Any]:
        """Get all values in a namespace.

        Args:
            namespace: Namespace prefix

        Returns:
            Dict of values in namespace
        """
        value = self.get(namespace)
        if isinstance(value, dict):
            return value
        return {}

    def set_harness_context(
        self,
        harness_id: str,
        key: str,
        value: Any,
    ) -> None:
        """Set context for a specific harness.

        Args:
            harness_id: Harness identifier
            key: Context key within harness namespace
            value: Value to store
        """
        self.set(f"harnesses.{harness_id}.{key}", value)

    def get_harness_context(
        self,
        harness_id: str,
        key: Optional[str] = None,
    ) -> Any:
        """Get context for a specific harness.

        Args:
            harness_id: Harness identifier
            key: Optional key within harness namespace

        Returns:
            Context value or full harness context
        """
        if key:
            return self.get(f"harnesses.{harness_id}.{key}")
        return self.get_namespace(f"harnesses.{harness_id}")

    def get_all_harness_contexts(self) -> Dict[str, Dict[str, Any]]:
        """Get contexts for all harnesses.

        Returns:
            Dict mapping harness_id to context
        """
        return self.get_namespace("harnesses")

    def sync_to_harness(self, harness_id: str, state_dir: Path) -> None:
        """Sync shared context to a sub-harness's state directory.

        Creates a copy of the shared context in the harness's inbox.

        Args:
            harness_id: Harness identifier
            state_dir: Harness state directory
        """
        inbox_dir = state_dir / "inbox"
        inbox_dir.mkdir(parents=True, exist_ok=True)

        # Copy current shared context
        context = self._read_context()
        context_copy = state_dir / "shared_context.json"
        context_copy.write_text(json.dumps(context, indent=2))

        logger.debug(f"Synced context to harness {harness_id}")

    def merge_from_harness(
        self,
        harness_id: str,
        state_dir: Path,
    ) -> None:
        """Merge context updates from a sub-harness.

        Reads harness's output context and merges into shared context.

        Args:
            harness_id: Harness identifier
            state_dir: Harness state directory
        """
        output_file = state_dir / "output_context.json"
        if not output_file.exists():
            return

        try:
            harness_context = json.loads(output_file.read_text())
            for key, value in harness_context.items():
                self.set_harness_context(harness_id, key, value)
            logger.info(f"Merged context from harness {harness_id}")
        except Exception as e:
            logger.error(f"Failed to merge context from {harness_id}: {e}")

    def get_all(self) -> Dict[str, Any]:
        """Get all context data.

        Returns:
            Complete context dict
        """
        with self._lock:
            return self._read_context()

    def clear(self) -> None:
        """Clear all context data."""
        with self._lock:
            self._write_context({
                "_meta": {
                    "created_at": datetime.utcnow().isoformat(),
                    "version": 1,
                    "cleared_at": datetime.utcnow().isoformat(),
                }
            })
        logger.info("Shared context cleared")
