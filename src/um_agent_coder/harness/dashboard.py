"""
Dashboard for meta-harness status monitoring.

Provides aggregated progress display and per-harness status tracking.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .meta_state import MetaStateManager
from .result import HarnessStatus

logger = logging.getLogger(__name__)


class MetaHarnessDashboard:
    """
    Dashboard for monitoring meta-harness execution.

    Provides:
    - Aggregated progress bar
    - Per-harness status table
    - Alert aggregation
    - Log tailing

    Example:
        dashboard = MetaHarnessDashboard()

        # Print status table
        dashboard.print_status()

        # Get JSON status
        status = dashboard.get_status_json()

        # Tail logs for a harness
        logs = dashboard.get_harness_logs("auth-harness", tail=50)
    """

    def __init__(self, harness_dir: Path = Path(".harness")):
        """Initialize dashboard.

        Args:
            harness_dir: Base harness directory
        """
        self.harness_dir = Path(harness_dir)
        self.meta_state = MetaStateManager(
            db_path=str(self.harness_dir / "meta_state.db")
        )

    def get_status(self) -> Dict[str, Any]:
        """Get complete status as dictionary.

        Returns:
            Status dict with meta info and harness statuses
        """
        meta = self.meta_state.get_meta_state()
        harnesses = self.meta_state.get_all_harnesses()
        running = self.meta_state.get_running_harnesses()
        pending = self.meta_state.get_pending_harnesses()
        completed = self.meta_state.get_completed_harnesses()

        # Calculate aggregate progress
        total_progress = 0.0
        for h in harnesses:
            total_progress += h.get("progress", 0.0)
        avg_progress = total_progress / len(harnesses) if harnesses else 0.0

        return {
            "meta": {
                "started_at": meta.get("started_at"),
                "strategy": meta.get("strategy", "parallel"),
                "total_harnesses": meta.get("total_harnesses", 0),
                "completed_harnesses": meta.get("completed_harnesses", 0),
                "failed_harnesses": meta.get("failed_harnesses", 0),
            },
            "summary": {
                "running": len(running),
                "pending": len(pending),
                "completed": len(completed),
                "avg_progress": avg_progress,
            },
            "harnesses": harnesses,
        }

    def get_status_json(self, indent: int = 2) -> str:
        """Get status as formatted JSON.

        Args:
            indent: JSON indentation

        Returns:
            JSON string
        """
        return json.dumps(self.get_status(), indent=indent, default=str)

    def print_status(self) -> None:
        """Print formatted status table to stdout."""
        status = self.get_status()

        # Header
        print("\n" + "=" * 70)
        print("META-HARNESS STATUS DASHBOARD")
        print("=" * 70)

        # Meta info
        meta = status["meta"]
        print(f"\nStrategy: {meta['strategy'].upper()}")
        print(f"Started: {meta['started_at']}")
        print(
            f"Total: {meta['total_harnesses']} | "
            f"Completed: {meta['completed_harnesses']} | "
            f"Failed: {meta['failed_harnesses']}"
        )

        # Summary
        summary = status["summary"]
        print(f"\nCurrent: {summary['running']} running, {summary['pending']} pending")

        # Progress bar
        progress = summary["avg_progress"]
        bar_width = 40
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\nOverall Progress: [{bar}] {progress:.1%}")

        # Harness table
        print("\n" + "-" * 70)
        print(f"{'HARNESS ID':<20} {'STATUS':<12} {'PROGRESS':<10} {'TASK':<25}")
        print("-" * 70)

        for h in status["harnesses"]:
            harness_id = h["harness_id"][:18]
            status_str = h.get("status", "unknown")[:10]
            progress_val = h.get("progress", 0.0)
            progress_str = f"{progress_val:.1%}"
            current_task = (h.get("current_task") or "")[:23]

            # Color status (ANSI codes)
            status_colored = status_str
            if status_str == "running":
                status_colored = f"\033[32m{status_str}\033[0m"  # Green
            elif status_str == "completed":
                status_colored = f"\033[34m{status_str}\033[0m"  # Blue
            elif status_str == "failed":
                status_colored = f"\033[31m{status_str}\033[0m"  # Red
            elif status_str == "pending":
                status_colored = f"\033[33m{status_str}\033[0m"  # Yellow

            print(f"{harness_id:<20} {status_colored:<12} {progress_str:<10} {current_task:<25}")

        print("-" * 70)

        # Alerts summary
        alerts = self._get_recent_alerts()
        if alerts:
            print(f"\nRecent Alerts ({len(alerts)}):")
            for alert in alerts[:5]:
                print(f"  [{alert.get('level', 'INFO')}] {alert.get('message', '')}")

        print("=" * 70 + "\n")

    def get_harness_status(self, harness_id: str) -> Optional[Dict[str, Any]]:
        """Get status for a specific harness.

        Args:
            harness_id: Harness identifier

        Returns:
            Harness status dict or None
        """
        return self.meta_state.get_harness(harness_id)

    def get_harness_logs(self, harness_id: str, tail: int = 100) -> List[str]:
        """Get recent logs for a harness.

        Args:
            harness_id: Harness identifier
            tail: Number of lines to return

        Returns:
            List of log lines
        """
        log_file = self.harness_dir / harness_id / "harness.log"
        if not log_file.exists():
            return []

        try:
            with open(log_file) as f:
                lines = f.readlines()
                return lines[-tail:]
        except Exception as e:
            logger.error(f"Failed to read logs for {harness_id}: {e}")
            return []

    def print_harness_logs(self, harness_id: str, tail: int = 50) -> None:
        """Print logs for a harness.

        Args:
            harness_id: Harness identifier
            tail: Number of lines to print
        """
        logs = self.get_harness_logs(harness_id, tail)

        print(f"\n--- Logs for {harness_id} (last {tail} lines) ---\n")
        for line in logs:
            print(line.rstrip())
        print(f"\n--- End of logs ---\n")

    def _get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts from all harnesses.

        Args:
            limit: Maximum alerts to return

        Returns:
            List of alert dicts
        """
        alerts = []

        # Collect alerts from all harness directories
        for harness_dir in self.harness_dir.iterdir():
            if not harness_dir.is_dir():
                continue

            alerts_file = harness_dir / "alerts.log"
            if not alerts_file.exists():
                continue

            try:
                with open(alerts_file) as f:
                    for line in f:
                        try:
                            alert = json.loads(line.strip())
                            alert["harness_id"] = harness_dir.name
                            alerts.append(alert)
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass

        # Sort by timestamp and limit
        alerts.sort(key=lambda a: a.get("timestamp", ""), reverse=True)
        return alerts[:limit]

    def get_stuck_harnesses(self, no_progress_minutes: int = 10) -> List[Dict[str, Any]]:
        """Get harnesses that appear stuck.

        Args:
            no_progress_minutes: Minutes without progress to be considered stuck

        Returns:
            List of stuck harness dicts
        """
        stuck = []
        running = self.meta_state.get_running_harnesses()

        for h in running:
            # Check progress history
            history = self.meta_state.get_progress_history(h["harness_id"], limit=5)
            if not history:
                continue

            # Check if progress has changed
            if len(history) >= 2:
                latest = history[0].get("progress", 0)
                oldest = history[-1].get("progress", 0)
                if latest == oldest:
                    stuck.append(h)

        return stuck

    def write_status_file(self, path: Optional[Path] = None) -> Path:
        """Write status to a JSON file.

        Args:
            path: Output path (defaults to .harness/status.json)

        Returns:
            Path to written file
        """
        if path is None:
            path = self.harness_dir / "status.json"

        status = self.get_status()
        status["generated_at"] = datetime.utcnow().isoformat()

        Path(path).write_text(json.dumps(status, indent=2, default=str))
        return path
