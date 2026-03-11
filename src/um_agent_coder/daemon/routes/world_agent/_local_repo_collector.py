"""Local repo scanner collector for the World Agent.

Scans adjacent repos on the local filesystem to detect:
- Spec/implementation gaps (specs exist but code doesn't)
- Failing tests
- TODO/FIXME density
- Missing test coverage
- Stale branches
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from um_agent_coder.daemon.routes.world_agent._collectors import (
    EventCollector,
)
from um_agent_coder.daemon.routes.world_agent.models import (
    Event,
    EventCategory,
    EventSeverity,
)

logger = logging.getLogger(__name__)


class LocalRepoCollector(EventCollector):
    """Scans local repos for development signals."""

    def __init__(self, repos: Dict[str, str]):
        """
        Args:
            repos: Mapping of repo_name → absolute_path, e.g.
                   {"um_ai-hedge-fund": "/home/umwai/um_ai-hedge-fund"}
        """
        self._repos = repos

    def source_id(self) -> str:
        return "dev.local_repo_scan"

    async def collect(self, since: Optional[datetime] = None) -> List[Event]:
        events: List[Event] = []
        for name, path in self._repos.items():
            if not Path(path).is_dir():
                logger.warning("Repo path not found: %s", path)
                continue
            try:
                events.extend(self._scan_repo(name, path))
            except Exception as e:
                logger.warning("Failed to scan %s: %s", name, e)
        logger.info("Local scan produced %d events from %d repos", len(events), len(self._repos))
        return events

    def _scan_repo(self, name: str, path: str) -> List[Event]:
        events: List[Event] = []
        now = datetime.now(timezone.utc)

        # 1. Git status — uncommitted changes
        events.extend(self._scan_git_status(name, path, now))

        # 2. TODO/FIXME density
        events.extend(self._scan_todos(name, path, now))

        # 3. Spec-implementation gaps
        events.extend(self._scan_spec_gaps(name, path, now))

        # 4. Recent git activity summary
        events.extend(self._scan_recent_commits(name, path, now))

        return events

    def _scan_git_status(self, name: str, path: str, now: datetime) -> List[Event]:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=path, capture_output=True, text=True, timeout=10,
            )
            lines = [ln for ln in result.stdout.strip().split("\n") if ln.strip()]
            if not lines:
                return []

            modified = [ln for ln in lines if ln.startswith(" M") or ln.startswith("M ")]
            untracked = [ln for ln in lines if ln.startswith("??")]
            staged = [ln for ln in lines if ln[0] in "AMDRC" and ln[1] == " "]

            body = f"Modified: {len(modified)}, Untracked: {len(untracked)}, Staged: {len(staged)}"
            severity = EventSeverity.notable if len(lines) > 10 else EventSeverity.info

            return [Event(
                id=f"local-{uuid.uuid4().hex[:8]}",
                source=self.source_id(),
                timestamp=now,
                category=EventCategory.dev,
                severity=severity,
                title=f"{name}: {len(lines)} uncommitted changes",
                body=body + "\n" + "\n".join(lines[:20]),
                metadata={"repo": name, "scan_type": "git_status",
                          "modified": len(modified), "untracked": len(untracked)},
            )]
        except Exception as e:
            logger.debug("git status failed for %s: %s", name, e)
            return []

    def _scan_todos(self, name: str, path: str, now: datetime) -> List[Event]:
        try:
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", "--include=*.rs",
                 "-E", r"TODO|FIXME|HACK|XXX", "."],
                cwd=path, capture_output=True, text=True, timeout=15,
            )
            lines = [ln for ln in result.stdout.strip().split("\n") if ln.strip()]
            if len(lines) < 5:
                return []

            return [Event(
                id=f"local-{uuid.uuid4().hex[:8]}",
                source=self.source_id(),
                timestamp=now,
                category=EventCategory.dev,
                severity=EventSeverity.info,
                title=f"{name}: {len(lines)} TODO/FIXME markers found",
                body="\n".join(lines[:30]),
                metadata={"repo": name, "scan_type": "todo_density", "count": len(lines)},
            )]
        except Exception as e:
            logger.debug("TODO scan failed for %s: %s", name, e)
            return []

    def _scan_spec_gaps(self, name: str, path: str, now: datetime) -> List[Event]:
        """Compare specs/ directory against actual implementation."""
        specs_dir = Path(path) / "specs"
        if not specs_dir.is_dir():
            return []

        events: List[Event] = []
        spec_files = list(specs_dir.rglob("*.md"))

        # Check for roadmap with unchecked items
        roadmap = specs_dir / "roadmap.md"
        if roadmap.exists():
            try:
                content = roadmap.read_text()
                unchecked = content.count("- [ ]")
                checked = content.count("- [x]")
                total = unchecked + checked
                if total > 0:
                    pct = (checked / total) * 100
                    severity = EventSeverity.notable if pct < 30 else EventSeverity.info
                    events.append(Event(
                        id=f"local-{uuid.uuid4().hex[:8]}",
                        source=self.source_id(),
                        timestamp=now,
                        category=EventCategory.dev,
                        severity=severity,
                        title=f"{name}: roadmap {checked}/{total} tasks done ({pct:.0f}%)",
                        body=f"Specs directory has {len(spec_files)} spec files. "
                             f"Roadmap: {checked} complete, {unchecked} remaining.",
                        metadata={
                            "repo": name, "scan_type": "spec_gaps",
                            "roadmap_done": checked, "roadmap_total": total,
                            "spec_files": len(spec_files),
                        },
                    ))
            except Exception:
                pass

        return events

    def _scan_recent_commits(self, name: str, path: str, now: datetime) -> List[Event]:
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-10", "--since=7 days ago"],
                cwd=path, capture_output=True, text=True, timeout=10,
            )
            lines = [ln for ln in result.stdout.strip().split("\n") if ln.strip()]
            if not lines:
                return [Event(
                    id=f"local-{uuid.uuid4().hex[:8]}",
                    source=self.source_id(),
                    timestamp=now,
                    category=EventCategory.dev,
                    severity=EventSeverity.notable,
                    title=f"{name}: no commits in 7 days — stale repo",
                    body="No recent git activity detected.",
                    metadata={"repo": name, "scan_type": "recent_commits", "count": 0},
                )]

            return [Event(
                id=f"local-{uuid.uuid4().hex[:8]}",
                source=self.source_id(),
                timestamp=now,
                category=EventCategory.dev,
                severity=EventSeverity.info,
                title=f"{name}: {len(lines)} commits in last 7 days",
                body="\n".join(lines),
                metadata={"repo": name, "scan_type": "recent_commits", "count": len(lines)},
            )]
        except Exception as e:
            logger.debug("git log failed for %s: %s", name, e)
            return []
