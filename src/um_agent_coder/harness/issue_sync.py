"""
GitHub issue synchronization for harness tasks.

Provides bidirectional sync between GitHub issues and harness tasks:
- Import issues with specific labels as tasks
- Close issues when tasks complete
- Update issue labels to reflect task status
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .github_client import GitHubClient, GitHubIssue
from .models import RalphConfig, Task, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of a sync operation."""

    success: bool
    tasks_imported: int = 0
    issues_closed: int = 0
    errors: List[str] = field(default_factory=list)


class IssueSyncManager:
    """Manage synchronization between GitHub issues and harness tasks.

    Enables workflow where:
    1. GitHub issues with a label (e.g., "harness") become tasks
    2. Task completion automatically closes the issue
    3. Task progress can be reflected in issue labels

    Example:
        sync = IssueSyncManager()

        # Import issues as tasks
        tasks = sync.import_issues_to_tasks(label="harness")

        # After task completion
        sync.close_issue_on_completion(task, output)
    """

    def __init__(
        self,
        github_client: Optional[GitHubClient] = None,
        default_label: str = "harness",
        in_progress_label: str = "harness:in-progress",
        completed_label: str = "harness:completed",
    ):
        """Initialize the sync manager.

        Args:
            github_client: GitHub client instance
            default_label: Label to filter issues for import
            in_progress_label: Label added when task starts
            completed_label: Label added when task completes
        """
        self.client = github_client or GitHubClient()
        self.default_label = default_label
        self.in_progress_label = in_progress_label
        self.completed_label = completed_label

    def import_issues_to_tasks(
        self,
        label: Optional[str] = None,
        phase: str = "GitHub Issues",
        default_timeout: int = 60,
        default_cli: str = "",
        enable_ralph: bool = False,
        ralph_max_iterations: int = 30,
    ) -> List[Task]:
        """Import GitHub issues as harness tasks.

        Args:
            label: Label to filter issues (defaults to default_label)
            phase: Phase name for imported tasks
            default_timeout: Default timeout in minutes
            default_cli: Default CLI backend
            enable_ralph: Enable ralph loop for imported tasks
            ralph_max_iterations: Max iterations if ralph enabled

        Returns:
            List of Task objects created from issues
        """
        label = label or self.default_label
        tasks = []

        issues = self.client.fetch_issues(labels=[label], state="open")
        logger.info(f"Found {len(issues)} issues with label '{label}'")

        for issue in issues:
            task = self._issue_to_task(
                issue=issue,
                phase=phase,
                default_timeout=default_timeout,
                default_cli=default_cli,
                enable_ralph=enable_ralph,
                ralph_max_iterations=ralph_max_iterations,
            )
            tasks.append(task)
            logger.info(f"Imported issue #{issue.number} as task {task.id}")

        return tasks

    def _issue_to_task(
        self,
        issue: GitHubIssue,
        phase: str,
        default_timeout: int,
        default_cli: str,
        enable_ralph: bool,
        ralph_max_iterations: int,
    ) -> Task:
        """Convert a GitHub issue to a Task.

        Args:
            issue: GitHub issue to convert
            phase: Phase name for the task
            default_timeout: Default timeout
            default_cli: Default CLI backend
            enable_ralph: Enable ralph loop
            ralph_max_iterations: Ralph max iterations

        Returns:
            Task object
        """
        # Generate task ID from issue number
        task_id = f"gh-{issue.number}"

        # Parse task properties from issue body
        props = self._parse_issue_properties(issue.body)

        # Build ralph config if enabled
        ralph_config = None
        if enable_ralph or props.get("ralph", False):
            ralph_config = RalphConfig(
                enabled=True,
                max_iterations=props.get("max_iterations", ralph_max_iterations),
                completion_promise=props.get("completion_promise", "COMPLETE"),
            )

        return Task(
            id=task_id,
            description=issue.to_task_description(),
            phase=phase,
            depends=props.get("depends", []),
            timeout_minutes=props.get("timeout", default_timeout),
            success_criteria=props.get("success", ""),
            cwd=props.get("cwd", "./"),
            cli=props.get("cli", default_cli),
            model=props.get("model", ""),
            ralph_config=ralph_config,
            # GitHub integration fields
            issue_url=issue.url,
            issue_number=issue.number,
            sync_to_github=True,
        )

    def _parse_issue_properties(self, body: str) -> dict:
        """Parse task properties from issue body.

        Looks for a markdown section like:
        ```
        ## Harness Config
        - timeout: 60min
        - cli: codex
        - ralph: true
        ```

        Args:
            body: Issue body text

        Returns:
            Dict of parsed properties
        """
        props = {}

        if not body:
            return props

        # Find harness config section
        config_pattern = r"##\s*Harness\s*Config\s*\n((?:[-*]\s*.+\n?)+)"
        match = re.search(config_pattern, body, re.IGNORECASE)

        if not match:
            return props

        config_text = match.group(1)

        # Parse property lines
        for line in config_text.split("\n"):
            line = line.strip()
            if not line.startswith(("-", "*")):
                continue

            line = line[1:].strip()

            if line.startswith("timeout:"):
                time_match = re.search(r"(\d+)", line)
                if time_match:
                    props["timeout"] = int(time_match.group(1))

            elif line.startswith("cli:"):
                props["cli"] = line.split(":", 1)[1].strip().lower()

            elif line.startswith("model:"):
                props["model"] = line.split(":", 1)[1].strip()

            elif line.startswith("cwd:"):
                props["cwd"] = line.split(":", 1)[1].strip()

            elif line.startswith("success:"):
                props["success"] = line.split(":", 1)[1].strip()

            elif line.startswith("depends:"):
                deps = line.split(":", 1)[1].strip()
                if deps.lower() != "none":
                    props["depends"] = [d.strip() for d in deps.split(",")]

            elif line.startswith("ralph:"):
                value = line.split(":", 1)[1].strip().lower()
                props["ralph"] = value in ("true", "yes", "1")

            elif line.startswith("max_iterations:"):
                iter_match = re.search(r"(\d+)", line)
                if iter_match:
                    props["max_iterations"] = int(iter_match.group(1))

            elif line.startswith("completion_promise:"):
                props["completion_promise"] = line.split(":", 1)[1].strip()

        return props

    def close_issue_on_completion(
        self,
        task: Task,
        output: str = "",
        add_summary: bool = True,
    ) -> bool:
        """Close a GitHub issue when its task completes.

        Args:
            task: The completed task
            output: Task output to include in comment
            add_summary: Whether to add a completion summary comment

        Returns:
            True if issue was closed successfully
        """
        if not task.issue_number:
            logger.debug(f"Task {task.id} has no associated issue")
            return True  # Not an error

        if not task.sync_to_github:
            logger.debug(f"Task {task.id} sync_to_github is disabled")
            return True

        # Build completion comment
        comment = ""
        if add_summary:
            comment = self._build_completion_comment(task, output)

        logger.info(f"Closing issue #{task.issue_number} for task {task.id}")

        return self.client.close_issue(
            issue_number=task.issue_number,
            comment=comment,
            reason="completed",
        )

    def _build_completion_comment(self, task: Task, output: str) -> str:
        """Build a completion comment for the issue.

        Args:
            task: The completed task
            output: Task output

        Returns:
            Formatted comment string
        """
        parts = [
            "## Task Completed by Harness",
            "",
            f"**Task ID:** `{task.id}`",
            f"**Completed at:** {datetime.utcnow().isoformat()}Z",
        ]

        if task.ralph_iterations:
            parts.append(f"**Ralph iterations:** {task.ralph_iterations}")

        if output:
            # Truncate long output
            output_preview = output[:2000]
            if len(output) > 2000:
                output_preview += "\n... (truncated)"

            parts.extend([
                "",
                "### Output",
                "```",
                output_preview,
                "```",
            ])

        parts.extend([
            "",
            "---",
            "*Closed automatically by um-agent-coder harness*",
        ])

        return "\n".join(parts)

    def update_issue_status(
        self,
        task: Task,
        status: TaskStatus,
    ) -> bool:
        """Update issue labels to reflect task status.

        Args:
            task: The task
            status: New task status

        Returns:
            True if labels were updated successfully
        """
        if not task.issue_number or not task.sync_to_github:
            return True

        try:
            if status == TaskStatus.IN_PROGRESS:
                # Add in-progress label
                self.client.add_label(task.issue_number, self.in_progress_label)
                self.client.remove_label(task.issue_number, self.completed_label)

            elif status == TaskStatus.COMPLETED:
                # Add completed label, remove in-progress
                self.client.add_label(task.issue_number, self.completed_label)
                self.client.remove_label(task.issue_number, self.in_progress_label)

            elif status == TaskStatus.FAILED:
                # Remove in-progress label
                self.client.remove_label(task.issue_number, self.in_progress_label)

            return True

        except Exception as e:
            logger.warning(f"Could not update labels for issue #{task.issue_number}: {e}")
            return False

    def sync_roadmap_to_issues(
        self,
        roadmap_path: Path,
        label: Optional[str] = None,
        phase: str = "GitHub Issues",
    ) -> SyncResult:
        """Sync GitHub issues into a roadmap file.

        This is a convenience method that:
        1. Fetches issues with the label
        2. Generates roadmap task entries
        3. Returns a summary (does not modify the file)

        Args:
            roadmap_path: Path to roadmap file
            label: Label to filter issues
            phase: Phase name for tasks

        Returns:
            SyncResult with import statistics
        """
        label = label or self.default_label
        result = SyncResult(success=True)

        try:
            tasks = self.import_issues_to_tasks(label=label, phase=phase)
            result.tasks_imported = len(tasks)

        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.exception(f"Error syncing issues to roadmap: {e}")

        return result

    def check_github_available(self) -> bool:
        """Check if GitHub CLI is available and authenticated.

        Returns:
            True if gh CLI is ready
        """
        return self.client.check_gh_cli()

    def generate_roadmap_entries(
        self,
        tasks: List[Task],
    ) -> str:
        """Generate roadmap markdown entries for tasks.

        Args:
            tasks: List of tasks to convert

        Returns:
            Markdown string for roadmap
        """
        lines = []

        for task in tasks:
            lines.append(f"- [ ] **{task.id}**: {task.description.split(chr(10))[0]}")
            lines.append(f"  - timeout: {task.timeout_minutes}min")

            if task.depends:
                lines.append(f"  - depends: {', '.join(task.depends)}")
            else:
                lines.append("  - depends: none")

            if task.success_criteria:
                lines.append(f"  - success: {task.success_criteria}")

            if task.cli:
                lines.append(f"  - cli: {task.cli}")

            if task.issue_number:
                lines.append(f"  - issue: #{task.issue_number}")
                lines.append("  - sync_github: true")

            if task.ralph_config and task.ralph_config.enabled:
                lines.append("  - ralph: true")
                lines.append(f"  - max_iterations: {task.ralph_config.max_iterations}")
                lines.append(f"  - completion_promise: {task.ralph_config.completion_promise}")

            lines.append("")

        return "\n".join(lines)
