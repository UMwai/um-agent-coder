"""
GitHub client for issue synchronization.

Uses the GitHub CLI (gh) to interact with GitHub issues,
enabling issue-task synchronization for the harness.
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GitHubIssue:
    """Represents a GitHub issue."""

    number: int
    title: str
    body: str
    labels: List[str] = field(default_factory=list)
    state: str = "open"
    url: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    assignees: List[str] = field(default_factory=list)
    milestone: Optional[str] = None

    @staticmethod
    def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string from GitHub API."""
        if not value:
            return None
        # GitHub returns ISO format with Z suffix for UTC
        # Convert to proper timezone-aware datetime
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    @classmethod
    def from_json(cls, data: dict) -> "GitHubIssue":
        """Create from GitHub API JSON response."""
        return cls(
            number=data.get("number", 0),
            title=data.get("title", ""),
            body=data.get("body", "") or "",
            labels=[label.get("name", "") for label in data.get("labels", [])],
            state=data.get("state", "open"),
            url=data.get("url", "") or data.get("html_url", ""),
            created_at=cls._parse_datetime(data.get("createdAt")),
            updated_at=cls._parse_datetime(data.get("updatedAt")),
            assignees=[a.get("login", "") for a in data.get("assignees", [])],
            milestone=data.get("milestone", {}).get("title") if data.get("milestone") else None,
        )

    def to_task_description(self) -> str:
        """Convert issue to a task description."""
        parts = [self.title]

        if self.body:
            # Truncate long bodies
            body = self.body[:1000]
            if len(self.body) > 1000:
                body += "..."
            parts.append(f"\n\n{body}")

        if self.labels:
            parts.append(f"\n\nLabels: {', '.join(self.labels)}")

        return "".join(parts)


class GitHubClient:
    """Client for GitHub operations using the gh CLI.

    Provides methods to fetch, close, and comment on issues
    for integration with the harness task system.

    Example:
        client = GitHubClient()

        # Fetch issues with a label
        issues = client.fetch_issues(labels=["harness"])

        # Close an issue when task completes
        client.close_issue(123, comment="Completed by harness")
    """

    def __init__(
        self,
        repo: Optional[str] = None,
        timeout_seconds: int = 30,
    ):
        """Initialize the GitHub client.

        Args:
            repo: Repository in owner/repo format (auto-detected if not provided)
            timeout_seconds: Timeout for gh commands
        """
        self.repo = repo
        self.timeout_seconds = timeout_seconds
        self._cached_repo: Optional[str] = None

    def get_repo(self) -> str:
        """Get the repository name, detecting from git if needed.

        Returns:
            Repository in owner/repo format
        """
        if self.repo:
            return self.repo

        if self._cached_repo:
            return self._cached_repo

        # Detect from git remote
        try:
            result = subprocess.run(
                ["gh", "repo", "view", "--json", "nameWithOwner"],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=True,
            )
            data = json.loads(result.stdout)
            self._cached_repo = data.get("nameWithOwner", "")
            return self._cached_repo
        except Exception as e:
            logger.warning(f"Could not detect repository: {e}")
            return ""

    def fetch_issues(
        self,
        labels: Optional[List[str]] = None,
        state: str = "open",
        limit: int = 100,
        assignee: Optional[str] = None,
        milestone: Optional[str] = None,
    ) -> List[GitHubIssue]:
        """Fetch issues from GitHub.

        Args:
            labels: Filter by labels
            state: Issue state (open, closed, all)
            limit: Maximum issues to fetch
            assignee: Filter by assignee
            milestone: Filter by milestone

        Returns:
            List of GitHubIssue objects
        """
        cmd = [
            "gh",
            "issue",
            "list",
            "--json",
            "number,title,body,labels,state,url,createdAt,updatedAt,assignees,milestone",
            "--limit",
            str(limit),
            "--state",
            state,
        ]

        if labels:
            for label in labels:
                cmd.extend(["--label", label])

        if assignee:
            cmd.extend(["--assignee", assignee])

        if milestone:
            cmd.extend(["--milestone", milestone])

        if self.repo:
            cmd.extend(["--repo", self.repo])

        try:
            logger.debug(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=True,
            )

            issues_data = json.loads(result.stdout)
            return [GitHubIssue.from_json(issue) for issue in issues_data]

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to fetch issues: {e.stderr}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse issues response: {e}")
            return []
        except subprocess.TimeoutExpired:
            logger.error("Timeout fetching issues")
            return []

    def get_issue(self, issue_number: int) -> Optional[GitHubIssue]:
        """Get a single issue by number.

        Args:
            issue_number: GitHub issue number

        Returns:
            GitHubIssue or None if not found
        """
        cmd = [
            "gh",
            "issue",
            "view",
            str(issue_number),
            "--json",
            "number,title,body,labels,state,url,createdAt,updatedAt,assignees,milestone",
        ]

        if self.repo:
            cmd.extend(["--repo", self.repo])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=True,
            )

            data = json.loads(result.stdout)
            return GitHubIssue.from_json(data)

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get issue #{issue_number}: {e.stderr}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse issue response: {e}")
            return None

    def close_issue(
        self,
        issue_number: int,
        comment: str = "",
        reason: str = "completed",
    ) -> bool:
        """Close a GitHub issue.

        Args:
            issue_number: GitHub issue number
            comment: Optional comment to add before closing
            reason: Close reason (completed, not_planned)

        Returns:
            True if successful
        """
        # Add comment first if provided
        if comment:
            if not self.add_issue_comment(issue_number, comment):
                logger.warning(f"Could not add closing comment to issue #{issue_number}")

        cmd = [
            "gh",
            "issue",
            "close",
            str(issue_number),
            "--reason",
            reason,
        ]

        if self.repo:
            cmd.extend(["--repo", self.repo])

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=True,
            )
            logger.info(f"Closed issue #{issue_number}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to close issue #{issue_number}: {e.stderr}")
            return False

    def add_issue_comment(
        self,
        issue_number: int,
        comment: str,
    ) -> bool:
        """Add a comment to an issue.

        Args:
            issue_number: GitHub issue number
            comment: Comment text

        Returns:
            True if successful
        """
        cmd = [
            "gh",
            "issue",
            "comment",
            str(issue_number),
            "--body",
            comment,
        ]

        if self.repo:
            cmd.extend(["--repo", self.repo])

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=True,
            )
            logger.info(f"Added comment to issue #{issue_number}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to comment on issue #{issue_number}: {e.stderr}")
            return False

    def add_label(
        self,
        issue_number: int,
        label: str,
    ) -> bool:
        """Add a label to an issue.

        Args:
            issue_number: GitHub issue number
            label: Label to add

        Returns:
            True if successful
        """
        cmd = [
            "gh",
            "issue",
            "edit",
            str(issue_number),
            "--add-label",
            label,
        ]

        if self.repo:
            cmd.extend(["--repo", self.repo])

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=True,
            )
            logger.info(f"Added label '{label}' to issue #{issue_number}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add label to issue #{issue_number}: {e.stderr}")
            return False

    def remove_label(
        self,
        issue_number: int,
        label: str,
    ) -> bool:
        """Remove a label from an issue.

        Args:
            issue_number: GitHub issue number
            label: Label to remove

        Returns:
            True if successful
        """
        cmd = [
            "gh",
            "issue",
            "edit",
            str(issue_number),
            "--remove-label",
            label,
        ]

        if self.repo:
            cmd.extend(["--repo", self.repo])

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=True,
            )
            logger.info(f"Removed label '{label}' from issue #{issue_number}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to remove label from issue #{issue_number}: {e.stderr}")
            return False

    def check_gh_cli(self) -> bool:
        """Check if gh CLI is available and authenticated.

        Returns:
            True if gh CLI is available and authenticated
        """
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.error("gh CLI not found. Install from https://cli.github.com/")
            return False
        except subprocess.TimeoutExpired:
            logger.error("gh auth status timed out")
            return False
