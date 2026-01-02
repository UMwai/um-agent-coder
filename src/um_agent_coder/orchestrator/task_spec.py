"""
Task Specification - Define and manage task requirements, repository targeting, and updates.

This module provides:
1. TaskSpec: Define task requirements, constraints, and expected outputs
2. RepoTarget: Target a specific repository for the task
3. TaskUpdate: Track updates and progress notifications
4. WebhookNotifier: Send updates to external services (n8n, Slack, etc.)
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import requests


class UpdateType(Enum):
    """Types of task updates."""

    STARTED = "started"
    PROGRESS = "progress"
    STEP_COMPLETE = "step_complete"
    CHECKPOINT = "checkpoint"
    APPROVAL_NEEDED = "approval_needed"
    ERROR = "error"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class RepoTarget:
    """
    Target a specific repository for task execution.

    The task will be executed in the context of this repository,
    with access to its files, git history, and structure.
    """

    path: str  # Local path or git URL
    branch: Optional[str] = None  # Branch to use (default: current/main)
    commit: Optional[str] = None  # Specific commit to checkout
    clone_if_remote: bool = True  # Clone if path is a URL
    temp_clone: bool = False  # Clone to temp directory

    # Cloned repo info (populated after setup)
    local_path: Optional[str] = None
    is_cloned: bool = False
    original_cwd: Optional[str] = None

    def setup(self) -> str:
        """
        Setup the repository target.

        Returns:
            Local path to the repository
        """
        self.original_cwd = os.getcwd()

        # Check if it's a URL
        is_url = self.path.startswith(("http://", "https://", "git@", "git://"))

        if is_url and self.clone_if_remote:
            # Clone the repository
            if self.temp_clone:
                clone_dir = tempfile.mkdtemp(prefix="um_agent_repo_")
            else:
                # Clone to current directory with repo name
                repo_name = self.path.rstrip("/").split("/")[-1]
                if repo_name.endswith(".git"):
                    repo_name = repo_name[:-4]
                clone_dir = os.path.join(os.getcwd(), repo_name)

            cmd = ["git", "clone", self.path, clone_dir]
            if self.branch:
                cmd.extend(["--branch", self.branch])

            subprocess.run(cmd, check=True, capture_output=True)
            self.local_path = clone_dir
            self.is_cloned = True

            # Checkout specific commit if specified
            if self.commit:
                subprocess.run(
                    ["git", "checkout", self.commit], cwd=clone_dir, check=True, capture_output=True
                )
        else:
            # Use local path
            self.local_path = os.path.abspath(self.path)

            if not os.path.exists(self.local_path):
                raise ValueError(f"Repository path does not exist: {self.local_path}")

            # Checkout branch/commit if specified
            if self.branch:
                subprocess.run(
                    ["git", "checkout", self.branch],
                    cwd=self.local_path,
                    check=True,
                    capture_output=True,
                )
            if self.commit:
                subprocess.run(
                    ["git", "checkout", self.commit],
                    cwd=self.local_path,
                    check=True,
                    capture_output=True,
                )

        return self.local_path

    def cleanup(self):
        """Cleanup cloned repository if it was temporary."""
        if self.is_cloned and self.temp_clone and self.local_path:
            import shutil

            shutil.rmtree(self.local_path, ignore_errors=True)

        # Restore original working directory
        if self.original_cwd:
            os.chdir(self.original_cwd)

    def get_repo_info(self) -> dict[str, Any]:
        """Get information about the repository."""
        if not self.local_path:
            return {}

        info = {
            "path": self.local_path,
            "is_git": os.path.exists(os.path.join(self.local_path, ".git")),
        }

        if info["is_git"]:
            try:
                # Get current branch
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=self.local_path,
                    capture_output=True,
                    text=True,
                )
                info["branch"] = result.stdout.strip()

                # Get current commit
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.local_path,
                    capture_output=True,
                    text=True,
                )
                info["commit"] = result.stdout.strip()

                # Get remote URL
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=self.local_path,
                    capture_output=True,
                    text=True,
                )
                info["remote"] = result.stdout.strip()
            except Exception:
                pass

        return info


@dataclass
class TaskUpdate:
    """A single task update/notification."""

    task_id: str
    update_type: UpdateType
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: Optional[dict[str, Any]] = None
    step_id: Optional[str] = None
    progress_pct: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "type": self.update_type.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "data": self.data,
            "step_id": self.step_id,
            "progress_pct": self.progress_pct,
        }


class WebhookNotifier:
    """
    Send task updates to external services via webhooks.

    Supports:
    - Generic HTTP webhooks
    - n8n workflows
    - Slack
    - Discord
    - Custom handlers
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        slack_webhook: Optional[str] = None,
        discord_webhook: Optional[str] = None,
        n8n_webhook: Optional[str] = None,
        custom_handler: Optional[Callable[[TaskUpdate], None]] = None,
    ):
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}
        self.slack_webhook = slack_webhook
        self.discord_webhook = discord_webhook
        self.n8n_webhook = n8n_webhook
        self.custom_handler = custom_handler

        self.update_history: list[TaskUpdate] = []

    def notify(self, update: TaskUpdate):
        """Send notification to all configured channels."""
        self.update_history.append(update)

        # Custom handler
        if self.custom_handler:
            try:
                self.custom_handler(update)
            except Exception as e:
                print(f"Custom handler error: {e}")

        # Generic webhook
        if self.webhook_url:
            self._send_webhook(self.webhook_url, update.to_dict())

        # n8n webhook
        if self.n8n_webhook:
            self._send_webhook(self.n8n_webhook, update.to_dict())

        # Slack
        if self.slack_webhook:
            self._send_slack(update)

        # Discord
        if self.discord_webhook:
            self._send_discord(update)

    def _send_webhook(self, url: str, payload: dict[str, Any]):
        """Send to generic webhook."""
        try:
            requests.post(url, json=payload, headers=self.headers, timeout=10)
        except Exception as e:
            print(f"Webhook error: {e}")

    def _send_slack(self, update: TaskUpdate):
        """Send Slack notification."""
        emoji = {
            UpdateType.STARTED: ":rocket:",
            UpdateType.PROGRESS: ":hourglass:",
            UpdateType.STEP_COMPLETE: ":white_check_mark:",
            UpdateType.CHECKPOINT: ":floppy_disk:",
            UpdateType.APPROVAL_NEEDED: ":hand:",
            UpdateType.ERROR: ":x:",
            UpdateType.COMPLETED: ":tada:",
            UpdateType.CANCELLED: ":no_entry:",
        }.get(update.update_type, ":information_source:")

        payload = {
            "text": f"{emoji} *{update.task_id}* - {update.message}",
            "attachments": [
                {
                    "color": "#36a64f" if update.update_type == UpdateType.COMPLETED else "#3AA3E3",
                    "fields": [
                        {"title": "Type", "value": update.update_type.value, "short": True},
                        {
                            "title": "Progress",
                            "value": (
                                f"{update.progress_pct:.0f}%" if update.progress_pct else "N/A"
                            ),
                            "short": True,
                        },
                    ],
                }
            ],
        }

        try:
            requests.post(self.slack_webhook, json=payload, timeout=10)
        except Exception as e:
            print(f"Slack error: {e}")

    def _send_discord(self, update: TaskUpdate):
        """Send Discord notification."""
        color = {
            UpdateType.STARTED: 0x3498DB,
            UpdateType.PROGRESS: 0xF39C12,
            UpdateType.STEP_COMPLETE: 0x2ECC71,
            UpdateType.CHECKPOINT: 0x9B59B6,
            UpdateType.APPROVAL_NEEDED: 0xE74C3C,
            UpdateType.ERROR: 0xE74C3C,
            UpdateType.COMPLETED: 0x2ECC71,
            UpdateType.CANCELLED: 0x95A5A6,
        }.get(update.update_type, 0x3498DB)

        payload = {
            "embeds": [
                {
                    "title": f"Task Update: {update.task_id}",
                    "description": update.message,
                    "color": color,
                    "fields": [
                        {"name": "Type", "value": update.update_type.value, "inline": True},
                        {
                            "name": "Progress",
                            "value": (
                                f"{update.progress_pct:.0f}%" if update.progress_pct else "N/A"
                            ),
                            "inline": True,
                        },
                    ],
                    "timestamp": update.timestamp,
                }
            ]
        }

        try:
            requests.post(self.discord_webhook, json=payload, timeout=10)
        except Exception as e:
            print(f"Discord error: {e}")


@dataclass
class TaskSpec:
    """
    Task Specification - Define requirements, constraints, and expected outputs.

    A TaskSpec provides structured requirements for complex tasks, including:
    - What needs to be done (objectives)
    - How to do it (constraints, approach)
    - What success looks like (deliverables, acceptance criteria)
    - Where to do it (target repository)
    - How to track progress (webhooks, callbacks)

    Usage:
        spec = TaskSpec(
            name="Implement Authentication",
            description="Add JWT authentication to the API",
            objectives=[
                "Create auth middleware",
                "Add login/logout endpoints",
                "Protect existing routes"
            ],
            deliverables=[
                "src/auth/middleware.py",
                "src/auth/routes.py",
                "tests/test_auth.py"
            ],
            repo=RepoTarget(path="/path/to/project"),
            webhook_url="https://n8n.example.com/webhook/task-updates"
        )

        result = orchestrator.run_with_spec(spec)
    """

    name: str
    description: str

    # What to do
    objectives: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    # Expected outputs
    deliverables: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)

    # Repository targeting
    repo: Optional[RepoTarget] = None

    # Model preferences
    prefer_models: Optional[dict[str, str]] = None  # {"research": "gemini", "code": "codex"}

    # Execution settings
    max_iterations: int = 10
    timeout_minutes: int = 60
    require_approval: bool = False
    approval_steps: list[str] = field(default_factory=list)

    # Notifications
    webhook_url: Optional[str] = None
    slack_webhook: Optional[str] = None
    discord_webhook: Optional[str] = None
    n8n_webhook: Optional[str] = None
    on_update: Optional[Callable[[TaskUpdate], None]] = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    priority: int = 5  # 1-10
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Runtime state
    task_id: Optional[str] = None
    notifier: Optional[WebhookNotifier] = None

    def __post_init__(self):
        """Initialize notifier if webhooks are configured."""
        if any(
            [
                self.webhook_url,
                self.slack_webhook,
                self.discord_webhook,
                self.n8n_webhook,
                self.on_update,
            ]
        ):
            self.notifier = WebhookNotifier(
                webhook_url=self.webhook_url,
                slack_webhook=self.slack_webhook,
                discord_webhook=self.discord_webhook,
                n8n_webhook=self.n8n_webhook,
                custom_handler=self.on_update,
            )

    def to_prompt(self) -> str:
        """Convert spec to a detailed prompt for the orchestrator."""
        parts = [
            f"# Task: {self.name}",
            "",
            "## Description",
            self.description,
            "",
        ]

        if self.objectives:
            parts.append("## Objectives")
            for obj in self.objectives:
                parts.append(f"- {obj}")
            parts.append("")

        if self.requirements:
            parts.append("## Requirements")
            for req in self.requirements:
                parts.append(f"- {req}")
            parts.append("")

        if self.constraints:
            parts.append("## Constraints")
            for con in self.constraints:
                parts.append(f"- {con}")
            parts.append("")

        if self.deliverables:
            parts.append("## Expected Deliverables")
            for deliv in self.deliverables:
                parts.append(f"- {deliv}")
            parts.append("")

        if self.acceptance_criteria:
            parts.append("## Acceptance Criteria")
            for crit in self.acceptance_criteria:
                parts.append(f"- {crit}")
            parts.append("")

        if self.repo and self.repo.local_path:
            parts.append("## Target Repository")
            repo_info = self.repo.get_repo_info()
            parts.append(f"- Path: {repo_info.get('path', 'N/A')}")
            if repo_info.get("branch"):
                parts.append(f"- Branch: {repo_info['branch']}")
            if repo_info.get("remote"):
                parts.append(f"- Remote: {repo_info['remote']}")
            parts.append("")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "objectives": self.objectives,
            "requirements": self.requirements,
            "constraints": self.constraints,
            "deliverables": self.deliverables,
            "acceptance_criteria": self.acceptance_criteria,
            "repo": (
                {
                    "path": self.repo.path if self.repo else None,
                    "branch": self.repo.branch if self.repo else None,
                    "local_path": self.repo.local_path if self.repo else None,
                }
                if self.repo
                else None
            ),
            "prefer_models": self.prefer_models,
            "max_iterations": self.max_iterations,
            "timeout_minutes": self.timeout_minutes,
            "require_approval": self.require_approval,
            "tags": self.tags,
            "priority": self.priority,
            "created_at": self.created_at,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        """Create TaskSpec from dictionary."""
        repo_data = data.pop("repo", None)
        repo = None
        if repo_data and repo_data.get("path"):
            repo = RepoTarget(
                path=repo_data["path"],
                branch=repo_data.get("branch"),
            )

        return cls(repo=repo, **data)

    @classmethod
    def from_yaml(cls, path: str) -> "TaskSpec":
        """Load TaskSpec from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "TaskSpec":
        """Load TaskSpec from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: str):
        """Save TaskSpec to file (JSON or YAML based on extension)."""
        data = self.to_dict()

        if path.endswith(".yaml") or path.endswith(".yml"):
            import yaml

            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    def notify(self, update_type: UpdateType, message: str, **kwargs):
        """Send a notification about this task."""
        if self.notifier:
            update = TaskUpdate(
                task_id=self.task_id or self.name,
                update_type=update_type,
                message=message,
                **kwargs,
            )
            self.notifier.notify(update)

    def setup_repo(self) -> Optional[str]:
        """Setup the target repository and return its path."""
        if self.repo:
            return self.repo.setup()
        return None

    def cleanup_repo(self):
        """Cleanup the target repository."""
        if self.repo:
            self.repo.cleanup()


def create_spec_template(output_path: str = "task_spec.yaml"):
    """Create a template TaskSpec YAML file."""
    template = """# Task Specification Template
# Save this file and customize for your task

name: "My Task Name"
description: |
  Detailed description of what needs to be accomplished.
  Can be multiple lines.

objectives:
  - First objective
  - Second objective
  - Third objective

requirements:
  - Must use Python 3.9+
  - Must include unit tests
  - Must follow existing code style

constraints:
  - Do not modify existing API contracts
  - Keep backwards compatibility
  - Maximum 500 lines of new code

deliverables:
  - src/feature/main.py
  - src/feature/utils.py
  - tests/test_feature.py

acceptance_criteria:
  - All tests pass
  - Code coverage > 80%
  - No linting errors

# Repository targeting (optional)
repo:
  path: "/path/to/local/repo"  # Or git URL
  branch: "main"  # Optional

# Model preferences (optional)
prefer_models:
  research: "gemini"
  code: "codex"
  review: "claude"

# Execution settings
max_iterations: 10
timeout_minutes: 60
require_approval: false
approval_steps: []  # e.g., ["final_review", "deploy"]

# Notifications (optional)
webhook_url: null  # Generic webhook
slack_webhook: null  # Slack incoming webhook
discord_webhook: null  # Discord webhook
n8n_webhook: null  # n8n workflow webhook

# Metadata
tags:
  - feature
  - backend
priority: 5  # 1-10
"""

    with open(output_path, "w") as f:
        f.write(template)

    print(f"Created template at: {output_path}")
    return output_path
