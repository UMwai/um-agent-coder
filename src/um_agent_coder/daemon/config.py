"""Daemon configuration via pydantic-settings with UM_DAEMON_ env var prefix."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class DaemonSettings(BaseSettings):
    """Configuration for the daemon service. All values can be set via
    environment variables with the UM_DAEMON_ prefix, e.g. UM_DAEMON_PORT=9090.
    """

    model_config = {"env_prefix": "UM_DAEMON_"}

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    log_level: str = "info"

    # Database
    db_path: str = "daemon_tasks.db"

    # Auth
    api_key: Optional[str] = None

    # GitHub webhook
    github_webhook_secret: Optional[str] = None
    github_token: Optional[str] = None

    # Slack
    slack_signing_secret: Optional[str] = None
    slack_bot_token: Optional[str] = None

    # Discord
    discord_public_key: Optional[str] = None
    discord_bot_token: Optional[str] = None

    # Notification defaults
    default_webhook_url: Optional[str] = None
    default_slack_webhook: Optional[str] = None
    default_discord_webhook: Optional[str] = None

    # Query proxy
    codex_model: str = "gpt-5.2"
    gemini_model: str = "gemini-3-flash-preview"
    gemini_auto_models: str = (
        "gemini-3-flash-preview,gemini-3-pro-preview,gemini-3.1-pro-preview"
    )
    query_rate_limit: int = 60  # requests/min/provider

    # Worker
    max_concurrent_tasks: int = 2
    task_timeout_seconds: int = 3600

    # Orchestrator
    checkpoint_dir: str = ".pipeline_checkpoints"
    verbose: bool = True

    @property
    def db_full_path(self) -> Path:
        return Path(self.db_path).resolve()
