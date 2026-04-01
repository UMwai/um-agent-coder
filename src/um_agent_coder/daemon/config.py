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
    discord_application_id: Optional[str] = None

    # Notification defaults
    default_webhook_url: Optional[str] = None
    default_slack_webhook: Optional[str] = None
    default_discord_webhook: Optional[str] = None

    # GCP
    gcp_project_id: Optional[str] = None

    # Query proxy
    codex_model: str = "gpt-5.2"
    gemini_model: str = "gemini-3-flash-preview"
    gemini_auto_models: str = "gemini-3-flash-preview,gemini-3-pro-preview,gemini-3.1-pro-preview"
    query_rate_limit: int = 60  # requests/min/provider

    # Worker
    max_concurrent_tasks: int = 2
    task_timeout_seconds: int = 3600

    # Gemini Intelligence Layer
    gemini_enhance_enabled: bool = True
    gemini_self_eval_enabled: bool = True
    gemini_self_eval_threshold: float = 0.6
    gemini_self_eval_model: str = "auto"  # "auto" = match generation model, or explicit model name
    gemini_max_retries: int = 2
    gemini_session_max_tokens: int = 500_000
    gemini_session_ttl_hours: int = 24
    gemini_batch_max_concurrent: int = 5
    gemini_agent_max_steps: int = 10
    gemini_complexity_threshold: int = 50

    # Model registry — update these env vars when new models ship
    gemini_model_flash: str = "gemini-3-flash-preview"
    gemini_model_pro: str = "gemini-3-pro-preview"
    gemini_model_pro_latest: str = "gemini-3.1-pro-preview"
    gemini_eval_model: str = "gemini-3-flash-preview"  # general eval (fast)
    gemini_accuracy_eval_model: str = "gemini-3.1-pro-preview"  # accuracy + fulfillment eval

    # Checklist evaluation (Phase 1)
    gemini_checklist_eval_enabled: bool = True

    # File extraction + syntax validation (Phase 2)
    gemini_file_extraction_enabled: bool = True
    gemini_syntax_validation_enabled: bool = True

    # Mistake library (Phase 3)
    gemini_mistake_library_enabled: bool = True
    gemini_mistake_library_top_k: int = 10
    gemini_mistake_library_min_occurrences: int = 2

    # Webhook notifications (Phase 4)
    gemini_webhook_timeout_seconds: int = 30
    gemini_webhook_max_retries: int = 3

    # Auto eval_context (Phase 5)
    gemini_auto_eval_context_enabled: bool = True
    gemini_auto_eval_context_max_files: int = 100

    # Pre-generation checklist
    gemini_pregen_checklist_enabled: bool = True
    gemini_pregen_checklist_in_prompt: bool = True
    gemini_pregen_checklist_max_checks: int = 40

    # Firestore persistence (opt-in)
    gemini_firestore_enabled: bool = False
    gemini_firestore_collection: str = "iteration_runs"

    # Iteration runner
    gemini_iterate_max_iterations: int = 5
    gemini_iterate_score_threshold: float = 0.85
    gemini_iterate_eval_models: str = "gemini-3.1-pro-preview"  # comma-separated
    gemini_iterate_generation_model: str = "gemini-3.1-pro-preview"
    gemini_iterate_temperature: float = 0.7
    gemini_iterate_max_tokens: int = 65536
    gemini_iterate_max_strategies: int = 2  # max fix strategies per retry
    gemini_iterate_oscillation_window: int = 3  # steps to detect plateau
    gemini_iterate_oscillation_spread: float = 0.03  # max score spread = plateau

    # Knowledge Base
    kb_firestore_collection: str = "knowledge_base"
    kb_auto_extract_enabled: bool = True
    kb_max_inject_items: int = 5
    kb_extract_model: str = "gemini-3-flash-preview"

    # OpenAI (for World Agent multi-provider routing)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-5.4"

    # Anthropic (for World Agent multi-provider routing)
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-4-6-20250627"

    # World Agent
    world_agent_enabled: bool = False
    world_agent_github_repos: str = ""  # Comma-separated "owner/repo"
    world_agent_local_repos: str = ""  # Comma-separated "name=/path" pairs
    world_agent_relevance_threshold: float = 0.3
    world_agent_goals_path: str = "goals/"
    world_agent_orientation_model: str = ""  # Empty = use gemini_model_flash
    world_agent_decide_model: str = ""  # Empty = use gemini_model_pro
    world_agent_llm_provider: str = ""  # Empty = auto-detect from model name
    world_agent_max_events_per_batch: int = 100

    # LLM rate-limit throttling
    llm_inter_call_delay: int = 10  # seconds between LLM calls in a cycle
    after_hours_skip_trade_recs: bool = True
    after_hours_cutoff_utc: int = 21  # 21:00 UTC = 4pm ET

    # Codex CLI fallback (when Gemini is rate-limited)
    codex_fallback_enabled: bool = True
    codex_cli_path: str = "codex"
    codex_fallback_timeout: int = 45

    # Command Center push bridge
    command_center_url: str = ""  # e.g. "https://command-center-staging-23o5bq3bfq-uc.a.run.app"

    # Orchestrator
    checkpoint_dir: str = ".pipeline_checkpoints"
    verbose: bool = True

    @property
    def db_full_path(self) -> Path:
        return Path(self.db_path).resolve()
