"""FastAPI application factory and uvicorn runner."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from um_agent_coder.daemon.config import DaemonSettings
from um_agent_coder.daemon.database import Database
from um_agent_coder.daemon.models import HealthResponse

logger = logging.getLogger(__name__)

# Module-level singletons (set during lifespan)
_settings: Optional[DaemonSettings] = None
_db: Optional[Database] = None
_worker = None  # type: Optional[um_agent_coder.daemon.worker.TaskWorker]
_gemini_client = None  # type: Optional[um_agent_coder.daemon.gemini_client.GeminiCodeAssistClient]
_llm_router = None  # type: Optional[um_agent_coder.daemon.llm_router.LLMRouter]


def get_settings() -> DaemonSettings:
    global _settings
    if _settings is None:
        _settings = DaemonSettings()
    return _settings


def get_db() -> Database:
    assert _db is not None, "Database not initialized. Is the app running?"
    return _db


def get_worker():
    assert _worker is not None, "Worker not initialized. Is the app running?"
    return _worker


def get_gemini_client():
    from um_agent_coder.daemon.gemini_client import GeminiCodeAssistClient

    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiCodeAssistClient()
    return _gemini_client


def get_llm_router():
    """Get the multi-provider LLM router (Gemini + OpenAI + Anthropic)."""
    from um_agent_coder.daemon.llm_router import LLMRouter

    global _llm_router
    if _llm_router is None:
        settings = get_settings()
        gemini = None
        try:
            gemini = get_gemini_client()
        except Exception:
            pass
        _llm_router = LLMRouter(
            gemini_client=gemini,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            anthropic_api_key=settings.anthropic_api_key,
            anthropic_model=settings.anthropic_model,
        )
    return _llm_router


async def reset_gemini_client(refresh_token: str) -> str | None:
    """Hot-reload the Gemini client with a new refresh token.

    Creates a new client, verifies it works via load_project(), then replaces
    the module-level singleton. Returns the tier string on success.
    """
    from um_agent_coder.daemon.gemini_client import GeminiCodeAssistClient

    global _gemini_client

    new_client = GeminiCodeAssistClient(refresh_token=refresh_token)
    await new_client.load_project()  # raises on failure

    # Close old client
    if _gemini_client:
        try:
            await _gemini_client.close()
        except Exception:
            pass

    _gemini_client = new_client
    logger.info("Gemini client hot-reloaded (tier=%s)", new_client.tier)
    return new_client.tier


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle manager."""
    global _settings, _db, _worker, _gemini_client, _llm_router

    _settings = DaemonSettings()
    logger.info("Starting daemon with settings: port=%s, db=%s", _settings.port, _settings.db_path)

    # Database
    _db = Database(_settings.db_path)
    await _db.connect()
    logger.info("Database connected: %s", _settings.db_path)

    # Gemini client (generativelanguage API)
    from um_agent_coder.daemon.gemini_client import GeminiCodeAssistClient

    try:
        _gemini_client = GeminiCodeAssistClient()
        await _gemini_client.load_project()
        logger.info("Gemini client initialized (tier=%s)", _gemini_client.tier)
    except Exception as e:
        logger.warning("Gemini client init failed (queries will retry on demand): %s", e)
        _gemini_client = None

    # LLM Router (multi-provider: Gemini + OpenAI + Anthropic)
    from um_agent_coder.daemon.llm_router import LLMRouter

    _llm_router = LLMRouter(
        gemini_client=_gemini_client,
        openai_api_key=_settings.openai_api_key,
        openai_model=_settings.openai_model,
        anthropic_api_key=_settings.anthropic_api_key,
        anthropic_model=_settings.anthropic_model,
    )
    logger.info("LLM router initialized: %s", _llm_router.available_providers)

    # Worker
    from um_agent_coder.daemon.worker import TaskWorker

    _worker = TaskWorker(_settings, _db)
    await _worker.start()
    logger.info("Worker started (max_concurrent=%s)", _settings.max_concurrent_tasks)

    yield

    # Shutdown
    logger.info("Shutting down daemon...")
    await _worker.stop()
    if _llm_router:
        await _llm_router.close()
    if _gemini_client:
        await _gemini_client.close()
    await _db.close()
    logger.info("Daemon stopped.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from um_agent_coder import __version__

    app = FastAPI(
        title="um-agent-daemon",
        description="24/7 AI coding agent daemon service",
        version=__version__,
        lifespan=lifespan,
    )

    # Root redirect to dashboard
    from fastapi.responses import RedirectResponse

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/ui")

    # Health check
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health():
        pending = await _db.count_tasks(status="pending") if _db else 0
        running = await _db.count_tasks(status="running") if _db else 0
        return HealthResponse(
            status="ok",
            version=__version__,
            tasks_pending=pending,
            tasks_running=running,
        )

    # Mount routes
    from um_agent_coder.daemon.routes.auth_gemini import router as auth_gemini_router
    from um_agent_coder.daemon.routes.discord import router as discord_router
    from um_agent_coder.daemon.routes.gemini import router as gemini_router
    from um_agent_coder.daemon.routes.github import router as github_router
    from um_agent_coder.daemon.routes.kb import router as kb_router
    from um_agent_coder.daemon.routes.query import router as query_router
    from um_agent_coder.daemon.routes.slack import router as slack_router
    from um_agent_coder.daemon.routes.tasks import router as tasks_router
    from um_agent_coder.daemon.routes.ui import router as ui_router

    app.include_router(tasks_router)
    app.include_router(github_router)
    app.include_router(slack_router)
    app.include_router(discord_router)
    app.include_router(ui_router)
    app.include_router(query_router)
    app.include_router(auth_gemini_router)
    app.include_router(gemini_router)
    app.include_router(kb_router)

    from um_agent_coder.daemon.routes.world_agent import router as world_agent_router

    app.include_router(world_agent_router)

    # Serve static files
    from pathlib import Path

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


def main():
    """Entry point for `um-agent-daemon` console script."""
    import uvicorn

    settings = DaemonSettings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run(
        "um_agent_coder.daemon.app:create_app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level,
        factory=True,
    )


if __name__ == "__main__":
    main()
