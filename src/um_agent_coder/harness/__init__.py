"""
24/7 CLI Harness

Autonomous task execution via Codex, Gemini, or Claude CLI with roadmap-driven
planning, state persistence, and continuous growth mode.

Supported CLIs:
- codex: OpenAI Codex CLI (ChatGPT Pro) - gpt-5.2
- gemini: Google Gemini CLI - gemini-3-pro, gemini-3-flash
- claude: Anthropic Claude CLI - claude-opus-4.5
"""

from .models import Task, Phase, Roadmap, TaskStatus, ExecutionResult
from .roadmap_parser import RoadmapParser
from .executors import (
    BaseCLIExecutor,
    CLIBackend,
    CodexExecutor,
    GeminiExecutor,
    ClaudeExecutor,
    create_executor,
)
from .state import StateManager
from .growth import GrowthLoop

__all__ = [
    "Task",
    "Phase",
    "Roadmap",
    "TaskStatus",
    "ExecutionResult",
    "RoadmapParser",
    "BaseCLIExecutor",
    "CLIBackend",
    "CodexExecutor",
    "GeminiExecutor",
    "ClaudeExecutor",
    "create_executor",
    "StateManager",
    "GrowthLoop",
]
