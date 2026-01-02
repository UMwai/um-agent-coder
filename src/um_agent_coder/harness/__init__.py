"""
24/7 CLI Harness

Autonomous task execution via Codex, Gemini, or Claude CLI with roadmap-driven
planning, state persistence, and continuous growth mode.

Supported CLIs:
- codex: OpenAI Codex CLI (ChatGPT Pro) - gpt-5.2
- gemini: Google Gemini CLI - gemini-3-pro, gemini-3-flash
- claude: Anthropic Claude CLI - claude-opus-4.5
"""

from .executors import (
    BaseCLIExecutor,
    ClaudeExecutor,
    CLIBackend,
    CodexExecutor,
    GeminiExecutor,
    create_executor,
)
from .growth import GrowthLoop
from .models import ExecutionResult, Phase, Roadmap, Task, TaskStatus
from .roadmap_parser import RoadmapParser
from .state import StateManager

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
