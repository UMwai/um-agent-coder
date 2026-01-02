"""
Multi-CLI executors for the 24/7 harness.

Supports:
- Codex CLI (OpenAI/ChatGPT Pro) - gpt-5.2
- Gemini CLI (Google) - gemini-3-pro, gemini-3-flash
- Claude CLI (Anthropic) - claude-opus-4.5
"""

import logging
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union

from .models import ExecutionResult, Task

logger = logging.getLogger(__name__)


class CLIBackend(Enum):
    """Available CLI backends."""
    CODEX = "codex"
    GEMINI = "gemini"
    CLAUDE = "claude"


class BaseCLIExecutor(ABC):
    """Base class for CLI executors."""

    CLI_NAME: str = ""
    DEFAULT_MODEL: str = ""
    DEFAULT_TIMEOUT: int = 1800  # 30 minutes

    def __init__(self, model: Optional[str] = None, timeout: int = None):
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self._conversations: dict[str, str] = {}

        # Verify CLI is available
        if not self._check_cli_available():
            logger.warning(f"{self.CLI_NAME} CLI not found in PATH")

    def _check_cli_available(self) -> bool:
        """Check if the CLI is installed and available."""
        return shutil.which(self.CLI_NAME) is not None

    def execute(self, task: Task, context: str = "") -> ExecutionResult:
        """Execute a task via the CLI."""
        start_time = time.time()
        prompt = self._build_prompt(task, context)

        try:
            result = self._run_cli(prompt, task.cwd)
            duration = time.time() - start_time

            return ExecutionResult(
                success=result.success,
                output=result.output,
                error=result.error,
                conversation_id=result.conversation_id,
                duration_seconds=duration,
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"Task {task.id} timed out after {self.timeout}s")
            return ExecutionResult(
                success=False,
                output="",
                error=f"Task timed out after {self.timeout} seconds",
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"{self.CLI_NAME} execution failed for task {task.id}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                duration_seconds=duration,
            )

    def _build_prompt(self, task: Task, context: str = "") -> str:
        """Build a structured prompt for the CLI."""
        prompt_parts = [
            f"## Task: {task.description}",
            "",
            f"Task ID: {task.id}",
            f"Phase: {task.phase}",
        ]

        if task.success_criteria:
            prompt_parts.extend([
                "",
                "## Success Criteria",
                task.success_criteria,
            ])

        if context:
            prompt_parts.extend([
                "",
                "## Context",
                context,
            ])

        prompt_parts.extend([
            "",
            "## Instructions",
            "1. Complete this task fully",
            "2. Verify success criteria are met",
            "3. Report any issues or blockers",
            "4. If task cannot be completed, explain why",
        ])

        return "\n".join(prompt_parts)

    @abstractmethod
    def _build_command(self, prompt: str, cwd: str) -> list[str]:
        """Build the CLI command. Override in subclasses."""
        pass

    def _run_cli(self, prompt: str, cwd: str) -> ExecutionResult:
        """Run the CLI command."""
        cmd = self._build_command(prompt, cwd)

        logger.info(f"Starting {self.CLI_NAME}: {' '.join(cmd[:5])}...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd if cwd and cwd != "./" else None,
            timeout=self.timeout,
        )

        output = result.stdout
        error = result.stderr
        success = result.returncode == 0 and not self._detect_failure(output, error)

        return ExecutionResult(
            success=success,
            output=output,
            error=error,
        )

    def _detect_failure(self, output: str, error: str) -> bool:
        """Analyze output to detect task failure."""
        failure_indicators = [
            "error:",
            "failed:",
            "could not complete",
            "unable to",
            "permission denied",
            "exception:",
            "traceback",
        ]

        combined = (output + error).lower()

        for indicator in failure_indicators:
            if indicator in combined:
                if f"task {indicator}" in combined or combined.count(indicator) > 2:
                    return True

        return False

    def verify_success(self, task: Task, result: ExecutionResult) -> bool:
        """Verify that the task's success criteria were met."""
        if not task.success_criteria:
            return result.success

        verify_prompt = f"""
Verify if the following success criteria was met:

Criteria: {task.success_criteria}

Task output:
{result.output[:2000]}

Respond with only "YES" or "NO" followed by a brief explanation.
"""
        verify_result = self._run_cli(verify_prompt, task.cwd)
        return "yes" in verify_result.output.lower()[:50]


class CodexExecutor(BaseCLIExecutor):
    """Execute tasks via Codex CLI (OpenAI/ChatGPT Pro)."""

    CLI_NAME = "codex"
    DEFAULT_MODEL = "gpt-5.2"

    def __init__(
        self,
        model: str = "gpt-5.2",
        reasoning_effort: str = "high",
        sandbox: str = "danger-full-access",
        approval_policy: str = "never",
        timeout: int = 1800,
    ):
        super().__init__(model=model, timeout=timeout)
        self.reasoning_effort = reasoning_effort
        self.sandbox = sandbox
        self.approval_policy = approval_policy

    def _build_command(self, prompt: str, cwd: str) -> list[str]:
        """Build the Codex CLI command."""
        cmd = [
            "codex",
            "-m", self.model,
            "-a", self.approval_policy,  # Changed from --approval-policy to -a
        ]

        if self.sandbox:
            cmd.extend(["-s", self.sandbox])  # Changed --sandbox to -s

        # Config format: -c key=value (TOML-parsed value)
        cmd.extend(["-c", f"model_reasoning_effort={self.reasoning_effort}"])

        if cwd and cwd != "./":
            cmd.extend(["-C", cwd])  # Changed --cwd to -C (--cd)

        cmd.append(prompt)
        return cmd


class GeminiExecutor(BaseCLIExecutor):
    """Execute tasks via Gemini CLI (Google)."""

    CLI_NAME = "gemini"
    DEFAULT_MODEL = "gemini-3-pro"

    def __init__(
        self,
        model: str = "gemini-3-pro",
        timeout: int = 1800,
    ):
        super().__init__(model=model, timeout=timeout)

    def _build_command(self, prompt: str, cwd: str) -> list[str]:
        """Build the Gemini CLI command."""
        # Gemini CLI uses: gemini [options] [query..]
        cmd = ["gemini"]

        # Add model if not auto
        if self.model and self.model != "auto":
            cmd.extend(["-m", self.model])

        # Use YOLO mode for non-interactive execution
        cmd.append("-y")  # Auto-accept all actions

        # Prompt is a positional argument, not a subcommand
        cmd.append(prompt)

        return cmd


class ClaudeExecutor(BaseCLIExecutor):
    """Execute tasks via Claude CLI (Anthropic)."""

    CLI_NAME = "claude"
    DEFAULT_MODEL = "claude-opus-4.5"

    def __init__(
        self,
        model: str = "claude-opus-4.5",
        timeout: int = 1800,
        print_mode: bool = True,
    ):
        super().__init__(model=model, timeout=timeout)
        self.print_mode = print_mode

    def _build_command(self, prompt: str, cwd: str) -> list[str]:
        """Build the Claude CLI command."""
        cmd = ["claude"]

        # Use -p for non-interactive mode (short form of --print)
        if self.print_mode:
            cmd.append("-p")

        # Add model
        if self.model:
            cmd.extend(["--model", self.model])

        # Prompt is a positional argument
        cmd.append(prompt)

        return cmd


def create_executor(
    backend: Union[str, CLIBackend],
    model: Optional[str] = None,
    **kwargs
) -> BaseCLIExecutor:
    """Factory function to create the appropriate executor.

    Args:
        backend: CLI backend to use (codex, gemini, claude)
        model: Model to use (optional, uses default if not specified)
        **kwargs: Additional arguments passed to executor

    Returns:
        Configured executor instance
    """
    if isinstance(backend, str):
        backend = CLIBackend(backend.lower())

    executors = {
        CLIBackend.CODEX: CodexExecutor,
        CLIBackend.GEMINI: GeminiExecutor,
        CLIBackend.CLAUDE: ClaudeExecutor,
    }

    executor_class = executors.get(backend)
    if not executor_class:
        raise ValueError(f"Unknown backend: {backend}")

    if model:
        kwargs["model"] = model

    return executor_class(**kwargs)


# Convenience aliases
def codex_executor(**kwargs) -> CodexExecutor:
    """Create a Codex executor with default settings."""
    return CodexExecutor(**kwargs)


def gemini_executor(**kwargs) -> GeminiExecutor:
    """Create a Gemini executor with default settings."""
    return GeminiExecutor(**kwargs)


def claude_executor(**kwargs) -> ClaudeExecutor:
    """Create a Claude executor with default settings."""
    return ClaudeExecutor(**kwargs)
