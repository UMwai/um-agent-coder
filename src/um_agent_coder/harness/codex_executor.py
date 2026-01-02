"""
Codex CLI executor for the 24/7 harness.

Wraps the Codex CLI (via subprocess or MCP) to execute tasks autonomously.
Configured for gpt-5.2 with high reasoning effort.
"""

import logging
import subprocess
import time
from typing import Optional

from .models import ExecutionResult, Task

logger = logging.getLogger(__name__)


class CodexExecutor:
    """Execute tasks via Codex CLI."""

    DEFAULT_CONFIG = {
        "model": "gpt-5.2",
        "sandbox": "danger-full-access",
        "approval_policy": "never",  # Changed from 'approval-policy' to 'approval_policy'
        "model_reasoning_effort": "high",
    }

    def __init__(
        self,
        model: str = "gpt-5.2",
        reasoning_effort: str = "high",
        sandbox: str = "danger-full-access",
        approval_policy: str = "never",
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.sandbox = sandbox
        self.approval_policy = approval_policy

        # Track active conversations for continuity
        self._conversations: dict[str, str] = {}

    def execute(self, task: Task, context: str = "") -> ExecutionResult:
        """Execute a task via Codex CLI."""
        start_time = time.time()

        # Build the prompt for Codex
        prompt = self._build_prompt(task, context)

        try:
            # Check if we have an existing conversation to continue
            if task.conversation_id and task.conversation_id in self._conversations:
                result = self._continue_conversation(task.conversation_id, prompt, task.cwd)
            else:
                result = self._start_conversation(prompt, task.cwd)

            duration = time.time() - start_time

            # Store conversation ID for potential continuation
            if result.conversation_id:
                self._conversations[task.id] = result.conversation_id
                task.conversation_id = result.conversation_id

            return ExecutionResult(
                success=result.success,
                output=result.output,
                error=result.error,
                conversation_id=result.conversation_id,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"Codex execution failed for task {task.id}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                duration_seconds=duration,
            )

    def _build_prompt(self, task: Task, context: str = "") -> str:
        """Build a structured prompt for Codex."""
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

    def _start_conversation(self, prompt: str, cwd: str) -> ExecutionResult:
        """Start a new Codex conversation."""
        cmd = self._build_codex_command(prompt, cwd)

        logger.info(f"Starting Codex: {' '.join(cmd[:5])}...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd if cwd != "./" else None,
            timeout=1800,  # 30 minute timeout
        )

        output = result.stdout
        error = result.stderr

        # Try to extract conversation ID from output for continuation
        conversation_id = self._extract_conversation_id(output)

        # Determine success based on exit code and output analysis
        success = result.returncode == 0 and not self._detect_failure(output, error)

        return ExecutionResult(
            success=success,
            output=output,
            error=error,
            conversation_id=conversation_id,
        )

    def _continue_conversation(
        self, conversation_id: str, prompt: str, cwd: str
    ) -> ExecutionResult:
        """Continue an existing Codex conversation."""
        # Use codex with conversation continuation
        cmd = [
            "codex",
            "-m", self.model,
            "--conversation", conversation_id,
            prompt,
        ]

        logger.info(f"Continuing Codex conversation {conversation_id}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd if cwd != "./" else None,
            timeout=1800,
        )

        success = result.returncode == 0 and not self._detect_failure(
            result.stdout, result.stderr
        )

        return ExecutionResult(
            success=success,
            output=result.stdout,
            error=result.stderr,
            conversation_id=conversation_id,
        )

    def _build_codex_command(self, prompt: str, cwd: str) -> list[str]:
        """Build the codex CLI command."""
        cmd = [
            "codex",
            "-m", self.model,
            "-a", self.approval_policy,  # Changed from --approval-policy to -a
        ]

        # Add sandbox mode
        if self.sandbox:
            cmd.extend(["-s", self.sandbox])  # Changed --sandbox to -s

        # Add reasoning effort config
        cmd.extend(["-c", f"model_reasoning_effort={self.reasoning_effort}"])

        # Add working directory if specified
        if cwd and cwd != "./":
            cmd.extend(["-C", cwd])  # Changed --cwd to -C (--cd)

        # Add the prompt
        cmd.append(prompt)

        return cmd

    def _extract_conversation_id(self, output: str) -> Optional[str]:
        """Try to extract conversation ID from Codex output."""
        # Codex may output conversation ID in various formats
        # This is a placeholder - actual format depends on Codex version
        import re

        # Look for common patterns
        patterns = [
            r'conversation[_-]?id[:\s]+([a-zA-Z0-9-]+)',
            r'session[:\s]+([a-zA-Z0-9-]+)',
            r'"conversationId":\s*"([^"]+)"',
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _detect_failure(self, output: str, error: str) -> bool:
        """Analyze output to detect task failure."""
        failure_indicators = [
            "error:",
            "failed:",
            "could not complete",
            "unable to",
            "permission denied",
            "not found",
            "exception:",
            "traceback",
        ]

        combined = (output + error).lower()

        for indicator in failure_indicators:
            if indicator in combined:
                # Check if it's in a context that suggests actual failure
                # vs just mentioning errors in output
                if f"task {indicator}" in combined or combined.count(indicator) > 2:
                    return True

        return False

    def verify_success(self, task: Task, result: ExecutionResult) -> bool:
        """Verify that the task's success criteria were met."""
        if not task.success_criteria:
            return result.success

        # Use Codex to verify the success criteria
        verify_prompt = f"""
Verify if the following success criteria was met:

Criteria: {task.success_criteria}

Task output:
{result.output[:2000]}

Respond with only "YES" or "NO" followed by a brief explanation.
"""

        verify_result = self._start_conversation(verify_prompt, task.cwd)

        return "yes" in verify_result.output.lower()[:50]


class CodexMCPExecutor(CodexExecutor):
    """
    Execute tasks via Codex MCP tool.

    This variant uses the MCP protocol instead of direct CLI calls,
    which may be preferred when running within an MCP-aware environment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mcp_available = self._check_mcp_available()

    def _check_mcp_available(self) -> bool:
        """Check if MCP tools are available."""
        # In MCP environment, these would be injected
        # For standalone use, fall back to CLI
        return False  # Default to CLI for now

    def execute(self, task: Task, context: str = "") -> ExecutionResult:
        """Execute via MCP if available, otherwise CLI."""
        if self._mcp_available:
            return self._execute_mcp(task, context)
        return super().execute(task, context)

    def _execute_mcp(self, task: Task, context: str = "") -> ExecutionResult:
        """Execute via MCP protocol."""
        # This would be implemented when running within Claude Code
        # or another MCP-aware environment
        raise NotImplementedError("MCP execution not yet implemented")
