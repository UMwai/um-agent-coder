import shutil
import subprocess
from typing import Any

from um_agent_coder.tools.base import Tool, ToolResult


class CLIAgentTool(Tool):
    """
    Base class for tools that wrap a conversational CLI agent (e.g., claude, gemini).
    """

    def __init__(self, name: str, description: str, executable: str, args: list[str] = None):
        self.executable = executable
        self.base_args = args or []
        # Parent init handles name/description if we don't override, but here we want custom ones
        self._custom_name = name
        self._custom_desc = description
        super().__init__()
        self.name = self._custom_name
        self.description = self._custom_desc
        self._check_installed()

    def _check_installed(self):
        if not shutil.which(self.executable):
            print(f"Warning: '{self.executable}' not found in PATH. {self.name} may not work.")

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The command or prompt to send to the CLI agent.",
                },
                "context": {"type": "string", "description": "Optional context for the request."},
            },
            "required": ["prompt"],
        }

    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the CLI tool with the given prompt.
        """
        prompt = kwargs.get("prompt")
        context = kwargs.get("context")

        if not prompt:
            return ToolResult(success=False, data=None, error="Prompt is required")

        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nUser Request: {prompt}"

        cmd = [self.executable] + self.base_args + [full_prompt]

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                return ToolResult(
                    success=False, data=stderr.strip(), error=f"Exit code {process.returncode}"
                )

            return ToolResult(success=True, data=stdout.strip())
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ClaudeCodeTool(CLIAgentTool):
    def __init__(self):
        super().__init__(
            name="claude_code",
            description="Interacts with Anthropic's 'claude' CLI for coding tasks.",
            executable="claude",
            args=["--print"],
        )


class GeminiCLITool(CLIAgentTool):
    def __init__(self):
        super().__init__(
            name="gemini_cli",
            description="Interacts with Google's 'gemini' CLI.",
            executable="gemini",
            args=["prompt"],
        )


class CodexTool(CLIAgentTool):
    def __init__(self):
        super().__init__(
            name="codex_cli",
            description="Interacts with the local 'codex' CLI.",
            executable="codex",
            args=[],
        )
