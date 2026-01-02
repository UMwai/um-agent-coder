import shutil
import subprocess
from typing import Any, Optional

from um_agent_coder.llm.base import LLM
from um_agent_coder.models import ModelRegistry


class ClaudeCLIProvider(LLM):
    """
    LLM provider that wraps the Anthropic 'claude' CLI tool.
    Requires 'claude' to be installed and authenticated ('claude login').
    """

    def __init__(self, model: str = "claude-3-opus-20240229", **kwargs):
        self.model = model
        self.model_registry = ModelRegistry()

        # Check if claude CLI is available
        if not shutil.which("claude"):
            raise RuntimeError(
                "The 'claude' CLI tool is not found in PATH. Please install it and run 'claude login'."
            )

    def chat(self, prompt: str, messages: Optional[list[dict[str, str]]] = None) -> str:
        """
        Send a chat request via the claude CLI.

        Note: The CLI is primarily designed for interactive use or piping.
        We will use a simple pipe approach. Context (messages) handling
        via CLI is limited unless we concatenate them.
        """

        full_prompt = prompt

        # If there is history, prepend it to the prompt to simulate context
        # The CLI doesn't natively support a "messages" array argument like the API
        if messages:
            history_text = ""
            for msg in messages:
                role = msg["role"].upper()
                content = msg["content"]
                history_text += f"{role}: {content}\n\n"

            full_prompt = f"{history_text}USER: {prompt}"

        try:
            # Run claude CLI with the prompt piped to stdin
            # Using -p / --print to just print the response
            # Note: CLI flags might vary by version. Assuming standard behavior.
            # If the CLI is 'claude', we might need to check its help.
            # Usually: echo "prompt" | claude

            process = subprocess.Popen(
                ["claude", "--print", full_prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                return f"Error calling Claude CLI: {stderr}"

            return stdout.strip()

        except Exception as e:
            return f"Error executing Claude CLI: {str(e)}"

    def stream_chat(self, prompt: str, messages: Optional[list[dict[str, str]]] = None):
        """
        Stream chat from CLI.
        """
        # Streaming from a subprocess is possible but complex to handle reliably
        # with the CLI's formatting. For now, fall back to non-streaming.
        response = self.chat(prompt, messages)
        yield response

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model."""
        # The CLI usually defaults to the best model (Opus) or whatever is configured.
        # We'll assume Opus for metadata purposes if not specified.
        model_info = self.model_registry.get(self.model)
        if model_info:
            return {
                "name": model_info.name,
                "context_window": model_info.context_window,
                "cost_per_1k_input": 0,  # Subscription based
                "cost_per_1k_output": 0,
                "capabilities": model_info.capabilities,
            }
        return {"name": self.model, "info": "Consumer Subscription (CLI)"}
