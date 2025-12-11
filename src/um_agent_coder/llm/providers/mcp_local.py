"""
MCP-based LLM provider for local tools (Codex, Gemini CLI, Claude).

This provider routes requests to locally available MCP tools:
- Codex (OpenAI GPT-5/O3/O4-mini)
- Gemini CLI (Google Gemini 2.5 Pro/Flash)
- Claude (via claude CLI or API)

These tools run locally without requiring API keys when used through
Claude Code's MCP infrastructure.
"""

import subprocess
import json
import os
from typing import Dict, Any, Optional, List
from ..base import LLM


class MCPLocalLLM(LLM):
    """
    LLM provider that uses local MCP tools (Codex, Gemini, Claude).

    Supports three backends:
    - codex: Uses OpenAI Codex MCP server
    - gemini: Uses Gemini CLI MCP server
    - claude: Uses Claude CLI directly

    Usage:
        llm = MCPLocalLLM(backend="gemini", model="gemini-2.5-pro")
        response = llm.chat("Explain this code")
    """

    BACKENDS = {
        "codex": {
            "models": ["o3", "o4-mini", "gpt-4o"],
            "default_model": "o4-mini",
            "cost_per_1k_input": 0.0,  # Local, no cost
            "cost_per_1k_output": 0.0,
        },
        "gemini": {
            "models": ["gemini-3-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash"],
            "default_model": "gemini-3-pro-preview",
            "cost_per_1k_input": 0.0,
            "cost_per_1k_output": 0.0,
        },
        "claude": {
            "models": ["claude-sonnet", "claude-opus", "claude-haiku"],
            "default_model": "claude-sonnet",
            "cost_per_1k_input": 0.0,
            "cost_per_1k_output": 0.0,
        }
    }

    def __init__(
        self,
        backend: str = "gemini",
        model: Optional[str] = None,
        cwd: Optional[str] = None,
        sandbox: str = "read-only",
        **kwargs
    ):
        """
        Initialize MCP Local LLM provider.

        Args:
            backend: Which backend to use (codex, gemini, claude)
            model: Specific model to use (defaults to backend's default)
            cwd: Working directory for commands
            sandbox: Sandbox mode for codex (read-only, workspace-write, danger-full-access)
        """
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. Choose from: {list(self.BACKENDS.keys())}")

        self.backend = backend
        self.model = model or self.BACKENDS[backend]["default_model"]
        self.cwd = cwd or os.getcwd()
        self.sandbox = sandbox
        self.conversation_id = None  # For codex continuation

    def chat(self, prompt: str) -> str:
        """
        Send a prompt to the local MCP backend.

        Args:
            prompt: The prompt to send

        Returns:
            Response from the LLM
        """
        if self.backend == "codex":
            return self._chat_codex(prompt)
        elif self.backend == "gemini":
            return self._chat_gemini(prompt)
        elif self.backend == "claude":
            return self._chat_claude(prompt)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _chat_codex(self, prompt: str) -> str:
        """Chat using Codex MCP server."""
        # This would ideally use the MCP tool directly
        # For now, we'll format a request that can be used with the MCP

        # Build the request payload
        request = {
            "prompt": prompt,
            "cwd": self.cwd,
            "sandbox": self.sandbox,
            "approval-policy": "never",
        }

        if self.model:
            request["model"] = self.model

        if self.conversation_id:
            request["conversationId"] = self.conversation_id

        # Return formatted request for MCP tool usage
        # In actual MCP context, this would be called via mcp__codex__codex
        return self._execute_mcp_codex(request)

    def _chat_gemini(self, prompt: str) -> str:
        """Chat using Gemini CLI MCP server."""
        request = {
            "prompt": prompt,
        }

        if self.model:
            request["model"] = self.model

        return self._execute_mcp_gemini(request)

    def _chat_claude(self, prompt: str) -> str:
        """Chat using Claude CLI."""
        # Use claude CLI directly
        try:
            result = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=self.cwd
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Error: {result.stderr}"
        except FileNotFoundError:
            return "Error: claude CLI not found. Please install it."
        except subprocess.TimeoutExpired:
            return "Error: Request timed out"

    def _execute_mcp_codex(self, request: Dict[str, Any]) -> str:
        """
        Execute Codex via MCP.

        In the Claude Code environment, this would be called via:
        mcp__codex__codex(prompt=..., sandbox=..., approval-policy=...)

        For standalone usage, we fall back to subprocess.
        """
        try:
            # Use codex exec for non-interactive execution
            cmd = ["codex", "exec", "--json"]
            if request.get("model"):
                cmd.extend(["--model", request["model"]])
            if request.get("sandbox"):
                cmd.extend(["--sandbox", request["sandbox"]])

            # Add the prompt at the end
            cmd.append(request["prompt"])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=request.get("cwd", self.cwd)
            )

            if result.returncode == 0:
                # Parse JSONL output - last message typically has the response
                lines = result.stdout.strip().split('\n')
                for line in reversed(lines):
                    try:
                        data = json.loads(line)
                        if data.get("type") == "message" and data.get("content"):
                            return data["content"]
                        if data.get("response"):
                            return data["response"]
                    except json.JSONDecodeError:
                        continue
                return result.stdout.strip() or "Codex completed"
            else:
                return f"Error: {result.stderr}"

        except FileNotFoundError:
            # Return a placeholder that indicates MCP should be used
            return f"[MCP_CODEX_REQUEST]{json.dumps(request)}"
        except subprocess.TimeoutExpired:
            return "Error: Codex request timed out"

    def _execute_mcp_gemini(self, request: Dict[str, Any]) -> str:
        """
        Execute Gemini via MCP.

        In the Claude Code environment, this would be called via:
        mcp__gemini-cli__ask-gemini(prompt=..., model=...)

        For standalone usage, we fall back to subprocess.
        """
        try:
            # Try using gemini CLI if available
            cmd = ["gemini", request["prompt"]]
            if request.get("model"):
                cmd.extend(["-m", request["model"]])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=self.cwd
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Error: {result.stderr}"

        except FileNotFoundError:
            # Return a placeholder that indicates MCP should be used
            return f"[MCP_GEMINI_REQUEST]{json.dumps(request)}"
        except subprocess.TimeoutExpired:
            return "Error: Gemini request timed out"

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        backend_info = self.BACKENDS[self.backend]
        return {
            "backend": self.backend,
            "model": self.model,
            "cost_per_1k_input": backend_info["cost_per_1k_input"],
            "cost_per_1k_output": backend_info["cost_per_1k_output"],
            "available_models": backend_info["models"],
        }

    def set_conversation_id(self, conversation_id: str):
        """Set conversation ID for Codex continuation."""
        self.conversation_id = conversation_id


class MCPOrchestrator:
    """
    Orchestrates multiple MCP backends for task decomposition.

    Following the pattern from CLAUDE.md:
    - Gemini: Large context gathering, exploration, research
    - Codex: Code generation, implementation, planning
    - Claude: Final execution, orchestration, safety review

    Usage:
        orchestrator = MCPOrchestrator()

        # Gather context with Gemini (1M token context)
        context = orchestrator.gather_context("@src/ analyze the codebase structure")

        # Plan with Codex
        plan = orchestrator.plan("Implement feature X given this context: " + context)

        # Execute steps
        for step in plan:
            result = orchestrator.execute(step)
    """

    def __init__(self, cwd: Optional[str] = None):
        self.cwd = cwd or os.getcwd()
        self.gemini = MCPLocalLLM(backend="gemini", cwd=self.cwd)
        self.codex = MCPLocalLLM(backend="codex", cwd=self.cwd, sandbox="workspace-write")
        self.claude = MCPLocalLLM(backend="claude", cwd=self.cwd)

    def gather_context(self, prompt: str) -> str:
        """
        Use Gemini for large context gathering.
        Good for: reading large files, codebase exploration, research.
        """
        return self.gemini.chat(prompt)

    def plan(self, prompt: str) -> str:
        """
        Use Codex for planning and code generation.
        Good for: implementation plans, code generation, refactoring.
        """
        return self.codex.chat(prompt)

    def execute(self, prompt: str) -> str:
        """
        Use Claude for execution and orchestration.
        Good for: tool execution, safety review, final synthesis.
        """
        return self.claude.chat(prompt)

    def route(self, prompt: str, task_type: str = "auto") -> str:
        """
        Automatically route to the best backend based on task type.

        Args:
            prompt: The prompt to process
            task_type: One of "context", "plan", "execute", or "auto"
        """
        if task_type == "auto":
            task_type = self._classify_task(prompt)

        if task_type == "context":
            return self.gather_context(prompt)
        elif task_type == "plan":
            return self.plan(prompt)
        else:
            return self.execute(prompt)

    def _classify_task(self, prompt: str) -> str:
        """Classify task type based on prompt keywords."""
        prompt_lower = prompt.lower()

        # Context gathering keywords
        if any(kw in prompt_lower for kw in ["analyze", "explore", "search", "find", "read", "@"]):
            return "context"

        # Planning keywords
        if any(kw in prompt_lower for kw in ["plan", "implement", "create", "generate", "write code"]):
            return "plan"

        # Default to execute
        return "execute"
