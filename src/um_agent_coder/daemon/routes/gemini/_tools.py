"""Simulated tool definitions and registry for agent tool-use loop.

Since Code Assist API has no native function calling, tools are prompt-simulated.
The agent uses THOUGHT/ACTION/ANSWER format, and the server parses and executes tools.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolDef:
    """Definition of a tool available to the agent."""

    name: str
    description: str
    parameters: Dict[str, str]  # param_name -> description
    handler: Callable[..., str]


def _tool_file_read(path: str, max_lines: int = "100") -> str:
    """Read a file's contents."""
    try:
        max_lines = int(max_lines)
    except (ValueError, TypeError):
        max_lines = 100
    try:
        path = os.path.expanduser(path)
        with open(path) as f:
            lines = f.readlines()[:max_lines]
        return "".join(lines)
    except Exception as e:
        return f"Error reading {path}: {e}"


def _tool_file_list(path: str = ".", pattern: str = "*") -> str:
    """List files in a directory."""
    try:
        path = os.path.expanduser(path)
        entries = sorted(os.listdir(path))
        # Simple glob filtering
        if pattern != "*":
            import fnmatch

            entries = [e for e in entries if fnmatch.fnmatch(e, pattern)]
        return "\n".join(entries[:100]) or "(empty directory)"
    except Exception as e:
        return f"Error listing {path}: {e}"


def _tool_web_search(query: str) -> str:
    """Simulate web search (returns placeholder — real impl would use an API)."""
    return (
        f"[Web search for: {query}]\n"
        "Note: Web search is simulated. In production, integrate a search API.\n"
        f"Query: {query}"
    )


def _tool_code_execute(code: str, language: str = "python") -> str:
    """Execute code in a sandboxed subprocess."""
    if language != "python":
        return f"Only Python execution is supported, got: {language}"
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            cwd="/tmp",
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (30s limit)"
    except Exception as e:
        return f"Error executing code: {e}"


def _tool_summarize(text: str, max_length: str = "200") -> str:
    """Truncate/summarize text to a max length."""
    try:
        max_length = int(max_length)
    except (ValueError, TypeError):
        max_length = 200
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... (truncated)"


# --- Tool Registry ---

TOOL_REGISTRY: Dict[str, ToolDef] = {
    "file_read": ToolDef(
        name="file_read",
        description="Read the contents of a file",
        parameters={"path": "File path to read", "max_lines": "Max lines to read (default 100)"},
        handler=_tool_file_read,
    ),
    "file_list": ToolDef(
        name="file_list",
        description="List files in a directory",
        parameters={
            "path": "Directory path (default '.')",
            "pattern": "Glob pattern (default '*')",
        },
        handler=_tool_file_list,
    ),
    "web_search": ToolDef(
        name="web_search",
        description="Search the web for information",
        parameters={"query": "Search query string"},
        handler=_tool_web_search,
    ),
    "code_execute": ToolDef(
        name="code_execute",
        description="Execute Python code and return the output",
        parameters={
            "code": "Python code to execute",
            "language": "Programming language (only 'python' supported)",
        },
        handler=_tool_code_execute,
    ),
    "summarize": ToolDef(
        name="summarize",
        description="Summarize or truncate text to a shorter form",
        parameters={"text": "Text to summarize", "max_length": "Max characters (default 200)"},
        handler=_tool_summarize,
    ),
}


def get_tools(subset: Optional[List[str]] = None) -> Dict[str, ToolDef]:
    """Get tools, optionally filtered to a subset."""
    if subset is None:
        return TOOL_REGISTRY
    return {k: v for k, v in TOOL_REGISTRY.items() if k in subset}


def format_tools_for_prompt(tools: Dict[str, ToolDef]) -> str:
    """Format tool definitions for inclusion in a system prompt."""
    lines = ["Available tools:"]
    for name, tool in tools.items():
        params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
        lines.append(f"  - {name}({params}): {tool.description}")
    return "\n".join(lines)


def execute_tool(tool_name: str, args: Dict[str, Any], tools: Dict[str, ToolDef]) -> str:
    """Execute a tool by name with the given arguments."""
    tool = tools.get(tool_name)
    if not tool:
        return f"Error: Unknown tool '{tool_name}'. Available: {', '.join(tools.keys())}"
    try:
        return tool.handler(**args)
    except TypeError as e:
        return f"Error calling {tool_name}: {e}"
    except Exception as e:
        return f"Error in {tool_name}: {e}"
