# MCP Tool Usage Guide

This guide demonstrates how to use the enhanced MCP tool integration in the `mcp_local.py` provider.

## Overview

The `MCPLocalLLM` class now supports direct MCP tool invocations that mirror Claude Code's tool patterns:

1. **mcp__gemini-cli__ask-gemini** - Query Gemini with large context
2. **mcp__gemini-cli__brainstorm** - Brainstorm ideas with Gemini
3. **mcp__codex__codex** - Execute Codex for code generation/planning

## Basic Usage

### Using Individual MCP Tools

```python
from src.um_agent_coder.llm.providers.mcp_local import MCPLocalLLM

# Initialize with a specific backend
gemini = MCPLocalLLM(backend="gemini")
codex = MCPLocalLLM(backend="codex")

# Use Gemini for context gathering
response = gemini.mcp_gemini_ask(
    "Analyze the architecture of this codebase",
    files=["src/main.py", "src/agent/agent.py"]
)

# Use Gemini for brainstorming
ideas = gemini.mcp_gemini_brainstorm(
    topic="improving the agent's planning capabilities",
    context="Focus on multi-step task decomposition"
)

# Use Codex for planning
plan = codex.mcp_codex_plan(
    task="Add support for Anthropic Claude API",
    context="Current implementation uses OpenAI pattern"
)

# Use Codex for implementation
result = codex.mcp_codex_invoke(
    prompt="Generate unit tests for the Agent class",
    sandbox="workspace-write",
    approval_policy="never",
    model="o4-mini"
)
```

## Using the Orchestrator

The `MCPOrchestrator` class provides high-level workflows that automatically route tasks to the appropriate backend:

```python
from src.um_agent_coder.llm.providers.mcp_local import MCPOrchestrator

orchestrator = MCPOrchestrator()

# Gather context with Gemini
context = orchestrator.gather_context(
    "What is the current agent architecture?",
    files=["src/agent/agent.py", "src/llm/base.py"]
)

# Brainstorm with Gemini
ideas = orchestrator.brainstorm(
    topic="new features for the agent",
    context="Keep it simple and modular"
)

# Plan with Codex
plan = orchestrator.plan(
    task="Add streaming support to LLM providers",
    context=context
)

# Implement with Codex
implementation = orchestrator.implement(
    "Implement streaming support based on the plan above",
    sandbox="workspace-write"
)
```

## Full Multi-Model Workflow

Execute a complete workflow that uses all three backends:

```python
orchestrator = MCPOrchestrator()

results = orchestrator.full_workflow(
    user_request="Add support for Anthropic Claude API",
    files_to_analyze=["src/llm/providers/openai.py", "src/llm/base.py"]
)

# Results contain:
# - results["context"] - Context gathered by Gemini
# - results["plan"] - Implementation plan from Codex
# - results["implementation"] - Implementation from Codex
# - results["review"] - Review from Claude
```

## Automatic Task Routing

The orchestrator can automatically route tasks based on keywords:

```python
orchestrator = MCPOrchestrator()

# Automatically routes to Gemini for analysis
result = orchestrator.route("analyze the codebase structure")

# Automatically routes to Codex for planning
result = orchestrator.route("plan how to add new features")

# Automatically routes to Codex for implementation
result = orchestrator.route("implement the new feature")

# Automatically routes to Claude for execution
result = orchestrator.route("review and test the changes")
```

## MCP Tool Parameters

### Gemini Tools

**mcp_gemini_ask**:
- `prompt` (str): The query or prompt
- `model` (Optional[str]): Model override (default: gemini-3-pro-preview)
- `files` (Optional[List[str]]): File paths to include with @ syntax

**mcp_gemini_brainstorm**:
- `topic` (str): Topic to brainstorm about
- `context` (Optional[str]): Additional constraints or context
- `model` (Optional[str]): Model override

### Codex Tools

**mcp_codex_invoke**:
- `prompt` (str): Task description or instructions
- `model` (Optional[str]): Model choice (o3, o4-mini, gpt-4o)
- `sandbox` (Optional[str]): Sandbox mode (read-only, workspace-write, danger-full-access)
- `approval_policy` (str): Approval policy (never, auto, always) - default: "never"
- `conversation_id` (Optional[str]): Conversation ID for continuation
- `cwd` (Optional[str]): Working directory override

**mcp_codex_plan**:
- `task` (str): The task to plan for
- `context` (Optional[str]): Additional context
- `model` (Optional[str]): Model override

## Integration Patterns

### Pattern 1: Gemini -> Codex -> Claude

```python
# 1. Gather context with Gemini (1M token window)
context = orchestrator.gather_context(
    "@src/ analyze all Python files",
    files=["src/**/*.py"]
)

# 2. Plan with Codex
plan = orchestrator.plan(
    "Refactor the codebase for better modularity",
    context=context
)

# 3. Execute with Claude
result = orchestrator.execute(
    f"Execute this plan: {plan}"
)
```

### Pattern 2: Parallel Review

```python
# Get both Gemini and Codex perspectives
gemini_analysis = gemini.mcp_gemini_ask("Review this code for design issues")
codex_analysis = codex.mcp_codex_invoke("Review this code for implementation issues")

# Claude synthesizes
final_review = claude.chat(f"Synthesize these reviews:\n\nGemini: {gemini_analysis}\n\nCodex: {codex_analysis}")
```

## Notes

- All MCP tool methods handle fallback to subprocess execution if MCP is not available
- File references using `@path` syntax are supported in Gemini prompts
- Codex sandbox modes control what the model can access and modify
- The orchestrator automatically selects appropriate models and settings per backend
