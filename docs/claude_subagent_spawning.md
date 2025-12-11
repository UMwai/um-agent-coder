# Claude Code Subagent Spawning

This document describes the Claude Code subagent spawning functionality, which enables programmatic spawning of specialized Claude Code agents for different tasks.

## Overview

The `ClaudeCodeSubagentSpawner` class provides a high-level interface for spawning Claude Code subagents with:

1. **Native Task Tool Support**: Attempts to use Claude Code's native Task tool when running inside Claude Code environment
2. **Subprocess Fallback**: Falls back to subprocess execution when Task tool is unavailable
3. **Specialized Agent Types**: Pre-configured prompts for different agent types (Explore, code-reviewer, Architect, etc.)
4. **Result Collection**: Structured result handling with success/error tracking and timing information
5. **Context Passing**: Seamless context and data passing between agents

## Architecture

### Core Components

```
claude_subagent.py
├── ClaudeCodeSubagentSpawner    # Main spawner class
├── SubagentType (Enum)          # Types of agents (Explore, code-reviewer, etc.)
├── SubagentConfig (dataclass)   # Configuration for spawning
└── SubagentResult (dataclass)   # Result from spawned agents
```

### Integration with ParallelExecutor

The `ParallelExecutor` class has been enhanced with:

1. **New Execution Mode**: `ExecutionMode.CLAUDE_CODE_SPAWN`
2. **ClaudeCodeSubagentSpawner Integration**: Uses spawner for subagent execution
3. **Automatic Agent Type Mapping**: Maps ModelRole to appropriate SubagentType

## Usage

### Basic Usage

```python
from um_agent_coder.orchestrator import ClaudeCodeSubagentSpawner, SubagentType

# Create spawner
spawner = ClaudeCodeSubagentSpawner(
    use_task_tool=True,           # Try Task tool first
    fallback_to_subprocess=True,  # Fall back if unavailable
    verbose=True
)

# Spawn a generic task
result = spawner.spawn_task(
    prompt="Analyze the codebase structure",
    subagent_type=SubagentType.EXPLORE,
    context={"project_root": "/path/to/project"}
)

if result.success:
    print(f"Output: {result.output}")
else:
    print(f"Error: {result.error}")
```

### Specialized Agent Methods

#### 1. Explore Agent

Used for multi-round search and codebase exploration.

```python
result = spawner.spawn_explore_agent(
    query="Find all database queries in the codebase",
    files_pattern="**/*.py",
    max_iterations=10
)
```

#### 2. Code Reviewer Agent

Used for code review with specific criteria.

```python
result = spawner.spawn_code_reviewer(
    files=["src/agent/agent.py", "src/llm/base.py"],
    focus="Check for error handling and type safety",
    review_criteria=[
        "Exception handling",
        "Type hints",
        "Documentation",
        "Security issues"
    ]
)
```

#### 3. Architect Agent

Used for system design and architecture planning.

```python
result = spawner.spawn_architect_agent(
    task="Design a plugin system for LLM providers",
    existing_architecture="Base LLM class with provider implementations",
    constraints=[
        "Must support hot-reloading",
        "Backward compatible"
    ]
)
```

#### 4. Debugger Agent

Used for investigating and fixing issues.

```python
result = spawner.spawn_debugger_agent(
    issue_description="API calls failing with timeout errors",
    files_to_investigate=["src/api/client.py"],
    error_logs="Timeout after 30 seconds..."
)
```

#### 5. Tester Agent

Used for generating tests.

```python
result = spawner.spawn_tester_agent(
    code_to_test="src/agent/agent.py",
    test_type="unit",
    coverage_requirements="80% line coverage"
)
```

#### 6. Documenter Agent

Used for generating documentation.

```python
result = spawner.spawn_documenter_agent(
    target="src/llm/base.py",
    doc_type="api",
    audience="developers"
)
```

### Using with ParallelExecutor

The `ParallelExecutor` now supports Claude Code subagent spawning:

```python
from um_agent_coder.orchestrator import (
    ParallelExecutor,
    ExecutionMode,
    DecomposedTask
)

# Create executor with Claude Code spawner
executor = ParallelExecutor(
    execution_mode=ExecutionMode.CLAUDE_CODE_SPAWN,
    use_claude_code_spawner=True,
    claude_spawner_fallback=True,  # Fall back to subprocess
    verbose=True
)

# Execute decomposed task
result = executor.execute(decomposed_task)
```

## Agent Types

### SubagentType Enum

| Type | Value | Use Case | Recommended Model |
|------|-------|----------|-------------------|
| EXPLORE | "Explore" | Multi-round search, codebase exploration | Gemini (large context) |
| CODE_REVIEWER | "code-reviewer" | Code review, quality checks | Codex (code analysis) |
| ARCHITECT | "Architect" | System design, architecture | Claude (reasoning) |
| DEBUGGER | "Debugger" | Bug investigation and fixes | Codex (code analysis) |
| OPTIMIZER | "Optimizer" | Performance optimization | Codex (code generation) |
| TESTER | "Tester" | Test generation | Codex (code generation) |
| DOCUMENTER | "Documenter" | Documentation generation | Gemini (large context) |
| GENERIC | "Generic" | General purpose tasks | Claude (orchestration) |

## Execution Modes

The spawner supports two execution modes:

### 1. Task Tool Mode (Preferred)

When running inside Claude Code, the spawner attempts to use the native Task tool:

```python
spawner = ClaudeCodeSubagentSpawner(
    use_task_tool=True,
    fallback_to_subprocess=False  # Fail if Task tool unavailable
)
```

**Advantages:**
- Native integration with Claude Code
- Better resource management
- Direct access to Claude Code's tools

**Note:** Currently returns None and falls back (Task tool integration is a placeholder for future Claude Code support).

### 2. Subprocess Mode (Fallback)

Falls back to subprocess execution using generated Python scripts:

```python
spawner = ClaudeCodeSubagentSpawner(
    use_task_tool=False,  # Skip Task tool
    fallback_to_subprocess=True
)
```

**Advantages:**
- Works standalone without Claude Code
- Full process isolation
- Compatible with existing code

**Implementation:**
- Generates temporary Python scripts
- Spawns subprocess with timeout
- Collects stdout/stderr
- Cleans up temporary files

## Configuration

### SubagentConfig

```python
from um_agent_coder.orchestrator import SubagentConfig, SubagentType

config = SubagentConfig(
    subagent_type=SubagentType.EXPLORE,
    prompt="Find all API endpoints",
    working_directory="/path/to/project",
    timeout=600,  # seconds
    max_iterations=10,  # for explore agents
    focus_files=["src/api/*.py"],
    context={"key": "value"}
)
```

### Result Handling

```python
from um_agent_coder.orchestrator import SubagentResult

# Result structure
result = SubagentResult(
    success=True,
    output="Agent output here...",
    error=None,
    subagent_type="Explore",
    started_at="2025-01-15T10:30:00",
    completed_at="2025-01-15T10:32:30",
    duration_seconds=150.0,
    metadata={"iterations": 5}
)

# Convert to dict for serialization
result_dict = result.to_dict()
```

## Integration with Existing Code

### Model Role Mapping

The `ParallelExecutor` automatically maps `ModelRole` to `SubagentType`:

```python
MODEL_TO_SUBAGENT_TYPE = {
    ModelRole.GEMINI: SubagentType.EXPLORE,    # Large context
    ModelRole.CODEX: SubagentType.GENERIC,     # Code tasks
    ModelRole.CLAUDE: SubagentType.ARCHITECT,  # Orchestration
}
```

### Subprocess Script Generation

The spawner generates Python scripts that:

1. Set up the Python path
2. Import required modules
3. Initialize the appropriate LLM backend (based on agent type)
4. Build the prompt with context
5. Execute and return results

Example generated script:

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")

from um_agent_coder.llm.providers.mcp_local import MCPLocalLLM

# Determine backend based on agent type
backend = "gemini"  # For Explore agents

# Initialize model
model = MCPLocalLLM(backend=backend)

# Build prompt with context
prompt = """Your task prompt here"""
context = {"key": "value"}
if context:
    import json
    context_str = json.dumps(context, indent=2)
    prompt += f"\n\n--- CONTEXT ---\n{context_str}"

# Execute
try:
    result = model.chat(prompt)
    print(result)
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
```

## Examples

See `examples/claude_subagent_example.py` for complete working examples:

```bash
# Run all examples
python examples/claude_subagent_example.py

# Run specific example
python examples/claude_subagent_example.py 1  # Explore agent
python examples/claude_subagent_example.py 2  # Code reviewer
python examples/claude_subagent_example.py 5  # Parallel execution
```

## Best Practices

1. **Choose the Right Agent Type**
   - Use `EXPLORE` for broad codebase searches
   - Use `CODE_REVIEWER` for quality checks
   - Use `ARCHITECT` for design decisions
   - Use `DEBUGGER` for issue investigation

2. **Set Appropriate Timeouts**
   - Explore agents: 300-600 seconds
   - Code review: 300-900 seconds
   - Architecture: 600-1200 seconds
   - Quick tasks: 60-300 seconds

3. **Provide Good Context**
   - Pass relevant data from previous steps
   - Include file paths and patterns
   - Specify focus areas clearly

4. **Handle Errors Gracefully**
   ```python
   result = spawner.spawn_task(...)
   if not result.success:
       print(f"Error: {result.error}")
       # Handle failure
   ```

5. **Use Parallel Execution for Independence**
   - Run independent analyses in parallel
   - Use `ExecutionMode.CLAUDE_CODE_SPAWN` for isolation
   - Let ParallelExecutor handle dependencies

## Future Enhancements

1. **Native Task Tool Integration**
   - Implement actual Task tool spawning (currently placeholder)
   - Direct communication with Claude Code

2. **Advanced Agent Types**
   - Refactorer agent
   - Security auditor agent
   - Performance profiler agent

3. **Result Streaming**
   - Stream output as agents work
   - Progress callbacks
   - Partial results

4. **Agent Collaboration**
   - Multi-agent conversations
   - Shared context management
   - Consensus building

## Troubleshooting

### Subprocess Execution Fails

```python
# Enable verbose output
spawner = ClaudeCodeSubagentSpawner(verbose=True)

# Check checkpoint directory for logs
checkpoint_dir = ".subagent_results"
```

### Task Tool Not Available

```python
# Ensure fallback is enabled
spawner = ClaudeCodeSubagentSpawner(
    use_task_tool=True,
    fallback_to_subprocess=True  # Critical!
)
```

### Timeout Issues

```python
# Increase timeout for complex tasks
result = spawner.spawn_task(
    prompt="...",
    timeout=1200  # 20 minutes
)
```

## API Reference

See inline documentation in:
- `/home/umwai/um-agent-coder/src/um_agent_coder/orchestrator/claude_subagent.py`
- `/home/umwai/um-agent-coder/src/um_agent_coder/orchestrator/parallel_executor.py`
