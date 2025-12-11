# Claude Subagent Spawning - Quick Reference

## TL;DR

```python
from um_agent_coder.orchestrator import ClaudeCodeSubagentSpawner

# Create spawner
spawner = ClaudeCodeSubagentSpawner(verbose=True)

# Spawn an agent
result = spawner.spawn_explore_agent(
    query="Find all API endpoints",
    files_pattern="**/*.py"
)

print(result.output if result.success else result.error)
```

## Common Patterns

### 1. Explore Codebase

```python
spawner = ClaudeCodeSubagentSpawner()
result = spawner.spawn_explore_agent(
    query="Find all uses of the OpenAI API",
    files_pattern="**/*.py",
    max_iterations=10
)
```

### 2. Review Code

```python
result = spawner.spawn_code_reviewer(
    files=["src/agent/agent.py"],
    focus="Check error handling",
    review_criteria=["Exception handling", "Logging"]
)
```

### 3. Design Architecture

```python
result = spawner.spawn_architect_agent(
    task="Design a plugin system",
    constraints=["Hot-reload support", "Backward compatible"]
)
```

### 4. Debug Issues

```python
result = spawner.spawn_debugger_agent(
    issue_description="API timeout errors",
    files_to_investigate=["src/api/client.py"],
    error_logs="Timeout after 30s..."
)
```

### 5. Generate Tests

```python
result = spawner.spawn_tester_agent(
    code_to_test="src/agent/agent.py",
    test_type="unit",
    coverage_requirements="80%"
)
```

### 6. Generate Documentation

```python
result = spawner.spawn_documenter_agent(
    target="src/llm/base.py",
    doc_type="api",
    audience="developers"
)
```

### 7. Generic Task

```python
result = spawner.spawn_task(
    prompt="Analyze project structure",
    subagent_type=SubagentType.EXPLORE,
    context={"project": "um-agent-coder"}
)
```

## With ParallelExecutor

### Enable Claude Code Spawning

```python
from um_agent_coder.orchestrator import (
    ParallelExecutor,
    ExecutionMode
)

executor = ParallelExecutor(
    execution_mode=ExecutionMode.CLAUDE_CODE_SPAWN,
    use_claude_code_spawner=True,
    verbose=True
)

result = executor.execute(decomposed_task)
```

### Comparison of Execution Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `SEQUENTIAL` | One at a time | Debugging, simple tasks |
| `PARALLEL_THREADS` | ThreadPoolExecutor | In-process parallelism |
| `PARALLEL_ASYNC` | asyncio | Async I/O tasks |
| `SUBAGENT_SPAWN` | Legacy subprocess | Process isolation (old) |
| `CLAUDE_CODE_SPAWN` | Claude Code spawner | Process isolation (new) |

## Agent Types

| Type | Backend | Use For |
|------|---------|---------|
| `EXPLORE` | Gemini | Search, large context |
| `CODE_REVIEWER` | Codex | Code quality checks |
| `ARCHITECT` | Claude | Design, planning |
| `DEBUGGER` | Codex | Bug fixes |
| `TESTER` | Codex | Test generation |
| `DOCUMENTER` | Gemini | Documentation |
| `GENERIC` | Claude | General tasks |

## Configuration Options

```python
spawner = ClaudeCodeSubagentSpawner(
    use_task_tool=True,           # Try Task tool first
    fallback_to_subprocess=True,  # Fall back if unavailable
    working_directory=None,       # Default: os.getcwd()
    verbose=False,                # Print progress
    checkpoint_dir=".subagent_results"  # Results storage
)
```

## Result Structure

```python
class SubagentResult:
    success: bool              # True if succeeded
    output: Any               # Agent output
    error: Optional[str]      # Error message if failed
    subagent_type: str        # Type of agent
    started_at: str           # ISO timestamp
    completed_at: str         # ISO timestamp
    duration_seconds: float   # Execution time
    metadata: Dict            # Additional info
```

## Error Handling

```python
result = spawner.spawn_task(...)

if result.success:
    print(f"Success: {result.output}")
else:
    print(f"Failed: {result.error}")
    # Handle error gracefully
```

## Recommended Timeouts

| Task Type | Timeout (seconds) |
|-----------|-------------------|
| Quick search | 60-180 |
| Code review | 300-600 |
| Architecture | 600-1200 |
| Full exploration | 300-900 |
| Test generation | 300-600 |

## Examples Location

Full examples: `/home/umwai/um-agent-coder/examples/claude_subagent_example.py`

```bash
# Run all examples
python examples/claude_subagent_example.py

# Run specific example
python examples/claude_subagent_example.py 1
```

## Files Modified/Created

### New Files
- `src/um_agent_coder/orchestrator/claude_subagent.py` - Main implementation
- `docs/claude_subagent_spawning.md` - Full documentation
- `docs/claude_subagent_quick_reference.md` - This file
- `examples/claude_subagent_example.py` - Usage examples

### Modified Files
- `src/um_agent_coder/orchestrator/parallel_executor.py` - Added Claude Code spawn mode
- `src/um_agent_coder/orchestrator/__init__.py` - Export new classes

## Key Features

1. **Native Task Tool Support** (with fallback)
2. **Specialized Agent Types** (7 types)
3. **Prompt Formatting** (type-specific)
4. **Result Collection** (structured)
5. **Context Passing** (between agents)
6. **Parallel Execution** (via ParallelExecutor)
7. **Error Handling** (graceful failures)
8. **Checkpointing** (result persistence)

## Integration Points

```python
# Standalone usage
from um_agent_coder.orchestrator import ClaudeCodeSubagentSpawner
spawner = ClaudeCodeSubagentSpawner()

# With ParallelExecutor
from um_agent_coder.orchestrator import ParallelExecutor, ExecutionMode
executor = ParallelExecutor(
    execution_mode=ExecutionMode.CLAUDE_CODE_SPAWN
)

# With TaskDecomposer
from um_agent_coder.orchestrator import TaskDecomposer
decomposer = TaskDecomposer()
task = decomposer.decompose("...")
result = executor.execute(task)
```
