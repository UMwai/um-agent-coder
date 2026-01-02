# um-agent-coder

A multi-model AI coding agent with parallel execution, task decomposition, and orchestration for **long-running tasks**.

## Features

- **Multi-Model Orchestration**: Route tasks to Gemini (research), Codex (code), and Claude (synthesis)
- **Parallel Execution**: Execute independent subtasks concurrently with dependency tracking
- **Task Decomposition**: Break complex tasks into structured subtasks with model assignments
- **Subagent Spawning**: Spawn isolated subagent processes for true parallelization
- **Checkpointing**: Durable task state for pause/resume capabilities
- **Data Fetchers**: Built-in integrations for SEC EDGAR, Yahoo Finance, ClinicalTrials.gov, News APIs
- **MCP Integration**: Direct MCP tool invocation matching Claude Code patterns (no API keys needed)

## Installation

### Via pip (recommended)

```bash
# Install from GitHub
pip install git+https://github.com/UMwai/um-agent-coder.git

# Or with optional dependencies for specific LLM providers
pip install "um-agent-coder[openai] @ git+https://github.com/UMwai/um-agent-coder.git"
pip install "um-agent-coder[anthropic] @ git+https://github.com/UMwai/um-agent-coder.git"
pip install "um-agent-coder[all] @ git+https://github.com/UMwai/um-agent-coder.git"
```

### From source (development)

```bash
# Clone the repository
git clone https://github.com/UMwai/um-agent-coder.git
cd um-agent-coder

# Install in editable mode
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

## Quick Start

### CLI Usage

After installation, you can use the `um-agent` command:

```bash
# Simple task execution
um-agent "your task here"

# Multi-model orchestration with parallel execution
um-agent --orchestrate --parallel "analyze biotech M&A opportunities"

# Subagent process-based execution (true isolation)
um-agent --orchestrate --parallel --exec-mode subagent "complex task"

# With human approval checkpoints
um-agent --orchestrate --parallel --human-approval "task requiring review"

# Show all options
um-agent --help
```

### Python API

```python
from um_agent_coder import (
    MultiModelOrchestrator,
    ParallelExecutor,
    TaskDecomposer,
    MCPLocalLLM,
    ExecutionMode
)

# Create models (uses local MCP tools - no API keys needed)
gemini = MCPLocalLLM(backend="gemini")
codex = MCPLocalLLM(backend="codex")
claude = MCPLocalLLM(backend="claude")

# Create orchestrator
orchestrator = MultiModelOrchestrator(
    gemini=gemini, codex=codex, claude=claude
)

# Run a complex task
result = orchestrator.run("your complex task here")
print(result["output"])
```

## Using from Another Repository

If you're in a different repo and want to use um-agent-coder for long-running tasks:

### Option 1: Install as Package (Recommended)

```bash
# From your project directory
pip install git+https://github.com/UMwai/um-agent-coder.git

# Then use in Python
from um_agent_coder import (
    MultiModelOrchestrator,
    TaskDecomposer,
    ParallelExecutor,
    ClaudeCodeSubagentSpawner,
    MCPLocalLLM,
    ExecutionMode
)
```

### Option 2: Add to requirements.txt

```txt
# In your requirements.txt
um-agent-coder @ git+https://github.com/UMwai/um-agent-coder.git
```

### Option 3: Add as Git Submodule

```bash
# In your project repo
git submodule add https://github.com/UMwai/um-agent-coder.git vendor/um-agent-coder
pip install -e vendor/um-agent-coder
```

## Running Long-Running Tasks

### Example: Complex Research Task

```python
import sys
sys.path.insert(0, '/path/to/um-agent-coder/src')

from um_agent_coder.orchestrator import (
    MultiModelOrchestrator,
    ParallelExecutor,
    ExecutionMode
)
from um_agent_coder.llm.providers.mcp_local import MCPLocalLLM

# Create model instances (uses local MCP tools - NO API keys needed)
gemini = MCPLocalLLM(backend="gemini", model="gemini-3-pro")
codex = MCPLocalLLM(backend="codex", model="gpt-5.2")
claude = MCPLocalLLM(backend="claude", model="claude-opus-4.5")

# Create orchestrator
orchestrator = MultiModelOrchestrator(
    gemini=gemini,
    codex=codex,
    claude=claude,
    checkpoint_dir=".my_task_checkpoints",  # For pause/resume
    verbose=True
)

# Run a complex, long-running task
result = orchestrator.run(
    "Analyze biotech companies for M&A opportunities, including pipeline analysis and financial screening"
)

print(result["output"])
```

### Example: Parallel Subagent Execution

```python
from um_agent_coder.orchestrator import (
    ParallelExecutor,
    TaskDecomposer,
    ExecutionMode
)
from um_agent_coder.llm.providers.mcp_local import MCPLocalLLM

# Setup models
gemini = MCPLocalLLM(backend="gemini")
codex = MCPLocalLLM(backend="codex")
claude = MCPLocalLLM(backend="claude")

# Decompose a complex task
decomposer = TaskDecomposer(claude)
decomposed = decomposer.decompose(
    "Build a full-stack feature with API, database, and frontend",
    use_llm=True
)

# Execute in parallel with subagent processes
executor = ParallelExecutor(
    gemini_llm=gemini,
    codex_llm=codex,
    claude_llm=claude,
    execution_mode=ExecutionMode.SUBAGENT_SPAWN,  # Isolated processes
    max_workers=4,
    verbose=True
)

result = executor.execute(decomposed)
```

### Example: Resumable Long-Running Task

```python
from um_agent_coder.agent.enhanced_agent import EnhancedAgent
from um_agent_coder.llm.providers.mcp_local import MCPLocalLLM

# Create agent with checkpointing
agent = EnhancedAgent(
    llm=MCPLocalLLM(backend="claude"),
    checkpoint_dir=".agent_checkpoints"
)

# Start a long task
task_id = agent.execute("Refactor the entire codebase for better modularity")

# Later, if interrupted, resume:
agent.resume(task_id)

# List all tracked tasks
tasks = agent.list_tasks()
for task in tasks:
    print(f"{task['task_id']}: {task['status']}")
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--orchestrate` | Enable multi-model task decomposition |
| `--parallel` | Enable parallel execution of subtasks |
| `--exec-mode` | `sequential`, `threads`, `async`, `subagent` |
| `--human-approval` | Require human approval at checkpoints |
| `--verbose` | Print detailed progress |
| `--decompose-only` | Show task decomposition without executing |

## How It Works

1. **Task Decomposition**: Complex tasks are broken into subtasks with model assignments
   - Gemini: Research, large context analysis, exploration
   - Codex: Code generation, implementation, planning
   - Claude: Synthesis, judgment, final review

2. **Dependency Graph**: Subtasks are organized into parallel execution groups

3. **Parallel Execution**: Independent tasks run concurrently (threads, async, or subagent processes)

4. **Checkpointing**: State is saved after each step for pause/resume

5. **Result Aggregation**: Outputs flow between dependent tasks

## Project Structure

```
src/um_agent_coder/
├── orchestrator/               # Multi-model orchestration
│   ├── task_decomposer.py      # Breaks tasks into subtasks
│   ├── parallel_executor.py    # Parallel execution engine
│   ├── claude_subagent.py      # Subagent process spawning
│   ├── multi_model.py          # Pipeline orchestration
│   └── data_fetchers.py        # SEC, Yahoo Finance, etc.
├── persistence/                # Checkpointing for durability
├── llm/providers/mcp_local.py  # MCP-based LLM provider
└── agent/                      # Core agent with planning

docs/
├── mcp_tool_usage.md           # MCP integration guide
├── claude_subagent_spawning.md # Subagent guide
└── claude_subagent_quick_reference.md

examples/
├── mcp_orchestration_example.py
└── claude_subagent_example.py
```

## Ralph Loop - Autonomous Iterative Execution

The Ralph Loop enables tasks to iterate autonomously until completion criteria are met. Perfect for complex tasks that require multiple refinement cycles.

### Quick Start

Define a ralph-enabled task in your roadmap:

```markdown
## Tasks

- [ ] **feat-001**: Implement user auth with tests. Output <promise>AUTH_COMPLETE</promise> when done.
  - ralph: true
  - max_iterations: 30
  - completion_promise: AUTH_COMPLETE
  - success: Login/logout work, JWT tokens issued, 10+ unit tests pass
  - cli: codex
```

Run the harness:

```bash
python -m src.um_agent_coder.harness --roadmap specs/roadmap.md
```

The task loops until `AUTH_COMPLETE` is detected in the output or max iterations reached.

### Configuration Options

| Property | Default | Description |
|----------|---------|-------------|
| `ralph` | false | Enable Ralph Loop for task |
| `max_iterations` | 30 | Max iterations before giving up |
| `completion_promise` | "COMPLETE" | Text to detect for completion |

### CLI Flags

```bash
# Override default iterations
python -m src.um_agent_coder.harness --ralph-default-iterations 50

# Override default promise
python -m src.um_agent_coder.harness --ralph-default-promise "TASK_DONE"
```

See [docs/ralph-loop.md](docs/ralph-loop.md) for detailed documentation and [examples/ralph-prompts/](examples/ralph-prompts/) for templates.

## Recommended Models

| Provider | Model | Use Case |
|----------|-------|----------|
| OpenAI | **gpt-5.2** | Best reasoning, complex coding tasks |
| Anthropic | **claude-opus-4.5** | Deep analysis, synthesis, review |
| Google | **gemini-3-pro** | Large context (1M+), research |
| Google | **gemini-3-flash** | Fast, cost-effective tasks |

For ChatGPT Pro subscribers, use Codex CLI with `gpt-5.2` for unlimited usage.
