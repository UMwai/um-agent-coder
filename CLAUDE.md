# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Agent
```bash
# Via installed CLI
um-agent "YOUR_PROMPT"
um-agent --orchestrate --parallel "complex task"

# Via module
python -m src.um_agent_coder "YOUR_PROMPT"
```

### Running the 24/7 CLI Harness
```bash
python -m src.um_agent_coder.harness --roadmap specs/roadmap.md           # Default: Codex CLI
python -m src.um_agent_coder.harness --roadmap specs/roadmap.md --cli gemini
python -m src.um_agent_coder.harness --roadmap specs/roadmap.md --cli claude
python -m src.um_agent_coder.harness --roadmap specs/roadmap.md --parallel # Parallel execution
python -m src.um_agent_coder.harness --roadmap specs/roadmap.md --parallel --max-parallel 8
python -m src.um_agent_coder.harness --dry-run                             # Preview only
python -m src.um_agent_coder.harness --daemon                              # 24/7 mode
python -m src.um_agent_coder.harness --status                              # Check status
python -m src.um_agent_coder.harness --reset                               # Reset state
```

### Development
```bash
pip install -e ".[dev]"                    # Install with dev dependencies
pytest                                     # Run all tests
pytest tests/test_router_mock.py -v        # Run single test file
black src/                                 # Format code
isort src/                                 # Sort imports
ruff check src/                            # Lint
mypy src/                                  # Type check
```

### Configuration
Config file: `config/config.yaml` (auto-created from example if missing)
Environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`

## Architecture

### Core Layers

```
src/um_agent_coder/
├── cli.py                    # CLI entry point (um-agent command)
├── config.py                 # YAML config with dot notation (e.g., llm.openai.api_key)
├── llm/
│   ├── base.py               # Abstract LLM base class
│   ├── factory.py            # LLMFactory.create(provider, config)
│   └── providers/            # openai, anthropic, google, claude_cli, google_adc, mcp_local
├── agent/
│   ├── agent.py              # Basic Agent class
│   ├── enhanced_agent.py     # EnhancedAgent with checkpointing
│   ├── planner.py            # Task planning
│   └── router.py             # Multi-agent routing
├── orchestrator/
│   ├── multi_model.py        # MultiModelOrchestrator - routes to Gemini/Codex/Claude
│   ├── task_decomposer.py    # Breaks tasks into subtasks with model assignments
│   ├── parallel_executor.py  # ExecutionMode: sequential, threads, async, subagent
│   ├── claude_subagent.py    # Spawn isolated Claude Code subagents
│   └── data_fetchers.py      # SEC EDGAR, Yahoo Finance, ClinicalTrials, News
├── persistence/
│   └── checkpointer.py       # TaskCheckpointer for pause/resume
├── harness/                  # 24/7 autonomous execution
│   ├── main.py               # Daemon loop
│   ├── roadmap_parser.py     # Parse specs/roadmap.md
│   ├── executors.py          # CodexExecutor, GeminiExecutor, ClaudeExecutor
│   ├── state.py              # SQLite persistence (.harness/state.db)
│   └── growth.py             # Auto-generate improvement tasks
└── tools/                    # file_tools, code_tools, system_tools, data_tools
```

### Execution Flow

1. **Task Decomposition**: Complex tasks → subtasks with model assignments
   - Gemini: Research, large context (1M+), exploration
   - Codex: Code generation, implementation
   - Claude: Synthesis, judgment, final review

2. **Parallel Execution**: Independent subtasks run concurrently via `ExecutionMode`
   - `SEQUENTIAL`: One at a time
   - `THREADS`: ThreadPoolExecutor
   - `ASYNC`: asyncio
   - `SUBAGENT_SPAWN`: Isolated Claude Code processes

3. **Checkpointing**: State saved after each step for resume capability

### LLM Providers

| Provider | Class | Auth |
|----------|-------|------|
| openai | OpenAILLM | API key |
| anthropic | AnthropicLLM | API key |
| google | GoogleLLM | API key |
| claude_cli | ClaudeCLIProvider | Anthropic OAuth |
| google_adc | GoogleADCProvider | gcloud auth |
| mcp_local | MCPLocalLLM | Local MCP tools (no API key) |

### Python API
```python
from um_agent_coder import (
    MultiModelOrchestrator, MCPLocalLLM, ExecutionMode
)

gemini = MCPLocalLLM(backend="gemini")
codex = MCPLocalLLM(backend="codex")
claude = MCPLocalLLM(backend="claude")

orchestrator = MultiModelOrchestrator(gemini=gemini, codex=codex, claude=claude)
result = orchestrator.run("complex task")
```

## 24/7 CLI Harness

Autonomous task execution via Codex, Gemini, or Claude CLI with roadmap-driven planning.

### Roadmap Format (`specs/roadmap.md`)
```markdown
## Tasks
- [ ] **task-001**: Task description
  - timeout: 15min
  - depends: none
  - success: How to verify
  - cli: codex              # Optional: codex, gemini, claude
  - model: gpt-5.2          # Optional: override model
```

### Supported CLIs
| CLI | Default Model | Use Case |
|-----|---------------|----------|
| codex | gpt-5.2 | Implementation, builds |
| gemini | gemini-3-pro | Analysis, research |
| claude | claude-opus-4.5 | Complex reasoning |

### State Files
- `.harness/state.db` - SQLite task state
- `.harness/ralph_state.db` - Ralph iteration history
- `.harness/harness.log` - Execution logs

### Ralph Loop Tasks

Enable iterative autonomous execution until completion criteria are met:

```markdown
## Tasks
- [ ] **task-001**: Implement feature X. Output <promise>FEATURE_COMPLETE</promise> when done.
  - ralph: true
  - max_iterations: 30
  - completion_promise: FEATURE_COMPLETE
  - timeout: 60min
  - success: Tests pass, coverage > 80%
  - cli: codex
```

Ralph-specific CLI flags:
```bash
python -m src.um_agent_coder.harness --ralph-default-iterations 50
python -m src.um_agent_coder.harness --ralph-default-promise "TASK_DONE"
```

| Ralph Property | Default | Description |
|----------------|---------|-------------|
| `ralph` | false | Enable Ralph Loop |
| `max_iterations` | 30 | Max loop iterations |
| `completion_promise` | "COMPLETE" | Promise text to detect |

See `docs/ralph-loop.md` for detailed documentation.

### Autonomous Loop (24/7 Unattended Execution)

Full autonomous execution with progress detection, stuck recovery, and multi-CLI routing:

```bash
# Enable autonomous mode
python -m src.um_agent_coder.harness --roadmap specs/roadmap.md --autonomous

# With time limit
python -m src.um_agent_coder.harness --roadmap specs/roadmap.md --autonomous --max-time 8h

# With specific CLIs and Opus limit
python -m src.um_agent_coder.harness --roadmap specs/roadmap.md --cli codex,gemini --opus-limit 50
```

#### Autonomous Task Definition
```markdown
## Tasks
- [ ] **auth-001**: Implement JWT authentication
  - ralph: true
  - max_iterations: 200
  - max_time: 4h
  - cli: codex,gemini
  - stuck_after: 5
  - completion_promise: AUTH_COMPLETE
  - success: All auth tests pass
```

#### Autonomous CLI Flags
| Flag | Default | Description |
|------|---------|-------------|
| `--autonomous` | - | Enable full autonomous mode |
| `--max-iterations` | 1000 | Maximum total iterations |
| `--max-time` | None | Time limit (e.g., "8h", "30m") |
| `--cli` | "auto" | Enabled CLIs ("codex,gemini" or "auto") |
| `--opus-limit` | 50 | Daily Opus iteration limit |
| `--progress-threshold` | 0.15 | Progress score threshold |
| `--stuck-after` | 3 | No-progress iterations before recovery |
| `--recovery-budget` | 20 | Behind-the-scenes recovery iterations |
| `--context-window` | 5 | Raw iterations to keep |
| `--alert-every` | 10 | Status alert interval |
| `--pause-on-critical` | false | Pause on critical alerts |
| `--watch-workspace` | false | Enable file watchers |
| `--enable-inbox` | false | Enable instruction queue |

#### Environment Control
```bash
# Request graceful stop
touch .harness/stop

# Request pause (resume on file removal)
touch .harness/pause

# Drop instructions during execution
echo "Focus on error handling" > .harness/inbox/001-priority.txt
```

#### Additional State Files
- `.harness/alerts.log` - Alert history
- `.harness/status.json` - Current status (JSON)

See `docs/autonomous-loop.md` for comprehensive documentation.

## Key Patterns

- **Config dot notation**: `config.get("llm.openai.api_key")`
- **New LLM provider**: Inherit from `LLM` base class in `llm/providers/`
- **Extension**: Provider selection via `llm.provider` config value
- **Multi-agent routing**: Configure roles in `multi_agent_router` config section
