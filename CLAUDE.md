# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Vision

um-agent-coder is a **meta-harness** - a harness that can manage other harnesses. Core capabilities:
- **Meta-Harness** (50%): Spawn, coordinate, aggregate sub-harnesses
- **Autonomous Loop** (30%): 24/7 unattended execution with recovery
- **Foundation** (20%): CLI routing, progress detection, stuck recovery

## GKE Deployment

UMClaw runs on GKE Autopilot as part of the AI hedge fund ecosystem.

| Property | Value |
|----------|-------|
| **Cluster** | `um-hedge-fund` in `us-central1`, project `aivestor-480814` |
| **Namespace** | `shared-services` |
| **Pod** | `umclaw` (500m CPU, 1Gi memory) |
| **Image** | `us-central1-docker.pkg.dev/aivestor-480814/um-hedge-fund/umclaw` |
| **Internal DNS** | `http://umclaw.shared-services.svc.cluster.local:8080` |

**CI/CD:** Push to `main` triggers `.github/workflows/cd-gke.yaml` which builds the Docker image via Cloud Build and deploys to the GKE `shared-services` namespace.

**Role:** Powers the Consensus Engine (Brain 2) for the hedge fund -- dual-brain trade validation. Every trade decision passes through UMClaw's Gemini-based AI reasoning, scoring each decision 0-1 before execution is approved.

**Note:** The Cloud Run service (`um-agent-daemon`) still exists but GKE is the primary deployment for hedge fund integration.

## Specifications

All technical specs are in `specs/`:

| Spec | Purpose |
|------|---------|
| `specs/README.md` | Index and reading guide |
| `specs/architecture/overview.md` | System architecture |
| `specs/architecture/interfaces.md` | Component contracts |
| `specs/features/meta-harness/spec.md` | **Meta-harness specification** |
| `specs/features/autonomous-loop/spec.md` | Autonomous execution |
| `specs/features/foundation/*.md` | CLI routing, progress, recovery |

## Prompts for Building This Repo

Use prompts in `prompts/self-build/` for AI-driven development:

```bash
# Build meta-harness feature
cat prompts/self-build/implement-meta-harness.md

# Build autonomous loop enhancements
cat prompts/self-build/implement-autonomous-loop.md

# Review and test
cat prompts/self-build/review-and-test.md
```

User templates for running harness on other projects: `prompts/user-templates/`

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

## UMClaw (Cloud Intelligence API)

UMClaw is the deployed FastAPI daemon — the cloud-hosted intelligence layer for um-agent-coder. It provides prompt enhancement, iterative refinement, multi-turn sessions, autonomous world agent cycles, and webhook integrations.

### Access

| Environment | URL |
|-------------|-----|
| **Production** | `https://um-agent-daemon-23o5bq3bfq-uc.a.run.app` |
| **Local** | `http://localhost:8080` |

```bash
# Start locally
pip install -e ".[daemon]"
um-agent-daemon

# Deploy to Cloud Run
gcloud run deploy um-agent-daemon --source . --region us-central1 --project aivestor-480814 --quiet
```

Auth: Optional `X-API-Key` header (enforced only when `UM_DAEMON_API_KEY` is set).

### API Endpoints

#### System
| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Service health + task counts |
| `GET` | `/ui` | Web dashboard |

#### Intelligence Layer (`/api/gemini`)
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/gemini/enhance` | Prompt enhancement + self-eval retry loop |
| `POST` | `/api/gemini/evaluate` | Standalone 5-dimension eval (accuracy, completeness, clarity, actionability, fulfillment) |
| `POST` | `/api/gemini/iterate` | Iteration run: generate → evaluate → strategize → retry (background) |
| `GET` | `/api/gemini/iterate/{id}` | Get iteration status + steps + final score |
| `POST` | `/api/gemini/iterate/batch` | Batch iteration runs |
| `GET` | `/api/gemini/iterations` | List all iteration runs |
| `POST` | `/api/gemini/sessions` | Create multi-turn conversation session |
| `POST` | `/api/gemini/sessions/{id}/message` | Send message in session |
| `POST` | `/api/gemini/batch` | Batch query processing (background) |
| `POST` | `/api/gemini/agent` | Agentic tool-use loop |
| `POST` | `/api/gemini/extract-files` | Extract code blocks from markdown |
| `POST` | `/api/gemini/extract-context` | Auto-extract eval context from Python source |
| `POST` | `/api/gemini/goal-validate/checklist` | Decompose goal into verifiable criteria |
| `POST` | `/api/gemini/goal-validate/score` | Score output against goal criteria |
| `GET` | `/api/gemini/models` | List models (optional `?probe=true`) |

#### Tasks (`/api/tasks`)
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/tasks` | Create background task |
| `GET` | `/api/tasks` | List tasks (filter by status) |
| `GET` | `/api/tasks/{id}` | Get task |
| `POST` | `/api/tasks/{id}/cancel` | Cancel task |
| `GET` | `/api/tasks/{id}/logs` | Get task logs |

#### Knowledge Base (`/api/kb`)
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/kb/items` | Create KB item |
| `GET` | `/api/kb/items` | List/filter items |
| `POST` | `/api/kb/search` | Semantic search |
| `POST` | `/api/kb/extract` | Auto-extract KB candidates from code |

#### World Agent (`/api/world-agent`)
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/world-agent/cycle` | Execute autonomous cycle (orient → decide → act) |
| `GET` | `/api/world-agent/status` | World state (goals, active cycle, lessons) |
| `POST` | `/api/world-agent/goals` | Create goal |
| `GET` | `/api/world-agent/goals` | List goals |
| `POST` | `/api/world-agent/goals/load-yaml` | Load goals from YAML |
| `POST` | `/api/world-agent/journal/generate` | Generate daily journal |
| `POST` | `/api/world-agent/learn/reflect` | Reflect on lessons |
| `POST` | `/api/world-agent/repos/{owner}/{repo}/pr` | Create GitHub PR |

#### Query Proxy (`/api/query`)
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/query` | Direct query to Gemini/Codex |
| `GET` | `/api/query/models` | List available models |

#### Webhooks
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/webhooks/github` | GitHub events (issue_comment `/agent`, PR opened) |
| `POST` | `/slack/events` | Slack @mentions |
| `POST` | `/webhooks/discord` | Discord `/agent` slash command |

### Daemon Source Layout

```
src/um_agent_coder/daemon/
├── app.py              # FastAPI factory + lifespan
├── config.py           # DaemonSettings (UM_DAEMON_* env vars)
├── database.py         # SQLite interface
├── gemini_client.py    # Gemini API client
├── worker.py           # Background task executor
├── routes/
│   ├── gemini/         # Intelligence Layer (enhance, iterate, eval, sessions, agent)
│   ├── kb/             # Knowledge Base (Firestore-backed CRUD + search)
│   ├── world_agent/    # World Agent (autonomous cycles, goals, journal, GitHub)
│   ├── tasks.py        # Task CRUD
│   ├── query.py        # Query proxy
│   ├── github.py       # GitHub webhook
│   ├── slack.py        # Slack webhook
│   ├── discord.py      # Discord webhook
│   └── ui.py           # Web dashboard
```

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UM_DAEMON_PORT` | 8080 | Listen port |
| `UM_DAEMON_API_KEY` | — | API key (auth disabled if unset) |
| `UM_DAEMON_GEMINI_MODEL` | gemini-3-flash-preview | Default model |
| `UM_DAEMON_GEMINI_ITERATE_MAX_ITERATIONS` | 5 | Max iteration steps |
| `UM_DAEMON_GEMINI_ITERATE_SCORE_THRESHOLD` | 0.85 | Quality threshold |
| `UM_DAEMON_GEMINI_FIRESTORE_ENABLED` | false | Persistence (true on Cloud Run) |
| `UM_DAEMON_WORLD_AGENT_ENABLED` | false | Enable world agent |
| `UM_DAEMON_WORLD_AGENT_GITHUB_REPOS` | — | Comma-separated `owner/repo` |
| `UM_DAEMON_KB_AUTO_EXTRACT_ENABLED` | true | Auto-extract KB items |

See `docs/daemon-api.md` for full API reference and `src/um_agent_coder/daemon/config.py` for all config options.

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

## Handling Ambiguous Prompts

When a user prompt is ambiguous or unclear, **always use the AskUserQuestion tool** to clarify before proceeding. Examples of ambiguity:

| Ambiguous Prompt | Clarification Needed |
|------------------|---------------------|
| "Add authentication" | OAuth vs JWT vs session-based? |
| "Improve performance" | Which component? What metric? |
| "Fix the bug" | Which bug? What's the expected behavior? |
| "Add tests" | Unit, integration, or e2e? Coverage target? |
| "Refactor this" | What's the target architecture? |

Use AskUserQuestion with 2-4 clear options when:
- Multiple valid implementation approaches exist
- Requirements are underspecified
- Architectural decisions need user input
- Trade-offs require user preference (speed vs memory, etc.)

## Key Patterns

- **Config dot notation**: `config.get("llm.openai.api_key")`
- **New LLM provider**: Inherit from `LLM` base class in `llm/providers/`
- **Extension**: Provider selection via `llm.provider` config value
- **Multi-agent routing**: Configure roles in `multi_agent_router` config section
