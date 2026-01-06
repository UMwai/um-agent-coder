# Architecture Overview

> **Status**: Current + Planned
> **Last Updated**: January 2026

## System Vision

um-agent-coder is a **multi-model AI coding agent framework** designed for orchestrating complex, long-running development tasks. The core vision is a **meta-harness** that can manage other harnesses, enabling truly autonomous multi-project and multi-strategy execution.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           META-HARNESS LAYER                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                     │
│  │ HarnessManager│   │SharedContext │   │  Strategies  │                     │
│  │              │   │              │   │ PARALLEL     │                     │
│  │ spawn()      │   │ artifacts    │   │ PIPELINE     │                     │
│  │ coordinate() │   │ context      │   │ RACE         │                     │
│  │ wait_for()   │   │ events       │   │ VOTING       │                     │
│  └──────────────┘   └──────────────┘   └──────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   SUB-HARNESS 1     │  │   SUB-HARNESS 2     │  │   SUB-HARNESS N     │
│   (subprocess)      │  │   (subprocess)      │  │   (subprocess)      │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HARNESS CORE LAYER                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │RoadmapParser │   │   Executor   │   │ RalphLoop    │   │   State      │  │
│  │              │   │              │   │              │   │   Manager    │  │
│  │ parse()      │   │ execute()    │   │ iterate()    │   │              │  │
│  │ validate()   │   │ verify()     │   │ detect()     │   │ persist()    │  │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AUTONOMOUS LAYER                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │  Progress    │   │   Stuck      │   │  Recovery    │   │   Context    │  │
│  │  Detector    │   │   Detector   │   │  Manager     │   │   Manager    │  │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │  CLI Router  │   │   Alert      │   │ Environment  │   │  Runaway     │  │
│  │              │   │   Manager    │   │   Manager    │   │  Detector    │  │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLI BACKEND LAYER                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                     │
│  │CodexExecutor │   │GeminiExecutor│   │ClaudeExecutor│                     │
│  │              │   │              │   │              │                     │
│  │ gpt-5.2      │   │ gemini-3-pro │   │ claude-opus  │                     │
│  │ implementation│  │ research/1M  │   │ reasoning    │                     │
│  └──────────────┘   └──────────────┘   └──────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Layer Descriptions

### Meta-Harness Layer (NEW)

**Purpose**: Orchestrate multiple harnesses for complex, multi-project workflows.

| Component | Responsibility |
|-----------|----------------|
| HarnessManager | Spawn, monitor, coordinate sub-harnesses |
| SharedContext | Cross-harness context and artifact sharing |
| Strategies | PARALLEL, PIPELINE, RACE, VOTING coordination |

**Spec**: `specs/features/meta-harness/spec.md`

### Harness Core Layer

**Purpose**: Execute roadmap-driven tasks with persistence and recovery.

| Component | Responsibility |
|-----------|----------------|
| RoadmapParser | Parse markdown roadmap into tasks |
| Executor | Execute tasks via CLI backends |
| RalphLoop | Iterative execution until completion |
| StateManager | SQLite persistence for crash recovery |

### Autonomous Layer

**Purpose**: Enable 24/7 unattended execution with self-recovery.

| Component | Responsibility |
|-----------|----------------|
| ProgressDetector | Multi-signal progress scoring |
| StuckDetector | Detect no-progress state |
| RecoveryManager | Prompt mutation, model escalation |
| ContextManager | Rolling window + summarization |
| CLIRouter | Route tasks to optimal CLI backend |
| AlertManager | Notifications and logging |
| EnvironmentManager | File watchers, instruction queue |
| RunawayDetector | Detect infinite loops |

**Spec**: `specs/features/autonomous-loop/spec.md`

### CLI Backend Layer

**Purpose**: Unified interface to multiple LLM CLI tools.

| Backend | Model | Use Case |
|---------|-------|----------|
| CodexExecutor | gpt-5.2 | Implementation, builds |
| GeminiExecutor | gemini-3-pro | Research, large context (1M) |
| ClaudeExecutor | claude-opus-4.5 | Complex reasoning |

**Spec**: `specs/architecture/interfaces.md`

## Data Flow

### Single Harness Execution

```
Roadmap.md
    │
    ▼
┌───────────────┐
│ RoadmapParser │ → List[Task]
└───────────────┘
    │
    ▼
┌───────────────┐
│  StateManager │ → Load/Resume
└───────────────┘
    │
    ▼
┌───────────────┐     ┌───────────────┐
│  CLIRouter    │ ──→ │   Executor    │ ──→ CLI (codex/gemini/claude)
└───────────────┘     └───────────────┘
    │                         │
    ▼                         ▼
┌───────────────┐     ┌───────────────┐
│ProgressDetect │     │ RalphLoop     │ (if ralph: true)
└───────────────┘     └───────────────┘
    │                         │
    ▼                         ▼
┌───────────────┐     ┌───────────────┐
│StuckRecovery  │     │PromiseDetect  │
└───────────────┘     └───────────────┘
```

### Meta-Harness Execution

```
Meta-Roadmap.md
    │
    ▼
┌───────────────────┐
│  HarnessManager   │
└───────────────────┘
    │
    ├── spawn("auth-harness") ──────→ Subprocess A
    ├── spawn("catalog-harness") ───→ Subprocess B
    └── spawn("checkout-harness") ──→ Subprocess C
                                          │
    ┌─────────────────────────────────────┘
    │
    ▼
┌───────────────────┐
│ coordinate(       │
│   [A, B, C],      │
│   PARALLEL        │
│ )                 │
└───────────────────┘
    │
    ▼
┌───────────────────┐
│ AggregatedResult  │
└───────────────────┘
```

## File System Structure

```
project/
├── specs/                      # Specifications
│   ├── architecture/
│   └── features/
│
├── prompts/                    # Actionable prompts
│   ├── self-build/
│   └── user-templates/
│
├── src/um_agent_coder/
│   ├── harness/                # Harness implementation
│   │   ├── main.py             # Entry point
│   │   ├── manager.py          # HarnessManager (meta)
│   │   ├── executors.py        # CLI executors
│   │   ├── state.py            # StateManager
│   │   ├── ralph/              # Ralph loop
│   │   ├── autonomous/         # Autonomous components
│   │   └── strategies/         # Coordination strategies
│   │
│   ├── llm/                    # LLM providers
│   │   └── providers/
│   │
│   └── orchestrator/           # Task orchestration
│
├── .harness/                   # Runtime state
│   ├── state.db                # Task state (single harness)
│   ├── meta-state.db           # Meta-harness state
│   ├── shared/                 # Cross-harness shared
│   │   ├── context.json
│   │   └── artifacts/
│   └── {harness-id}/           # Per-sub-harness state
│       ├── state.db
│       ├── harness.log
│       └── inbox/
│
└── config/
    └── config.yaml             # Configuration
```

## Key Design Decisions

### 1. Subprocess Isolation for Sub-Harnesses

Sub-harnesses run as separate Python processes, not threads.

**Why**:
- True isolation (failure in one doesn't crash others)
- Independent memory spaces
- Can run on different machines (future)
- Clean state separation

### 2. SQLite for State Persistence

All state is persisted in SQLite databases.

**Why**:
- ACID guarantees
- No external dependencies
- WAL mode for concurrent access
- Easy backup/restore

### 3. File-Based Communication

Sub-harnesses communicate via files (.harness/inbox/, .harness/stop).

**Why**:
- Works across process boundaries
- No complex IPC
- Human-inspectable
- Survives process restarts

### 4. Strategy Pattern for Coordination

Coordination strategies are pluggable.

**Why**:
- Easy to add new strategies
- Clear interface contract
- Testable in isolation
- User can implement custom

## Extension Points

| Extension | Interface | Example |
|-----------|-----------|---------|
| New CLI Backend | `BaseCLIExecutor` | Add OpenAI o3 |
| New Strategy | `BaseStrategy` | Add A/B testing |
| New Progress Signal | `ProgressSignal` | Add test coverage |
| New Recovery Strategy | `RecoveryStrategy` | Add web search |

## Performance Considerations

| Concern | Solution |
|---------|----------|
| Many sub-harnesses | Limit max_concurrent |
| Large context | Gemini (1M), summarization |
| Rate limits | Per-CLI rate limiters |
| Disk I/O | WAL mode, batch writes |

## Security Model

| Boundary | Control |
|----------|---------|
| Sub-harness isolation | Separate process, optional sandbox |
| File access | Working directory scoping |
| CLI execution | Approval policies (codex: never) |
| Secrets | Environment variables only |

---

## Related Documents

- `specs/architecture/interfaces.md` - Component contracts
- `specs/features/meta-harness/spec.md` - Meta-harness specification
- `specs/features/autonomous-loop/spec.md` - Autonomous execution

*Last Updated: January 2026*
