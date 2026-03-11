# World-Aware Autonomous Agent Specification

> **Priority**: CRITICAL
> **Status**: Draft
> **Version**: 0.1.0
> **Date**: March 2026

## Vision

A proactive, event-driven agent that monitors the external world, perceives changes relevant to high-level goals, formulates plans, and dispatches work across multiple projects simultaneously. Built on an OODA (Observe-Orient-Decide-Act) loop with adaptive frequency, deployed on GCP Cloud Run + Cloud Run Jobs.

This layer sits **above** the existing meta-harness — the meta-harness is the "hands" that execute work, while the world agent is the "eyes and brain" that decide **what** to work on and **why**.

## Problem Statement

The current system is reactive: it executes tasks from a roadmap. It has no ability to:
- Monitor external events (markets, news, code changes, APIs)
- Autonomously decide what to work on next based on changing conditions
- Reprioritize across multiple projects when the world changes
- Pursue abstract goals that require ongoing adaptation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GOAL STORE                               │
│  Hybrid: natural language + structured priority/constraints  │
│  Persisted in Firestore                                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  OODA CYCLE (Cloud Run Service)               │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │  OBSERVE     │  │  ORIENT      │  │  DECIDE          │    │
│  │  (Perceive)  │→ │  (Analyze)   │→ │  (Plan)          │    │
│  │              │  │              │  │                  │    │
│  │  Event       │  │  Relevance   │  │  Task Generation │    │
│  │  Collectors  │  │  Filtering   │  │  Prioritization  │    │
│  │  ↓           │  │  ↓           │  │  ↓               │    │
│  │  Event Store │  │  World State │  │  Task Queue      │    │
│  └─────────────┘  └──────────────┘  └────────┬─────────┘    │
│                                               │              │
│  ┌────────────────────────────────────────────▼─────────┐    │
│  │  ACT (Dispatch)                                       │    │
│  │  Spawn Cloud Run Jobs per project/task                │    │
│  │  Monitor progress, handle failures                    │    │
│  └───────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐    │
│  │  ADAPTIVE SCHEDULER                                    │    │
│  │  Adjusts cycle frequency based on:                     │    │
│  │  - Event volume (more events = faster cycles)          │    │
│  │  - Goal urgency (deadlines approaching = faster)       │    │
│  │  - Time of day (market hours vs overnight)             │    │
│  │  - Active worker count (more workers = more checking)  │    │
│  └───────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  CR Job:     │  │  CR Job:     │  │  CR Job:     │
│  Project A   │  │  Project B   │  │  Project N   │
│  (container) │  │  (container) │  │  (container) │
│              │  │              │  │              │
│  Uses existing│  │  Uses existing│  │  Uses existing│
│  iteration   │  │  iteration   │  │  iteration   │
│  loop        │  │  loop        │  │  loop        │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Core Concepts

### 1. Goal Store

Goals are the agent's north star. They're defined in a hybrid format: natural language for the LLM to interpret, plus structured metadata for the scheduler and planner.

```yaml
# goals/portfolio-alpha.yaml
goal:
  id: "portfolio-alpha"
  name: "Maximize Portfolio Alpha"
  description: |
    Continuously identify and act on market opportunities.
    Build and refine quantitative models. Monitor positions
    and risk exposure. Generate actionable trade signals.
  priority: 1          # 1 = highest
  status: active
  constraints:
    - "Paper trading only until backtested > 6 months"
    - "Max drawdown: 15%"
    - "No single position > 10% of portfolio"
  kpis:
    - metric: "sharpe_ratio"
      target: "> 1.5"
      current: null
    - metric: "win_rate"
      target: "> 55%"
      current: null
  projects:
    - repo: "um-agent-coder"
      role: "Infrastructure and agent capabilities"
    - repo: "trading-signals"
      role: "Signal generation and backtesting"
  event_sources:
    - "financial.sec_filings"
    - "financial.earnings"
    - "financial.price_alerts"
    - "news.market"
  schedule:
    active_hours: "09:00-16:30 ET"  # Market hours
    frequency_active: "5min"
    frequency_idle: "1h"
```

```yaml
# goals/agent-capabilities.yaml
goal:
  id: "agent-capabilities"
  name: "Continuously Improve Agent Capabilities"
  description: |
    Monitor the AI/ML ecosystem for new tools, models, and techniques.
    Evaluate and integrate improvements. Keep the agent framework
    at the cutting edge.
  priority: 2
  status: active
  constraints:
    - "Don't break existing functionality"
    - "All changes must pass CI"
  kpis:
    - metric: "eval_scores"
      target: "> 0.95 across all evals"
      current: "0.93"
  projects:
    - repo: "um-agent-coder"
      role: "Core framework"
  event_sources:
    - "dev.github_releases"
    - "news.ai_research"
    - "dev.dependency_updates"
  schedule:
    frequency_active: "30min"
    frequency_idle: "4h"
```

### 2. Event Collectors (Observe)

Pluggable collectors that fetch external data on schedule. Each collector implements a simple interface:

```python
class EventCollector(ABC):
    """Base class for all event collectors."""

    @abstractmethod
    async def collect(self) -> list[Event]:
        """Fetch new events since last collection."""
        ...

    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this source (e.g., 'financial.sec_filings')."""
        ...
```

#### Built-in Collectors

| Collector | Source ID | Data | Frequency |
|-----------|-----------|------|-----------|
| **SEC EDGAR** | `financial.sec_filings` | 10-K, 10-Q, 8-K filings | 15min during market hours |
| **Yahoo Finance** | `financial.price_alerts` | Price moves, volume spikes | 5min during market hours |
| **Earnings Calendar** | `financial.earnings` | Upcoming/recent earnings | 1h |
| **GitHub Events** | `dev.github_events` | PRs, issues, releases across repos | 5min |
| **Dependency Watch** | `dev.dependency_updates` | New versions of key deps | 4h |
| **News/RSS** | `news.market` | Financial news from configured feeds | 15min |
| **AI Research** | `news.ai_research` | ArXiv, HuggingFace, model releases | 2h |
| **CI/CD Status** | `dev.ci_status` | Build/deploy status across repos | 5min |

#### Event Schema

```python
@dataclass
class Event:
    id: str                     # Unique event ID
    source: str                 # Collector source_id
    timestamp: datetime         # When event occurred
    category: str               # "financial", "dev", "news"
    severity: str               # "info", "notable", "urgent", "critical"
    title: str                  # Human-readable summary
    body: str                   # Full event data (JSON-serializable)
    metadata: dict              # Source-specific metadata
    related_goals: list[str]    # Goal IDs this might relate to (pre-filter)
```

### 3. World State (Orient)

The orientation layer maintains a running "world state" — an LLM-generated summary of what's happening and what matters. This prevents the decision layer from being overwhelmed by raw events.

```python
@dataclass
class WorldState:
    """Compressed representation of the current world."""
    last_updated: datetime
    summary: str                    # LLM-generated narrative summary
    active_signals: list[Signal]    # Events that warrant attention
    project_status: dict[str, ProjectStatus]  # Per-project state
    goal_progress: dict[str, GoalProgress]    # Per-goal tracking

@dataclass
class Signal:
    """A filtered, interpreted event that may require action."""
    event_id: str
    goal_id: str
    relevance_score: float      # 0-1, LLM-assessed
    interpretation: str         # What this means for the goal
    suggested_action: str       # What could be done about it
    urgency: str                # "immediate", "today", "this_week", "backlog"
```

**Orientation Process:**
1. Collect raw events from all active collectors
2. Batch events and send to LLM with current goals + recent world state
3. LLM filters noise, assesses relevance, generates signals
4. Update world state with new signals, expire old ones
5. Track goal progress based on worker reports

### 4. Task Generation (Decide)

The decision layer converts signals into executable tasks, prioritized across all projects.

```python
@dataclass
class PlannedTask:
    """A task ready for dispatch to a worker."""
    id: str
    goal_id: str
    project: str                # Which repo/project
    title: str
    description: str            # Detailed task description for the worker
    priority: int               # Computed from goal priority + signal urgency
    estimated_effort: str       # "small", "medium", "large"
    dependencies: list[str]     # Other task IDs that must complete first
    cli: str                    # "codex", "gemini", "claude"
    model: str | None           # Optional model override
    timeout: str                # Max execution time
    success_criteria: str       # How to verify completion
    context: dict               # Relevant world state to pass to worker
```

**Decision Process:**
1. Review active signals from world state
2. Check current worker status (what's already running)
3. LLM generates/updates task queue considering:
   - Goal priorities and constraints
   - Signal urgency
   - Resource availability (worker slots, API quotas)
   - Dependencies between tasks
4. Output: ordered task queue with dispatch instructions

### 5. Worker Dispatch (Act)

Workers are Cloud Run Jobs that execute tasks using the existing iteration loop.

```python
class WorkerDispatcher:
    """Manages Cloud Run Job workers for task execution."""

    async def dispatch(self, task: PlannedTask) -> WorkerHandle:
        """Spawn a Cloud Run Job for this task."""
        ...

    async def check_status(self, handle: WorkerHandle) -> WorkerStatus:
        """Check worker progress."""
        ...

    async def collect_results(self, handle: WorkerHandle) -> WorkerResult:
        """Get results from completed worker."""
        ...

    async def cancel(self, handle: WorkerHandle) -> None:
        """Cancel a running worker."""
        ...
```

**Worker Container:**
Each worker container includes:
- The `um-agent-coder` codebase
- Task description and context from the planner
- Access to the project repo (cloned or mounted)
- The existing Gemini iteration loop for task execution
- Ability to report progress back to the planner via Firestore

### 6. Adaptive Scheduler

The scheduler dynamically adjusts the OODA cycle frequency:

```python
class AdaptiveScheduler:
    """Adjusts OODA cycle frequency based on conditions."""

    def compute_interval(self) -> timedelta:
        """Determine time until next OODA cycle."""
        base_interval = self._get_base_interval()  # From goal schedules

        # Factors that speed up the cycle
        if self.pending_events > 10:
            base_interval *= 0.5
        if self.urgent_signals > 0:
            base_interval *= 0.25
        if self.is_market_hours():
            base_interval *= 0.5

        # Factors that slow down the cycle
        if self.active_workers == 0 and self.pending_events == 0:
            base_interval *= 2.0
        if self.is_overnight():
            base_interval *= 4.0

        return clamp(base_interval, min=timedelta(minutes=1), max=timedelta(hours=4))
```

## Data Model

### Persistence (Firestore)

```
firestore/
├── goals/                    # Goal definitions
│   ├── {goal_id}/
│   │   ├── config            # Goal YAML as document
│   │   └── progress/         # Historical KPI snapshots
├── events/                   # Raw event log
│   └── {date}/
│       └── {event_id}        # Event documents (TTL: 30 days)
├── world_state/              # Current world state
│   └── current               # Latest WorldState document
├── tasks/                    # Task queue and history
│   ├── pending/              # Tasks waiting for dispatch
│   ├── active/               # Currently executing
│   └── completed/            # Historical (TTL: 90 days)
├── workers/                  # Worker status tracking
│   └── {worker_id}/
│       ├── status            # Running, completed, failed
│       ├── progress          # Progress reports from worker
│       └── result            # Final output
└── system/                   # System metadata
    ├── scheduler_state       # Last cycle time, interval
    └── resource_usage        # API quotas, costs
```

### Cost Tracking

Every LLM call and API call is tracked for cost awareness:

```python
@dataclass
class CostEntry:
    timestamp: datetime
    component: str          # "orient", "decide", "worker-proj-a"
    model: str              # "gemini-2.5-flash", "gpt-5.2"
    input_tokens: int
    output_tokens: int
    estimated_cost: float   # USD
    goal_id: str | None
```

## GCP Infrastructure

### Cloud Run Service: `world-agent-planner`

The always-on OODA brain. Runs as a single Cloud Run service with 1 min instance.

```yaml
# cloud-run-planner.yaml
service:
  name: world-agent-planner
  region: us-central1
  min_instances: 1        # Always on
  max_instances: 1        # Single brain
  cpu: 1
  memory: 512Mi
  timeout: 300s           # Per-request timeout
  env:
    - FIRESTORE_PROJECT: aivestor-480814
    - GOAL_CONFIG_PATH: gs://um-agent-config/goals/
```

**Endpoints:**
- `POST /cycle` — Trigger an OODA cycle (called by Cloud Scheduler or self-scheduled)
- `GET /status` — Current world state, active workers, next cycle time
- `POST /goals` — CRUD for goals
- `POST /goals/{id}/override` — Temporary priority override for a goal
- `GET /events` — Recent event log
- `POST /nudge` — Short-term natural language steering instruction (expires in 24h)
- `POST /dispatch` — Manually dispatch a task
- `POST /pause` / `POST /resume` — Emergency controls
- `GET /costs` — Cost tracking dashboard data

### Cloud Run Jobs: `world-agent-worker-{project}`

Per-project execution containers. Spawned by the planner, run to completion.

```yaml
# cloud-run-worker.yaml
job:
  name: world-agent-worker
  region: us-central1
  cpu: 2
  memory: 2Gi
  timeout: 3600s          # 1 hour max per task
  max_retries: 1
  env:
    - TASK_ID: (injected at dispatch)
    - FIRESTORE_PROJECT: aivestor-480814
    - PROJECT_REPO: (injected at dispatch)
```

### Cloud Scheduler

Backup heartbeat to ensure the planner runs even when self-scheduling fails:

```yaml
scheduler:
  name: world-agent-heartbeat
  schedule: "*/15 * * * *"   # Every 15 minutes
  target:
    uri: https://world-agent-planner-xxx.run.app/cycle
    method: POST
    body: '{"source": "heartbeat"}'
```

### Supporting Infrastructure

| Service | Purpose | Estimated Cost |
|---------|---------|---------------|
| Cloud Run (planner) | Always-on OODA brain | ~$5-10/mo |
| Cloud Run Jobs (workers) | Per-task execution | ~$0.10/task-hour |
| Firestore | State persistence | ~$1-5/mo |
| Cloud Scheduler | Heartbeat | Free tier |
| Secret Manager | API keys | Free tier |
| Artifact Registry | Container images | ~$1/mo |
| Cloud Storage | Goal configs, artifacts | ~$1/mo |
| **Total baseline** | | **~$10-20/mo** |

## Configuration

### System Config

```yaml
# config/world-agent.yaml
world_agent:
  enabled: true

  scheduler:
    min_interval: "1min"
    max_interval: "4h"
    heartbeat_interval: "15min"

  orientation:
    model: "gemini-2.5-flash"       # Fast, cheap for filtering
    max_events_per_batch: 50
    signal_relevance_threshold: 0.3  # Minimum relevance to keep

  decision:
    model: "gemini-2.5-pro"          # Smarter for planning
    max_concurrent_workers: 5
    task_ttl: "24h"                  # Stale task expiry

  workers:
    default_timeout: "1h"
    default_cli: "codex"
    container_image: "us-central1-docker.pkg.dev/aivestor-480814/um-agent/worker:latest"

  costs:
    daily_budget: 10.00              # USD
    alert_threshold: 0.8             # Alert at 80% of budget
    pause_at_limit: true             # Stop spawning workers at budget

  goals_path: "goals/"              # Directory of goal YAML files
```

## Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Goal**: Planner service running on Cloud Run, with manual goal management and a single event collector.

- [ ] Define `Event`, `Goal`, `WorldState`, `PlannedTask` data models
- [ ] Implement `GoalStore` with Firestore persistence
- [ ] Implement `EventCollector` base class + one collector (GitHub Events)
- [ ] Implement basic `WorldState` orientation (LLM filters events against goals)
- [ ] Build `POST /cycle` endpoint that runs Observe → Orient
- [ ] Deploy planner to Cloud Run with Cloud Scheduler heartbeat
- [ ] Basic `/status` endpoint

### Phase 2: Decision & Dispatch (Week 3-4)

**Goal**: Planner can generate tasks and dispatch Cloud Run Job workers.

- [ ] Implement decision layer (LLM generates `PlannedTask` from signals)
- [ ] Build `WorkerDispatcher` for Cloud Run Jobs
- [ ] Worker container that runs existing iteration loop with injected task
- [ ] Worker → Firestore progress reporting
- [ ] Planner monitors worker status on each cycle
- [ ] Task queue management (priority, dependencies, dedup)

### Phase 3: Event Collectors (Week 5-6)

**Goal**: Full suite of event collectors covering financial, dev, and news sources.

- [ ] SEC EDGAR collector (10-K, 10-Q, 8-K)
- [ ] Yahoo Finance collector (price moves, volume)
- [ ] Earnings calendar collector
- [ ] Dependency watch collector (PyPI, npm)
- [ ] News/RSS collector (configurable feeds)
- [ ] AI research collector (ArXiv, HuggingFace)
- [ ] CI/CD status collector (GitHub Actions)

### Phase 4: Adaptive Scheduler & Cost Control (Week 7-8)

**Goal**: Smart scheduling and budget awareness.

- [ ] Adaptive scheduler implementation
- [ ] Cost tracking per component and goal
- [ ] Daily budget enforcement
- [ ] Alert system (Discord/Slack webhook for budget + critical events)
- [ ] Goal KPI tracking and progress visualization

### Phase 5: Multi-Project Execution (Week 9-10)

**Goal**: Workers can operate across multiple repos with proper isolation.

- [ ] Worker container supports arbitrary repo cloning
- [ ] Cross-project context sharing via Firestore
- [ ] Dependency-aware task scheduling across projects
- [ ] Coordinated releases/deploys across repos
- [ ] Project-level resource quotas

### Phase 6: Learning & Self-Improvement (Week 11-12)

**Goal**: Agent learns from outcomes and improves over time.

- [ ] Outcome tracking (did the task achieve its goal?)
- [ ] Strategy learning (which approaches work for which signal types?)
- [ ] Goal auto-refinement (suggest KPI adjustments based on data)
- [ ] Collector tuning (adjust frequency based on signal yield)
- [ ] Self-evaluation report generation

## Security & Safety

### Guardrails

1. **Budget hard limits**: System pauses when daily spend exceeds threshold
2. **Human-in-the-loop for high-stakes**: Tasks flagged as "high-risk" require approval
3. **No real money**: Financial goals default to paper trading only
4. **Audit trail**: Every decision, dispatch, and result logged to Firestore
5. **Kill switch**: `POST /pause` stops all workers and future cycles
6. **Goal constraints are hard limits**: LLM planning respects constraint fields

### Authentication

- Cloud Run service uses IAM for inter-service auth
- Firestore uses service account with minimal permissions
- API keys stored in Secret Manager
- External API keys (OpenAI, Anthropic, etc.) in Secret Manager

## API Reference

### POST /cycle

Trigger an OODA cycle. Called by Cloud Scheduler or self-scheduled.

```json
// Request
{ "source": "heartbeat" | "self" | "manual" }

// Response
{
  "cycle_id": "cycle-2026-03-10-1430",
  "events_collected": 12,
  "signals_generated": 3,
  "tasks_created": 1,
  "workers_dispatched": 1,
  "next_cycle_in": "5m",
  "cost_this_cycle": 0.02
}
```

### GET /status

Current system state.

```json
{
  "world_state_summary": "Markets flat, AAPL earnings tomorrow...",
  "active_signals": [...],
  "active_workers": [
    {
      "worker_id": "w-abc123",
      "project": "trading-signals",
      "task": "Backtest momentum strategy on Q1 data",
      "started": "2026-03-10T14:00:00Z",
      "progress": 0.6
    }
  ],
  "pending_tasks": 2,
  "next_cycle": "2026-03-10T14:35:00Z",
  "daily_cost": 1.23,
  "daily_budget": 10.00,
  "goals": [...]
}
```

### POST /goals

Create or update a goal.

### GET /events?since=&source=&severity=

Query event history.

### POST /dispatch

Manually dispatch a task (bypasses decision layer).

### POST /pause / POST /resume

Emergency controls.

### GET /costs?period=today|week|month

Cost tracking.

## Design Decisions

### 1. Worker Communication: Planner + Shared Context Bus

Workers write discoveries and context to a shared Firestore collection (`/context_bus/`). The planner curates and distributes relevant context, but workers can also read the bus directly for time-sensitive cross-project information.

```
Worker A (trading-signals) → writes to /context_bus/
                                ↓ (planner curates next cycle)
                                ↓ (worker B can also read directly)
Worker B (um-agent-coder)  ← reads from /context_bus/
```

This balances centralized control (planner decides what's relevant) with speed (workers don't have to wait for the next cycle for urgent cross-project context).

### 2. Goal Conflicts: LLM Arbitration + Budget Splitting

When goals conflict, the system uses a two-tier approach:

1. **Budget splitting**: Resources allocated proportional to goal priority weights. Both goals make progress, neither gets starved.
2. **LLM arbitration for hard conflicts**: When a specific action contradicts two goals, the planner sends both goal docs (full context, constraints, KPIs) to the LLM for reasoned tradeoff analysis.

The key insight: the planner always refers back to the high-level goal documents as the source of truth. Goals aren't just labels — they're rich documents that encode intent, constraints, and priorities the LLM can reason over.

### 3. Context Window: Goal-Scoped Rolling History

Each goal maintains its own rolling history window, tuned to the domain:

| Domain | History Window | Rationale |
|--------|---------------|-----------|
| Financial | 7 days | Markets have multi-day patterns, earnings cycles |
| Dev/GitHub | 24 hours | Code changes are typically recent-context |
| News/Research | 3 days | Trends develop over days |
| CI/CD | 6 hours | Build status is highly time-sensitive |

The orientation LLM receives: current world state summary + goal-scoped history for the goals relevant to the current batch of events. This prevents context bloat while preserving domain-appropriate memory.

### 4. Human Steering: Autonomous-First with Override Controls

The system is autonomous by default — it runs without human intervention. But three steering mechanisms exist for when the human behind the servers needs to adjust:

| Mechanism | Scope | Latency | Use Case |
|-----------|-------|---------|----------|
| **Nudges** (`POST /nudge`) | Short-term, ad-hoc | Next cycle | "Focus on X today", "Investigate this event" |
| **Priority overrides** (`POST /goals/{id}/override`) | Medium-term | Next cycle | Boost/suppress a goal's priority temporarily |
| **Goal file edits** | Long-term, structural | Next reload | Change objectives, constraints, KPIs |

Nudges are natural language instructions injected into the planner's context for the next cycle. They expire after 24h by default (configurable). Priority overrides persist until manually reverted. Goal file edits are permanent.

```python
@dataclass
class Nudge:
    instruction: str        # "Focus on AAPL earnings analysis today"
    created: datetime
    expires: datetime       # Default: 24h from creation
    priority: str           # "low", "normal", "urgent"
    source: str             # "api", "discord", "cli"
```

### 5. Eval Framework (Open — Phase 6)

How to evaluate planner decision quality over time is deferred to Phase 6 (Learning & Self-Improvement). Initial approach will likely involve:
- Outcome tracking: did dispatched tasks achieve their stated success criteria?
- Signal-to-action ratio: what % of signals led to useful work?
- Goal KPI trajectory: are KPIs trending in the right direction?
- Cost efficiency: cost per unit of goal progress
