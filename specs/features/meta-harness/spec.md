# Meta-Harness Specification

> **Priority**: CRITICAL (50%)
> **Status**: Draft
> **Version**: 1.0.0

## Vision

A meta-orchestration layer that spawns, coordinates, and aggregates results from independent sub-harnesses. Each sub-harness has its own roadmap, state, CLI backend, and execution context. The meta-harness enables truly autonomous, parallel execution across complex multi-project or multi-strategy workflows.

## Primary Use Cases

### 1. Multi-Project Orchestration

Orchestrate work across multiple repositories simultaneously.

```
Meta-Harness
├── Sub-Harness: frontend-repo/
│   ├── roadmap: frontend/roadmap.md
│   ├── cli: codex
│   └── tasks: UI components, API integration
├── Sub-Harness: backend-repo/
│   ├── roadmap: backend/roadmap.md
│   ├── cli: codex
│   └── tasks: REST API, database schema
└── Sub-Harness: infra-repo/
    ├── roadmap: infra/roadmap.md
    ├── cli: gemini
    └── tasks: Terraform, CI/CD pipelines
```

**Benefits**:
- Shared context passing between projects
- Unified progress tracking across repos
- Coordinated releases

### 2. Task Decomposition

Break complex tasks into independent sub-projects.

```
Goal: "Build e-commerce platform"
    ↓
Meta-Harness decomposes:
├── auth-harness     → implements authentication
├── catalog-harness  → implements product catalog
├── checkout-harness → implements checkout flow
└── Parent aggregates when all complete
```

**Benefits**:
- Each sub-harness has focused scope
- Independent failure isolation
- Parallel execution where possible

### 3. Parallel Strategy Exploration

Try multiple approaches, evaluate, and pick winner.

```
Goal: "Implement caching layer"
    ↓
Meta-Harness spawns:
├── redis-branch     → implements with Redis
├── memcached-branch → implements with Memcached
└── in-memory-branch → implements with local cache

Evaluation:
- Tests pass?
- Performance benchmarks
- Code complexity

Winner: redis-branch (merged)
Losers: terminated, branches deleted
```

**Benefits**:
- Explore options without commitment
- Data-driven decision making
- No wasted work on dead ends

---

## Core Concepts

### HarnessManager

Central coordinator that manages sub-harness lifecycle.

```python
class HarnessManager:
    """Spawns, monitors, and coordinates sub-harnesses."""

    def __init__(self, config: MetaHarnessConfig):
        self.config = config
        self.handles: Dict[str, HarnessHandle] = {}
        self.shared_context = SharedContext()

    def spawn_harness(
        self,
        harness_id: str,
        roadmap: Path,
        working_dir: Path,
        cli: str = "auto",
        parent_context: Optional[Dict] = None
    ) -> HarnessHandle:
        """
        Spawn an independent sub-harness.

        Args:
            harness_id: Unique identifier for this sub-harness
            roadmap: Path to the roadmap.md file
            working_dir: Working directory for execution
            cli: CLI backend (codex, gemini, claude, auto)
            parent_context: Context passed from parent

        Returns:
            HarnessHandle for async control
        """
        pass

    def wait_for(
        self,
        handles: List[HarnessHandle],
        timeout: Optional[timedelta] = None
    ) -> List[HarnessResult]:
        """Wait for specified harnesses to complete."""
        pass

    def wait_for_any(
        self,
        handles: List[HarnessHandle],
        timeout: Optional[timedelta] = None
    ) -> HarnessResult:
        """Wait for first harness to complete (for RACE strategy)."""
        pass

    def coordinate(
        self,
        handles: List[HarnessHandle],
        strategy: CoordinationStrategy
    ) -> AggregatedResult:
        """Coordinate harnesses using specified strategy."""
        pass

    def broadcast_instruction(self, instruction: str):
        """Send instruction to all running sub-harnesses."""
        pass

    def request_stop_all(self):
        """Request graceful stop of all sub-harnesses."""
        pass
```

### HarnessHandle

Async control interface for a running sub-harness.

```python
@dataclass
class HarnessHandle:
    """Handle to control and monitor a sub-harness."""

    harness_id: str
    pid: int
    working_dir: Path
    state_db: Path  # .harness/{harness_id}/state.db

    # Observable state
    status: HarnessStatus  # PENDING, RUNNING, COMPLETED, FAILED, STOPPED
    progress: float        # 0.0 - 1.0
    current_task: Optional[str]
    current_iteration: int

    # Results (populated on completion)
    result: Optional[HarnessResult]
    error: Optional[str]

    def send_instruction(self, instruction: str):
        """Send instruction mid-flight (writes to .harness/inbox/)."""
        pass

    def request_pause(self):
        """Request pause (creates .harness/pause file)."""
        pass

    def request_stop(self):
        """Request graceful stop (creates .harness/stop file)."""
        pass

    def force_kill(self, timeout: int = 30):
        """Force kill after timeout."""
        pass

    def get_logs(self, tail: int = 100) -> List[str]:
        """Get recent log lines."""
        pass

    def get_alerts(self) -> List[Alert]:
        """Get alerts from sub-harness."""
        pass

class HarnessStatus(Enum):
    PENDING = "pending"      # Spawned but not started
    RUNNING = "running"      # Actively executing
    PAUSED = "paused"        # Paused (waiting for resume)
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"        # Failed with error
    STOPPED = "stopped"      # Stopped by request
```

### CoordinationStrategy

Strategies for coordinating multiple sub-harnesses.

```python
class CoordinationStrategy(Enum):
    PARALLEL = "parallel"    # All run simultaneously, aggregate all results
    PIPELINE = "pipeline"    # Sequential, output feeds next input
    RACE = "race"            # First to complete wins, others terminated
    VOTING = "voting"        # Multiple complete, pick best by criteria

@dataclass
class ParallelConfig:
    """Config for PARALLEL strategy."""
    max_concurrent: int = 10
    fail_fast: bool = False  # Stop all on first failure?

@dataclass
class PipelineConfig:
    """Config for PIPELINE strategy."""
    stages: List[str]  # Ordered list of harness_ids
    pass_context: bool = True  # Pass output as input to next

@dataclass
class RaceConfig:
    """Config for RACE strategy."""
    min_progress_to_win: float = 0.8  # Must reach 80% to count as winner
    terminate_losers: bool = True

@dataclass
class VotingConfig:
    """Config for VOTING strategy."""
    min_votes: int = 2  # Minimum harnesses that must complete
    selection_criteria: SelectionCriteria  # How to pick winner

class SelectionCriteria(Enum):
    FIRST_COMPLETE = "first"     # First to complete wins
    BEST_TESTS = "tests"         # Most tests passing
    BEST_PERFORMANCE = "perf"    # Best performance metrics
    HUMAN_REVIEW = "human"       # Human picks winner
```

### SharedContext

Cross-harness context sharing.

```python
@dataclass
class SharedContext:
    """Shared state across sub-harnesses."""

    # Global context
    global_vars: Dict[str, Any]

    # Per-harness outputs (for pipeline)
    harness_outputs: Dict[str, HarnessOutput]

    # Shared artifacts
    artifacts: Dict[str, Artifact]

    # Event log
    events: List[CrossHarnessEvent]

    def set(self, key: str, value: Any, harness_id: Optional[str] = None):
        """Set context value (optionally scoped to harness)."""
        pass

    def get(self, key: str, harness_id: Optional[str] = None) -> Any:
        """Get context value."""
        pass

    def publish_artifact(self, name: str, path: Path, harness_id: str):
        """Publish artifact for other harnesses to consume."""
        pass

    def consume_artifact(self, name: str) -> Path:
        """Consume artifact published by another harness."""
        pass

@dataclass
class Artifact:
    """Shared file artifact between harnesses."""
    name: str
    source_harness: str
    path: Path
    created_at: datetime
    metadata: Dict[str, Any]
```

### HarnessResult

Result from a completed sub-harness.

```python
@dataclass
class HarnessResult:
    """Result from a completed sub-harness."""

    harness_id: str
    status: HarnessStatus

    # Execution stats
    total_iterations: int
    total_duration: timedelta
    tasks_completed: int
    tasks_failed: int

    # Output
    final_output: str
    artifacts: List[Artifact]

    # Metrics (for voting/selection)
    metrics: HarnessMetrics

    # Error info (if failed)
    error: Optional[str]
    traceback: Optional[str]

@dataclass
class HarnessMetrics:
    """Metrics for comparing harness results."""
    tests_passed: int
    tests_failed: int
    test_coverage: float
    performance_score: float  # Benchmark results
    code_quality_score: float  # Linting/complexity
    lines_of_code: int
```

---

## Architecture

### Directory Structure

```
project/
├── .harness/
│   ├── meta-state.db           # Meta-harness state
│   ├── shared/
│   │   ├── context.json        # Shared context
│   │   └── artifacts/          # Shared artifacts
│   │
│   ├── {harness-id-1}/         # Sub-harness 1 state
│   │   ├── state.db
│   │   ├── harness.log
│   │   ├── alerts.log
│   │   └── inbox/
│   │
│   └── {harness-id-2}/         # Sub-harness 2 state
│       ├── state.db
│       └── ...
```

### Process Model

```
Meta-Harness Process (Python)
│
├── spawn_harness("auth") ──────────┐
│                                   │
│                                   ▼
│                           Sub-Process: auth
│                           ├── python -m harness --roadmap auth/roadmap.md
│                           ├── Working dir: ./auth/
│                           └── State: .harness/auth/state.db
│
├── spawn_harness("catalog") ───────┐
│                                   │
│                                   ▼
│                           Sub-Process: catalog
│                           ├── python -m harness --roadmap catalog/roadmap.md
│                           └── Working dir: ./catalog/
│
├── Monitor all handles
│   ├── Poll status every N seconds
│   ├── Aggregate progress
│   └── Handle failures
│
└── Coordinate(PARALLEL)
    ├── Wait for all
    └── Aggregate results
```

### State Synchronization

```python
class MetaStateManager:
    """Manages meta-harness and sub-harness state."""

    def __init__(self, harness_dir: Path):
        self.meta_db = harness_dir / "meta-state.db"
        self.shared_dir = harness_dir / "shared"

    def register_sub_harness(self, handle: HarnessHandle):
        """Register new sub-harness in meta-state."""
        pass

    def update_progress(self, harness_id: str, progress: float):
        """Update sub-harness progress."""
        pass

    def sync_context(self):
        """Sync shared context to all sub-harnesses."""
        # Write context.json
        # Notify sub-harnesses via inbox
        pass

    def aggregate_results(self, handles: List[HarnessHandle]) -> AggregatedResult:
        """Aggregate results from completed sub-harnesses."""
        pass
```

### Meta-State Schema

```sql
-- .harness/meta-state.db

CREATE TABLE meta_harness_state (
    id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,           -- running, completed, failed
    strategy TEXT NOT NULL,         -- parallel, pipeline, race, voting
    started_at TEXT,
    completed_at TEXT,
    config_json TEXT                -- Strategy configuration
);

CREATE TABLE sub_harnesses (
    harness_id TEXT PRIMARY KEY,
    pid INTEGER,
    status TEXT NOT NULL,
    working_dir TEXT NOT NULL,
    roadmap_path TEXT NOT NULL,
    cli TEXT,
    progress REAL DEFAULT 0.0,
    current_task TEXT,
    current_iteration INTEGER DEFAULT 0,
    started_at TEXT,
    completed_at TEXT,
    result_json TEXT
);

CREATE TABLE cross_harness_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,       -- spawn, progress, complete, fail, instruction
    harness_id TEXT,
    details_json TEXT
);

CREATE TABLE shared_artifacts (
    name TEXT PRIMARY KEY,
    source_harness TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT
);
```

---

## Configuration

### Roadmap Format for Meta-Harness

```markdown
# Meta-Harness Roadmap

## Strategy
coordination: parallel
fail_fast: false
max_concurrent: 5

## Sub-Harnesses

### auth-harness
- working_dir: ./auth
- roadmap: auth/roadmap.md
- cli: codex
- depends: none

### catalog-harness
- working_dir: ./catalog
- roadmap: catalog/roadmap.md
- cli: codex
- depends: none

### checkout-harness
- working_dir: ./checkout
- roadmap: checkout/roadmap.md
- cli: codex
- depends: auth-harness, catalog-harness

## Aggregation
on_complete: merge_all
output_dir: ./dist
```

### CLI Interface

```bash
# Run as meta-harness
python -m src.um_agent_coder.harness \
    --meta \
    --roadmap specs/meta-roadmap.md \
    --strategy parallel \
    --max-concurrent 5

# With pipeline strategy
python -m src.um_agent_coder.harness \
    --meta \
    --roadmap specs/meta-roadmap.md \
    --strategy pipeline

# Race mode (first wins)
python -m src.um_agent_coder.harness \
    --meta \
    --roadmap specs/meta-roadmap.md \
    --strategy race \
    --min-progress 0.8

# Status check
python -m src.um_agent_coder.harness --meta --status

# Stop all sub-harnesses
python -m src.um_agent_coder.harness --meta --stop-all
```

### YAML Configuration

```yaml
# config/meta-harness.yaml

meta_harness:
  enabled: true
  max_concurrent_harnesses: 10

  # Default strategy
  default_strategy: parallel

  # Monitoring
  poll_interval_seconds: 5
  progress_report_interval: 30

  # Failure handling
  fail_fast: false
  retry_failed_harnesses: true
  max_retries_per_harness: 2

  # Resource limits
  max_total_iterations: 10000
  max_total_time: "24h"

  # Shared context
  shared_context_sync_interval: 10
  artifact_dir: ".harness/shared/artifacts"
```

---

## Implementation Phases

### Phase 1: Single Sub-Harness Spawn (Proof of Concept)

**Goal**: Demonstrate spawning and monitoring a single sub-harness.

**Deliverables**:
- [ ] `HarnessManager.spawn_harness()` - spawn subprocess
- [ ] `HarnessHandle` - basic status monitoring
- [ ] `HarnessResult` - capture result on completion
- [ ] Integration with existing harness as subprocess

**Files to Create/Modify**:
- `src/um_agent_coder/harness/manager.py` (new)
- `src/um_agent_coder/harness/handle.py` (new)
- `src/um_agent_coder/harness/main.py` (modify for subprocess mode)

### Phase 2: Parallel Multi-Harness

**Goal**: Run multiple sub-harnesses in parallel with independent state.

**Deliverables**:
- [ ] Spawn multiple sub-harnesses concurrently
- [ ] Independent state.db per sub-harness
- [ ] Progress aggregation across all
- [ ] `wait_for()` and `wait_for_any()`

**Files to Create/Modify**:
- `src/um_agent_coder/harness/manager.py` (extend)
- `src/um_agent_coder/harness/meta_state.py` (new)

### Phase 3: Cross-Harness Coordination

**Goal**: Enable sub-harnesses to share context and artifacts.

**Deliverables**:
- [ ] `SharedContext` implementation
- [ ] Artifact publishing/consumption
- [ ] Dependency resolution (harness B waits for harness A)
- [ ] `broadcast_instruction()`

**Files to Create/Modify**:
- `src/um_agent_coder/harness/shared_context.py` (new)
- `src/um_agent_coder/harness/artifacts.py` (new)

### Phase 4: Strategy Implementations

**Goal**: Implement all coordination strategies.

**Deliverables**:
- [ ] `PARALLEL` strategy with aggregation
- [ ] `PIPELINE` strategy with context passing
- [ ] `RACE` strategy with termination
- [ ] `VOTING` strategy with selection criteria

**Files to Create/Modify**:
- `src/um_agent_coder/harness/strategies/` (new directory)
- `src/um_agent_coder/harness/strategies/parallel.py`
- `src/um_agent_coder/harness/strategies/pipeline.py`
- `src/um_agent_coder/harness/strategies/race.py`
- `src/um_agent_coder/harness/strategies/voting.py`

### Phase 5: Unified Monitoring Dashboard

**Goal**: Single view of all sub-harness activity.

**Deliverables**:
- [ ] `--meta --status` CLI command
- [ ] Aggregated progress bar
- [ ] Per-harness status table
- [ ] Alert aggregation
- [ ] Optional: web dashboard (future)

**Files to Create/Modify**:
- `src/um_agent_coder/harness/dashboard.py` (new)
- `src/um_agent_coder/harness/cli.py` (extend)

---

## Success Criteria

### Functional Requirements

1. **Spawn**: Can spawn N independent sub-harnesses
2. **Monitor**: Can track progress of all sub-harnesses
3. **Coordinate**: Supports PARALLEL, PIPELINE, RACE, VOTING
4. **Share**: Sub-harnesses can share context and artifacts
5. **Control**: Can pause/stop individual or all sub-harnesses
6. **Aggregate**: Results are aggregated into unified output

### Non-Functional Requirements

1. **Isolation**: Sub-harness failure doesn't crash meta-harness
2. **Scalability**: Can handle 10+ concurrent sub-harnesses
3. **Observability**: Clear visibility into all sub-harness activity
4. **Resumability**: Can resume meta-harness after interruption

---

## Related Specifications

- [architecture/overview.md](../../architecture/overview.md) - System overview
- [architecture/interfaces.md](../../architecture/interfaces.md) - Executor interfaces
- [autonomous-loop/spec.md](../autonomous-loop/spec.md) - Autonomous execution within each sub-harness

---

*Last Updated: January 2026*
