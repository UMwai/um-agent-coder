# Meta-Harness Implementation Roadmap

> **Reference**: [spec.md](spec.md)
> **Priority**: CRITICAL (50%)

## Overview

This roadmap defines the phased implementation of the meta-harness feature - the ability for a harness to spawn, coordinate, and aggregate results from independent sub-harnesses.

## Implementation Phases

### Phase 1: Single Sub-Harness Spawn

**Goal**: Prove we can spawn and monitor a single sub-harness as a subprocess.

**Tasks**:
- [ ] **1.1** Create `HarnessManager` class skeleton
- [ ] **1.2** Implement `spawn_harness()` using subprocess
- [ ] **1.3** Create `HarnessHandle` with basic status monitoring
- [ ] **1.4** Implement `wait_for()` for single handle
- [ ] **1.5** Capture `HarnessResult` on completion
- [ ] **1.6** Add `--subprocess` flag to existing harness for sub-harness mode

**Files to Create**:
```
src/um_agent_coder/harness/
├── manager.py          # HarnessManager class
├── handle.py           # HarnessHandle class
└── result.py           # HarnessResult dataclass
```

**Success Criteria**:
- Can spawn a harness subprocess that runs a simple roadmap
- Can monitor subprocess status (running/completed/failed)
- Can capture output when subprocess completes

---

### Phase 2: Parallel Multi-Harness

**Goal**: Run multiple sub-harnesses in parallel with independent state.

**Tasks**:
- [ ] **2.1** Extend `spawn_harness()` to track multiple handles
- [ ] **2.2** Implement isolated state directories: `.harness/{harness_id}/`
- [ ] **2.3** Create `MetaStateManager` for meta-harness state
- [ ] **2.4** Implement `wait_for()` for multiple handles
- [ ] **2.5** Implement `wait_for_any()` (first to complete)
- [ ] **2.6** Add progress aggregation across all sub-harnesses
- [ ] **2.7** Handle sub-harness failures gracefully

**Files to Create/Modify**:
```
src/um_agent_coder/harness/
├── manager.py          # Extend for multi-harness
├── meta_state.py       # NEW: MetaStateManager
└── state.py            # Modify for isolated directories
```

**Success Criteria**:
- Can spawn 3+ sub-harnesses simultaneously
- Each sub-harness has independent state.db
- Progress shows aggregate across all sub-harnesses
- Failure in one doesn't crash others

---

### Phase 3: Cross-Harness Coordination

**Goal**: Enable sub-harnesses to share context and artifacts.

**Tasks**:
- [ ] **3.1** Implement `SharedContext` class
- [ ] **3.2** Create shared context file: `.harness/shared/context.json`
- [ ] **3.3** Implement artifact publishing: `publish_artifact()`
- [ ] **3.4** Implement artifact consumption: `consume_artifact()`
- [ ] **3.5** Add dependency resolution (harness B waits for harness A)
- [ ] **3.6** Implement `broadcast_instruction()` to all sub-harnesses
- [ ] **3.7** Context sync on interval (push updates to sub-harnesses)

**Files to Create**:
```
src/um_agent_coder/harness/
├── shared_context.py   # SharedContext class
├── artifacts.py        # Artifact management
└── dependencies.py     # Dependency resolution
```

**Success Criteria**:
- Sub-harness A can publish artifact, sub-harness B can consume it
- Sub-harness B can wait for sub-harness A to complete before starting
- Global context changes propagate to all sub-harnesses

---

### Phase 4: Coordination Strategies

**Goal**: Implement all coordination strategies: PARALLEL, PIPELINE, RACE, VOTING.

**Tasks**:
- [ ] **4.1** Create strategy interface: `CoordinationStrategy`
- [ ] **4.2** Implement `ParallelStrategy` - run all, aggregate results
- [ ] **4.3** Implement `PipelineStrategy` - sequential with context passing
- [ ] **4.4** Implement `RaceStrategy` - first to complete wins
- [ ] **4.5** Implement `VotingStrategy` - multiple complete, pick best
- [ ] **4.6** Add strategy configuration parsing from roadmap
- [ ] **4.7** Implement result aggregation per strategy

**Files to Create**:
```
src/um_agent_coder/harness/strategies/
├── __init__.py
├── base.py             # Abstract strategy interface
├── parallel.py
├── pipeline.py
├── race.py
└── voting.py
```

**Success Criteria**:
- Each strategy correctly coordinates sub-harnesses
- Pipeline passes output of stage N as input to stage N+1
- Race terminates losers when winner completes
- Voting selects best based on criteria (tests passed, performance)

---

### Phase 5: Unified Monitoring Dashboard

**Goal**: Single view of all sub-harness activity.

**Tasks**:
- [ ] **5.1** Extend CLI: `--meta --status` command
- [ ] **5.2** Create aggregated progress display
- [ ] **5.3** Create per-harness status table
- [ ] **5.4** Aggregate alerts from all sub-harnesses
- [ ] **5.5** Add `--meta --stop-all` command
- [ ] **5.6** Add `--meta --logs {harness_id}` command

**Files to Create/Modify**:
```
src/um_agent_coder/harness/
├── dashboard.py        # NEW: Status display
└── main.py             # Extend CLI
```

**Success Criteria**:
- Single command shows status of all sub-harnesses
- Can see which sub-harness is stuck or failing
- Can stop all sub-harnesses with one command

---

## Dependency Graph

```
Phase 1 (Spawn)
    │
    ▼
Phase 2 (Parallel)
    │
    ├───────────────────┐
    ▼                   ▼
Phase 3 (Coordination)  Phase 4 (Strategies)
    │                   │
    └─────────┬─────────┘
              ▼
        Phase 5 (Dashboard)
```

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Subprocess crashes | Isolate state, implement restart |
| Resource exhaustion | Limit max_concurrent |
| State corruption | WAL mode SQLite, atomic writes |
| Context race conditions | Lock files for shared context |
| Deadlock in dependencies | Cycle detection in dependency graph |

## Estimated Scope

| Phase | Files | Complexity |
|-------|-------|------------|
| Phase 1 | 3 new | Low |
| Phase 2 | 2 new, 2 modify | Medium |
| Phase 3 | 3 new | Medium |
| Phase 4 | 6 new | High |
| Phase 5 | 2 new, 1 modify | Medium |

---

*Last Updated: January 2026*
