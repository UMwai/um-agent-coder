# Implement Meta-Harness

## Context

- **Reference**: `specs/features/meta-harness/spec.md`
- **Roadmap**: `specs/features/meta-harness/roadmap.md`
- **Priority**: CRITICAL (50%)
- **Scope**: ~15 new files, ~5 modified files

## Ralph Loop Integration

This prompt is designed for iterative execution via Ralph loop.

**Completion Signal**: When ALL phases are complete and verified:
```
<promise>COMPLETE</promise>
```

**Progress Signals**: Output after each phase:
```
<progress>Phase N complete: [description]</progress>
```

**Do NOT output the completion promise until**:
- All 5 phases are implemented
- All success criteria are met
- Tests pass

## Goal

Implement the meta-harness feature - a harness that can spawn, coordinate, and aggregate results from independent sub-harnesses. This enables multi-project orchestration, task decomposition, and parallel strategy exploration.

## Spec Summary

### Core Classes to Implement

1. **HarnessManager** (`src/um_agent_coder/harness/manager.py`)
   - `spawn_harness(harness_id, roadmap, working_dir, cli)` → HarnessHandle
   - `wait_for(handles)` → List[HarnessResult]
   - `wait_for_any(handles)` → HarnessResult
   - `coordinate(handles, strategy)` → AggregatedResult
   - `broadcast_instruction(instruction)`
   - `request_stop_all()`

2. **HarnessHandle** (`src/um_agent_coder/harness/handle.py`)
   - Properties: harness_id, pid, status, progress, current_task
   - Methods: send_instruction(), request_pause(), request_stop(), get_logs()

3. **CoordinationStrategy** (`src/um_agent_coder/harness/strategies/`)
   - PARALLEL: All run simultaneously, aggregate all results
   - PIPELINE: Sequential, output feeds next input
   - RACE: First to complete wins, others terminated
   - VOTING: Multiple complete, pick best by criteria

4. **SharedContext** (`src/um_agent_coder/harness/shared_context.py`)
   - Cross-harness context sharing
   - Artifact publishing/consumption

### Directory Structure After Implementation

```
src/um_agent_coder/harness/
├── manager.py              # HarnessManager
├── handle.py               # HarnessHandle
├── result.py               # HarnessResult
├── meta_state.py           # MetaStateManager
├── shared_context.py       # SharedContext
├── artifacts.py            # Artifact management
├── dependencies.py         # Dependency resolution
├── dashboard.py            # Status display
└── strategies/
    ├── __init__.py
    ├── base.py             # Abstract strategy
    ├── parallel.py
    ├── pipeline.py
    ├── race.py
    └── voting.py
```

## Implementation Checklist

### Phase 1: Single Sub-Harness Spawn

- [ ] Create `src/um_agent_coder/harness/manager.py`
  - [ ] Implement `HarnessManager.__init__()`
  - [ ] Implement `spawn_harness()` using subprocess.Popen
  - [ ] Store handle in `self.handles` dict

- [ ] Create `src/um_agent_coder/harness/handle.py`
  - [ ] Define `HarnessHandle` dataclass
  - [ ] Define `HarnessStatus` enum
  - [ ] Implement `send_instruction()` (write to .harness/{id}/inbox/)
  - [ ] Implement `request_stop()` (write to .harness/{id}/stop)
  - [ ] Implement status polling from subprocess

- [ ] Create `src/um_agent_coder/harness/result.py`
  - [ ] Define `HarnessResult` dataclass
  - [ ] Define `HarnessMetrics` dataclass

- [ ] Modify `src/um_agent_coder/harness/main.py`
  - [ ] Add `--subprocess` flag for running as sub-harness
  - [ ] When subprocess mode, use isolated state dir

- [ ] Test: spawn single sub-harness, wait for completion

### Phase 2: Parallel Multi-Harness

- [ ] Extend `manager.py`
  - [ ] Track multiple handles in `self.handles`
  - [ ] Implement `wait_for(handles)` using polling
  - [ ] Implement `wait_for_any(handles)` (first to complete)

- [ ] Create `src/um_agent_coder/harness/meta_state.py`
  - [ ] Define `MetaStateManager`
  - [ ] Create meta-state.db schema
  - [ ] Implement `register_sub_harness()`
  - [ ] Implement `update_progress()`

- [ ] Modify state isolation
  - [ ] Each sub-harness uses `.harness/{harness_id}/state.db`
  - [ ] Logs go to `.harness/{harness_id}/harness.log`

- [ ] Test: spawn 3 sub-harnesses in parallel, wait for all

### Phase 3: Cross-Harness Coordination

- [ ] Create `src/um_agent_coder/harness/shared_context.py`
  - [ ] Define `SharedContext` dataclass
  - [ ] Implement `set()` and `get()` methods
  - [ ] Write to `.harness/shared/context.json`
  - [ ] Implement context sync to sub-harnesses

- [ ] Create `src/um_agent_coder/harness/artifacts.py`
  - [ ] Define `Artifact` dataclass
  - [ ] Implement `publish_artifact()`
  - [ ] Implement `consume_artifact()`
  - [ ] Store in `.harness/shared/artifacts/`

- [ ] Create `src/um_agent_coder/harness/dependencies.py`
  - [ ] Parse `depends:` from roadmap
  - [ ] Implement dependency graph
  - [ ] Implement cycle detection
  - [ ] Implement `get_ready_harnesses()`

- [ ] Extend `manager.py`
  - [ ] Implement `broadcast_instruction()`
  - [ ] Respect dependencies in spawn order

- [ ] Test: harness B depends on harness A, context passes

### Phase 4: Coordination Strategies

- [ ] Create `src/um_agent_coder/harness/strategies/base.py`
  - [ ] Define `BaseStrategy` abstract class
  - [ ] Define `execute(handles) → AggregatedResult`

- [ ] Create `strategies/parallel.py`
  - [ ] Spawn all, wait for all
  - [ ] Aggregate results
  - [ ] Handle fail_fast option

- [ ] Create `strategies/pipeline.py`
  - [ ] Execute sequentially
  - [ ] Pass output as context to next stage
  - [ ] Stop on failure

- [ ] Create `strategies/race.py`
  - [ ] Execute all, wait_for_any
  - [ ] Terminate losers
  - [ ] Return winner result

- [ ] Create `strategies/voting.py`
  - [ ] Execute all, wait for min_votes
  - [ ] Score results by criteria
  - [ ] Return best result

- [ ] Extend `manager.py`
  - [ ] Implement `coordinate(handles, strategy)`
  - [ ] Strategy factory from config

- [ ] Test: each strategy with 3 sub-harnesses

### Phase 5: Unified Dashboard

- [ ] Create `src/um_agent_coder/harness/dashboard.py`
  - [ ] Aggregated progress bar
  - [ ] Per-harness status table
  - [ ] Alert aggregation

- [ ] Extend CLI in `main.py`
  - [ ] `--meta` flag to enable meta-harness mode
  - [ ] `--meta --status` to show dashboard
  - [ ] `--meta --stop-all` to stop all
  - [ ] `--meta --logs {harness_id}` to show logs

- [ ] Test: run meta-harness, check status in another terminal

## Success Criteria

1. **Functional**:
   - [ ] Can spawn 3+ sub-harnesses in parallel
   - [ ] Each sub-harness has isolated state
   - [ ] PARALLEL strategy aggregates all results
   - [ ] PIPELINE strategy passes context between stages
   - [ ] RACE strategy terminates losers
   - [ ] Dependencies are respected

2. **Observability**:
   - [ ] `--meta --status` shows all sub-harness status
   - [ ] Can see which sub-harness is stuck
   - [ ] Logs are per-harness in `.harness/{id}/`

3. **Reliability**:
   - [ ] Sub-harness failure doesn't crash meta-harness
   - [ ] Can resume meta-harness after interruption
   - [ ] Clean shutdown on Ctrl+C

## References

- `specs/features/meta-harness/spec.md` - Full specification
- `specs/features/meta-harness/roadmap.md` - Phased implementation
- `specs/architecture/interfaces.md` - Executor interface
- `src/um_agent_coder/harness/main.py` - Existing harness entry point
- `src/um_agent_coder/harness/state.py` - Existing state management

## Prompting Tips

When implementing:

1. **Start with Phase 1** - Get a single spawn working first
2. **Test each phase** before moving to next
3. **Use existing patterns** - Look at how `state.py` manages SQLite
4. **Subprocess isolation** - Use `subprocess.Popen` with proper env
5. **Status polling** - Poll subprocess every 5 seconds

Example spawn command:
```python
subprocess.Popen(
    [sys.executable, "-m", "src.um_agent_coder.harness",
     "--roadmap", str(roadmap),
     "--subprocess",
     "--harness-id", harness_id],
    cwd=working_dir,
    env={**os.environ, "HARNESS_STATE_DIR": str(state_dir)}
)
```

---

*Use this prompt with: `cat prompts/self-build/implement-meta-harness.md | claude-code`*
