# Autonomous Loop Implementation for um-agent-coder

Build a fully autonomous loop capability into the um-agent-coder harness. This enables iterative task execution where agents self-loop, detect progress, recover from being stuck, and respond to real-time environmental changes—removing the human bottleneck.

**Full Specification**: [`specs/autonomous-loop-spec.md`](specs/autonomous-loop-spec.md)

## Overview

### Core Vision
Maximize velocity by enabling unlimited Codex/Gemini usage autonomously while preserving scarce Opus tokens for complex reasoning. Agents should:
- Run until manually stopped, time limit reached, or goal completed
- Detect progress via multi-signal scoring
- Self-recover when stuck (prompt mutation → model escalation → branch exploration)
- Respond to real-time environmental inputs (file changes, instruction queue, env vars)
- Route intelligently between CLIs based on model strengths

### Requirements

#### Core Functionality (Existing Ralph Loop)
- Completion promise detection in executor output
- Re-feed mechanism that preserves file state between iterations
- Iteration tracking with configurable max-iterations
- Safety escape hatches when max iterations exceeded

#### New Autonomous Features
- Multi-signal progress detection (output diff, file changes, explicit markers, checklist)
- Time-based termination (`--max-time 8h`)
- Stuck recovery system (20 cycles triggers alternative solution search)
- Context accumulation (rolling window + summarization)
- Multi-CLI router with auto mode (`--cli codex,gemini` or `--cli auto`)
- Environmental awareness (file watchers, instruction queue, env vars)
- Alert system (CLI notifications + file logging)
- Runaway prevention

---

## Phase 1: Completion Promise Detection (Existing)

Implement output parsing to detect completion promises in CLI executor responses.

### Tasks
1. Create `src/um_agent_coder/harness/ralph/` module directory
2. Implement `promise_detector.py`:
   - `PromiseDetector` class with configurable promise patterns
   - Support for `<promise>TEXT</promise>` format
   - Support for plain text exact matching
   - Return `DetectionResult` with `found`, `promise_text`, `match_type`
3. Add unit tests in `tests/harness/test_promise_detector.py`
4. Run tests and verify all pass

### Verification
```bash
pytest tests/harness/test_promise_detector.py -v --cov=src/um_agent_coder/harness/ralph
```

### Success Criteria
- [ ] PromiseDetector correctly identifies `<promise>COMPLETE</promise>` in output
- [ ] PromiseDetector correctly identifies plain text promises
- [ ] All unit tests pass with >90% coverage
- [ ] No linting errors (`ruff check src/um_agent_coder/harness/ralph/`)

---

## Phase 2: Iteration Tracker (Existing)

Implement iteration state management with safety limits.

### Tasks
1. Implement `src/um_agent_coder/harness/ralph/iteration_tracker.py`:
   - `IterationTracker` class with `task_id`, `max_iterations`, `current_iteration`, `start_time`, `iteration_history`
   - Methods: `increment()`, `can_continue()`, `get_summary()`
   - `IterationRecord` dataclass
2. Add SQLite persistence in `src/um_agent_coder/harness/ralph/persistence.py`
3. Add unit tests in `tests/harness/test_iteration_tracker.py`
4. Run tests and verify all pass

### Verification
```bash
pytest tests/harness/test_iteration_tracker.py -v --cov=src/um_agent_coder/harness/ralph
```

### Success Criteria
- [ ] IterationTracker enforces max_iterations limit
- [ ] State persists to SQLite and survives restart
- [ ] Iteration history is queryable
- [ ] All unit tests pass with >90% coverage

---

## Phase 3: Ralph Executor Wrapper (Existing)

Create executor wrapper that implements the re-feed loop.

### Tasks
1. Implement `src/um_agent_coder/harness/ralph/executor.py`:
   - `RalphExecutor` class wrapping base executors
   - Loop until promise detected or max iterations reached
   - Preserve working directory state between iterations
2. Implement `RalphResult` dataclass
3. Add integration tests in `tests/harness/test_ralph_executor.py`
4. Run tests and verify all pass

### Verification
```bash
pytest tests/harness/test_ralph_executor.py -v --cov=src/um_agent_coder/harness/ralph
```

### Success Criteria
- [ ] RalphExecutor loops until promise detected or max reached
- [ ] File state preserved between iterations
- [ ] Each iteration logged with timing
- [ ] All integration tests pass with >85% coverage

---

## Phase 4: Roadmap Parser Extension (Existing)

Extend roadmap parser to support ralph-style and autonomous task definitions.

### Tasks
1. Update `src/um_agent_coder/harness/roadmap_parser.py` with new fields:
   - `ralph: true`, `max_iterations: N`, `completion_promise: TEXT`
   - `max_time: DURATION`, `cli: LIST`, `stuck_after: N`
2. Update `src/um_agent_coder/harness/models.py` with `RalphConfig` and `AutonomousConfig`
3. Add parser tests in `tests/harness/test_roadmap_parser.py`
4. Run tests and verify all pass

### Verification
```bash
pytest tests/harness/test_roadmap_parser.py -v --cov=src/um_agent_coder/harness
```

### Success Criteria
- [ ] Parser recognizes ralph-specific fields
- [ ] Parser recognizes autonomous loop fields
- [ ] Invalid configurations raise clear errors
- [ ] All parser tests pass with >90% coverage

---

## Phase 5: Harness Integration (Existing)

Wire ralph executor into main harness loop.

### Tasks
1. Update `src/um_agent_coder/harness/main.py`:
   - Check for ralph config in `_execute_task()`
   - Wrap executor in `RalphExecutor` when enabled
2. Add CLI flags: `--ralph-default-iterations`, `--ralph-default-promise`
3. Update `src/um_agent_coder/harness/state.py` with `ralph_iterations` tracking
4. Add integration tests in `tests/harness/test_main_ralph.py`
5. Run all harness tests

### Verification
```bash
pytest tests/harness/ -v --cov=src/um_agent_coder/harness
```

### Success Criteria
- [ ] Harness correctly identifies and executes ralph tasks
- [ ] Non-ralph tasks unaffected
- [ ] CLI flags work correctly
- [ ] All harness tests pass with >85% coverage

---

## Phase 6: Progress Detection System (NEW)

Implement multi-signal progress detection per `specs/autonomous-loop-spec.md` Section 2.

### Tasks
1. Create `src/um_agent_coder/harness/autonomous/progress_detector.py`:
   - `ProgressSignal` dataclass with `output_diff_score`, `file_changes_score`, `explicit_markers`, `checklist_progress`
   - `calculate_progress_score()` with weighted signals (30/30/25/15)
   - Output diff algorithm using `difflib.SequenceMatcher`
   - File changes detection via `git diff --stat`
2. Create `src/um_agent_coder/harness/autonomous/progress_markers.py`:
   - Extract `<progress>...</progress>` tags from output
3. Add unit tests in `tests/harness/test_progress_detector.py`:
   - Test each signal component
   - Test weighted combination
   - Test no-progress threshold (< 0.15)
4. Run tests and verify all pass

### Verification
```bash
pytest tests/harness/test_progress_detector.py -v --cov=src/um_agent_coder/harness/autonomous
```

### Success Criteria
- [ ] Progress score correctly combines all signals
- [ ] Output diff detects similarity accurately
- [ ] File changes integrates with git
- [ ] Explicit markers extracted correctly
- [ ] All unit tests pass with >90% coverage

---

## Phase 7: Stuck Recovery System (NEW)

Implement stuck detection and recovery per `specs/autonomous-loop-spec.md` Section 3.

### Tasks
1. Create `src/um_agent_coder/harness/autonomous/stuck_detector.py`:
   - Track consecutive no-progress iterations
   - Trigger recovery after N iterations (default: 3, behind-scenes: 20)
2. Create `src/um_agent_coder/harness/autonomous/recovery/`:
   - `prompt_mutator.py`: Rephrase, decompose, constrain mutations
   - `model_escalator.py`: Escalation order (gemini-flash → gemini-pro → codex → sonnet → opus)
   - `branch_explorer.py`: Fork into 2-3 parallel approaches
   - `recovery_manager.py`: Orchestrate recovery strategies
3. Add unit tests in `tests/harness/test_stuck_recovery.py`:
   - Test stuck detection
   - Test each recovery strategy
   - Test recovery flow
4. Run tests and verify all pass

### Verification
```bash
pytest tests/harness/test_stuck_recovery.py -v --cov=src/um_agent_coder/harness/autonomous
```

### Success Criteria
- [ ] Stuck detected after 3 consecutive no-progress iterations
- [ ] Prompt mutations generate valid alternatives
- [ ] Model escalation follows correct order
- [ ] Branch exploration runs in parallel
- [ ] All unit tests pass with >85% coverage

---

## Phase 8: Context Management (NEW)

Implement rolling window + summarization per `specs/autonomous-loop-spec.md` Section 6.

### Tasks
1. Create `src/um_agent_coder/harness/autonomous/context_manager.py`:
   - `IterationContext` dataclass
   - `LoopContext` with rolling window of raw iterations
   - `add_iteration()` maintaining window size (default: 5)
2. Create `src/um_agent_coder/harness/autonomous/context_summarizer.py`:
   - Summarize older iterations using cheap model (gemini-flash)
   - Re-summarize every N iterations (default: 10)
   - `build_contextual_prompt()` combining summary + recent raw
3. Add unit tests in `tests/harness/test_context_manager.py`:
   - Test rolling window
   - Test summarization trigger
   - Test prompt building
4. Run tests and verify all pass

### Verification
```bash
pytest tests/harness/test_context_manager.py -v --cov=src/um_agent_coder/harness/autonomous
```

### Success Criteria
- [ ] Rolling window maintains correct size
- [ ] Older iterations summarized before removal
- [ ] Context prompts include summary + recent
- [ ] All unit tests pass with >90% coverage

---

## Phase 9: Multi-CLI Router (NEW)

Implement CLI routing per `specs/autonomous-loop-spec.md` Section 7.

### Tasks
1. Create `src/um_agent_coder/harness/autonomous/cli_router.py`:
   - Parse `--cli codex,gemini` CLI list
   - `TaskAnalyzer` to analyze task characteristics
   - `AutoRouter` with routing logic:
     - Stuck recovery → smartest available
     - Large context → Gemini (1M)
     - Implementation → Codex
     - Research → Gemini
     - Complex reasoning → Claude (if available)
     - Default → cheapest enabled
2. Create `src/um_agent_coder/harness/autonomous/opus_guard.py`:
   - Daily Opus iteration limit (default: 50)
   - Track usage and enforce limits
3. Add unit tests in `tests/harness/test_cli_router.py`:
   - Test CLI list parsing
   - Test task analysis
   - Test routing decisions
   - Test Opus guard
4. Run tests and verify all pass

### Verification
```bash
pytest tests/harness/test_cli_router.py -v --cov=src/um_agent_coder/harness/autonomous
```

### Success Criteria
- [ ] CLI list parsing works (`codex,gemini` → set)
- [ ] Auto-router selects appropriate CLI per task
- [ ] Opus guard enforces daily limits
- [ ] All unit tests pass with >90% coverage

---

## Phase 10: Environmental Awareness (NEW)

Implement environmental inputs per `specs/autonomous-loop-spec.md` Section 4.

### Tasks
1. Create `src/um_agent_coder/harness/autonomous/environment/`:
   - `file_watcher.py`: Watch workspace for file changes (using watchdog)
   - `instruction_queue.py`: Read from `.harness/inbox/` directory
   - `env_monitor.py`: Check `HARNESS_*` environment variables
   - `environment_manager.py`: Aggregate all inputs
2. Add instruction queue protocol:
   - Text files in `.harness/inbox/` processed by filename order
   - Move to `.harness/inbox/processed/` after handling
3. Add unit tests in `tests/harness/test_environment.py`:
   - Test file watcher events
   - Test instruction queue polling
   - Test env var detection
4. Run tests and verify all pass

### Verification
```bash
pytest tests/harness/test_environment.py -v --cov=src/um_agent_coder/harness/autonomous
```

### Success Criteria
- [ ] File watcher detects workspace changes
- [ ] Instruction queue processes files correctly
- [ ] Environment variable changes detected
- [ ] All unit tests pass with >85% coverage

---

## Phase 11: Alert System (NEW)

Implement alerts per `specs/autonomous-loop-spec.md` Section 5.

### Tasks
1. Create `src/um_agent_coder/harness/autonomous/alerts/`:
   - `alert_manager.py`: Severity levels (INFO, WARNING, CRITICAL, SUCCESS, ERROR)
   - `runaway_detector.py`: Detect potential infinite loops
   - CLI notification (colored terminal output)
   - File logging to `.harness/alerts.log`
2. Add alert types:
   - `iteration_milestone`, `no_progress`, `stuck_recovery`
   - `approaching_limit`, `model_escalation`, `runaway_detected`
   - `goal_complete`, `fatal_error`
3. Add unit tests in `tests/harness/test_alerts.py`:
   - Test each alert type
   - Test file logging
   - Test runaway detection
4. Run tests and verify all pass

### Verification
```bash
pytest tests/harness/test_alerts.py -v --cov=src/um_agent_coder/harness/autonomous
```

### Success Criteria
- [ ] Alerts display in terminal with colors
- [ ] Alerts logged to `.harness/alerts.log`
- [ ] Runaway detection identifies loops
- [ ] All unit tests pass with >90% coverage

---

## Phase 12: Autonomous Loop Integration (NEW)

Wire all autonomous features into enhanced executor.

### Tasks
1. Create `src/um_agent_coder/harness/autonomous/executor.py`:
   - `AutonomousExecutor` extending `RalphExecutor`
   - Integrate: progress detection, stuck recovery, context management, CLI routing, environment awareness, alerts
   - Support `--max-time` termination
   - Support resume with re-evaluation
2. Update `src/um_agent_coder/harness/main.py`:
   - Add new CLI flags per spec Section 11
   - Add `--autonomous` shorthand mode
   - Detect autonomous config and use `AutonomousExecutor`
3. Update `src/um_agent_coder/harness/state.py`:
   - Add `loop_state`, `loop_iterations`, `recovery_attempts`, `env_events` tables
4. Add integration tests in `tests/harness/test_autonomous_executor.py`:
   - Test full autonomous loop
   - Test time-based termination
   - Test stuck recovery integration
   - Test resume behavior
5. Run all tests

### Verification
```bash
pytest tests/harness/ -v --cov=src/um_agent_coder/harness --cov-report=term-missing
```

### Success Criteria
- [ ] Autonomous executor integrates all features
- [ ] Time-based termination works
- [ ] Resume re-evaluates state correctly
- [ ] All integration tests pass with >80% coverage

---

## Phase 13: Monitoring & Logging (NEW)

Implement real-time monitoring per `specs/autonomous-loop-spec.md` Section 10.

### Tasks
1. Create `src/um_agent_coder/harness/autonomous/monitoring/`:
   - `realtime_logger.py`: Stream logs to terminal and file
   - `status_reporter.py`: Periodic status summaries
2. Enhance `--status` command:
   - Show current iteration, progress, CLI, elapsed time
   - Show recent progress markers
3. Add unit tests in `tests/harness/test_monitoring.py`
4. Run tests and verify all pass

### Verification
```bash
pytest tests/harness/test_monitoring.py -v --cov=src/um_agent_coder/harness/autonomous
```

### Success Criteria
- [ ] Real-time logs stream correctly
- [ ] Status summaries generated at intervals
- [ ] `--status` shows current state
- [ ] All unit tests pass with >90% coverage

---

## Phase 14: Documentation & Examples

Create comprehensive documentation for autonomous loop.

### Tasks
1. Update `docs/ralph-loop.md`:
   - Add autonomous loop section
   - Document all new CLI flags
   - Add configuration examples
2. Create `docs/autonomous-loop.md`:
   - Overview and architecture
   - Progress detection explained
   - Stuck recovery strategies
   - Multi-CLI routing guide
   - Environmental awareness guide
   - Troubleshooting
3. Update `README.md`:
   - Add autonomous loop section
   - Quick start examples
   - Link to detailed docs
4. Update `CLAUDE.md`:
   - Add autonomous harness commands
   - Document new CLI flags
   - Update architecture section

### Verification
```bash
# Verify docs render correctly
cat docs/autonomous-loop.md
cat docs/ralph-loop.md
```

### Success Criteria
- [ ] `docs/autonomous-loop.md` complete and accurate
- [ ] `docs/ralph-loop.md` updated with autonomous features
- [ ] `README.md` includes autonomous loop section
- [ ] `CLAUDE.md` documents all new commands and flags

---

## Phase 15: End-to-End Validation

Validate complete implementation with real execution.

### Tasks
1. Create test roadmap `specs/autonomous-test-roadmap.md`:
   ```markdown
   ## Tasks
   - [ ] **auto-test-001**: Implement a simple feature with tests
     - ralph: true
     - max_iterations: 50
     - max_time: 30m
     - cli: codex,gemini
     - stuck_after: 5
     - completion_promise: FEATURE_COMPLETE
     - success: Feature works, tests pass, coverage > 80%
   ```
2. Run harness with autonomous test task:
   ```bash
   python -m src.um_agent_coder.harness \
     --roadmap specs/autonomous-test-roadmap.md \
     --autonomous \
     --dry-run
   ```
3. Execute for real and verify:
   - Progress detection working
   - Stuck recovery triggers if needed
   - Context accumulates correctly
   - CLI routing makes sensible choices
4. Clean up test artifacts
5. Run full test suite with coverage:
   ```bash
   pytest tests/ -v --cov=src/um_agent_coder --cov-report=term-missing --cov-fail-under=80
   ```
6. Run linting and type checking:
   ```bash
   ruff check src/
   mypy src/
   ```

### Verification
```bash
pytest tests/ -v --cov=src/um_agent_coder --cov-fail-under=80 && ruff check src/ && mypy src/
```

### Success Criteria
- [ ] End-to-end autonomous loop completes successfully
- [ ] Progress detection works correctly
- [ ] Stuck recovery triggers and resolves issues
- [ ] Context accumulates between iterations
- [ ] CLI routing selects appropriate models
- [ ] All tests pass with >80% overall coverage
- [ ] No linting errors
- [ ] No type errors

---

## Phase 16: CI/CD Pipeline Validation

Ensure all GitHub Actions workflows pass on main.

### Tasks
1. Review existing workflows in `.github/workflows/`
2. Update workflows if needed:
   - Add coverage threshold check (>80%)
   - Add autonomous loop test job
   - Ensure all new modules tested
3. Push changes and verify pipeline passes:
   ```bash
   git push origin main
   gh run list --limit 5
   gh run view <run-id>
   ```
4. Fix any failing checks
5. Verify badge shows passing

### Verification
```bash
gh run list --branch main --status completed --limit 1 | grep -q "completed"
```

### Success Criteria
- [ ] All GitHub Actions workflows pass on main
- [ ] Coverage threshold (>80%) enforced in CI
- [ ] New modules included in test coverage
- [ ] CI completes without errors

---

## Final Checklist

Before outputting completion promise, verify **ALL** of the following:

### Code Quality
- [ ] All new code has type hints
- [ ] No commented-out code
- [ ] No TODO comments left unaddressed
- [ ] Consistent code style with existing codebase
- [ ] All imports organized (`isort`)
- [ ] All code formatted (`black`)

### Testing
- [ ] Unit tests for PromiseDetector (>90% coverage)
- [ ] Unit tests for IterationTracker (>90% coverage)
- [ ] Integration tests for RalphExecutor (>85% coverage)
- [ ] Unit tests for ProgressDetector (>90% coverage)
- [ ] Unit tests for StuckRecovery (>85% coverage)
- [ ] Unit tests for ContextManager (>90% coverage)
- [ ] Unit tests for CLIRouter (>90% coverage)
- [ ] Unit tests for Environment (>85% coverage)
- [ ] Unit tests for Alerts (>90% coverage)
- [ ] Integration tests for AutonomousExecutor (>80% coverage)
- [ ] Unit tests for Monitoring (>90% coverage)
- [ ] End-to-end tests for harness integration
- [ ] All tests pass: `pytest tests/ -v --cov-fail-under=80`

### Documentation
- [ ] `docs/autonomous-loop.md` exists and is complete
- [ ] `docs/ralph-loop.md` updated
- [ ] `README.md` updated with autonomous loop section
- [ ] `CLAUDE.md` includes all new commands and flags

### Integration
- [ ] Harness correctly executes autonomous tasks
- [ ] Non-autonomous tasks still work (backward compatibility)
- [ ] All CLI flags functional
- [ ] State persistence working
- [ ] Resume re-evaluates correctly

### CI/CD
- [ ] All GitHub Actions workflows pass on main
- [ ] Coverage threshold enforced (>80%)
- [ ] No linting errors: `ruff check src/`
- [ ] No type errors: `mypy src/`

---

## Workflow Instructions

For each phase:
1. Read existing code to understand patterns
2. Implement the feature following existing conventions
3. Write tests before or alongside implementation
4. Run tests: `pytest tests/harness/ -v --cov`
5. Fix any failures before proceeding
6. Run linting: `ruff check src/`
7. Run type checking: `mypy src/`
8. Fix any errors before proceeding to next phase

If blocked on any phase:
- Document the blocker clearly
- Attempt alternative approaches
- After 3 failed attempts, move to next phase and note the issue

---

## Completion Promise

When **ALL** of the following are true:
1. ALL phases (1-16) are complete
2. ALL Final Checklist items are verified (checked)
3. ALL tests pass with >80% coverage
4. ALL GitHub Actions workflows pass on main
5. ALL documentation is updated

Output: `<promise>AUTONOMOUS_LOOP_IMPLEMENTATION_COMPLETE</promise>`
