# Ralph Loop - Autonomous Iterative Task Execution

The Ralph Loop is an autonomous execution pattern that enables tasks to iterate until completion criteria are met. Named after the "Ralph Wiggum" technique, it allows the harness to continuously re-execute a task, checking for a completion "promise" in the output after each iteration.

## Overview

Traditional task execution runs once and reports success or failure. Ralph Loop tasks run repeatedly until:

1. The completion promise is detected in the output
2. Maximum iterations are reached
3. An unrecoverable error occurs

This is useful for tasks that require iterative refinement, such as:

- Implementing features with test coverage requirements
- Fixing bugs that may require multiple attempts
- Refactoring code until quality metrics are met
- Writing documentation until approval criteria are satisfied

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    Ralph Loop Flow                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐                                        │
│  │ Start Task   │                                        │
│  └──────┬───────┘                                        │
│         │                                                │
│         ▼                                                │
│  ┌──────────────┐     No      ┌──────────────┐          │
│  │ Can Continue?├────────────►│ Return Fail  │          │
│  │(iter < max)  │             │(max exceeded)│          │
│  └──────┬───────┘             └──────────────┘          │
│         │ Yes                                            │
│         ▼                                                │
│  ┌──────────────┐                                        │
│  │   Execute    │                                        │
│  │  CLI Task    │                                        │
│  └──────┬───────┘                                        │
│         │                                                │
│         ▼                                                │
│  ┌──────────────┐     Yes     ┌──────────────┐          │
│  │ Promise      ├────────────►│ Return       │          │
│  │ Detected?    │             │ Success      │          │
│  └──────┬───────┘             └──────────────┘          │
│         │ No                                             │
│         │                                                │
│         └────────────────────┐                          │
│                              │                          │
│                              ▼                          │
│                       ┌──────────────┐                  │
│                       │  Increment   │                  │
│                       │  Iteration   │                  │
│                       └──────┬───────┘                  │
│                              │                          │
│                              └──────────► (loop back)   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Configuration

### Roadmap Task Definition

Enable Ralph Loop for a task in your roadmap file:

```markdown
## Tasks

### Phase 1: Implementation

- [ ] **task-001**: Implement user authentication with tests
  - ralph: true
  - max_iterations: 30
  - completion_promise: AUTH_FEATURE_COMPLETE
  - timeout: 60min
  - success: All tests pass, coverage > 80%
  - cli: codex
```

### Task Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `ralph` | bool | false | Enable Ralph Loop for this task |
| `max_iterations` | int | 30 | Maximum iterations before giving up |
| `completion_promise` | string | "COMPLETE" | Text to detect in output for completion |
| `timeout` | int | 30 | Timeout in minutes per iteration |
| `cli` | string | (global) | CLI backend: `codex`, `gemini`, `claude` |

### CLI Flags

Override defaults for all ralph tasks:

```bash
# Set default max iterations
python -m src.um_agent_coder.harness \
  --roadmap specs/roadmap.md \
  --ralph-default-iterations 50

# Set default completion promise
python -m src.um_agent_coder.harness \
  --roadmap specs/roadmap.md \
  --ralph-default-promise "TASK_DONE"
```

## Promise Detection

The harness detects completion promises in two formats:

### XML Format (Recommended)

```
<promise>FEATURE_COMPLETE</promise>
```

The XML format is more reliable as it's unambiguous and unlikely to appear in regular output.

### Plain Text Format

```
FEATURE_COMPLETE
```

Plain text matching is case-insensitive by default.

### Custom Promise Text

Choose promise text that:

1. Is unique and won't appear accidentally in code or logs
2. Describes the completion state clearly
3. Uses uppercase with underscores for readability

Good examples:
- `FEATURE_X_IMPLEMENTATION_COMPLETE`
- `BUG_FIX_VERIFIED`
- `REFACTORING_DONE_ALL_TESTS_PASS`

Bad examples:
- `DONE` (too common)
- `complete` (may appear in code)
- `success` (ambiguous)

## Best Practices

### 1. Write Clear Task Descriptions

The AI needs to understand what "done" means:

```markdown
- [ ] **task-001**: Implement login API endpoint
  - ralph: true
  - completion_promise: LOGIN_API_COMPLETE
  - success: POST /api/login accepts email/password, returns JWT, has unit tests
```

### 2. Set Appropriate Iteration Limits

| Task Complexity | Recommended max_iterations |
|-----------------|---------------------------|
| Simple bug fix | 5-10 |
| Feature implementation | 15-30 |
| Complex refactoring | 30-50 |
| Large feature with tests | 50+ |

### 3. Use Specific Success Criteria

The success criteria helps the AI understand when to output the promise:

```markdown
# Bad - vague
- success: Feature works

# Good - specific
- success: API returns 200 for valid credentials, 401 for invalid, has 5+ unit tests
```

### 4. Match CLI to Task Type

| Task Type | Recommended CLI |
|-----------|----------------|
| Code implementation | codex |
| Research/analysis | gemini |
| Complex reasoning | claude |

### 5. Instruct the AI About the Promise

Include instructions in the task description:

```markdown
- [ ] **task-001**: Implement feature X. Output <promise>FEATURE_COMPLETE</promise> when all tests pass.
  - ralph: true
  - completion_promise: FEATURE_COMPLETE
```

## State Persistence

Ralph Loop state is persisted to SQLite, enabling:

- **Resume after interruption**: If the harness crashes or is stopped, it resumes from the last iteration
- **Iteration history**: All iterations are logged with timestamps and output snippets
- **Progress tracking**: Query iteration counts and durations

### State Files

```
.harness/
├── state.db          # Main harness state (includes ralph iterations)
└── ralph_state.db    # Detailed iteration history
```

### Checking Status

```bash
# View harness statistics including ralph iterations
python -m src.um_agent_coder.harness --status
```

## Troubleshooting

### Task Never Completes

**Symptoms**: Task keeps iterating without outputting promise

**Solutions**:
1. Make the success criteria clearer
2. Add explicit promise output instructions to the task description
3. Check that the promise text isn't being matched accidentally early
4. Increase verbosity to see what the AI is outputting

### Task Completes Too Early

**Symptoms**: Promise detected before task is actually complete

**Solutions**:
1. Use more specific promise text (e.g., `MY_UNIQUE_TASK_123_COMPLETE`)
2. Use XML format (`<promise>...</promise>`)
3. Set `require_xml_format: true` in config

### Max Iterations Exceeded

**Symptoms**: Task fails after max iterations

**Solutions**:
1. Increase `max_iterations`
2. Break task into smaller subtasks
3. Clarify success criteria
4. Use a more capable model/CLI

### State Corruption

**Symptoms**: Strange behavior after restart

**Solutions**:
```bash
# Reset all state
python -m src.um_agent_coder.harness --reset

# Or manually delete state files
rm -rf .harness/
```

## Examples

### Feature Implementation

```markdown
- [ ] **feat-001**: Add password reset functionality
  - ralph: true
  - max_iterations: 25
  - completion_promise: PASSWORD_RESET_COMPLETE
  - timeout: 45min
  - success: POST /api/reset-password sends email, token expires in 24h, has integration tests
  - cli: codex
```

### Bug Fix

```markdown
- [ ] **fix-001**: Fix race condition in user service
  - ralph: true
  - max_iterations: 15
  - completion_promise: RACE_CONDITION_FIXED
  - timeout: 30min
  - success: Concurrent user creation no longer causes duplicates, regression test added
  - cli: codex
```

### Code Refactoring

```markdown
- [ ] **refactor-001**: Extract authentication into separate module
  - ralph: true
  - max_iterations: 40
  - completion_promise: AUTH_MODULE_EXTRACTED
  - timeout: 60min
  - success: auth/ module exists, all auth logic moved, imports updated, all tests pass
  - cli: claude
```

## API Reference

### RalphConfig

```python
@dataclass
class RalphConfig:
    enabled: bool = True
    max_iterations: int = 30
    completion_promise: str = "COMPLETE"
    require_xml_format: bool = True
```

### RalphResult

```python
@dataclass
class RalphResult:
    success: bool
    iterations: int
    total_duration: timedelta
    final_output: str
    reason: Optional[str] = None
    promise_text: Optional[str] = None
    error: Optional[str] = None
```

### Key Classes

| Class | Description |
|-------|-------------|
| `PromiseDetector` | Detects completion promises in output |
| `IterationTracker` | Tracks iteration state and history |
| `RalphPersistence` | SQLite persistence for iteration state |
| `RalphExecutor` | Main executor wrapper implementing the loop |

## Integration with Harness

The Ralph Loop integrates seamlessly with the existing harness:

```bash
# Ralph and non-ralph tasks can coexist
python -m src.um_agent_coder.harness --roadmap specs/roadmap.md

# Parallel execution supports ralph tasks
python -m src.um_agent_coder.harness --roadmap specs/roadmap.md --parallel
```

Non-ralph tasks execute normally with a single attempt. Ralph tasks loop until completion or max iterations.

---

## Autonomous Loop (Advanced)

The Autonomous Loop extends Ralph Loop with advanced features for 24/7 unattended execution:

- **Multi-signal progress detection**: Tracks output changes, file modifications, explicit markers
- **Stuck recovery**: Automatic recovery when no progress is detected
- **Context management**: Rolling window + summarization to manage conversation length
- **Multi-CLI routing**: Intelligent routing between Codex, Gemini, and Claude
- **Environmental awareness**: File watchers, instruction queue, environment variables
- **Alert system**: CLI notifications, file logging, runaway detection

### Enabling Autonomous Mode

```bash
# Full autonomous mode with all features
python -m src.um_agent_coder.harness \
  --roadmap specs/roadmap.md \
  --autonomous

# Or with specific configurations
python -m src.um_agent_coder.harness \
  --roadmap specs/roadmap.md \
  --max-time 8h \
  --max-iterations 500 \
  --cli codex,gemini \
  --progress-threshold 0.15 \
  --stuck-after 3
```

### Autonomous Task Definition

```markdown
- [ ] **task-001**: Implement JWT authentication
  - ralph: true
  - max_iterations: 200
  - max_time: 4h
  - cli: codex,gemini
  - stuck_after: 5
  - completion_promise: AUTH_COMPLETE
  - success: All auth tests pass, tokens validate correctly

  Goal: Create a complete JWT authentication system including:
  1. User registration endpoint
  2. Login endpoint returning JWT
  3. Middleware for protected routes
  4. Token refresh mechanism

  Output `<progress>...</progress>` after each component.
  Output `<promise>AUTH_COMPLETE</promise>` when all tests pass.
```

### Progress Detection

The autonomous loop uses multi-signal progress detection:

| Signal | Weight | Description |
|--------|--------|-------------|
| Output diff | 30% | How different is current output from previous |
| File changes | 30% | Git diff showing actual code changes |
| Explicit markers | 25% | `<progress>...</progress>` tags in output |
| Checklist progress | 15% | Subtasks completed (if defined) |

Progress score = weighted sum of signals (0.0 to 1.0)

### Stuck Recovery

When progress falls below threshold for consecutive iterations:

1. **Prompt Mutation**: Rephrase the task with different emphasis
2. **Model Escalation**: Try a more capable model (gemini → codex → claude)
3. **Branch Exploration**: Try alternative approaches in parallel

Configuration:
```bash
--stuck-after 3          # Trigger recovery after 3 no-progress iterations
--recovery-budget 20     # Behind-the-scenes recovery iterations
```

### CLI Routing

The router selects the optimal CLI based on task analysis:

| CLI | Model | Best For |
|-----|-------|----------|
| codex | gpt-5.2 | Implementation, code generation |
| gemini | gemini-3-pro | Research, large context analysis |
| claude | claude-opus-4.5 | Complex reasoning, judgment calls |

```bash
--cli codex,gemini       # Enable specific CLIs
--cli auto               # Auto-route based on task type
--opus-limit 50          # Daily Opus iteration limit
```

### Environmental Awareness

The harness can respond to real-time changes:

**Stop/Pause Files**:
```bash
# Create stop file to request graceful stop
touch .harness/stop

# Create pause file to pause execution
touch .harness/pause
```

**Instruction Queue**:
```bash
# Drop instructions during execution
echo "Focus on error handling next" > .harness/inbox/001-instruction.txt
```

**Environment Variables**:
```bash
export HARNESS_PAUSE=true    # Pause execution
export HARNESS_PRIORITY=fix  # Prioritize fix tasks
```

### Monitoring

```bash
# Check status from another terminal
python -m src.um_agent_coder.harness --status

# Output:
# Task: task-001 (Implement authentication)
# Status: RUNNING
# Iteration: 47/unlimited
# Progress: 0.35 (last 5 avg)
# CLI: codex (gpt-5.2)
# Elapsed: 2h 15m
```

**Log Files**:
```
.harness/
├── harness.log      # Execution logs
├── alerts.log       # Alert history
└── status.json      # Current status (JSON)
```

### Autonomous Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 1000 | Maximum total iterations |
| `max_time` | None | Time limit (e.g., "8h", "30m") |
| `progress_threshold` | 0.15 | Minimum progress to count as making progress |
| `stuck_after` | 3 | Iterations without progress before recovery |
| `recovery_budget` | 20 | Behind-the-scenes recovery iterations |
| `context_window` | 5 | Raw iterations to keep in context |
| `summarize_every` | 10 | Re-summarize interval |
| `alert_every` | 10 | Status alert interval |
| `pause_on_critical` | false | Pause on critical alerts |

### Python API

```python
from um_agent_coder.harness.autonomous import (
    AutonomousExecutor,
    AutonomousConfig,
    AutonomousResult,
    TerminationReason,
)

# Configure autonomous execution
config = AutonomousConfig(
    max_iterations=200,
    max_time_seconds=4 * 3600,  # 4 hours
    progress_threshold=0.15,
    stuck_after=3,
    cli_spec="codex,gemini",
    completion_promise="TASK_COMPLETE",
)

# Create executor with CLI backends
executor = AutonomousExecutor(
    executors={"codex": codex_executor, "gemini": gemini_executor},
    config=config,
    workspace_path=Path("."),
)

# Execute task
result = executor.execute(task)

if result.success:
    print(f"Completed in {result.iterations} iterations")
else:
    print(f"Failed: {result.termination_reason}")
```

### Key Classes

| Class | Description |
|-------|-------------|
| `AutonomousExecutor` | Main executor integrating all autonomous features |
| `ProgressDetector` | Multi-signal progress detection |
| `StuckDetector` | Detects when loop is stuck |
| `RecoveryManager` | Coordinates stuck recovery strategies |
| `CLIRouter` | Routes tasks to optimal CLI |
| `ContextManager` | Manages conversation context |
| `EnvironmentManager` | Monitors environment changes |
| `AlertManager` | Handles alerts and notifications |
| `RealTimeLogger` | Streams logs to terminal and file |
| `StatusReporter` | Generates status summaries |
