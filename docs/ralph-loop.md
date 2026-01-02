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
