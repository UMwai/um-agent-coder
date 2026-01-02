# Autonomous Loop - 24/7 Unattended Task Execution

The Autonomous Loop enables fully autonomous task execution that can run indefinitely until completion, time limits, or manual intervention. It builds on the Ralph Loop foundation with advanced features for production-grade autonomous operation.

## Overview

### Core Vision

Maximize velocity by enabling unlimited Codex/Gemini usage autonomously while preserving scarce Opus tokens for complex reasoning. Agents should:

- Run until manually stopped, time limit reached, or goal completed
- Detect progress via multi-signal scoring
- Self-recover when stuck (prompt mutation → model escalation → branch exploration)
- Respond to real-time environmental inputs
- Route intelligently between CLIs based on model strengths

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AutonomousExecutor                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Progress   │  │   Stuck     │  │   Context   │              │
│  │  Detector   │  │  Recovery   │  │   Manager   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │    CLI      │  │ Environment │  │   Alert     │              │
│  │   Router    │  │   Manager   │  │   System    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐                               │
│  │  Real-Time  │  │   Status    │                               │
│  │   Logger    │  │  Reporter   │                               │
│  └─────────────┘  └─────────────┘                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```bash
# Run with autonomous mode enabled
python -m src.um_agent_coder.harness \
  --roadmap specs/roadmap.md \
  --autonomous

# With time limit
python -m src.um_agent_coder.harness \
  --roadmap specs/roadmap.md \
  --autonomous \
  --max-time 8h

# With specific CLIs
python -m src.um_agent_coder.harness \
  --roadmap specs/roadmap.md \
  --cli codex,gemini \
  --max-iterations 500
```

### Task Definition

```markdown
## Tasks

- [ ] **auth-001**: Implement JWT authentication for the API
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

## Progress Detection

The autonomous loop uses multi-signal progress detection to determine if meaningful work is being done.

### Signals

| Signal | Weight | Description |
|--------|--------|-------------|
| **Output Diff** | 30% | How different is current output from previous iteration |
| **File Changes** | 30% | Git diff showing actual code modifications |
| **Explicit Markers** | 25% | `<progress>...</progress>` tags in output |
| **Checklist Progress** | 15% | Subtasks completed (if defined) |

### Progress Score

Progress score = weighted sum of all signals (0.0 to 1.0)

- **Score > 0.15**: Making progress, continue normally
- **Score < 0.15**: No significant progress detected
- **3 consecutive no-progress**: Trigger stuck recovery

### Using Progress Markers

Include explicit progress markers in your task output:

```
<progress>Completed user registration endpoint</progress>
<progress>Added JWT token generation</progress>
<progress>Implemented middleware for protected routes</progress>
```

## Stuck Recovery

When the loop detects no progress for consecutive iterations, it automatically attempts recovery.

### Recovery Strategies

Recovery attempts strategies in this order:

1. **Prompt Mutation**
   - Rephrase: Reword the task with different emphasis
   - Decompose: Break into smaller subtasks
   - Constrain: Add specific constraints or examples

2. **Model Escalation**
   - Escalate to more capable models:
   - gemini-flash → gemini-pro → codex (gpt-5.2) → sonnet → opus

3. **Branch Exploration**
   - Fork into 2-3 parallel alternative approaches
   - Select best-performing branch based on progress score

### Configuration

```bash
--stuck-after 3          # Iterations without progress before recovery (default: 3)
--recovery-budget 20     # Behind-the-scenes recovery iterations (default: 20)
```

### Recovery Budget

The recovery budget allows up to 20 iterations for behind-the-scenes recovery attempts. These iterations:

- Don't count toward the main iteration limit
- Are invisible to the user unless they check logs
- Allow the loop to try alternative approaches without human intervention

## Multi-CLI Routing

The router automatically selects the optimal CLI based on task characteristics.

### Available CLIs

| CLI | Model | Strengths |
|-----|-------|-----------|
| **codex** | gpt-5.2 | Implementation, code generation, debugging |
| **gemini** | gemini-3-pro | Research, large context (1M tokens), analysis |
| **claude** | claude-opus-4.5 | Complex reasoning, judgment, synthesis |

### Routing Logic

- **Stuck recovery**: Use smartest available model
- **Large context needed**: Route to Gemini (1M context)
- **Implementation tasks**: Route to Codex
- **Research/analysis**: Route to Gemini
- **Complex reasoning**: Route to Claude (if available)
- **Default**: Use cheapest enabled CLI

### Opus Guard

Claude Opus tokens are expensive. The Opus Guard limits daily usage:

```bash
--opus-limit 50          # Maximum Opus iterations per day (default: 50)
```

### CLI Specification

```bash
--cli codex              # Single CLI
--cli codex,gemini       # Multiple CLIs
--cli auto               # Auto-route based on task type
```

## Context Management

The autonomous loop manages context to avoid token limits while preserving important information.

### Rolling Window

Keep the last N iterations in raw form (default: 5):

```bash
--context-window 5       # Raw iterations to keep
```

### Summarization

Older iterations are summarized to preserve context while reducing tokens:

```bash
--summarize-every 10     # Re-summarize interval
```

### Context Structure

```
┌────────────────────────────────────────┐
│            Contextual Prompt            │
├────────────────────────────────────────┤
│  ┌──────────────────────────────────┐  │
│  │         Task Summary              │  │
│  │   (accumulated from iterations)   │  │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │     Recent Iterations (raw)       │  │
│  │   Iteration N-4: ...              │  │
│  │   Iteration N-3: ...              │  │
│  │   Iteration N-2: ...              │  │
│  │   Iteration N-1: ...              │  │
│  │   Iteration N:   ...              │  │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │      Current Task Prompt          │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

## Environmental Awareness

The harness can respond to real-time changes during execution.

### Stop/Pause Files

```bash
# Request graceful stop
touch .harness/stop

# Request pause (will resume on file removal)
touch .harness/pause
```

### Instruction Queue

Drop instructions during execution:

```bash
# Create instruction file
echo "Focus on error handling next" > .harness/inbox/001-high-priority.txt

# Priority from filename:
# 000-* : Urgent (process immediately)
# 001-* : High priority
# 002-* : Normal priority
```

### Environment Variables

```bash
export HARNESS_PAUSE=true     # Pause execution
export HARNESS_PRIORITY=fix   # Prioritize fix tasks
export HARNESS_LOG_LEVEL=DEBUG
```

## Alert System

### Alert Types

| Type | Severity | Description |
|------|----------|-------------|
| `iteration_milestone` | INFO | Every N iterations |
| `no_progress` | WARNING | No progress detected |
| `stuck_recovery` | INFO | Recovery attempt started |
| `approaching_limit` | WARNING | Near iteration/time limit |
| `model_escalation` | INFO | Escalated to better model |
| `runaway_detected` | CRITICAL | Possible infinite loop |
| `goal_complete` | SUCCESS | Task completed successfully |
| `fatal_error` | ERROR | Unrecoverable error |

### Alert Configuration

```bash
--alert-every 10         # Status alert interval
--pause-on-critical      # Pause on critical alerts
```

### Alert Output

Alerts display in terminal with colors:
- INFO: Blue
- WARNING: Yellow
- ERROR/CRITICAL: Red
- SUCCESS: Green

Alerts also logged to `.harness/alerts.log`.

### Runaway Detection

The runaway detector watches for:

1. **Too many iterations without time limit**: Warning after 500 iterations
2. **Repeated identical outputs**: Critical if last 5 outputs are identical
3. **Suspiciously fast iterations**: Warning if iterations much faster than initial

## Monitoring

### Status Command

Check status from another terminal:

```bash
python -m src.um_agent_coder.harness --status
```

Output:
```
Task: auth-001 (Implement JWT authentication)
Status: RUNNING
Iteration: 47/unlimited
Progress: 0.35 (last 5 avg)
CLI: codex (gpt-5.2)
Elapsed: 2h 15m
Last markers: "Completed login endpoint", "Added JWT validation"
```

### Log Files

```
.harness/
├── harness.log      # Execution logs
├── alerts.log       # Alert history
├── status.json      # Current status (JSON format)
├── state.db         # SQLite task state
└── ralph_state.db   # Iteration history
```

### Real-Time Logs

Logs stream to terminal with color coding:
- Iteration start: Blue
- Iteration complete: Green (good progress), Yellow (low progress)
- Progress markers: Cyan
- Stuck detection: Bold Red
- Recovery: Magenta
- Goal complete: Bold Green

## Configuration Reference

### CLI Flags

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
| `--summarize-every` | 10 | Re-summarize interval |
| `--alert-every` | 10 | Status alert interval |
| `--pause-on-critical` | false | Pause on critical alerts |
| `--watch-workspace` | false | Enable file watchers |
| `--enable-inbox` | false | Enable instruction queue |

### Task Properties

| Property | Default | Description |
|----------|---------|-------------|
| `ralph` | false | Enable iterative execution |
| `max_iterations` | 30 | Maximum iterations for task |
| `max_time` | None | Time limit for task |
| `cli` | (global) | CLI to use for task |
| `stuck_after` | 3 | Iterations before stuck recovery |
| `completion_promise` | "COMPLETE" | Promise text to detect |

## Python API

### Basic Usage

```python
from pathlib import Path
from um_agent_coder.harness.autonomous import (
    AutonomousExecutor,
    AutonomousConfig,
    AutonomousResult,
    TerminationReason,
)

# Configure
config = AutonomousConfig(
    max_iterations=200,
    max_time_seconds=4 * 3600,  # 4 hours
    progress_threshold=0.15,
    stuck_after=3,
    cli_spec="codex,gemini",
    completion_promise="TASK_COMPLETE",
)

# Create executor
executor = AutonomousExecutor(
    executors={"codex": codex_executor, "gemini": gemini_executor},
    config=config,
    workspace_path=Path("."),
)

# Execute
result = executor.execute(task)

if result.success:
    print(f"Completed in {result.iterations} iterations")
    print(f"Promise: {result.promise_text}")
else:
    print(f"Failed: {result.termination_reason}")
    print(f"After {result.iterations} iterations")
```

### Key Classes

| Class | Description |
|-------|-------------|
| `AutonomousExecutor` | Main executor integrating all features |
| `AutonomousConfig` | Configuration for autonomous execution |
| `AutonomousResult` | Execution result with termination info |
| `ProgressDetector` | Multi-signal progress detection |
| `StuckDetector` | Detects consecutive no-progress iterations |
| `RecoveryManager` | Coordinates stuck recovery strategies |
| `CLIRouter` | Routes tasks to optimal CLI |
| `ContextManager` | Manages conversation context |
| `EnvironmentManager` | Monitors environment changes |
| `AlertManager` | Handles alerts and notifications |
| `RealTimeLogger` | Streams logs to terminal and file |
| `StatusReporter` | Generates status summaries |

## Troubleshooting

### Task Never Completes

**Symptoms**: Task keeps iterating without outputting promise

**Solutions**:
1. Make success criteria clearer in task description
2. Add explicit promise output instructions
3. Check if promise text appears accidentally early
4. Increase verbosity to see what AI is outputting
5. Try different CLI (codex vs gemini)

### Stuck Recovery Not Working

**Symptoms**: Loop stays stuck despite recovery attempts

**Solutions**:
1. Increase recovery budget (`--recovery-budget 30`)
2. Lower stuck threshold (`--stuck-after 2`)
3. Enable more CLIs for escalation
4. Break task into smaller subtasks
5. Check logs for recovery attempt details

### High Token Usage

**Symptoms**: Excessive API costs

**Solutions**:
1. Enable Opus Guard (`--opus-limit 50`)
2. Use auto-routing (`--cli auto`)
3. Reduce context window (`--context-window 3`)
4. Increase summarization frequency (`--summarize-every 5`)

### Runaway Detected

**Symptoms**: Runaway detection alerts

**Solutions**:
1. Set time limit (`--max-time 8h`)
2. Review task for infinite loop potential
3. Check if output is actually changing
4. Increase iteration warning threshold in config

## Best Practices

### 1. Clear Task Definitions

Write clear, specific task descriptions with explicit success criteria.

### 2. Use Progress Markers

Include `<progress>...</progress>` tags in your prompts to help track progress.

### 3. Set Appropriate Limits

Always set reasonable iteration and/or time limits for production use.

### 4. Monitor Early Runs

Watch the first few iterations to catch issues early.

### 5. Use Auto-Routing

Let the router select the best CLI unless you have specific requirements.

### 6. Check Logs

Review `.harness/harness.log` for detailed execution history.

### 7. Test Incrementally

Start with smaller tasks before running complex multi-hour jobs.
