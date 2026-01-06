# Autonomous Loop Specification

> **Version**: 1.0.0-draft
> **Status**: Draft for Review
> **Author**: Claude Code + User Interview
> **Date**: 2026-01-02

## Executive Summary

Enhance the existing Ralph Loop system to enable truly autonomous agent execution that removes the human bottleneck. Agents should self-loop, detect progress, recover from being stuck, and respond to real-time environmental changes—all while intelligently routing between multiple CLI backends based on model strengths and token efficiency.

**Core Principle**: Maximize velocity by enabling unlimited Codex/Gemini usage autonomously while preserving scarce Opus tokens for complex reasoning.

---

## Table of Contents

1. [Termination Conditions](#1-termination-conditions)
2. [Progress Detection System](#2-progress-detection-system)
3. [Stuck Recovery System](#3-stuck-recovery-system)
4. [Environmental Awareness](#4-environmental-awareness)
5. [Alert System](#5-alert-system)
6. [Context Management](#6-context-management)
7. [Multi-CLI Router](#7-multi-cli-router)
8. [Configuration Schema](#8-configuration-schema)
9. [State Management](#9-state-management)
10. [Monitoring & Logging](#10-monitoring--logging)
11. [CLI Interface](#11-cli-interface)
12. [Implementation Phases](#12-implementation-phases)

---

## 1. Termination Conditions

The autonomous loop terminates when ANY of these conditions are met:

| Condition | Description | Priority |
|-----------|-------------|----------|
| Manual Stop | User sends SIGINT (Ctrl+C) or writes to stop file | Highest |
| Goal Complete | Promise detected in output (existing Ralph behavior) | High |
| Time Limit | Predefined duration exceeded (e.g., `--max-time 8h`) | High |
| Iteration Limit | Max iterations reached (e.g., `--max-iterations 1000`) | Medium |
| Runaway Alert | Alert threshold triggered and mode is `pause` | Medium |

### 1.1 Stop Mechanisms

```python
class TerminationCondition(Enum):
    MANUAL_STOP = "manual_stop"           # SIGINT or stop file
    GOAL_COMPLETE = "goal_complete"       # Promise detected
    TIME_LIMIT = "time_limit"             # Duration exceeded
    ITERATION_LIMIT = "iteration_limit"   # Max iterations
    ALERT_PAUSE = "alert_pause"           # Runaway prevention
    FATAL_ERROR = "fatal_error"           # Unrecoverable error
```

### 1.2 Stop File Convention

Write to `.harness/stop` to gracefully stop the loop:
```bash
echo "stop" > .harness/stop  # Graceful stop after current iteration
echo "abort" > .harness/stop # Immediate abort
```

---

## 2. Progress Detection System

**Strategy**: Combination of multiple signals weighted together (user choice: E)

### 2.1 Progress Signals

| Signal | Weight | Description |
|--------|--------|-------------|
| Output Diff | 30% | Compare iteration N output to N-1 similarity |
| File Changes | 30% | Track git diff / file modifications |
| Explicit Markers | 25% | Agent outputs `<progress>...</progress>` tags |
| Goal Checklist | 15% | Subtasks checked off (if defined) |

### 2.2 Progress Score Calculation

```python
@dataclass
class ProgressSignal:
    output_diff_score: float      # 0.0 = identical, 1.0 = completely different
    file_changes_score: float     # 0.0 = no changes, 1.0 = significant changes
    explicit_markers: List[str]   # Extracted <progress> content
    checklist_progress: float     # 0.0 = none, 1.0 = all complete

def calculate_progress_score(signal: ProgressSignal) -> float:
    """Returns 0.0 (no progress) to 1.0 (significant progress)"""
    weights = {
        'output_diff': 0.30,
        'file_changes': 0.30,
        'explicit_markers': 0.25,
        'checklist': 0.15
    }

    marker_score = min(1.0, len(signal.explicit_markers) * 0.5)

    return (
        weights['output_diff'] * signal.output_diff_score +
        weights['file_changes'] * signal.file_changes_score +
        weights['explicit_markers'] * marker_score +
        weights['checklist'] * signal.checklist_progress
    )
```

### 2.3 No-Progress Threshold

- **Threshold**: `progress_score < 0.15` = no meaningful progress
- **Stuck Detection**: 3 consecutive iterations with no progress triggers stuck recovery
- **Configurable**: `--progress-threshold 0.15` and `--stuck-after 3`

### 2.4 Output Diff Algorithm

```python
from difflib import SequenceMatcher

def output_diff_score(prev_output: str, curr_output: str) -> float:
    """Higher score = more different = more progress"""
    if not prev_output:
        return 1.0  # First iteration always has progress

    # Normalize outputs (strip whitespace, lowercase for comparison)
    prev_norm = normalize(prev_output)
    curr_norm = normalize(curr_output)

    similarity = SequenceMatcher(None, prev_norm, curr_norm).ratio()
    return 1.0 - similarity  # Convert similarity to difference score
```

### 2.5 File Changes Detection

```python
def file_changes_score(workspace: Path) -> float:
    """Detect meaningful file changes since last iteration"""
    result = subprocess.run(
        ["git", "diff", "--stat", "--cached", "HEAD~1"],
        capture_output=True, cwd=workspace
    )

    if not result.stdout:
        return 0.0

    # Parse diff stats: files changed, insertions, deletions
    stats = parse_git_diff_stat(result.stdout)

    # Normalize to 0-1 score (cap at 100 lines changed = 1.0)
    total_changes = stats.insertions + stats.deletions
    return min(1.0, total_changes / 100)
```

---

## 3. Stuck Recovery System

**Strategy**: Prompt mutation + Model escalation + Branch exploration (user choices: A, B, E)

### 3.1 Recovery Trigger

When stuck detected (N iterations with no progress, default N=20 for behind-the-scenes recovery):

```
Stuck Detected (20 cycles no progress)
    ↓
┌─────────────────────────────────────┐
│     STUCK RECOVERY SYSTEM           │
├─────────────────────────────────────┤
│ 1. Prompt Mutation (3 attempts)     │
│ 2. Model Escalation (if still stuck)│
│ 3. Branch Exploration (parallel)    │
│ 4. Human Escalation (last resort)   │
└─────────────────────────────────────┘
```

### 3.2 Recovery Strategies

#### Strategy A: Prompt Mutation

```python
class PromptMutator:
    """Generate alternative phrasings of the goal"""

    mutations = [
        "rephrase",      # Reword the goal differently
        "decompose",     # Break into smaller steps
        "constrain",     # Add specific constraints
        "examples",      # Add concrete examples
        "negative",      # Specify what NOT to do
    ]

    def mutate(self, original_prompt: str, mutation_type: str) -> str:
        """Use LLM to generate mutated prompt"""
        system = f"Rewrite this task using the '{mutation_type}' strategy..."
        # Use cheapest available model for mutation
        return self.llm.generate(system, original_prompt)
```

#### Strategy B: Model Escalation

```python
ESCALATION_ORDER = [
    ("gemini", "gemini-3-flash"),    # Cheapest, fastest
    ("gemini", "gemini-3-pro"),      # Better reasoning
    ("codex", "gpt-5.2"),            # Strong implementation
    ("claude", "claude-sonnet-4"),   # Good balance
    ("claude", "claude-opus-4.5"),   # Most capable (use sparingly)
]

def escalate_model(current_cli: str, current_model: str) -> Tuple[str, str]:
    """Move to next model in escalation order"""
    current_idx = find_index(current_cli, current_model)
    if current_idx < len(ESCALATION_ORDER) - 1:
        return ESCALATION_ORDER[current_idx + 1]
    return None  # Already at highest
```

#### Strategy E: Branch Exploration

```python
@dataclass
class ExplorationBranch:
    branch_id: str
    approach: str           # Description of approach
    prompt_variant: str     # Mutated prompt
    cli: str               # CLI to use
    max_iterations: int    # Budget for this branch

class BranchExplorer:
    """Fork into parallel approaches, pick winner"""

    def explore(self, goal: str, context: str) -> List[ExplorationBranch]:
        """Generate 2-3 parallel exploration branches"""
        branches = [
            ExplorationBranch(
                branch_id="branch-a",
                approach="bottom-up implementation",
                prompt_variant=self.mutate(goal, "decompose"),
                cli="codex",
                max_iterations=10
            ),
            ExplorationBranch(
                branch_id="branch-b",
                approach="research-first",
                prompt_variant=self.mutate(goal, "research"),
                cli="gemini",
                max_iterations=10
            ),
            ExplorationBranch(
                branch_id="branch-c",
                approach="constraint-driven",
                prompt_variant=self.mutate(goal, "constrain"),
                cli="codex",
                max_iterations=10
            ),
        ]
        return branches

    def execute_branches(self, branches: List[ExplorationBranch]) -> ExplorationBranch:
        """Run branches in parallel, return best performer"""
        results = parallel_execute(branches)
        return max(results, key=lambda r: r.progress_score)
```

### 3.3 Recovery Flow

```python
class StuckRecoveryManager:
    def recover(self, task: Task, context: LoopContext) -> RecoveryResult:
        # Stage 1: Prompt Mutations (3 attempts with current model)
        for mutation_type in ["rephrase", "decompose", "constrain"]:
            mutated = self.mutator.mutate(task.goal, mutation_type)
            result = self.execute_with_prompt(mutated, max_iterations=5)
            if result.made_progress:
                return RecoveryResult(success=True, new_prompt=mutated)

        # Stage 2: Model Escalation
        next_model = escalate_model(context.current_cli, context.current_model)
        if next_model:
            result = self.execute_with_model(next_model, max_iterations=10)
            if result.made_progress:
                return RecoveryResult(success=True, escalated_to=next_model)

        # Stage 3: Branch Exploration (parallel)
        branches = self.explorer.explore(task.goal, context.summary)
        winner = self.explorer.execute_branches(branches)
        if winner.progress_score > 0.3:
            return RecoveryResult(success=True, branch=winner)

        # Stage 4: Human Escalation
        self.alert_manager.escalate("Stuck after all recovery attempts")
        return RecoveryResult(success=False, needs_human=True)
```

---

## 4. Environmental Awareness

**Strategy**: Full environmental awareness (user choice: E - all inputs)

### 4.1 Input Sources

| Source | Description | Check Frequency |
|--------|-------------|-----------------|
| File Watchers | React to workspace file changes | Real-time (inotify/fsevents) |
| Instruction Queue | Read from `.harness/inbox/` directory | Every iteration |
| Environment Vars | Check for mode/config changes | Every iteration |
| API Events | Webhook receiver (future) | Event-driven |

### 4.2 File Watcher System

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class WorkspaceWatcher(FileSystemEventHandler):
    """Watch for file changes that should influence the loop"""

    def __init__(self, loop_context: LoopContext):
        self.context = loop_context
        self.ignore_patterns = [
            ".harness/*",
            ".git/*",
            "__pycache__/*",
            "*.pyc",
        ]

    def on_modified(self, event):
        if self.should_ignore(event.src_path):
            return

        self.context.file_events.append(FileEvent(
            type="modified",
            path=event.src_path,
            timestamp=datetime.now()
        ))

        # Signal loop to incorporate changes
        self.context.environment_changed.set()

    def on_created(self, event):
        # Similar handling for new files
        pass
```

### 4.3 Instruction Queue

Users can drop instructions mid-loop:

```bash
# Add instruction to queue
echo "Focus on error handling next" > .harness/inbox/001-instruction.txt
echo "Skip the tests for now" > .harness/inbox/002-instruction.txt

# High-priority instruction (processed immediately)
echo "URGENT: Stop working on feature X" > .harness/inbox/000-urgent.txt
```

```python
class InstructionQueue:
    """Process queued instructions between iterations"""

    def __init__(self, inbox_path: Path):
        self.inbox = inbox_path
        self.processed_path = inbox_path / "processed"

    def poll(self) -> List[Instruction]:
        """Get pending instructions, sorted by priority (filename)"""
        instructions = []
        for file in sorted(self.inbox.glob("*.txt")):
            if file.name == "processed":
                continue
            instructions.append(Instruction(
                id=file.stem,
                content=file.read_text(),
                priority=self._parse_priority(file.name)
            ))
        return instructions

    def mark_processed(self, instruction: Instruction):
        """Move to processed folder"""
        src = self.inbox / f"{instruction.id}.txt"
        dst = self.processed_path / f"{instruction.id}.txt"
        shutil.move(src, dst)
```

### 4.4 Environment Variables

```python
ENV_VARS_MONITORED = {
    "HARNESS_MODE": "normal|turbo|conservative",  # Execution speed
    "HARNESS_PAUSE": "true|false",                # Pause loop
    "HARNESS_CLI": "codex|gemini|claude|auto",    # Override CLI
    "HARNESS_PRIORITY": "speed|quality|cost",     # Optimization target
}

def check_environment_changes(context: LoopContext) -> List[EnvChange]:
    """Detect environment variable changes"""
    changes = []
    for var, valid_values in ENV_VARS_MONITORED.items():
        current = os.environ.get(var)
        previous = context.env_snapshot.get(var)
        if current != previous:
            changes.append(EnvChange(var=var, old=previous, new=current))
            context.env_snapshot[var] = current
    return changes
```

### 4.5 Incorporating Environmental Changes

```python
def build_iteration_prompt(
    task: Task,
    context: LoopContext,
    env_changes: List[EnvChange],
    new_instructions: List[Instruction],
    file_events: List[FileEvent]
) -> str:
    """Build prompt incorporating all environmental inputs"""

    sections = [
        f"## Goal\n{task.goal}",
        f"## Progress Summary\n{context.progress_summary}",
    ]

    if env_changes:
        sections.append(f"## Environment Changes\n" +
            "\n".join(f"- {c.var}: {c.old} → {c.new}" for c in env_changes))

    if new_instructions:
        sections.append(f"## New Instructions (incorporate these)\n" +
            "\n".join(f"- {i.content}" for i in new_instructions))

    if file_events:
        sections.append(f"## File Changes Detected\n" +
            "\n".join(f"- {e.type}: {e.path}" for e in file_events))

    sections.append(
        "## Output Requirements\n"
        "- Output `<progress>description</progress>` to indicate progress made\n"
        "- Output `<promise>COMPLETE</promise>` when goal is fully achieved\n"
    )

    return "\n\n".join(sections)
```

---

## 5. Alert System

**Strategy**: CLI notification + File marker (user choices: A, C)

### 5.1 Alert Types

| Alert Type | Severity | Trigger |
|------------|----------|---------|
| `iteration_milestone` | INFO | Every N iterations (configurable) |
| `no_progress` | WARNING | 3+ iterations without progress |
| `stuck_recovery` | WARNING | Stuck recovery triggered |
| `approaching_limit` | WARNING | 80% of time/iteration limit |
| `model_escalation` | INFO | Model escalated to more capable |
| `runaway_detected` | CRITICAL | Potential infinite loop |
| `goal_complete` | SUCCESS | Promise detected |
| `fatal_error` | ERROR | Unrecoverable error |

### 5.2 Alert Manager

```python
@dataclass
class Alert:
    type: str
    severity: Literal["INFO", "WARNING", "CRITICAL", "SUCCESS", "ERROR"]
    message: str
    timestamp: datetime
    iteration: int
    context: Dict[str, Any] = field(default_factory=dict)

class AlertManager:
    def __init__(self, config: AlertConfig):
        self.config = config
        self.alert_log = Path(".harness/alerts.log")
        self.alerts: List[Alert] = []

    def alert(self, alert_type: str, message: str, severity: str, **context):
        alert = Alert(
            type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            iteration=context.get("iteration", 0),
            context=context
        )

        # Always log to file
        self._write_to_file(alert)

        # CLI notification (real-time)
        self._cli_notify(alert)

        # Track for status queries
        self.alerts.append(alert)

        # Check if should pause
        if severity == "CRITICAL" and self.config.pause_on_critical:
            raise PauseRequested(alert)

    def _write_to_file(self, alert: Alert):
        """Append to alerts.log for external monitoring"""
        with open(self.alert_log, "a") as f:
            f.write(json.dumps({
                "type": alert.type,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "iteration": alert.iteration,
                "context": alert.context
            }) + "\n")

    def _cli_notify(self, alert: Alert):
        """Print to terminal with color coding"""
        colors = {
            "INFO": "\033[94m",      # Blue
            "WARNING": "\033[93m",   # Yellow
            "CRITICAL": "\033[91m",  # Red
            "SUCCESS": "\033[92m",   # Green
            "ERROR": "\033[91m",     # Red
        }
        reset = "\033[0m"
        color = colors.get(alert.severity, "")

        print(f"{color}[{alert.severity}] {alert.type}: {alert.message}{reset}")
```

### 5.3 Runaway Detection

```python
class RunawayDetector:
    """Detect potential infinite loops or runaway execution"""

    def __init__(self, config: RunawayConfig):
        self.config = config
        self.iteration_times: List[float] = []
        self.output_hashes: List[str] = []

    def check(self, iteration: int, duration: float, output: str) -> Optional[Alert]:
        self.iteration_times.append(duration)
        self.output_hashes.append(hash_output(output))

        # Check 1: Too many iterations without time limit
        if iteration > self.config.max_iterations_warning and not self.config.has_time_limit:
            return Alert(
                type="runaway_detected",
                severity="WARNING",
                message=f"Reached {iteration} iterations without time limit set"
            )

        # Check 2: Repeated identical outputs
        if self._detect_output_loop():
            return Alert(
                type="runaway_detected",
                severity="CRITICAL",
                message="Detected repeated identical outputs - possible infinite loop"
            )

        # Check 3: Iterations getting faster (no work being done)
        if self._detect_speedup_pattern():
            return Alert(
                type="runaway_detected",
                severity="WARNING",
                message="Iterations completing unusually fast - may not be doing work"
            )

        return None

    def _detect_output_loop(self, window: int = 5) -> bool:
        """Check if last N outputs are identical"""
        if len(self.output_hashes) < window:
            return False
        recent = self.output_hashes[-window:]
        return len(set(recent)) == 1
```

---

## 6. Context Management

**Strategy**: Rolling window + Hybrid summarization (user choices: A, E)

### 6.1 Context Structure

```python
@dataclass
class IterationContext:
    iteration_number: int
    timestamp: datetime
    cli_used: str
    model_used: str
    prompt: str
    output: str
    progress_score: float
    progress_markers: List[str]
    file_changes: List[str]
    duration_seconds: float

@dataclass
class LoopContext:
    task: Task
    iterations: List[IterationContext]      # Rolling window of recent
    summary: str                             # Compressed history
    total_iterations: int
    start_time: datetime
    current_cli: str
    current_model: str
    env_snapshot: Dict[str, str]
    file_events: List[FileEvent]
    environment_changed: threading.Event
```

### 6.2 Rolling Window

```python
class ContextManager:
    def __init__(self, config: ContextConfig):
        self.config = config
        self.raw_window_size = config.raw_window_size  # Default: 5
        self.summarize_every = config.summarize_every  # Default: 10

    def add_iteration(self, context: LoopContext, iteration: IterationContext):
        """Add iteration, maintaining rolling window"""
        context.iterations.append(iteration)
        context.total_iterations += 1

        # Trim to window size
        if len(context.iterations) > self.raw_window_size:
            # Summarize oldest before removing
            to_summarize = context.iterations[0]
            self._incorporate_into_summary(context, to_summarize)
            context.iterations.pop(0)

        # Periodic full re-summarization
        if context.total_iterations % self.summarize_every == 0:
            context.summary = self._regenerate_summary(context)
```

### 6.3 Hybrid Summarization

```python
class ContextSummarizer:
    """Compress iteration history into summary"""

    def summarize(self, iterations: List[IterationContext], existing_summary: str) -> str:
        """Generate updated summary incorporating new iterations"""

        # Use cheap model for summarization
        prompt = f"""
        ## Existing Progress Summary
        {existing_summary}

        ## New Iterations to Incorporate
        {self._format_iterations(iterations)}

        ## Task
        Generate an updated progress summary that:
        1. Preserves key decisions and findings
        2. Notes what approaches worked/failed
        3. Tracks current state and next steps
        4. Is concise (max 500 words)
        """

        return self.llm.generate(prompt, model="gemini-3-flash")

    def _format_iterations(self, iterations: List[IterationContext]) -> str:
        """Format iterations for summarization"""
        lines = []
        for it in iterations:
            lines.append(f"### Iteration {it.iteration_number}")
            lines.append(f"- CLI: {it.cli_used}, Model: {it.model_used}")
            lines.append(f"- Progress: {it.progress_score:.2f}")
            if it.progress_markers:
                lines.append(f"- Markers: {', '.join(it.progress_markers)}")
            lines.append(f"- Output snippet: {it.output[:500]}...")
        return "\n".join(lines)
```

### 6.4 Building Prompt with Context

```python
def build_contextual_prompt(task: Task, context: LoopContext) -> str:
    """Build prompt with appropriate context"""

    sections = [
        f"# Goal\n{task.goal}",
    ]

    # Include summary of older history
    if context.summary:
        sections.append(f"# Progress Summary (iterations 1-{context.total_iterations - len(context.iterations)})\n{context.summary}")

    # Include raw recent iterations (rolling window)
    if context.iterations:
        recent = "\n\n".join([
            f"## Iteration {it.iteration_number}\n{it.output[:1000]}"
            for it in context.iterations[-3:]  # Last 3 raw
        ])
        sections.append(f"# Recent Iterations\n{recent}")

    # Current state
    sections.append(f"# Current State\n- Total iterations: {context.total_iterations}\n- Current CLI: {context.current_cli}")

    # Instructions
    sections.append(
        "# Instructions\n"
        "Continue working toward the goal. Output:\n"
        "- `<progress>what you accomplished</progress>` for progress updates\n"
        "- `<promise>COMPLETE</promise>` when goal is fully achieved"
    )

    return "\n\n".join(sections)
```

---

## 7. Multi-CLI Router

**Strategy**: User-defined CLI list + auto-router with model strength/efficiency routing

### 7.1 CLI Selection Modes

```bash
# Explicit single CLI
--cli codex

# Explicit multi-CLI (comma-separated)
--cli codex,gemini

# Auto-router (intelligent selection)
--cli auto
```

### 7.2 Model Strength & Efficiency Matrix

| CLI | Model | Strength | Token Cost | Use For |
|-----|-------|----------|------------|---------|
| gemini | gemini-3-flash | Speed | $ | Simple tasks, summarization |
| gemini | gemini-3-pro | Context (1M) | $$ | Research, large codebase |
| codex | gpt-5.2 | Implementation | $$ | Code generation, builds |
| claude | claude-sonnet-4 | Balance | $$$ | Complex tasks, review |
| claude | claude-opus-4.5 | Intelligence | $$$$ | Hardest problems, final review |

### 7.3 Auto-Router Logic

```python
@dataclass
class TaskAnalysis:
    requires_large_context: bool    # > 100k tokens context
    is_implementation: bool         # Writing code
    is_research: bool               # Exploring/researching
    is_complex_reasoning: bool      # Multi-step logic
    is_stuck_recovery: bool         # Recovering from stuck state
    estimated_difficulty: float     # 0.0 (trivial) to 1.0 (hardest)

class AutoRouter:
    """Route tasks to optimal CLI based on characteristics"""

    def __init__(self, enabled_clis: List[str]):
        self.enabled = set(enabled_clis)  # User-enabled CLIs only

    def route(self, analysis: TaskAnalysis) -> Tuple[str, str]:
        """Returns (cli, model) tuple"""

        # Priority 1: Stuck recovery → Use smartest available
        if analysis.is_stuck_recovery:
            return self._get_smartest()

        # Priority 2: Large context → Gemini (1M context)
        if analysis.requires_large_context and "gemini" in self.enabled:
            return ("gemini", "gemini-3-pro")

        # Priority 3: Implementation → Codex
        if analysis.is_implementation and "codex" in self.enabled:
            return ("codex", "gpt-5.2")

        # Priority 4: Research → Gemini
        if analysis.is_research and "gemini" in self.enabled:
            return ("gemini", "gemini-3-pro")

        # Priority 5: Complex reasoning → Claude (use sparingly)
        if analysis.is_complex_reasoning and analysis.estimated_difficulty > 0.7:
            if "claude" in self.enabled:
                return ("claude", "claude-opus-4.5")

        # Default: Cheapest enabled
        return self._get_cheapest()

    def _get_smartest(self) -> Tuple[str, str]:
        """Get smartest available model (for stuck recovery)"""
        if "claude" in self.enabled:
            return ("claude", "claude-opus-4.5")
        if "codex" in self.enabled:
            return ("codex", "gpt-5.2")
        return ("gemini", "gemini-3-pro")

    def _get_cheapest(self) -> Tuple[str, str]:
        """Get cheapest available model (for simple tasks)"""
        if "gemini" in self.enabled:
            return ("gemini", "gemini-3-flash")
        if "codex" in self.enabled:
            return ("codex", "gpt-5.2")
        return ("claude", "claude-sonnet-4")
```

### 7.4 Task Analysis

```python
class TaskAnalyzer:
    """Analyze task to determine routing"""

    IMPLEMENTATION_KEYWORDS = ["implement", "write", "create", "build", "add", "fix"]
    RESEARCH_KEYWORDS = ["research", "explore", "find", "investigate", "analyze"]
    COMPLEX_KEYWORDS = ["design", "architect", "optimize", "refactor", "debug complex"]

    def analyze(self, task: Task, context: LoopContext) -> TaskAnalysis:
        goal_lower = task.goal.lower()

        return TaskAnalysis(
            requires_large_context=self._estimate_context_size(context) > 100_000,
            is_implementation=any(kw in goal_lower for kw in self.IMPLEMENTATION_KEYWORDS),
            is_research=any(kw in goal_lower for kw in self.RESEARCH_KEYWORDS),
            is_complex_reasoning=any(kw in goal_lower for kw in self.COMPLEX_KEYWORDS),
            is_stuck_recovery=context.consecutive_no_progress >= 3,
            estimated_difficulty=self._estimate_difficulty(task, context)
        )

    def _estimate_difficulty(self, task: Task, context: LoopContext) -> float:
        """Estimate task difficulty 0.0-1.0"""
        factors = [
            0.3 if context.total_iterations > 20 else 0.0,  # Many iterations = harder
            0.2 if context.consecutive_no_progress > 0 else 0.0,  # Stuck = harder
            0.2 if len(task.goal) > 500 else 0.0,  # Long goal = complex
            0.3 if "complex" in task.goal.lower() else 0.0,  # Explicit complexity
        ]
        return min(1.0, sum(factors))
```

### 7.5 Token Efficiency: Opus Preservation

```python
class OpusGuard:
    """Preserve scarce Opus tokens"""

    def __init__(self, config: OpusConfig):
        self.daily_limit = config.daily_opus_iterations  # e.g., 50
        self.used_today = 0
        self.last_reset = date.today()

    def can_use_opus(self) -> bool:
        self._maybe_reset()
        return self.used_today < self.daily_limit

    def record_opus_use(self):
        self.used_today += 1

    def _maybe_reset(self):
        if date.today() != self.last_reset:
            self.used_today = 0
            self.last_reset = date.today()

# Integration with router
class AutoRouter:
    def route(self, analysis: TaskAnalysis) -> Tuple[str, str]:
        # ... other routing logic ...

        if analysis.is_complex_reasoning and "claude" in self.enabled:
            if self.opus_guard.can_use_opus():
                self.opus_guard.record_opus_use()
                return ("claude", "claude-opus-4.5")
            else:
                # Opus exhausted, fall back
                return ("codex", "gpt-5.2")
```

---

## 8. Configuration Schema

### 8.1 Full Configuration

```yaml
# config/autonomous-loop.yaml

loop:
  # Termination
  max_iterations: null          # null = unlimited
  max_time: "24h"               # Duration string: "30m", "8h", "24h", null

  # Progress detection
  progress_threshold: 0.15      # Below this = no progress
  stuck_after: 3                # Iterations before stuck recovery

  # Stuck recovery
  recovery:
    prompt_mutations: 3         # Mutation attempts before escalation
    branch_exploration: true    # Enable parallel branch exploration
    branch_count: 3             # Number of parallel branches
    branch_budget: 10           # Max iterations per branch

# CLI routing
cli:
  enabled: ["codex", "gemini"]  # Or ["auto"] for auto-routing
  default: "codex"              # Default when not auto-routing

  # Auto-router settings
  auto_router:
    prefer_cheap: true          # Prefer cheaper models when possible
    opus_daily_limit: 50        # Max Opus iterations per day

# Context management
context:
  raw_window_size: 5            # Recent iterations to keep raw
  summarize_every: 10           # Re-summarize every N iterations
  max_summary_tokens: 2000      # Summary size limit

# Alerts
alerts:
  milestone_every: 10           # Alert every N iterations
  pause_on_critical: true       # Pause loop on critical alerts
  log_file: ".harness/alerts.log"

# Environmental awareness
environment:
  watch_workspace: true         # Enable file watchers
  inbox_enabled: true           # Enable instruction queue
  inbox_path: ".harness/inbox"
  check_env_vars: true          # Monitor environment variables

# Runaway prevention
runaway:
  max_iterations_warning: 100   # Warn if no limit set and exceeded
  identical_output_window: 5    # Detect N identical outputs
  min_iteration_duration: 1.0   # Seconds - faster = suspicious
```

### 8.2 Task-Level Overrides (in roadmap.md)

```markdown
## Tasks

- [ ] **task-001**: Implement authentication system
  - timeout: 4h
  - max_iterations: 200
  - cli: codex,gemini
  - stuck_after: 5
  - completion_promise: AUTH_COMPLETE
  - success: All auth tests pass
```

---

## 9. State Management

### 9.1 Enhanced State Schema

```sql
-- Extends existing .harness/state.db

-- Loop state table
CREATE TABLE IF NOT EXISTS loop_state (
    task_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,                    -- running, paused, completed, failed
    current_iteration INTEGER DEFAULT 0,
    total_iterations INTEGER DEFAULT 0,
    start_time TEXT,
    last_update TEXT,
    current_cli TEXT,
    current_model TEXT,
    progress_summary TEXT,
    context_json TEXT,                       -- Serialized LoopContext
    termination_reason TEXT,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

-- Iteration history
CREATE TABLE IF NOT EXISTS loop_iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    iteration_number INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    cli_used TEXT,
    model_used TEXT,
    prompt_hash TEXT,                        -- For deduplication
    output_snippet TEXT,                     -- First 1000 chars
    output_hash TEXT,                        -- For loop detection
    progress_score REAL,
    progress_markers TEXT,                   -- JSON array
    file_changes TEXT,                       -- JSON array
    duration_seconds REAL,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

-- Recovery attempts
CREATE TABLE IF NOT EXISTS recovery_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    triggered_at_iteration INTEGER,
    strategy TEXT,                           -- mutation, escalation, branch
    details TEXT,                            -- JSON with strategy-specific data
    outcome TEXT,                            -- success, failed
    iterations_used INTEGER,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

-- Environmental events
CREATE TABLE IF NOT EXISTS env_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    event_type TEXT,                         -- file_change, instruction, env_var
    details TEXT,                            -- JSON
    incorporated_at_iteration INTEGER,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);
```

### 9.2 Resume Behavior

**Strategy**: Re-evaluate - read progress, decide next step fresh (user choice: B)

```python
class LoopResumer:
    """Resume loop by re-evaluating state"""

    def resume(self, task_id: str) -> LoopContext:
        # Load state from DB
        state = self.db.get_loop_state(task_id)
        iterations = self.db.get_iterations(task_id)

        # Rebuild context
        context = LoopContext(
            task=self.db.get_task(task_id),
            iterations=iterations[-self.config.raw_window_size:],
            summary=state.progress_summary,
            total_iterations=state.total_iterations,
            start_time=datetime.fromisoformat(state.start_time),
            current_cli=state.current_cli,
            current_model=state.current_model,
        )

        # Re-evaluate: analyze what was happening
        analysis = self.analyzer.analyze_resume_point(context)

        # Decide next action
        if analysis.was_stuck:
            # Continue stuck recovery
            return self._resume_recovery(context, analysis)
        elif analysis.was_making_progress:
            # Continue from where we left off
            return context
        else:
            # Re-orient: generate fresh assessment
            context.summary = self._generate_fresh_assessment(context)
            return context

    def _generate_fresh_assessment(self, context: LoopContext) -> str:
        """Generate fresh assessment of progress"""
        prompt = f"""
        Review this task resumption:

        Goal: {context.task.goal}
        Iterations completed: {context.total_iterations}
        Last summary: {context.summary}

        Generate a fresh assessment:
        1. What has been accomplished?
        2. What remains to be done?
        3. What should be the immediate next step?
        """
        return self.llm.generate(prompt, model="gemini-3-flash")
```

---

## 10. Monitoring & Logging

### 10.1 Real-Time Logs (Priority 1)

```python
class RealTimeLogger:
    """Stream logs to terminal and file"""

    def __init__(self, log_path: Path = Path(".harness/harness.log")):
        self.log_path = log_path
        self.console = Console()  # Rich console for formatting

    def log_iteration_start(self, iteration: int, cli: str, model: str):
        msg = f"[{datetime.now().isoformat()}] Iteration {iteration} starting (cli={cli}, model={model})"
        self._write(msg, style="bold blue")

    def log_iteration_complete(self, iteration: int, progress: float, duration: float):
        color = "green" if progress > 0.3 else "yellow" if progress > 0.1 else "red"
        msg = f"[{datetime.now().isoformat()}] Iteration {iteration} complete (progress={progress:.2f}, duration={duration:.1f}s)"
        self._write(msg, style=color)

    def log_progress_marker(self, iteration: int, marker: str):
        msg = f"[{datetime.now().isoformat()}] Progress: {marker}"
        self._write(msg, style="cyan")

    def log_stuck_detected(self, iteration: int, consecutive: int):
        msg = f"[{datetime.now().isoformat()}] STUCK DETECTED at iteration {iteration} ({consecutive} consecutive no-progress)"
        self._write(msg, style="bold red")

    def log_recovery_attempt(self, strategy: str, details: str):
        msg = f"[{datetime.now().isoformat()}] Recovery: {strategy} - {details}"
        self._write(msg, style="magenta")

    def _write(self, msg: str, style: str = None):
        # Console output
        self.console.print(msg, style=style)
        # File output
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")
```

### 10.2 Periodic Status Summaries (Priority 2)

```python
class StatusReporter:
    """Generate periodic status summaries"""

    def __init__(self, interval_iterations: int = 10):
        self.interval = interval_iterations

    def maybe_report(self, context: LoopContext) -> Optional[str]:
        if context.total_iterations % self.interval != 0:
            return None
        return self.generate_summary(context)

    def generate_summary(self, context: LoopContext) -> str:
        elapsed = datetime.now() - context.start_time
        recent_progress = [it.progress_score for it in context.iterations]
        avg_progress = sum(recent_progress) / len(recent_progress) if recent_progress else 0

        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    AUTONOMOUS LOOP STATUS                     ║
╠══════════════════════════════════════════════════════════════╣
║ Task: {context.task.id[:50]:<50} ║
║ Goal: {context.task.goal[:50]:<50} ║
╠══════════════════════════════════════════════════════════════╣
║ Iterations: {context.total_iterations:<10} Elapsed: {str(elapsed)[:10]:<15}        ║
║ Current CLI: {context.current_cli:<10} Model: {context.current_model:<20}   ║
║ Avg Progress (last {len(recent_progress)}): {avg_progress:.2f}                               ║
╠══════════════════════════════════════════════════════════════╣
║ Recent Progress Markers:                                      ║
{self._format_markers(context)}
╚══════════════════════════════════════════════════════════════╝
        """
        return summary
```

### 10.3 Status Command

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
# Last marker: "Completed login endpoint"
```

---

## 11. CLI Interface

### 11.1 New Flags

```bash
python -m src.um_agent_coder.harness \
    --roadmap specs/roadmap.md \
    # Termination
    --max-time 8h \                      # Time limit (e.g., 30m, 8h, 24h)
    --max-iterations 500 \               # Iteration limit (or 'unlimited')
    # CLI routing
    --cli codex,gemini \                 # Enabled CLIs (or 'auto')
    --opus-limit 50 \                    # Daily Opus iteration limit
    # Progress & recovery
    --progress-threshold 0.15 \          # Progress detection threshold
    --stuck-after 3 \                    # Iterations before stuck recovery
    --recovery-budget 20 \               # Behind-scenes recovery iterations
    # Context
    --context-window 5 \                 # Raw iterations to keep
    --summarize-every 10 \               # Re-summarize interval
    # Alerts
    --alert-every 10 \                   # Status alert interval
    --pause-on-critical \                # Pause on critical alerts
    # Environment
    --watch-workspace \                  # Enable file watchers
    --enable-inbox \                     # Enable instruction queue
    # Modes
    --autonomous \                       # Full autonomous mode (all features)
    --daemon                             # Run as background daemon
```

### 11.2 Shorthand Modes

```bash
# Quick autonomous mode with sensible defaults
python -m src.um_agent_coder.harness --autonomous --roadmap specs/roadmap.md

# Equivalent to:
# --max-time 24h --cli auto --watch-workspace --enable-inbox --alert-every 10

# Conservative mode (more human oversight)
python -m src.um_agent_coder.harness --conservative --roadmap specs/roadmap.md

# Equivalent to:
# --max-iterations 100 --alert-every 5 --pause-on-critical
```

### 11.3 Runtime Commands

While running, the harness responds to:

```bash
# Via inbox
echo "pause" > .harness/inbox/cmd-pause.txt      # Pause loop
echo "resume" > .harness/inbox/cmd-resume.txt    # Resume loop
echo "status" > .harness/inbox/cmd-status.txt    # Force status output
echo "stop" > .harness/stop                       # Graceful stop

# Via environment
export HARNESS_PAUSE=true                         # Pause loop
export HARNESS_CLI=claude                         # Switch CLI
```

---

## 12. Implementation Phases

### Phase 1: Core Enhancements (Foundation)
- [ ] Progress detection system (multi-signal)
- [ ] Context accumulation with rolling window + summarization
- [ ] Time-based termination (--max-time)
- [ ] Enhanced real-time logging

### Phase 2: Stuck Recovery
- [ ] No-progress detection
- [ ] Prompt mutation strategies
- [ ] Model escalation
- [ ] Branch exploration (parallel)

### Phase 3: Multi-CLI Router
- [ ] CLI list parsing (--cli codex,gemini)
- [ ] Auto-router with task analysis
- [ ] Opus preservation guard
- [ ] Per-iteration CLI selection

### Phase 4: Environmental Awareness
- [ ] File watchers (watchdog)
- [ ] Instruction queue (.harness/inbox/)
- [ ] Environment variable monitoring
- [ ] Runtime command handling

### Phase 5: Alerts & Monitoring
- [ ] Alert manager with severity levels
- [ ] Runaway detection
- [ ] Periodic status summaries
- [ ] --status command enhancement

### Phase 6: Polish & Integration
- [ ] Shorthand modes (--autonomous, --conservative)
- [ ] Configuration file support
- [ ] Resume re-evaluation logic
- [ ] Documentation updates

---

## Appendix A: Example Roadmap

```markdown
# Autonomous Loop Example Roadmap

## Tasks

- [ ] **auth-001**: Implement JWT authentication for the API
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

- [ ] **perf-002**: Optimize database queries
  - max_iterations: unlimited
  - max_time: 8h
  - cli: auto
  - completion_promise: PERF_OPTIMIZED
  - success: P95 latency < 100ms
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Progress Score** | 0.0-1.0 measure of iteration productivity |
| **Stuck Recovery** | Automated attempts to escape no-progress loops |
| **Rolling Window** | Last N iterations kept in raw form |
| **Branch Exploration** | Parallel execution of alternative approaches |
| **Opus Guard** | Rate limiter for expensive Claude Opus tokens |
| **Instruction Queue** | File-based system for injecting runtime commands |

---

*End of Specification*
