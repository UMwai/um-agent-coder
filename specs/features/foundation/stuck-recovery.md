# Stuck Recovery Specification

> **Priority**: MEDIUM (Foundation)
> **Status**: Implemented
> **Location**: `src/um_agent_coder/harness/autonomous/recovery/`

## Overview

Stuck recovery automatically attempts to escape no-progress loops using a tiered strategy: prompt mutation, model escalation, and branch exploration.

## When Recovery Triggers

Recovery triggers when:
- `consecutive_no_progress >= stuck_after` (default: 3)
- Progress score < `progress_threshold` (default: 0.15)

```python
class StuckDetector:
    def check(self, progress_score: float) -> StuckState:
        if progress_score >= self.threshold:
            self.consecutive_no_progress = 0
            return StuckState.PROGRESSING

        self.consecutive_no_progress += 1

        if self.consecutive_no_progress >= self.stuck_after:
            return StuckState.STUCK

        if self.consecutive_no_progress >= 2:
            return StuckState.WARNING

        return StuckState.PROGRESSING
```

## Recovery Strategies

### Recovery Flow

```
Stuck Detected
    │
    ▼
┌───────────────────────────────────────────────┐
│ Stage 1: Prompt Mutation (3 attempts)         │
│   └─ Try: rephrase → decompose → constrain    │
└───────────────────────────────────────────────┘
    │ Still stuck?
    ▼
┌───────────────────────────────────────────────┐
│ Stage 2: Model Escalation                     │
│   └─ Upgrade to smarter model                 │
└───────────────────────────────────────────────┘
    │ Still stuck?
    ▼
┌───────────────────────────────────────────────┐
│ Stage 3: Branch Exploration (parallel)        │
│   └─ Try 3 different approaches               │
└───────────────────────────────────────────────┘
    │ Still stuck?
    ▼
┌───────────────────────────────────────────────┐
│ Stage 4: Human Escalation                     │
│   └─ Alert user, request guidance             │
└───────────────────────────────────────────────┘
```

### Stage 1: Prompt Mutation

Try different phrasings of the goal:

```python
class PromptMutator:
    mutations = [
        "rephrase",      # Reword the goal
        "decompose",     # Break into smaller steps
        "constrain",     # Add specific constraints
        "examples",      # Add concrete examples
        "negative",      # Specify what NOT to do
    ]

    def mutate(self, original: str, mutation_type: str) -> str:
        prompt = f"""
        Rewrite this task using the '{mutation_type}' strategy:

        Original: {original}

        {self._get_strategy_instructions(mutation_type)}
        """
        return self.llm.generate(prompt)  # Use cheap model
```

**Mutation Strategies**:

| Strategy | Description | Example |
|----------|-------------|---------|
| rephrase | Same meaning, different words | "Create login" → "Build authentication endpoint" |
| decompose | Break into steps | "Create login" → "1. Create model, 2. Create endpoint, 3. Add validation" |
| constrain | Add specifics | "Create login" → "Create login using JWT, bcrypt, and FastAPI" |
| examples | Add concrete examples | "Create login" → "Create login like auth0.com/api/login" |
| negative | Say what not to do | "Create login" → "Create login (don't use sessions, don't store plain passwords)" |

### Stage 2: Model Escalation

Upgrade to a smarter model:

```python
ESCALATION_ORDER = [
    ("gemini", "gemini-3-flash"),    # Cheapest
    ("gemini", "gemini-3-pro"),      # Better
    ("codex", "gpt-5.2"),            # Strong
    ("claude", "claude-sonnet-4"),   # Good balance
    ("claude", "claude-opus-4.5"),   # Smartest
]

class ModelEscalator:
    def escalate(self, current: tuple) -> Optional[tuple]:
        idx = ESCALATION_ORDER.index(current)
        if idx < len(ESCALATION_ORDER) - 1:
            return ESCALATION_ORDER[idx + 1]
        return None  # Already at highest
```

### Stage 3: Branch Exploration

Try multiple approaches in parallel:

```python
@dataclass
class ExplorationBranch:
    branch_id: str
    approach: str
    prompt_variant: str
    cli: str
    max_iterations: int  # Budget for this branch

class BranchExplorer:
    def explore(self, goal: str, context: str) -> List[ExplorationBranch]:
        return [
            ExplorationBranch(
                branch_id="bottom-up",
                approach="Start with smallest component",
                prompt_variant=self.mutate(goal, "decompose"),
                cli="codex",
                max_iterations=10
            ),
            ExplorationBranch(
                branch_id="research-first",
                approach="Research before implementing",
                prompt_variant=f"First research how to: {goal}",
                cli="gemini",
                max_iterations=10
            ),
            ExplorationBranch(
                branch_id="constrained",
                approach="Add specific constraints",
                prompt_variant=self.mutate(goal, "constrain"),
                cli="codex",
                max_iterations=10
            ),
        ]

    def execute_branches(self, branches: List[ExplorationBranch]) -> ExplorationBranch:
        """Run in parallel, return best."""
        with ThreadPoolExecutor(max_workers=len(branches)) as executor:
            results = list(executor.map(self._run_branch, branches))
        return max(results, key=lambda r: r.progress_score)
```

### Stage 4: Human Escalation

When all automated recovery fails:

```python
class RecoveryManager:
    def recover(self, task: Task, context: LoopContext) -> RecoveryResult:
        # ... stages 1-3 ...

        # Stage 4: Human escalation
        self.alert_manager.alert(
            "stuck_unrecoverable",
            f"Task {task.id} stuck after all recovery attempts",
            severity="CRITICAL"
        )
        return RecoveryResult(success=False, needs_human=True)
```

## Recovery Budget

To prevent infinite recovery loops:

```python
class RecoveryConfig:
    prompt_mutations: int = 3       # Max mutation attempts
    branch_exploration: bool = True
    branch_count: int = 3           # Parallel branches
    branch_budget: int = 10         # Iterations per branch
    total_recovery_budget: int = 20 # Max recovery iterations total
```

## Configuration

### CLI Flags

```bash
--stuck-after 3           # Iterations before recovery
--recovery-budget 20      # Max recovery iterations
```

### YAML Config

```yaml
loop:
  stuck_after: 3

  recovery:
    prompt_mutations: 3
    branch_exploration: true
    branch_count: 3
    branch_budget: 10
```

## Metrics

Track recovery effectiveness:

| Metric | Description |
|--------|-------------|
| recovery_triggers | Times recovery was triggered |
| recovery_success | Times recovery resolved stuck |
| stage_success | Which stage resolved it |
| recovery_iterations | Iterations spent on recovery |

---

*Last Updated: January 2026*
