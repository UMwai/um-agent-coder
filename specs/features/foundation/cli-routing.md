# CLI Routing Specification

> **Priority**: MEDIUM (Foundation)
> **Status**: Implemented
> **Location**: `src/um_agent_coder/harness/autonomous/cli_router.py`

## Overview

CLI routing determines which CLI backend (Codex, Gemini, Claude) handles each task based on task characteristics and available resources.

## Routing Logic

### Task Analysis

Tasks are analyzed for:

```python
@dataclass
class TaskAnalysis:
    requires_large_context: bool    # > 100k tokens
    is_implementation: bool         # Writing code
    is_research: bool               # Exploring/researching
    is_complex_reasoning: bool      # Multi-step logic
    is_stuck_recovery: bool         # Recovering from stuck state
    estimated_difficulty: float     # 0.0 (trivial) to 1.0 (hardest)
```

### Keyword Detection

```python
IMPLEMENTATION_KEYWORDS = [
    "implement", "write", "create", "build", "add", "fix",
    "develop", "code", "program"
]

RESEARCH_KEYWORDS = [
    "research", "explore", "find", "investigate", "analyze",
    "search", "understand", "review"
]

COMPLEX_KEYWORDS = [
    "design", "architect", "optimize", "refactor", "debug complex",
    "integrate", "migrate"
]
```

### Routing Rules

Priority order:

1. **Stuck recovery** → Smartest available (Opus > Codex > Gemini)
2. **Large context (>100k)** → Gemini (1M context window)
3. **Implementation** → Codex (gpt-5.2)
4. **Research** → Gemini (gemini-3-pro)
5. **Complex reasoning** → Claude (if Opus budget available)
6. **Default** → Cheapest available

### Routing Table

| Task Type | Primary CLI | Model | Fallback |
|-----------|-------------|-------|----------|
| Research | gemini | gemini-3-pro | claude |
| Implementation | codex | gpt-5.2 | claude |
| Complex Reasoning | claude | claude-opus-4.5 | codex |
| Simple/Quick | gemini | gemini-3-flash | codex |
| Stuck Recovery | claude | claude-opus-4.5 | codex |

## Opus Preservation

Claude Opus tokens are expensive. OpusGuard limits daily usage:

```python
class OpusGuard:
    daily_limit: int = 50  # Default 50 iterations/day
    used_today: int = 0
    last_reset: date

    def can_use_opus(self) -> bool:
        self._maybe_reset()  # Reset at midnight
        return self.used_today < self.daily_limit

    def record_opus_use(self):
        self.used_today += 1
```

When Opus is exhausted:
- Falls back to claude-sonnet-4 or codex
- Logs warning about Opus exhaustion

## Configuration

### CLI Flag

```bash
# Explicit CLI
python -m harness --cli codex

# Multi-CLI (comma-separated)
python -m harness --cli codex,gemini

# Auto-routing
python -m harness --cli auto
```

### YAML Config

```yaml
cli:
  enabled: ["codex", "gemini", "claude"]
  default: "auto"

  auto_router:
    prefer_cheap: true
    opus_daily_limit: 50
```

### Per-Task Override

```markdown
- [ ] **task-001**: Research the codebase
  - cli: gemini

- [ ] **task-002**: Implement authentication
  - cli: codex
  - model: gpt-5.2
```

## Implementation

### CLIRouter Class

```python
class CLIRouter:
    def __init__(self, enabled_clis: List[str], config: RouterConfig):
        self.enabled = set(enabled_clis)
        self.config = config
        self.opus_guard = OpusGuard(config.opus_daily_limit)
        self.analyzer = TaskAnalyzer()
        self.auto_router = AutoRouter(enabled_clis, self.opus_guard)

    def route(self, task: Task, context: LoopContext) -> Tuple[str, str]:
        """Route task to (cli, model)."""
        if task.cli:
            # Explicit override
            return (task.cli, task.model or self._default_model(task.cli))

        if "auto" in self.enabled or len(self.enabled) > 1:
            # Auto-routing
            analysis = self.analyzer.analyze(task, context)
            return self.auto_router.route(analysis)

        # Single CLI mode
        cli = list(self.enabled)[0]
        return (cli, self._default_model(cli))
```

### AutoRouter Class

```python
class AutoRouter:
    def route(self, analysis: TaskAnalysis) -> Tuple[str, str]:
        # Priority 1: Stuck recovery
        if analysis.is_stuck_recovery:
            return self._get_smartest()

        # Priority 2: Large context
        if analysis.requires_large_context and "gemini" in self.enabled:
            return ("gemini", "gemini-3-pro")

        # Priority 3: Implementation
        if analysis.is_implementation and "codex" in self.enabled:
            return ("codex", "gpt-5.2")

        # Priority 4: Research
        if analysis.is_research and "gemini" in self.enabled:
            return ("gemini", "gemini-3-pro")

        # Priority 5: Complex reasoning (with Opus guard)
        if analysis.is_complex_reasoning and "claude" in self.enabled:
            if self.opus_guard.can_use_opus():
                self.opus_guard.record_opus_use()
                return ("claude", "claude-opus-4.5")

        # Default: Cheapest
        return self._get_cheapest()
```

## Metrics

Track routing decisions for optimization:

- Routes per CLI
- Opus usage per day
- Success rate per CLI
- Average duration per CLI

---

*Last Updated: January 2026*
