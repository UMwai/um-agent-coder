# Context Management Specification

> **Priority**: MEDIUM (Foundation)
> **Status**: Partially Implemented
> **Location**: `src/um_agent_coder/harness/autonomous/context_manager.py`

## Overview

Context management maintains execution context across iterations while keeping token usage bounded. Uses a rolling window of raw iterations plus LLM-based summarization of older history.

## Context Structure

```python
@dataclass
class IterationContext:
    """Context for a single iteration."""
    iteration_number: int
    timestamp: datetime
    cli_used: str
    model_used: str
    prompt: str                 # Truncated to 500 chars
    output: str                 # Truncated to 5000 chars
    progress_score: float
    progress_markers: List[str]
    file_changes: List[str]
    duration_seconds: float

@dataclass
class LoopContext:
    """Overall loop context."""
    task: Task
    iterations: List[IterationContext]  # Rolling window (last N)
    summary: str                          # Compressed older history
    total_iterations: int
    start_time: datetime
    current_cli: str
    current_model: str
    env_snapshot: Dict[str, str]
    file_events: List[FileEvent]
```

## Rolling Window

Keep last N iterations in raw form:

```python
class ContextManager:
    def __init__(self, config: ContextConfig):
        self.raw_window_size = config.raw_window_size  # Default: 5
        self.summarize_every = config.summarize_every  # Default: 10

    def add_iteration(self, context: LoopContext, iteration: IterationContext):
        context.iterations.append(iteration)
        context.total_iterations += 1

        # Trim to window size
        if len(context.iterations) > self.raw_window_size:
            oldest = context.iterations.pop(0)
            self._incorporate_into_summary(context, oldest)

        # Periodic re-summarization
        if context.total_iterations % self.summarize_every == 0:
            context.summary = self._regenerate_summary(context)
```

### Window Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| raw_window_size | 5 | Keep last N iterations in full detail |
| summarize_every | 10 | Re-summarize at N-iteration intervals |
| max_summary_tokens | 2000 | Maximum summary length |

## Summarization

### Approach

Use cheap LLM (gemini-3-flash) to summarize older iterations:

```python
class ContextSummarizer:
    def summarize(
        self,
        iterations: List[IterationContext],
        existing_summary: str
    ) -> str:
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
```

### What to Preserve

- **Key decisions**: "Chose JWT over sessions"
- **Working approaches**: "Using bcrypt for password hashing"
- **Failed approaches**: "Redis caching caused issues, reverted"
- **Current state**: "Login endpoint complete, logout pending"
- **Next steps**: "Need to implement refresh tokens"

### What to Discard

- Verbose output logs
- Temporary debugging info
- Repeated error messages
- Implementation details already in code

## Building Prompts with Context

```python
def build_contextual_prompt(task: Task, context: LoopContext) -> str:
    sections = [f"# Goal\n{task.goal}"]

    # Summary of older history
    if context.summary:
        start = context.total_iterations - len(context.iterations)
        sections.append(
            f"# Progress Summary (iterations 1-{start})\n{context.summary}"
        )

    # Raw recent iterations
    if context.iterations:
        recent = "\n\n".join([
            f"## Iteration {it.iteration_number}\n{it.output[:1000]}"
            for it in context.iterations[-3:]  # Last 3
        ])
        sections.append(f"# Recent Iterations\n{recent}")

    # Current state
    sections.append(
        f"# Current State\n"
        f"- Total iterations: {context.total_iterations}\n"
        f"- Current CLI: {context.current_cli}"
    )

    # Instructions
    sections.append(
        "# Instructions\n"
        "Continue working toward the goal. Output:\n"
        "- `<progress>what you accomplished</progress>` for progress\n"
        "- `<promise>COMPLETE</promise>` when goal is fully achieved"
    )

    return "\n\n".join(sections)
```

## Token Budget

Target: Keep total context under 50k tokens.

### Token Allocation

| Component | Max Tokens | Strategy |
|-----------|------------|----------|
| Goal | 1000 | Truncate if needed |
| Summary | 2000 | LLM summarization |
| Raw iterations (5) | 30000 | 6000 per iteration |
| Instructions | 500 | Fixed |
| **Total** | ~35000 | Leave buffer |

### Truncation Rules

```python
def truncate_iteration(iteration: IterationContext) -> IterationContext:
    return IterationContext(
        ...iteration,
        prompt=iteration.prompt[:500],      # 500 chars
        output=iteration.output[:5000],     # 5000 chars
    )
```

## Configuration

### CLI Flags

```bash
--context-window 5        # Raw iterations to keep
--summarize-every 10      # Re-summarize interval
```

### YAML Config

```yaml
context:
  raw_window_size: 5
  summarize_every: 10
  max_summary_tokens: 2000
```

## Edge Cases

| Case | Handling |
|------|----------|
| First iteration | No summary, only current iteration |
| Very long output | Truncate to 5000 chars |
| Empty output | Store as empty, note in summary |
| Summarization fails | Keep raw, retry next interval |

## Future Enhancements

1. **Semantic chunking**: Smarter truncation that preserves meaning
2. **Vector store**: RAG-based context retrieval
3. **Cross-task memory**: Share context between related tasks
4. **Adaptive windowing**: Adjust window size based on task complexity

---

*Last Updated: January 2026*
