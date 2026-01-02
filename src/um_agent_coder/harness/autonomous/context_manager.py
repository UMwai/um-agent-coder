"""Context management for autonomous loop.

Maintains iteration context with a rolling window of raw iterations
plus summarization of older iterations. This enables context to
accumulate between iterations without unbounded growth.

Reference: specs/autonomous-loop-spec.md Section 6
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class IterationContext:
    """Context for a single iteration."""

    iteration_number: int
    timestamp: datetime
    cli_used: str
    model_used: str
    prompt: str
    output: str
    progress_score: float
    progress_markers: list[str] = field(default_factory=list)
    file_changes: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "iteration_number": self.iteration_number,
            "timestamp": self.timestamp.isoformat(),
            "cli_used": self.cli_used,
            "model_used": self.model_used,
            "prompt": self.prompt,
            "output": self.output,
            "progress_score": self.progress_score,
            "progress_markers": self.progress_markers,
            "file_changes": self.file_changes,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IterationContext":
        """Deserialize from dictionary."""
        return cls(
            iteration_number=data["iteration_number"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            cli_used=data["cli_used"],
            model_used=data["model_used"],
            prompt=data["prompt"],
            output=data["output"],
            progress_score=data.get("progress_score", 0.0),
            progress_markers=data.get("progress_markers", []),
            file_changes=data.get("file_changes", []),
            duration_seconds=data.get("duration_seconds", 0.0),
        )

    def get_output_snippet(self, max_length: int = 500) -> str:
        """Get truncated output for summary."""
        if len(self.output) <= max_length:
            return self.output
        return self.output[:max_length] + "..."


@dataclass
class LoopContext:
    """Full context for the autonomous loop."""

    task_id: str
    goal: str
    iterations: list[IterationContext] = field(default_factory=list)
    summary: str = ""
    total_iterations: int = 0
    start_time: Optional[datetime] = None
    current_cli: str = ""
    current_model: str = ""
    env_snapshot: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "iterations": [it.to_dict() for it in self.iterations],
            "summary": self.summary,
            "total_iterations": self.total_iterations,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "current_cli": self.current_cli,
            "current_model": self.current_model,
            "env_snapshot": self.env_snapshot,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoopContext":
        """Deserialize from dictionary."""
        ctx = cls(
            task_id=data["task_id"],
            goal=data["goal"],
            summary=data.get("summary", ""),
            total_iterations=data.get("total_iterations", 0),
            current_cli=data.get("current_cli", ""),
            current_model=data.get("current_model", ""),
            env_snapshot=data.get("env_snapshot", {}),
        )
        if data.get("start_time"):
            ctx.start_time = datetime.fromisoformat(data["start_time"])
        ctx.iterations = [IterationContext.from_dict(it) for it in data.get("iterations", [])]
        return ctx


class ContextManager:
    """Manage context across iterations with rolling window.

    Maintains a rolling window of recent raw iterations and summarizes
    older iterations to prevent unbounded context growth.

    Args:
        raw_window_size: Number of recent iterations to keep raw.
        summarize_every: Re-summarize every N iterations.
        max_summary_tokens: Maximum summary size in tokens (estimated).
    """

    def __init__(
        self,
        raw_window_size: int = 5,
        summarize_every: int = 10,
        max_summary_tokens: int = 2000,
        summarizer=None,
    ):
        """Initialize context manager.

        Args:
            raw_window_size: Number of iterations to keep raw.
            summarize_every: Re-summarize interval.
            max_summary_tokens: Max summary size.
            summarizer: Optional ContextSummarizer instance.
        """
        self.raw_window_size = raw_window_size
        self.summarize_every = summarize_every
        self.max_summary_tokens = max_summary_tokens
        self.summarizer = summarizer or ContextSummarizer()

    def add_iteration(
        self,
        context: LoopContext,
        iteration: IterationContext,
    ) -> None:
        """Add an iteration to context, maintaining window.

        Args:
            context: The loop context to update.
            iteration: The new iteration to add.
        """
        context.iterations.append(iteration)
        context.total_iterations += 1

        # Update current cli/model
        context.current_cli = iteration.cli_used
        context.current_model = iteration.model_used

        # Trim to window size
        if len(context.iterations) > self.raw_window_size:
            # Summarize oldest before removing
            to_summarize = context.iterations[0]
            context.summary = self.summarizer.incorporate_iteration(context.summary, to_summarize)
            context.iterations.pop(0)

        # Periodic full re-summarization
        if context.total_iterations % self.summarize_every == 0:
            context.summary = self.summarizer.regenerate_summary(context)

    def build_prompt(
        self,
        context: LoopContext,
        include_recent_count: int = 3,
    ) -> str:
        """Build contextual prompt for next iteration.

        Args:
            context: The loop context.
            include_recent_count: Number of recent iterations to include raw.

        Returns:
            Formatted prompt with context.
        """
        sections = [
            f"# Goal\n{context.goal}",
        ]

        # Include summary of older history
        if context.summary:
            summarized_count = context.total_iterations - len(context.iterations)
            sections.append(
                f"# Progress Summary (iterations 1-{summarized_count})\n{context.summary}"
            )

        # Include raw recent iterations
        if context.iterations:
            recent = context.iterations[-include_recent_count:]
            recent_text = "\n\n".join(
                [
                    f"## Iteration {it.iteration_number}\n"
                    f"Progress: {it.progress_score:.2f}\n"
                    f"Output:\n{it.get_output_snippet(1000)}"
                    for it in recent
                ]
            )
            sections.append(f"# Recent Iterations\n{recent_text}")

        # Current state
        sections.append(
            f"# Current State\n"
            f"- Total iterations: {context.total_iterations}\n"
            f"- Current CLI: {context.current_cli}\n"
            f"- Current Model: {context.current_model}"
        )

        # Instructions
        sections.append(
            "# Instructions\n"
            "Continue working toward the goal. Output:\n"
            "- `<progress>what you accomplished</progress>` for progress updates\n"
            "- `<promise>COMPLETE</promise>` when goal is fully achieved"
        )

        return "\n\n".join(sections)

    def get_summary_for_recovery(self, context: LoopContext) -> str:
        """Get a summary suitable for recovery context.

        Args:
            context: The loop context.

        Returns:
            Concise summary for recovery.
        """
        recent_markers = []
        for it in context.iterations[-3:]:
            recent_markers.extend(it.progress_markers)

        summary_parts = [
            f"Goal: {context.goal}",
            f"Iterations: {context.total_iterations}",
        ]

        if context.summary:
            summary_parts.append(f"Summary: {context.summary[:500]}")

        if recent_markers:
            summary_parts.append(f"Recent progress: {', '.join(recent_markers[:5])}")

        return "\n".join(summary_parts)


class ContextSummarizer:
    """Summarize iteration history.

    Uses templates or LLM to compress iteration history into
    concise summaries.
    """

    def __init__(self, llm=None, max_summary_words: int = 500):
        """Initialize summarizer.

        Args:
            llm: Optional LLM for intelligent summarization.
            max_summary_words: Maximum words in summary.
        """
        self.llm = llm
        self.max_summary_words = max_summary_words

    def incorporate_iteration(
        self,
        existing_summary: str,
        iteration: IterationContext,
    ) -> str:
        """Incorporate a single iteration into existing summary.

        Args:
            existing_summary: Current summary.
            iteration: Iteration to incorporate.

        Returns:
            Updated summary.
        """
        # Extract key info from iteration
        new_info = []
        if iteration.progress_markers:
            new_info.append(
                f"Iteration {iteration.iteration_number}: {', '.join(iteration.progress_markers)}"
            )
        elif iteration.progress_score > 0.15:
            new_info.append(
                f"Iteration {iteration.iteration_number}: Made progress (score: {iteration.progress_score:.2f})"
            )

        if not new_info:
            return existing_summary

        # Append to summary
        if existing_summary:
            return f"{existing_summary}\n{new_info[0]}"
        return new_info[0]

    def regenerate_summary(self, context: LoopContext) -> str:
        """Regenerate full summary from context.

        Args:
            context: The loop context to summarize.

        Returns:
            New comprehensive summary.
        """
        if self.llm is not None:
            return self._llm_summarize(context)

        return self._template_summarize(context)

    def _template_summarize(self, context: LoopContext) -> str:
        """Generate summary using template."""
        lines = [f"Progress summary after {context.total_iterations} iterations:"]

        # Collect all progress markers
        all_markers = []
        for it in context.iterations:
            all_markers.extend(it.progress_markers)

        if all_markers:
            lines.append("Accomplishments:")
            for marker in all_markers[-10:]:  # Last 10 markers
                lines.append(f"- {marker}")

        # Add recent progress scores
        if context.iterations:
            recent_scores = [it.progress_score for it in context.iterations[-5:]]
            avg_score = sum(recent_scores) / len(recent_scores)
            lines.append(f"Recent average progress: {avg_score:.2f}")

        # Preserve existing summary content (truncated)
        if context.summary:
            existing_truncated = context.summary[:300]
            lines.append(f"Previous: {existing_truncated}")

        # Truncate to max words
        summary = "\n".join(lines)
        words = summary.split()
        if len(words) > self.max_summary_words:
            summary = " ".join(words[: self.max_summary_words]) + "..."

        return summary

    def _llm_summarize(self, context: LoopContext) -> str:
        """Generate summary using LLM."""
        prompt = f"""
        Summarize the progress on this task:

        Goal: {context.goal}
        Total iterations: {context.total_iterations}
        Existing summary: {context.summary}

        Recent iterations:
        {self._format_recent_iterations(context)}

        Generate a concise summary (max {self.max_summary_words} words) that:
        1. Preserves key decisions and findings
        2. Notes what approaches worked/failed
        3. Tracks current state and next steps
        """

        return self.llm.generate(prompt)

    def _format_recent_iterations(self, context: LoopContext) -> str:
        """Format recent iterations for summarization."""
        lines = []
        for it in context.iterations[-5:]:
            lines.append(
                f"Iteration {it.iteration_number}: "
                f"progress={it.progress_score:.2f}, "
                f"markers={it.progress_markers}"
            )
        return "\n".join(lines)
