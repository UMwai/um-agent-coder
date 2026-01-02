"""Tests for context management system."""

from datetime import datetime, timedelta

import pytest

from src.um_agent_coder.harness.autonomous.context_manager import (
    ContextManager,
    ContextSummarizer,
    IterationContext,
    LoopContext,
)


class TestIterationContext:
    """Tests for IterationContext."""

    def test_basic_creation(self):
        """Test basic context creation."""
        ctx = IterationContext(
            iteration_number=1,
            timestamp=datetime.now(),
            cli_used="codex",
            model_used="gpt-5.2",
            prompt="Test prompt",
            output="Test output",
            progress_score=0.5,
        )
        assert ctx.iteration_number == 1
        assert ctx.cli_used == "codex"
        assert ctx.progress_score == 0.5

    def test_serialization(self):
        """Test serialization to dict."""
        ctx = IterationContext(
            iteration_number=1,
            timestamp=datetime.now(),
            cli_used="codex",
            model_used="gpt-5.2",
            prompt="Test",
            output="Output",
            progress_score=0.5,
            progress_markers=["step1", "step2"],
            file_changes=["file1.py"],
            duration_seconds=10.5,
        )

        data = ctx.to_dict()
        assert data["iteration_number"] == 1
        assert data["cli_used"] == "codex"
        assert data["progress_markers"] == ["step1", "step2"]

    def test_deserialization(self):
        """Test deserialization from dict."""
        data = {
            "iteration_number": 5,
            "timestamp": datetime.now().isoformat(),
            "cli_used": "gemini",
            "model_used": "gemini-3-pro",
            "prompt": "Prompt",
            "output": "Output",
            "progress_score": 0.7,
            "progress_markers": ["done"],
            "file_changes": [],
            "duration_seconds": 5.0,
        }

        ctx = IterationContext.from_dict(data)
        assert ctx.iteration_number == 5
        assert ctx.cli_used == "gemini"
        assert ctx.progress_score == 0.7

    def test_output_snippet_short(self):
        """Test output snippet with short output."""
        ctx = IterationContext(
            iteration_number=1,
            timestamp=datetime.now(),
            cli_used="codex",
            model_used="gpt-5.2",
            prompt="",
            output="Short output",
            progress_score=0.0,
        )
        assert ctx.get_output_snippet() == "Short output"

    def test_output_snippet_long(self):
        """Test output snippet truncates long output."""
        long_output = "x" * 1000
        ctx = IterationContext(
            iteration_number=1,
            timestamp=datetime.now(),
            cli_used="codex",
            model_used="gpt-5.2",
            prompt="",
            output=long_output,
            progress_score=0.0,
        )
        snippet = ctx.get_output_snippet(max_length=100)
        assert len(snippet) == 103  # 100 + "..."
        assert snippet.endswith("...")


class TestLoopContext:
    """Tests for LoopContext."""

    def test_basic_creation(self):
        """Test basic loop context creation."""
        ctx = LoopContext(
            task_id="task-001",
            goal="Implement feature X",
        )
        assert ctx.task_id == "task-001"
        assert ctx.goal == "Implement feature X"
        assert len(ctx.iterations) == 0
        assert ctx.total_iterations == 0

    def test_serialization(self):
        """Test loop context serialization."""
        ctx = LoopContext(
            task_id="task-001",
            goal="Test goal",
            summary="Some summary",
            total_iterations=5,
            current_cli="codex",
            current_model="gpt-5.2",
        )
        ctx.start_time = datetime.now()
        ctx.iterations.append(
            IterationContext(
                iteration_number=1,
                timestamp=datetime.now(),
                cli_used="codex",
                model_used="gpt-5.2",
                prompt="P",
                output="O",
                progress_score=0.3,
            )
        )

        data = ctx.to_dict()
        assert data["task_id"] == "task-001"
        assert data["total_iterations"] == 5
        assert len(data["iterations"]) == 1

    def test_deserialization(self):
        """Test loop context deserialization."""
        data = {
            "task_id": "task-002",
            "goal": "Another goal",
            "summary": "Summary text",
            "total_iterations": 10,
            "start_time": datetime.now().isoformat(),
            "current_cli": "gemini",
            "current_model": "gemini-3-pro",
            "env_snapshot": {"VAR": "value"},
            "iterations": [],
        }

        ctx = LoopContext.from_dict(data)
        assert ctx.task_id == "task-002"
        assert ctx.total_iterations == 10
        assert ctx.env_snapshot == {"VAR": "value"}


class TestContextManager:
    """Tests for ContextManager."""

    def test_default_settings(self):
        """Test default manager settings."""
        manager = ContextManager()
        assert manager.raw_window_size == 5
        assert manager.summarize_every == 10

    def test_custom_settings(self):
        """Test custom manager settings."""
        manager = ContextManager(
            raw_window_size=3,
            summarize_every=5,
            max_summary_tokens=1000,
        )
        assert manager.raw_window_size == 3
        assert manager.summarize_every == 5

    def test_add_iteration(self):
        """Test adding iteration."""
        manager = ContextManager(raw_window_size=3)
        ctx = LoopContext(task_id="test", goal="Test goal")

        iteration = IterationContext(
            iteration_number=1,
            timestamp=datetime.now(),
            cli_used="codex",
            model_used="gpt-5.2",
            prompt="P",
            output="O",
            progress_score=0.5,
        )

        manager.add_iteration(ctx, iteration)

        assert len(ctx.iterations) == 1
        assert ctx.total_iterations == 1
        assert ctx.current_cli == "codex"
        assert ctx.current_model == "gpt-5.2"

    def test_rolling_window(self):
        """Test rolling window trims old iterations."""
        manager = ContextManager(raw_window_size=3)
        ctx = LoopContext(task_id="test", goal="Test")

        # Add 5 iterations
        for i in range(5):
            iteration = IterationContext(
                iteration_number=i + 1,
                timestamp=datetime.now(),
                cli_used="codex",
                model_used="gpt-5.2",
                prompt="P",
                output="O",
                progress_score=0.3,
                progress_markers=[f"step{i+1}"],
            )
            manager.add_iteration(ctx, iteration)

        # Should only keep last 3
        assert len(ctx.iterations) == 3
        assert ctx.iterations[0].iteration_number == 3
        assert ctx.iterations[-1].iteration_number == 5
        assert ctx.total_iterations == 5

    def test_summary_accumulates(self):
        """Test summary accumulates from trimmed iterations."""
        manager = ContextManager(raw_window_size=2)
        ctx = LoopContext(task_id="test", goal="Test")

        # Add iterations with progress markers
        for i in range(4):
            iteration = IterationContext(
                iteration_number=i + 1,
                timestamp=datetime.now(),
                cli_used="codex",
                model_used="gpt-5.2",
                prompt="P",
                output="O",
                progress_score=0.5,
                progress_markers=[f"completed_step_{i+1}"],
            )
            manager.add_iteration(ctx, iteration)

        # Summary should contain info from trimmed iterations
        assert ctx.summary  # Should have some summary
        assert len(ctx.iterations) == 2  # Only 2 in window

    def test_build_prompt_basic(self):
        """Test building basic prompt."""
        manager = ContextManager()
        ctx = LoopContext(task_id="test", goal="Implement feature X")
        ctx.current_cli = "codex"
        ctx.current_model = "gpt-5.2"

        prompt = manager.build_prompt(ctx)

        assert "# Goal" in prompt
        assert "Implement feature X" in prompt
        assert "# Instructions" in prompt
        assert "<progress>" in prompt
        assert "<promise>" in prompt

    def test_build_prompt_with_summary(self):
        """Test prompt includes summary."""
        manager = ContextManager()
        ctx = LoopContext(
            task_id="test",
            goal="Test goal",
            summary="Previous work: completed step 1 and step 2",
            total_iterations=10,
        )
        # Add some iterations to window
        for i in range(3):
            ctx.iterations.append(
                IterationContext(
                    iteration_number=i + 8,
                    timestamp=datetime.now(),
                    cli_used="codex",
                    model_used="gpt-5.2",
                    prompt="P",
                    output="Output text",
                    progress_score=0.4,
                )
            )

        prompt = manager.build_prompt(ctx)

        assert "# Progress Summary" in prompt
        assert "Previous work" in prompt

    def test_build_prompt_with_iterations(self):
        """Test prompt includes recent iterations."""
        manager = ContextManager()
        ctx = LoopContext(task_id="test", goal="Test")

        ctx.iterations.append(
            IterationContext(
                iteration_number=1,
                timestamp=datetime.now(),
                cli_used="codex",
                model_used="gpt-5.2",
                prompt="P",
                output="First iteration output",
                progress_score=0.3,
            )
        )
        ctx.total_iterations = 1

        prompt = manager.build_prompt(ctx, include_recent_count=1)

        assert "# Recent Iterations" in prompt
        assert "First iteration output" in prompt

    def test_get_summary_for_recovery(self):
        """Test getting recovery summary."""
        manager = ContextManager()
        ctx = LoopContext(
            task_id="test",
            goal="Complete the task",
            summary="Made progress on steps 1-3",
            total_iterations=5,
        )
        ctx.iterations.append(
            IterationContext(
                iteration_number=5,
                timestamp=datetime.now(),
                cli_used="codex",
                model_used="gpt-5.2",
                prompt="P",
                output="O",
                progress_score=0.5,
                progress_markers=["step 4 done"],
            )
        )

        summary = manager.get_summary_for_recovery(ctx)

        assert "Goal: Complete the task" in summary
        assert "Iterations: 5" in summary


class TestContextSummarizer:
    """Tests for ContextSummarizer."""

    def test_default_settings(self):
        """Test default summarizer settings."""
        summarizer = ContextSummarizer()
        assert summarizer.max_summary_words == 500
        assert summarizer.llm is None

    def test_incorporate_iteration_with_markers(self):
        """Test incorporating iteration with markers."""
        summarizer = ContextSummarizer()
        existing = "Previous summary"

        iteration = IterationContext(
            iteration_number=3,
            timestamp=datetime.now(),
            cli_used="codex",
            model_used="gpt-5.2",
            prompt="P",
            output="O",
            progress_score=0.5,
            progress_markers=["completed step 3"],
        )

        new_summary = summarizer.incorporate_iteration(existing, iteration)

        assert "Previous summary" in new_summary
        assert "Iteration 3" in new_summary
        assert "completed step 3" in new_summary

    def test_incorporate_iteration_with_progress_score(self):
        """Test incorporating iteration with only progress score."""
        summarizer = ContextSummarizer()

        iteration = IterationContext(
            iteration_number=2,
            timestamp=datetime.now(),
            cli_used="codex",
            model_used="gpt-5.2",
            prompt="P",
            output="O",
            progress_score=0.5,
            progress_markers=[],
        )

        new_summary = summarizer.incorporate_iteration("", iteration)

        assert "Iteration 2" in new_summary
        assert "progress" in new_summary.lower()

    def test_incorporate_iteration_no_progress(self):
        """Test incorporating iteration with no progress."""
        summarizer = ContextSummarizer()

        iteration = IterationContext(
            iteration_number=1,
            timestamp=datetime.now(),
            cli_used="codex",
            model_used="gpt-5.2",
            prompt="P",
            output="O",
            progress_score=0.1,  # Below threshold
            progress_markers=[],
        )

        new_summary = summarizer.incorporate_iteration("existing", iteration)
        assert new_summary == "existing"  # No change

    def test_regenerate_summary_template(self):
        """Test regenerating summary with template."""
        summarizer = ContextSummarizer()

        ctx = LoopContext(task_id="test", goal="Test goal", total_iterations=5)
        ctx.iterations = [
            IterationContext(
                iteration_number=i,
                timestamp=datetime.now(),
                cli_used="codex",
                model_used="gpt-5.2",
                prompt="P",
                output="O",
                progress_score=0.4,
                progress_markers=[f"step {i}"],
            )
            for i in range(1, 4)
        ]

        summary = summarizer.regenerate_summary(ctx)

        assert "Progress summary" in summary
        assert "Accomplishments" in summary

    def test_summary_truncation(self):
        """Test summary is truncated to max words."""
        summarizer = ContextSummarizer(max_summary_words=10)

        ctx = LoopContext(task_id="test", goal="Test", total_iterations=100)
        ctx.iterations = [
            IterationContext(
                iteration_number=i,
                timestamp=datetime.now(),
                cli_used="codex",
                model_used="gpt-5.2",
                prompt="P",
                output="O",
                progress_score=0.5,
                progress_markers=[f"very long marker text for step {i}"],
            )
            for i in range(1, 20)
        ]

        summary = summarizer.regenerate_summary(ctx)
        words = summary.split()
        # Should be truncated (with ...)
        assert len(words) <= 15  # 10 + some buffer for "..."


class TestContextManagerIntegration:
    """Integration tests for context management."""

    def test_full_workflow(self):
        """Test complete context management workflow."""
        manager = ContextManager(raw_window_size=3, summarize_every=5)
        ctx = LoopContext(task_id="test", goal="Complete the feature")
        ctx.start_time = datetime.now()

        # Simulate 10 iterations
        for i in range(10):
            iteration = IterationContext(
                iteration_number=i + 1,
                timestamp=datetime.now(),
                cli_used="codex" if i % 2 == 0 else "gemini",
                model_used="gpt-5.2" if i % 2 == 0 else "gemini-3-pro",
                prompt=f"Iteration {i+1} prompt",
                output=f"Iteration {i+1} output",
                progress_score=0.3 + (i * 0.05),
                progress_markers=[f"completed_step_{i+1}"] if i % 2 == 0 else [],
                duration_seconds=10.0,
            )
            manager.add_iteration(ctx, iteration)

        # Check state after iterations
        assert ctx.total_iterations == 10
        assert len(ctx.iterations) == 3  # Window size
        assert ctx.summary  # Should have accumulated summary

        # Build prompt should work
        prompt = manager.build_prompt(ctx)
        assert "# Goal" in prompt
        assert "Complete the feature" in prompt

        # Serialize/deserialize should preserve state
        data = ctx.to_dict()
        restored = LoopContext.from_dict(data)
        assert restored.total_iterations == 10
        assert len(restored.iterations) == 3

    def test_context_persists_through_recovery(self):
        """Test context works with recovery flow."""
        manager = ContextManager(raw_window_size=5)
        ctx = LoopContext(task_id="test", goal="Difficult task")

        # Add iterations showing no progress
        for i in range(5):
            iteration = IterationContext(
                iteration_number=i + 1,
                timestamp=datetime.now(),
                cli_used="codex",
                model_used="gpt-5.2",
                prompt="Stuck prompt",
                output="No progress output",
                progress_score=0.1,
                progress_markers=[],
            )
            manager.add_iteration(ctx, iteration)

        # Get recovery summary
        recovery_summary = manager.get_summary_for_recovery(ctx)
        assert "Difficult task" in recovery_summary
        assert "5" in recovery_summary  # Iteration count
