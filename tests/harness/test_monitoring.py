"""Tests for monitoring module."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from um_agent_coder.harness.autonomous.monitoring import (
    LogLevel,
    RealTimeLogger,
    StatusFormat,
    StatusReporter,
)
from um_agent_coder.harness.autonomous.monitoring.status_reporter import (
    LoopStatus,
    MetricsCollector,
)
from um_agent_coder.harness.autonomous.context_manager import (
    IterationContext,
    LoopContext,
)


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_all_levels(self):
        """Test all log levels exist."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"
        assert LogLevel.SUCCESS.value == "SUCCESS"


class TestRealTimeLogger:
    """Tests for RealTimeLogger."""

    def test_basic_init(self):
        """Test basic initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = RealTimeLogger(
                log_path=log_path,
                console_output=False,
            )
            assert logger.log_path == log_path
            assert not logger.console_output
            assert logger.file_output

    def test_log_creates_file(self):
        """Test that logging creates file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs" / "test.log"
            logger = RealTimeLogger(
                log_path=log_path,
                console_output=False,
            )
            logger.info("Test message")
            assert log_path.exists()

    def test_log_levels(self):
        """Test different log levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = RealTimeLogger(
                log_path=log_path,
                console_output=False,
                min_level=LogLevel.INFO,
            )

            logger.debug("Debug message")  # Should not be logged
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
            logger.success("Success message")

            # Debug should not be logged due to min_level
            assert len(logger.entries) == 5
            assert logger.entries[0].level == LogLevel.INFO
            assert logger.entries[1].level == LogLevel.WARNING

    def test_log_iteration_events(self):
        """Test iteration event logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = RealTimeLogger(
                log_path=log_path,
                console_output=False,
            )

            logger.log_iteration_start(1, "codex", "gpt-5.2")
            logger.log_iteration_complete(1, 0.45, 10.5)
            logger.log_progress_marker(1, "Completed module")

            assert len(logger.entries) == 3
            assert "Iteration 1 starting" in logger.entries[0].message
            assert "progress=0.45" in logger.entries[1].message

    def test_log_stuck_and_recovery(self):
        """Test stuck and recovery logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = RealTimeLogger(
                log_path=log_path,
                console_output=False,
            )

            logger.log_stuck_detected(10, 3)
            logger.log_recovery_attempt(10, "prompt_mutation", "Rephrasing task")
            logger.log_recovery_result(11, True, "prompt_mutation")

            assert len(logger.entries) == 3
            assert "STUCK DETECTED" in logger.entries[0].message
            assert logger.entries[0].level == LogLevel.WARNING

    def test_log_goal_complete(self):
        """Test goal completion logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = RealTimeLogger(
                log_path=log_path,
                console_output=False,
            )

            logger.log_goal_complete(25, "FEATURE_COMPLETE", 1500.0)

            assert len(logger.entries) == 1
            assert "GOAL COMPLETE" in logger.entries[0].message
            assert logger.entries[0].level == LogLevel.SUCCESS

    def test_log_termination(self):
        """Test termination logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = RealTimeLogger(
                log_path=log_path,
                console_output=False,
            )

            logger.log_termination(100, "iteration_limit", False)

            assert len(logger.entries) == 1
            assert "iteration_limit" in logger.entries[0].message
            assert logger.entries[0].level == LogLevel.WARNING

    def test_log_alert(self):
        """Test alert logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = RealTimeLogger(
                log_path=log_path,
                console_output=False,
            )

            logger.log_alert(5, "runaway_detected", "WARNING", "Too many fast iterations")

            assert len(logger.entries) == 1
            assert "runaway_detected" in logger.entries[0].message

    def test_get_recent_entries(self):
        """Test getting recent entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = RealTimeLogger(
                log_path=log_path,
                console_output=False,
            )

            for i in range(10):
                logger.info(f"Message {i}")

            recent = logger.get_recent_entries(5)
            assert len(recent) == 5
            assert "Message 5" in recent[0].message

    def test_get_entries_by_level(self):
        """Test filtering entries by level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = RealTimeLogger(
                log_path=log_path,
                console_output=False,
            )

            logger.info("Info 1")
            logger.warning("Warning 1")
            logger.info("Info 2")
            logger.warning("Warning 2")
            logger.error("Error 1")

            warnings = logger.get_entries_by_level(LogLevel.WARNING)
            assert len(warnings) == 2
            assert all(e.level == LogLevel.WARNING for e in warnings)

    def test_clear_entries(self):
        """Test clearing entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = RealTimeLogger(
                log_path=log_path,
                console_output=False,
            )

            logger.info("Test message")
            assert len(logger.entries) == 1

            logger.clear()
            assert len(logger.entries) == 0


class TestLoopStatus:
    """Tests for LoopStatus."""

    def test_to_dict(self):
        """Test serialization."""
        status = LoopStatus(
            task_id="test-001",
            task_description="Test task",
            status="RUNNING",
            current_iteration=10,
            max_iterations=100,
            elapsed=timedelta(minutes=30),
            current_cli="codex",
            current_model="gpt-5.2",
            avg_progress=0.45,
            recent_markers=["Step 1 done", "Step 2 done"],
            stuck_state="progressing",
            recovery_attempts=1,
            alerts_issued=2,
        )

        data = status.to_dict()
        assert data["task_id"] == "test-001"
        assert data["current_iteration"] == 10
        assert data["elapsed_seconds"] == 1800


class TestStatusReporter:
    """Tests for StatusReporter."""

    def test_basic_init(self):
        """Test basic initialization."""
        reporter = StatusReporter(interval_iterations=10)
        assert reporter.interval == 10

    def test_should_report(self):
        """Test should_report logic."""
        reporter = StatusReporter(interval_iterations=10)

        assert not reporter.should_report(0)
        assert not reporter.should_report(5)
        assert reporter.should_report(10)
        assert reporter.should_report(20)
        assert not reporter.should_report(15)

    def test_generate_summary_text(self):
        """Test text format summary."""
        reporter = StatusReporter(interval_iterations=10)
        context = _create_test_context(iterations=10)

        summary = reporter.generate_summary(context, StatusFormat.TEXT)

        assert "AUTONOMOUS LOOP STATUS" in summary
        assert "test-task" in summary

    def test_generate_summary_json(self):
        """Test JSON format summary."""
        reporter = StatusReporter(interval_iterations=10)
        context = _create_test_context(iterations=10)

        summary = reporter.generate_summary(context, StatusFormat.JSON)

        data = json.loads(summary)
        assert data["task_id"] == "test-task"
        assert "current_iteration" in data

    def test_generate_summary_brief(self):
        """Test brief format summary."""
        reporter = StatusReporter(interval_iterations=10)
        context = _create_test_context(iterations=10)

        summary = reporter.generate_summary(context, StatusFormat.BRIEF)

        assert "test-task" in summary
        assert "iter=" in summary
        assert "progress=" in summary

    def test_maybe_report_returns_none(self):
        """Test maybe_report returns None when not at interval."""
        reporter = StatusReporter(interval_iterations=10)
        context = _create_test_context(iterations=5)

        result = reporter.maybe_report(context)
        assert result is None

    def test_maybe_report_returns_summary(self):
        """Test maybe_report returns summary at interval."""
        reporter = StatusReporter(interval_iterations=10)
        context = _create_test_context(iterations=10)

        result = reporter.maybe_report(context)
        assert result is not None
        assert "test-task" in result

    def test_write_and_read_status_file(self):
        """Test writing and reading status file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / "status.json"
            reporter = StatusReporter(status_path=status_path)
            context = _create_test_context(iterations=10)

            reporter.write_status_file(context)

            assert status_path.exists()

            status = reporter.read_status_file()
            assert status is not None
            assert status["task_id"] == "test-task"

    def test_format_status_output(self):
        """Test formatting status output."""
        status_data = {
            "task_id": "test-001",
            "task_description": "Test task description",
            "status": "RUNNING",
            "current_iteration": 25,
            "max_iterations": 100,
            "elapsed_seconds": 3600,
            "avg_progress": 0.35,
            "current_cli": "codex",
            "current_model": "gpt-5.2",
            "recent_markers": ["Step 1", "Step 2"],
        }

        output = StatusReporter.format_status_output(status_data)

        assert "test-001" in output
        assert "RUNNING" in output
        assert "25/100" in output


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_record_iteration(self):
        """Test recording iteration metrics."""
        collector = MetricsCollector()

        collector.record_iteration(
            duration=10.5,
            progress=0.45,
            cli="codex",
            model="gpt-5.2",
        )

        assert len(collector.iteration_durations) == 1
        assert collector.iteration_durations[0] == 10.5
        assert collector.cli_usage["codex"] == 1

    def test_record_multiple_iterations(self):
        """Test recording multiple iterations."""
        collector = MetricsCollector()

        for i in range(5):
            collector.record_iteration(
                duration=10.0 + i,
                progress=0.3 + (i * 0.1),
                cli="codex" if i % 2 == 0 else "gemini",
                model="gpt-5.2" if i % 2 == 0 else "gemini-3-pro",
            )

        assert len(collector.iteration_durations) == 5
        assert collector.cli_usage["codex"] == 3
        assert collector.cli_usage["gemini"] == 2

    def test_record_recovery(self):
        """Test recording recovery attempts."""
        collector = MetricsCollector()

        collector.record_recovery(success=True)
        collector.record_recovery(success=False)
        collector.record_recovery(success=True)

        assert collector.recovery_attempts == 3
        assert collector.successful_recoveries == 2

    def test_record_tokens(self):
        """Test recording token usage."""
        collector = MetricsCollector()

        collector.record_tokens(1000)
        collector.record_tokens(500)

        assert collector.total_tokens_used == 1500

    def test_get_summary(self):
        """Test getting metrics summary."""
        collector = MetricsCollector()

        for i in range(10):
            collector.record_iteration(
                duration=10.0,
                progress=0.5,
                cli="codex",
                model="gpt-5.2",
            )
        collector.record_recovery(success=True)
        collector.record_recovery(success=False)
        collector.record_tokens(5000)

        summary = collector.get_summary()

        assert summary["total_iterations"] == 10
        assert summary["total_duration_seconds"] == 100.0
        assert summary["avg_iteration_duration"] == 10.0
        assert summary["avg_progress"] == 0.5
        assert summary["recovery_attempts"] == 2
        assert summary["recovery_success_rate"] == 0.5
        assert summary["total_tokens_used"] == 5000

    def test_reset(self):
        """Test resetting metrics."""
        collector = MetricsCollector()

        collector.record_iteration(10.0, 0.5, "codex", "gpt-5.2")
        collector.record_recovery(success=True)
        collector.record_tokens(1000)

        collector.reset()

        assert len(collector.iteration_durations) == 0
        assert collector.recovery_attempts == 0
        assert collector.total_tokens_used == 0


def _create_test_context(iterations: int = 5) -> LoopContext:
    """Create a test LoopContext."""
    context = LoopContext(
        task_id="test-task",
        goal="Test task description",
        start_time=datetime.now() - timedelta(minutes=30),
    )

    for i in range(iterations):
        iteration = IterationContext(
            iteration_number=i + 1,
            timestamp=datetime.now(),
            cli_used="codex",
            model_used="gpt-5.2",
            prompt="Test prompt",
            output="Test output",
            progress_score=0.4,
            progress_markers=["marker1"] if i % 2 == 0 else [],
            duration_seconds=10.0,
        )
        context.iterations.append(iteration)
        context.total_iterations = i + 1

    return context
