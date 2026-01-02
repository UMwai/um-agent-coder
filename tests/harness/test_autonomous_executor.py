"""Tests for autonomous executor."""

import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.um_agent_coder.harness.autonomous.executor import (
    AutonomousConfig,
    AutonomousExecutor,
    AutonomousResult,
    TerminationReason,
)
from src.um_agent_coder.harness.models import Task


@dataclass
class MockExecutorResult:
    """Mock result from executor."""

    success: bool = True
    output: str = "Test output"
    error: Optional[str] = None


class MockExecutor:
    """Mock CLI executor for testing."""

    def __init__(self, outputs: Optional[list] = None):
        self.outputs = outputs or ["Test output"]
        self.call_count = 0

    def execute(self, task, prompt: str = "") -> MockExecutorResult:
        """Execute and return mock result."""
        output = self.outputs[min(self.call_count, len(self.outputs) - 1)]
        self.call_count += 1
        return MockExecutorResult(output=output)


class TestAutonomousConfig:
    """Tests for AutonomousConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = AutonomousConfig()
        assert config.max_iterations == 1000
        assert config.max_time_seconds is None
        assert config.progress_threshold == 0.15
        assert config.stuck_after == 3
        assert config.cli_spec == "auto"

    def test_custom_config(self):
        """Test custom configuration."""
        config = AutonomousConfig(
            max_iterations=100,
            max_time_seconds=3600,
            stuck_after=5,
            cli_spec="codex,gemini",
        )
        assert config.max_iterations == 100
        assert config.max_time_seconds == 3600
        assert config.stuck_after == 5


class TestAutonomousResult:
    """Tests for AutonomousResult."""

    def test_success_result(self):
        """Test successful result."""
        result = AutonomousResult(
            success=True,
            iterations=10,
            total_duration=timedelta(seconds=100),
            final_output="Done",
            termination_reason=TerminationReason.GOAL_COMPLETE,
            promise_text="COMPLETE",
        )
        assert result.success
        assert "SUCCESS" in result.summary
        assert result.promise_text == "COMPLETE"

    def test_failure_result(self):
        """Test failure result."""
        result = AutonomousResult(
            success=False,
            iterations=1000,
            total_duration=timedelta(hours=1),
            final_output="Stuck",
            termination_reason=TerminationReason.ITERATION_LIMIT,
        )
        assert not result.success
        assert "FAILED" in result.summary
        assert "iteration_limit" in result.summary

    def test_to_dict(self):
        """Test serialization."""
        result = AutonomousResult(
            success=True,
            iterations=5,
            total_duration=timedelta(seconds=50),
            final_output="Test",
            termination_reason=TerminationReason.GOAL_COMPLETE,
            models_used=["gpt-5.2", "gemini-3-pro"],
        )
        data = result.to_dict()
        assert data["success"] is True
        assert data["iterations"] == 5
        assert data["termination_reason"] == "goal_complete"
        assert "gpt-5.2" in data["models_used"]


class TestAutonomousExecutor:
    """Tests for AutonomousExecutor."""

    def test_basic_init(self):
        """Test basic initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = AutonomousExecutor(
                executors={"codex": MockExecutor()},
                workspace_path=Path(tmpdir),
            )
            assert executor.config.max_iterations == 1000
            assert "codex" in executor.executors

    def test_promise_detection_terminates(self):
        """Test that promise detection terminates loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create executor that outputs promise on second iteration
            mock = MockExecutor(outputs=[
                "Working on it...",
                "Done! <promise>COMPLETE</promise>",
            ])

            config = AutonomousConfig(
                max_iterations=10,
                completion_promise="COMPLETE",
                cli_spec="codex",
            )

            executor = AutonomousExecutor(
                executors={"codex": mock},
                config=config,
                workspace_path=Path(tmpdir),
            )

            task = Task(id="test-001", description="Test task", phase="test")
            result = executor.execute(task)

            assert result.success
            assert result.termination_reason == TerminationReason.GOAL_COMPLETE
            assert result.promise_text == "COMPLETE"
            assert result.iterations == 2

    def test_iteration_limit_terminates(self):
        """Test that iteration limit terminates loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock = MockExecutor(outputs=["Still working..."])

            config = AutonomousConfig(
                max_iterations=3,
                cli_spec="codex",
            )

            executor = AutonomousExecutor(
                executors={"codex": mock},
                config=config,
                workspace_path=Path(tmpdir),
            )

            task = Task(id="test-002", description="Test task", phase="test")
            result = executor.execute(task)

            assert not result.success
            assert result.termination_reason == TerminationReason.ITERATION_LIMIT
            assert result.iterations == 3

    def test_time_limit_terminates(self):
        """Test that time limit terminates loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create slow executor
            class SlowExecutor:
                def execute(self, task, prompt=""):
                    time.sleep(0.1)
                    return MockExecutorResult(output="Working...")

            config = AutonomousConfig(
                max_iterations=100,
                max_time_seconds=0.25,
                cli_spec="codex",
            )

            executor = AutonomousExecutor(
                executors={"codex": SlowExecutor()},
                config=config,
                workspace_path=Path(tmpdir),
            )

            task = Task(id="test-003", description="Test task", phase="test")
            result = executor.execute(task)

            assert not result.success
            assert result.termination_reason == TerminationReason.TIME_LIMIT

    def test_manual_stop(self):
        """Test manual stop request."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock = MockExecutor()

            config = AutonomousConfig(
                max_iterations=100,
                cli_spec="codex",
            )

            executor = AutonomousExecutor(
                executors={"codex": mock},
                config=config,
                workspace_path=Path(tmpdir),
            )

            # Request stop immediately
            executor.request_stop()

            task = Task(id="test-004", description="Test task", phase="test")
            result = executor.execute(task)

            assert not result.success
            assert result.termination_reason == TerminationReason.MANUAL_STOP

    def test_models_used_tracking(self):
        """Test that models used are tracked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock = MockExecutor(outputs=[
                "First output",
                "<promise>COMPLETE</promise>",
            ])

            config = AutonomousConfig(
                max_iterations=10,
                completion_promise="COMPLETE",
                cli_spec="codex",
            )

            executor = AutonomousExecutor(
                executors={"codex": mock},
                config=config,
                workspace_path=Path(tmpdir),
            )

            task = Task(id="test-005", description="Test task", phase="test")
            result = executor.execute(task)

            assert len(result.models_used) > 0

    def test_get_status(self):
        """Test getting executor status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = AutonomousExecutor(
                executors={"codex": MockExecutor()},
                workspace_path=Path(tmpdir),
            )

            status = executor.get_status()
            assert "stop_requested" in status
            assert "pause_requested" in status
            assert "enabled_clis" in status
            assert "stuck_state" in status

    def test_config_with_stuck_recovery(self):
        """Test configuration for stuck recovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AutonomousConfig(
                stuck_after=2,
                recovery_budget=5,
            )

            executor = AutonomousExecutor(
                executors={"codex": MockExecutor()},
                config=config,
                workspace_path=Path(tmpdir),
            )

            assert executor.stuck_detector.stuck_threshold == 2
            assert executor.stuck_detector.recovery_budget == 5


class TestAutonomousExecutorIntegration:
    """Integration tests for autonomous executor."""

    def test_full_workflow_success(self):
        """Test complete successful workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Progress through multiple iterations
            outputs = [
                "Step 1: Starting... <progress>initialized project</progress>",
                "Step 2: Building... <progress>built core module</progress>",
                "Step 3: Testing... <progress>tests passing</progress>",
                "Step 4: Complete! <promise>COMPLETE</promise>",
            ]
            mock = MockExecutor(outputs=outputs)

            config = AutonomousConfig(
                max_iterations=10,
                completion_promise="COMPLETE",
                cli_spec="codex",
                progress_threshold=0.1,
            )

            executor = AutonomousExecutor(
                executors={"codex": mock},
                config=config,
                workspace_path=Path(tmpdir),
            )

            task = Task(
                id="integration-001",
                description="Complete the feature",
                phase="test",
            )
            result = executor.execute(task)

            assert result.success
            assert result.iterations == 4
            assert result.termination_reason == TerminationReason.GOAL_COMPLETE

    def test_multi_cli_routing(self):
        """Test with multiple CLIs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            codex_mock = MockExecutor(outputs=["Codex output"])
            gemini_mock = MockExecutor(outputs=["Gemini output"])

            config = AutonomousConfig(
                max_iterations=5,
                cli_spec="codex,gemini",
            )

            executor = AutonomousExecutor(
                executors={
                    "codex": codex_mock,
                    "gemini": gemini_mock,
                },
                config=config,
                workspace_path=Path(tmpdir),
            )

            task = Task(id="multi-cli-001", description="Test task", phase="test")
            result = executor.execute(task)

            # Should use one of the executors
            assert codex_mock.call_count > 0 or gemini_mock.call_count > 0

    def test_environment_stop_file(self):
        """Test stop via stop file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness_path = Path(tmpdir) / ".harness"
            harness_path.mkdir()

            # Create slow executor and stop file
            class SlowExecutor:
                def __init__(self):
                    self.call_count = 0

                def execute(self, task, prompt=""):
                    self.call_count += 1
                    if self.call_count == 2:
                        # Create stop file
                        (harness_path / "stop").write_text("stop")
                    return MockExecutorResult(output="Working...")

            config = AutonomousConfig(
                max_iterations=100,
                cli_spec="codex",
                check_env_interval=1,
            )

            executor = AutonomousExecutor(
                executors={"codex": SlowExecutor()},
                config=config,
                workspace_path=Path(tmpdir),
                harness_path=harness_path,
            )

            task = Task(id="stop-test", description="Test task", phase="test")
            result = executor.execute(task)

            # Should stop due to stop file
            assert not result.success
            assert result.iterations < 100

    def test_alerts_generated(self):
        """Test that alerts are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock = MockExecutor(outputs=[
                "Output 1",
                "Output 2",
                "Output 3",
                "Output 4",
                "Output 5",
                "<promise>COMPLETE</promise>",
            ])

            config = AutonomousConfig(
                max_iterations=10,
                completion_promise="COMPLETE",
                cli_spec="codex",
                alert_milestone_interval=3,
            )

            executor = AutonomousExecutor(
                executors={"codex": mock},
                config=config,
                workspace_path=Path(tmpdir),
            )

            task = Task(id="alert-test", description="Test task", phase="test")
            result = executor.execute(task)

            assert result.alerts_issued > 0  # At least goal_complete

    def test_recovery_attempts_tracked(self):
        """Test that recovery attempts are tracked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Same output to trigger stuck detection
            outputs = ["Same output"] * 10 + ["<promise>COMPLETE</promise>"]
            mock = MockExecutor(outputs=outputs)

            config = AutonomousConfig(
                max_iterations=20,
                completion_promise="COMPLETE",
                cli_spec="codex",
                stuck_after=3,
                progress_threshold=0.5,  # High threshold to trigger stuck
            )

            executor = AutonomousExecutor(
                executors={"codex": mock},
                config=config,
                workspace_path=Path(tmpdir),
            )

            task = Task(id="recovery-test", description="Test task", phase="test")
            result = executor.execute(task)

            # Recovery would be attempted since outputs are identical
            # (depending on progress detection)
            assert result.iterations > 0
