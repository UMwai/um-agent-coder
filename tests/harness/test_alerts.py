"""Tests for alert system."""

import json
import tempfile
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from um_agent_coder.harness.autonomous.alerts import (
    Alert,
    AlertConfig,
    AlertManager,
    AlertSeverity,
    AlertType,
    PauseRequested,
    RunawayConfig,
    RunawayDetector,
)
from um_agent_coder.harness.autonomous.alerts.runaway_detector import (
    RunawayState,
    hash_output,
)


class TestAlert:
    """Tests for Alert dataclass."""

    def test_basic_creation(self):
        """Test basic alert creation."""
        alert = Alert(
            alert_type=AlertType.NO_PROGRESS.value,
            severity=AlertSeverity.WARNING,
            message="No progress detected",
            iteration=5,
        )
        assert alert.alert_type == "no_progress"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "No progress detected"
        assert alert.iteration == 5

    def test_serialization(self):
        """Test alert serialization."""
        alert = Alert(
            alert_type=AlertType.GOAL_COMPLETE.value,
            severity=AlertSeverity.SUCCESS,
            message="Task complete",
            iteration=10,
            context={"promise": "COMPLETE"},
        )
        data = alert.to_dict()
        assert data["type"] == "goal_complete"
        assert data["severity"] == "SUCCESS"
        assert data["context"]["promise"] == "COMPLETE"

    def test_deserialization(self):
        """Test alert deserialization."""
        data = {
            "type": "fatal_error",
            "severity": "ERROR",
            "message": "Something failed",
            "timestamp": datetime.now().isoformat(),
            "iteration": 20,
            "context": {"error": "details"},
        }
        alert = Alert.from_dict(data)
        assert alert.alert_type == "fatal_error"
        assert alert.severity == AlertSeverity.ERROR
        assert alert.context["error"] == "details"

    def test_to_json(self):
        """Test JSON serialization."""
        alert = Alert(
            alert_type="test",
            severity=AlertSeverity.INFO,
            message="Test message",
        )
        json_str = alert.to_json()
        data = json.loads(json_str)
        assert data["type"] == "test"


class TestAlertConfig:
    """Tests for AlertConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = AlertConfig()
        assert config.write_to_file is True
        assert config.cli_notify is True
        assert config.pause_on_critical is True
        assert config.milestone_interval == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = AlertConfig(
            milestone_interval=5,
            pause_on_critical=False,
            use_colors=False,
        )
        assert config.milestone_interval == 5
        assert config.pause_on_critical is False
        assert config.use_colors is False


class TestAlertManager:
    """Tests for AlertManager."""

    def test_basic_alert(self):
        """Test basic alert creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
            )
            manager = AlertManager(config)

            alert = manager.alert(
                AlertType.NO_PROGRESS.value,
                "Test alert",
                AlertSeverity.WARNING,
                iteration=5,
            )

            assert alert.alert_type == "no_progress"
            assert len(manager.alerts) == 1

    def test_writes_to_file(self):
        """Test alert writes to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "alerts.log"
            config = AlertConfig(
                alert_log_path=log_path,
                cli_notify=False,
            )
            manager = AlertManager(config)

            manager.alert("test", "Test message", AlertSeverity.INFO)

            assert log_path.exists()
            content = log_path.read_text()
            assert "Test message" in content

    def test_cli_notify_with_colors(self):
        """Test CLI notification with colors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=True,
                use_colors=True,
            )
            manager = AlertManager(config)

            with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
                manager.alert("test", "Color test", AlertSeverity.WARNING)
                output = mock_stderr.getvalue()
                assert "WARNING" in output
                assert "Color test" in output

    def test_cli_notify_without_colors(self):
        """Test CLI notification without colors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=True,
                use_colors=False,
            )
            manager = AlertManager(config)

            with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
                manager.alert("test", "No color", AlertSeverity.INFO)
                output = mock_stderr.getvalue()
                assert "[INFO]" in output

    def test_pause_on_critical(self):
        """Test pause on critical alert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
                pause_on_critical=True,
            )
            manager = AlertManager(config)

            with pytest.raises(PauseRequested) as exc_info:
                manager.alert("runaway", "Critical issue", AlertSeverity.CRITICAL)

            assert exc_info.value.alert.severity == AlertSeverity.CRITICAL

    def test_no_pause_when_disabled(self):
        """Test no pause when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
                pause_on_critical=False,
            )
            manager = AlertManager(config)

            # Should not raise
            alert = manager.alert("runaway", "Critical issue", AlertSeverity.CRITICAL)
            assert alert.severity == AlertSeverity.CRITICAL

    def test_custom_handler(self):
        """Test custom alert handler."""
        received_alerts = []

        def handler(alert):
            received_alerts.append(alert)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
                custom_handlers=[handler],
            )
            manager = AlertManager(config)

            manager.alert("test", "Handler test", AlertSeverity.INFO)

            assert len(received_alerts) == 1

    def test_convenience_methods(self):
        """Test convenience alert methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
                pause_on_critical=False,
            )
            manager = AlertManager(config)

            manager.info("Info message")
            manager.warning("Warning message")
            manager.error("Error message")
            manager.success("Success message")
            manager.critical("Critical message")

            assert len(manager.alerts) == 5
            severities = [a.severity for a in manager.alerts]
            assert AlertSeverity.INFO in severities
            assert AlertSeverity.WARNING in severities
            assert AlertSeverity.ERROR in severities
            assert AlertSeverity.SUCCESS in severities
            assert AlertSeverity.CRITICAL in severities

    def test_iteration_milestone(self):
        """Test iteration milestone alerts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
                milestone_interval=5,
            )
            manager = AlertManager(config)

            # Should not alert on iteration 3
            result = manager.iteration_milestone(3)
            assert result is None

            # Should alert on iteration 5
            result = manager.iteration_milestone(5)
            assert result is not None
            assert result.alert_type == AlertType.ITERATION_MILESTONE.value

            # Should alert on iteration 10
            result = manager.iteration_milestone(10)
            assert result is not None

    def test_no_progress_alert(self):
        """Test no progress alert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
            )
            manager = AlertManager(config)

            alert = manager.no_progress(5, iteration=10)
            assert alert.alert_type == AlertType.NO_PROGRESS.value
            assert "5" in alert.message

    def test_stuck_recovery_alert(self):
        """Test stuck recovery alert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
            )
            manager = AlertManager(config)

            alert = manager.stuck_recovery("prompt_mutation", iteration=15)
            assert alert.alert_type == AlertType.STUCK_RECOVERY.value
            assert "prompt_mutation" in alert.message

    def test_approaching_limit_alert(self):
        """Test approaching limit alert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
            )
            manager = AlertManager(config)

            alert = manager.approaching_limit("iterations", 80, 100, iteration=80)
            assert alert.alert_type == AlertType.APPROACHING_LIMIT.value
            assert "80%" in alert.message

    def test_model_escalation_alert(self):
        """Test model escalation alert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
            )
            manager = AlertManager(config)

            alert = manager.model_escalation("gpt-4", "opus", iteration=25)
            assert alert.alert_type == AlertType.MODEL_ESCALATION.value
            assert "gpt-4" in alert.message
            assert "opus" in alert.message

    def test_goal_complete_alert(self):
        """Test goal complete alert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
            )
            manager = AlertManager(config)

            alert = manager.goal_complete(30, "TASK_DONE")
            assert alert.alert_type == AlertType.GOAL_COMPLETE.value
            assert alert.severity == AlertSeverity.SUCCESS
            assert "TASK_DONE" in alert.message

    def test_get_alerts_filter(self):
        """Test getting alerts with filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
            )
            manager = AlertManager(config)

            manager.info("Info 1")
            manager.warning("Warning 1")
            manager.info("Info 2")

            # Filter by severity
            warnings = manager.get_alerts(severity=AlertSeverity.WARNING)
            assert len(warnings) == 1

            # Filter by type
            infos = manager.get_alerts(alert_type=AlertType.CUSTOM.value)
            assert len(infos) == 3

    def test_get_recent_alerts(self):
        """Test getting recent alerts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
            )
            manager = AlertManager(config)

            for i in range(20):
                manager.info(f"Alert {i}")

            recent = manager.get_recent_alerts(5)
            assert len(recent) == 5
            assert recent[-1].message == "Alert 19"

    def test_count_by_severity(self):
        """Test counting alerts by severity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
            )
            manager = AlertManager(config)

            manager.info("Info")
            manager.info("Info")
            manager.warning("Warning")

            counts = manager.count_by_severity()
            assert counts[AlertSeverity.INFO] == 2
            assert counts[AlertSeverity.WARNING] == 1

    def test_trim_alerts(self):
        """Test alerts are trimmed to max size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
                max_alerts_in_memory=10,
            )
            manager = AlertManager(config)

            for i in range(20):
                manager.info(f"Alert {i}")

            assert len(manager.alerts) == 10
            # Should keep most recent
            assert manager.alerts[-1].message == "Alert 19"

    def test_get_summary(self):
        """Test getting alert summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
            )
            manager = AlertManager(config)

            manager.info("Test")
            manager.warning("Test")

            summary = manager.get_summary()
            assert summary["total"] == 2
            assert "by_severity" in summary
            assert "recent" in summary


class TestHashOutput:
    """Tests for hash_output function."""

    def test_hash_basic(self):
        """Test basic hashing."""
        h1 = hash_output("test output")
        h2 = hash_output("test output")
        assert h1 == h2

    def test_hash_different(self):
        """Test different outputs have different hashes."""
        h1 = hash_output("output 1")
        h2 = hash_output("output 2")
        assert h1 != h2

    def test_hash_case_insensitive(self):
        """Test hashing is case insensitive."""
        h1 = hash_output("Test Output")
        h2 = hash_output("test output")
        assert h1 == h2

    def test_hash_strips_whitespace(self):
        """Test hashing strips whitespace."""
        h1 = hash_output("  test  ")
        h2 = hash_output("test")
        assert h1 == h2


class TestRunawayState:
    """Tests for RunawayState."""

    def test_serialization(self):
        """Test state serialization."""
        state = RunawayState(
            iteration_times=[1.0, 2.0, 3.0],
            output_hashes=["abc", "def"],
            last_check_iteration=5,
            warnings_issued=2,
            critical_issued=False,
        )
        data = state.to_dict()
        assert data["iteration_times"] == [1.0, 2.0, 3.0]
        assert data["warnings_issued"] == 2

    def test_deserialization(self):
        """Test state deserialization."""
        data = {
            "iteration_times": [1.0, 2.0],
            "output_hashes": ["a", "b"],
            "last_check_iteration": 10,
            "warnings_issued": 1,
            "critical_issued": True,
        }
        state = RunawayState.from_dict(data)
        assert state.last_check_iteration == 10
        assert state.critical_issued is True


class TestRunawayConfig:
    """Tests for RunawayConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = RunawayConfig()
        assert config.max_iterations_warning == 500
        assert config.output_loop_window == 5
        assert config.min_iteration_seconds == 0.5


class TestRunawayDetector:
    """Tests for RunawayDetector."""

    def test_no_runaway_initially(self):
        """Test no runaway detected initially."""
        detector = RunawayDetector()
        result = detector.check(1, 1.0, "Output 1")
        assert result is None

    def test_detect_output_loop(self):
        """Test detecting output loop."""
        config = RunawayConfig(
            output_loop_window=3,
            enable_speedup_detection=False,
            enable_iteration_warning=False,
        )
        detector = RunawayDetector(config)

        # Same output 3 times
        for i in range(3):
            result = detector.check(i + 1, 1.0, "same output")

        assert result is not None
        assert result.severity == AlertSeverity.CRITICAL
        assert "loop" in result.message.lower()

    def test_no_loop_different_outputs(self):
        """Test no loop with different outputs."""
        config = RunawayConfig(
            output_loop_window=3,
            enable_speedup_detection=False,
            enable_iteration_warning=False,
        )
        detector = RunawayDetector(config)

        # Different outputs
        for i in range(5):
            result = detector.check(i + 1, 1.0, f"output {i}")

        assert result is None

    def test_detect_speedup(self):
        """Test detecting speedup pattern."""
        config = RunawayConfig(
            speedup_window=3,
            speedup_threshold=0.5,
            enable_output_loop_detection=False,
            enable_iteration_warning=False,
        )
        detector = RunawayDetector(config)

        # Start with slow iterations
        for i in range(3):
            detector.check(i + 1, 2.0, f"output {i}")

        # Then fast iterations
        for i in range(3, 10):
            result = detector.check(i + 1, 0.3, f"output {i}")

        # Should eventually detect speedup
        assert result is not None or detector.get_warning_count() > 0

    def test_detect_too_fast(self):
        """Test detecting iterations too fast."""
        config = RunawayConfig(
            min_iteration_seconds=1.0,
            speedup_window=3,
            enable_output_loop_detection=False,
            enable_iteration_warning=False,
        )
        detector = RunawayDetector(config)

        # Very fast iterations
        for i in range(10):
            result = detector.check(i + 1, 0.1, f"output {i}")

        assert result is not None

    def test_iteration_limit_warning(self):
        """Test iteration limit warning."""
        config = RunawayConfig(
            max_iterations_warning=10,
            has_time_limit=False,
            enable_output_loop_detection=False,
            enable_speedup_detection=False,
        )
        detector = RunawayDetector(config)

        # Many iterations
        result = None
        for i in range(15):
            result = detector.check(i + 1, 1.0, f"output {i}")

        assert result is not None
        assert "iterations" in result.message.lower()

    def test_no_warning_with_time_limit(self):
        """Test no warning when time limit set."""
        config = RunawayConfig(
            max_iterations_warning=10,
            has_time_limit=True,
            enable_output_loop_detection=False,
            enable_speedup_detection=False,
        )
        detector = RunawayDetector(config)

        # Many iterations
        result = None
        for i in range(15):
            result = detector.check(i + 1, 1.0, f"output {i}")

        assert result is None

    def test_has_detected_runaway(self):
        """Test has_detected_runaway."""
        config = RunawayConfig(
            output_loop_window=3,
            enable_speedup_detection=False,
            enable_iteration_warning=False,
        )
        detector = RunawayDetector(config)

        assert not detector.has_detected_runaway()

        # Trigger critical
        for i in range(3):
            detector.check(i + 1, 1.0, "same")

        assert detector.has_detected_runaway()

    def test_get_statistics(self):
        """Test getting statistics."""
        detector = RunawayDetector()

        for i in range(5):
            detector.check(i + 1, float(i + 1), f"output {i}")

        stats = detector.get_statistics()
        assert stats["total_iterations_tracked"] == 5
        assert stats["avg_iteration_time"] == 3.0  # (1+2+3+4+5)/5
        assert stats["min_iteration_time"] == 1.0
        assert stats["max_iteration_time"] == 5.0

    def test_reset(self):
        """Test reset."""
        detector = RunawayDetector()

        for i in range(5):
            detector.check(i + 1, 1.0, f"output {i}")

        detector.reset()

        assert len(detector.state.iteration_times) == 0
        assert len(detector.state.output_hashes) == 0


class TestAlertSystemIntegration:
    """Integration tests for alert system."""

    def test_full_workflow(self):
        """Test complete alert workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            alert_config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
                pause_on_critical=False,
                milestone_interval=5,
            )
            manager = AlertManager(alert_config)

            runaway_config = RunawayConfig(
                output_loop_window=3,
                enable_speedup_detection=False,
                enable_iteration_warning=False,
            )
            detector = RunawayDetector(runaway_config)

            # Simulate iterations
            for i in range(10):
                # Check milestones
                manager.iteration_milestone(i + 1)

                # Check runaway
                runaway_alert = detector.check(i + 1, 1.0, f"output {i}")
                if runaway_alert:
                    manager.alert(
                        runaway_alert.alert_type,
                        runaway_alert.message,
                        runaway_alert.severity,
                        iteration=runaway_alert.iteration,
                    )

            # Should have milestone alerts at 5 and 10
            milestones = manager.get_alerts(alert_type=AlertType.ITERATION_MILESTONE.value)
            assert len(milestones) == 2

    def test_runaway_triggers_critical(self):
        """Test runaway detection triggers critical alert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            alert_config = AlertConfig(
                alert_log_path=Path(tmpdir) / "alerts.log",
                cli_notify=False,
                pause_on_critical=True,
            )
            manager = AlertManager(alert_config)

            runaway_config = RunawayConfig(
                output_loop_window=3,
                enable_speedup_detection=False,
                enable_iteration_warning=False,
            )
            detector = RunawayDetector(runaway_config)

            # Trigger runaway
            with pytest.raises(PauseRequested):
                for i in range(5):
                    runaway_alert = detector.check(i + 1, 1.0, "same output")
                    if runaway_alert:
                        manager.runaway_detected(
                            runaway_alert.message,
                            iteration=i + 1,
                        )
