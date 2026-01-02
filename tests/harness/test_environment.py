"""Tests for environmental awareness system."""

import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.um_agent_coder.harness.autonomous.environment import (
    FileEvent,
    FileEventType,
    WorkspaceWatcher,
    Instruction,
    InstructionPriority,
    InstructionQueue,
    EnvChange,
    EnvMonitor,
    MONITORED_ENV_VARS,
    EnvironmentManager,
    EnvironmentState,
)
from src.um_agent_coder.harness.autonomous.environment.file_watcher import (
    DEFAULT_IGNORE_PATTERNS,
    PollingWatcher,
)


class TestFileEvent:
    """Tests for FileEvent."""

    def test_basic_creation(self):
        """Test basic event creation."""
        event = FileEvent(
            event_type=FileEventType.MODIFIED,
            path="/test/file.py",
        )
        assert event.event_type == FileEventType.MODIFIED
        assert event.path == "/test/file.py"
        assert event.is_directory is False

    def test_serialization(self):
        """Test event serialization."""
        event = FileEvent(
            event_type=FileEventType.CREATED,
            path="/test/new.py",
            is_directory=False,
            dest_path=None,
        )
        data = event.to_dict()
        assert data["event_type"] == "created"
        assert data["path"] == "/test/new.py"

    def test_deserialization(self):
        """Test event deserialization."""
        data = {
            "event_type": "deleted",
            "path": "/test/old.py",
            "timestamp": datetime.now().isoformat(),
            "is_directory": False,
        }
        event = FileEvent.from_dict(data)
        assert event.event_type == FileEventType.DELETED
        assert event.path == "/test/old.py"

    def test_move_event_with_dest(self):
        """Test move event with destination path."""
        event = FileEvent(
            event_type=FileEventType.MOVED,
            path="/test/old.py",
            dest_path="/test/new.py",
        )
        assert event.dest_path == "/test/new.py"


class TestWorkspaceWatcher:
    """Tests for WorkspaceWatcher."""

    def test_should_ignore_harness(self):
        """Test ignoring .harness directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = WorkspaceWatcher(Path(tmpdir))
            harness_path = Path(tmpdir) / ".harness" / "state.db"
            assert watcher.should_ignore(str(harness_path))

    def test_should_ignore_git(self):
        """Test ignoring .git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = WorkspaceWatcher(Path(tmpdir))
            git_path = Path(tmpdir) / ".git" / "objects"
            assert watcher.should_ignore(str(git_path))

    def test_should_ignore_pycache(self):
        """Test ignoring __pycache__ directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = WorkspaceWatcher(Path(tmpdir))
            pycache_path = Path(tmpdir) / "__pycache__" / "module.pyc"
            assert watcher.should_ignore(str(pycache_path))

    def test_should_not_ignore_source(self):
        """Test not ignoring source files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = WorkspaceWatcher(Path(tmpdir))
            src_path = Path(tmpdir) / "src" / "module.py"
            assert not watcher.should_ignore(str(src_path))

    def test_custom_ignore_patterns(self):
        """Test custom ignore patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = WorkspaceWatcher(
                Path(tmpdir),
                ignore_patterns=["*.log", "temp/*"],
            )
            assert watcher.should_ignore(str(Path(tmpdir) / "debug.log"))
            assert watcher.should_ignore(str(Path(tmpdir) / "temp" / "file.txt"))
            assert not watcher.should_ignore(str(Path(tmpdir) / "src" / "main.py"))

    def test_add_event_manually(self):
        """Test adding events manually."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = WorkspaceWatcher(Path(tmpdir))
            event = FileEvent(
                event_type=FileEventType.MODIFIED,
                path="/test/file.py",
            )
            watcher.add_event(event)
            events = watcher.get_events()
            assert len(events) == 1
            assert events[0].event_type == FileEventType.MODIFIED

    def test_has_changes(self):
        """Test has_changes detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = WorkspaceWatcher(Path(tmpdir))
            assert not watcher.has_changes()

            watcher.add_event(
                FileEvent(event_type=FileEventType.CREATED, path="/test.py")
            )
            assert watcher.has_changes()

    def test_get_events_clears(self):
        """Test get_events clears by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = WorkspaceWatcher(Path(tmpdir))
            watcher.add_event(
                FileEvent(event_type=FileEventType.MODIFIED, path="/test.py")
            )
            events1 = watcher.get_events()
            events2 = watcher.get_events()
            assert len(events1) == 1
            assert len(events2) == 0

    def test_get_events_no_clear(self):
        """Test get_events without clearing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = WorkspaceWatcher(Path(tmpdir))
            watcher.add_event(
                FileEvent(event_type=FileEventType.MODIFIED, path="/test.py")
            )
            events1 = watcher.get_events(clear=False)
            events2 = watcher.get_events(clear=False)
            assert len(events1) == 1
            assert len(events2) == 1

    def test_get_modified_files(self):
        """Test getting modified files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = WorkspaceWatcher(Path(tmpdir))
            watcher.add_event(
                FileEvent(event_type=FileEventType.CREATED, path="/new.py")
            )
            watcher.add_event(
                FileEvent(event_type=FileEventType.MODIFIED, path="/changed.py")
            )
            watcher.add_event(
                FileEvent(event_type=FileEventType.DELETED, path="/removed.py")
            )

            modified = watcher.get_modified_files()
            assert "/new.py" in modified
            assert "/changed.py" in modified
            assert "/removed.py" not in modified

    def test_callback_invoked(self):
        """Test callback is invoked on event."""
        events_received = []

        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = WorkspaceWatcher(
                Path(tmpdir),
                callback=lambda e: events_received.append(e),
            )
            watcher.add_event(
                FileEvent(event_type=FileEventType.MODIFIED, path="/test.py")
            )
            assert len(events_received) == 1


class TestPollingWatcher:
    """Tests for PollingWatcher."""

    def test_detect_new_file(self):
        """Test detecting new files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = PollingWatcher(Path(tmpdir))

            # Initial scan
            watcher.scan()

            # Create file
            new_file = Path(tmpdir) / "new.py"
            new_file.write_text("content")

            # Scan again
            events = watcher.scan()

            assert len(events) == 1
            assert events[0].event_type == FileEventType.CREATED
            assert "new.py" in events[0].path

    def test_detect_modified_file(self):
        """Test detecting modified files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = Path(tmpdir) / "existing.py"
            existing_file.write_text("original")

            watcher = PollingWatcher(Path(tmpdir))
            watcher.scan()  # Initial scan

            # Modify file
            time.sleep(0.1)  # Ensure mtime changes
            existing_file.write_text("modified")

            events = watcher.scan()
            assert len(events) == 1
            assert events[0].event_type == FileEventType.MODIFIED

    def test_detect_deleted_file(self):
        """Test detecting deleted files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = Path(tmpdir) / "existing.py"
            existing_file.write_text("content")

            watcher = PollingWatcher(Path(tmpdir))
            watcher.scan()  # Initial scan

            # Delete file
            existing_file.unlink()

            events = watcher.scan()
            assert len(events) == 1
            assert events[0].event_type == FileEventType.DELETED

    def test_respects_ignore_patterns(self):
        """Test ignoring patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = PollingWatcher(
                Path(tmpdir),
                ignore_patterns=["*.log"],
            )
            watcher.scan()

            # Create ignored file
            (Path(tmpdir) / "debug.log").write_text("log")
            # Create tracked file
            (Path(tmpdir) / "main.py").write_text("code")

            events = watcher.scan()
            assert len(events) == 1
            assert "main.py" in events[0].path


class TestInstruction:
    """Tests for Instruction."""

    def test_basic_creation(self):
        """Test basic instruction creation."""
        inst = Instruction(
            id="001",
            content="Test instruction",
        )
        assert inst.id == "001"
        assert inst.content == "Test instruction"
        assert inst.priority == InstructionPriority.NORMAL

    def test_serialization(self):
        """Test instruction serialization."""
        inst = Instruction(
            id="002",
            content="High priority",
            priority=InstructionPriority.HIGH,
        )
        data = inst.to_dict()
        assert data["id"] == "002"
        assert data["priority"] == InstructionPriority.HIGH.value

    def test_deserialization(self):
        """Test instruction deserialization."""
        data = {
            "id": "003",
            "content": "Urgent task",
            "priority": 0,
            "timestamp": datetime.now().isoformat(),
        }
        inst = Instruction.from_dict(data)
        assert inst.id == "003"
        assert inst.priority == InstructionPriority.URGENT


class TestInstructionQueue:
    """Tests for InstructionQueue."""

    def test_empty_queue(self):
        """Test empty queue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = InstructionQueue(Path(tmpdir) / "inbox")
            assert queue.poll() == []
            assert not queue.has_pending()

    def test_add_and_poll(self):
        """Test adding and polling instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = InstructionQueue(Path(tmpdir) / "inbox")

            inst = queue.add_instruction("Test instruction")
            assert inst.content == "Test instruction"

            pending = queue.poll()
            assert len(pending) == 1
            assert pending[0].content == "Test instruction"

    def test_priority_sorting(self):
        """Test instructions sorted by priority."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = InstructionQueue(Path(tmpdir) / "inbox")

            queue.add_instruction("Low", InstructionPriority.LOW, "zzz-low")
            queue.add_instruction("Normal", InstructionPriority.NORMAL, "005-normal")
            queue.add_instruction("Urgent", InstructionPriority.URGENT, "000-urgent")

            pending = queue.poll()
            assert len(pending) == 3
            assert pending[0].priority == InstructionPriority.URGENT
            assert pending[1].priority == InstructionPriority.NORMAL
            assert pending[2].priority == InstructionPriority.LOW

    def test_mark_processed(self):
        """Test marking instruction as processed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = InstructionQueue(Path(tmpdir) / "inbox")
            inst = queue.add_instruction("Process me")

            result = queue.mark_processed(inst)
            assert result is True
            assert not queue.has_pending()

            # Check processed directory
            processed = list((Path(tmpdir) / "inbox" / "processed").glob("*.txt"))
            assert len(processed) == 1

    def test_poll_urgent(self):
        """Test polling only urgent instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = InstructionQueue(Path(tmpdir) / "inbox")

            queue.add_instruction("Normal task", InstructionPriority.NORMAL, "005-normal")
            queue.add_instruction("Urgent task", InstructionPriority.URGENT, "000-urgent")

            urgent = queue.poll_urgent()
            assert len(urgent) == 1
            assert urgent[0].content == "Urgent task"

    def test_parse_priority_from_filename(self):
        """Test parsing priority from filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = InstructionQueue(Path(tmpdir) / "inbox")

            assert queue._parse_priority("000-urgent.txt") == InstructionPriority.URGENT
            assert queue._parse_priority("001-high.txt") == InstructionPriority.HIGH
            assert queue._parse_priority("005-normal.txt") == InstructionPriority.NORMAL
            assert queue._parse_priority("zzz-low.txt") == InstructionPriority.LOW
            assert queue._parse_priority("regular.txt") == InstructionPriority.NORMAL

    def test_get_queue_status(self):
        """Test getting queue status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = InstructionQueue(Path(tmpdir) / "inbox")

            queue.add_instruction("Urgent", InstructionPriority.URGENT, "000-urgent")
            queue.add_instruction("Normal", InstructionPriority.NORMAL, "005-normal")

            status = queue.get_queue_status()
            assert status["total_pending"] == 2
            assert status["urgent"] == 1
            assert status["normal"] == 1

    def test_has_urgent(self):
        """Test has_urgent check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = InstructionQueue(Path(tmpdir) / "inbox")

            assert not queue.has_urgent()

            queue.add_instruction("Urgent!", InstructionPriority.URGENT, "000-urgent")
            assert queue.has_urgent()


class TestEnvChange:
    """Tests for EnvChange."""

    def test_basic_creation(self):
        """Test basic change creation."""
        change = EnvChange(
            var="HARNESS_MODE",
            old_value="normal",
            new_value="turbo",
        )
        assert change.var == "HARNESS_MODE"
        assert change.is_set
        assert not change.is_unset

    def test_is_unset(self):
        """Test detecting unset."""
        change = EnvChange(
            var="HARNESS_PAUSE",
            old_value="true",
            new_value=None,
        )
        assert change.is_unset
        assert not change.is_set

    def test_serialization(self):
        """Test serialization."""
        change = EnvChange(
            var="HARNESS_CLI",
            old_value="codex",
            new_value="gemini",
        )
        data = change.to_dict()
        assert data["var"] == "HARNESS_CLI"
        assert data["old_value"] == "codex"
        assert data["new_value"] == "gemini"


class TestEnvMonitor:
    """Tests for EnvMonitor."""

    def test_initial_snapshot(self):
        """Test initial snapshot is taken."""
        monitor = EnvMonitor()
        snapshot = monitor.get_snapshot()
        # Should have entries for all monitored vars
        assert "HARNESS_MODE" in snapshot

    def test_detect_change(self):
        """Test detecting environment change."""
        with patch.dict(os.environ, {}, clear=False):
            monitor = EnvMonitor()

            # Set a new value
            os.environ["HARNESS_MODE"] = "turbo"

            changes = monitor.check_changes()
            mode_changes = [c for c in changes if c.var == "HARNESS_MODE"]
            assert len(mode_changes) == 1
            assert mode_changes[0].new_value == "turbo"

            # Clean up
            del os.environ["HARNESS_MODE"]

    def test_no_change_no_report(self):
        """Test no change results in empty list."""
        monitor = EnvMonitor()
        # First check establishes baseline
        monitor.check_changes()
        # Second check with no changes
        changes = monitor.check_changes()
        assert changes == []

    def test_is_paused(self):
        """Test is_paused check."""
        with patch.dict(os.environ, {"HARNESS_PAUSE": "true"}, clear=False):
            monitor = EnvMonitor()
            assert monitor.is_paused()

        with patch.dict(os.environ, {"HARNESS_PAUSE": "false"}, clear=False):
            monitor = EnvMonitor()
            assert not monitor.is_paused()

    def test_should_stop(self):
        """Test should_stop check."""
        with patch.dict(os.environ, {"HARNESS_STOP": "true"}, clear=False):
            monitor = EnvMonitor()
            assert monitor.should_stop()

    def test_get_mode(self):
        """Test get_mode."""
        with patch.dict(os.environ, {"HARNESS_MODE": "conservative"}, clear=False):
            monitor = EnvMonitor()
            assert monitor.get_mode() == "conservative"

    def test_get_cli_override(self):
        """Test get_cli_override."""
        with patch.dict(os.environ, {"HARNESS_CLI": "gemini"}, clear=False):
            monitor = EnvMonitor()
            assert monitor.get_cli_override() == "gemini"

        with patch.dict(os.environ, {"HARNESS_CLI": "auto"}, clear=False):
            monitor = EnvMonitor()
            assert monitor.get_cli_override() is None

    def test_validation(self):
        """Test value validation."""
        with patch.dict(os.environ, {"HARNESS_MODE": "invalid"}, clear=False):
            monitor = EnvMonitor(validate=True)
            os.environ["HARNESS_MODE"] = "invalid"
            changes = monitor.check_changes()
            # Should detect change but mark as invalid
            mode_changes = [c for c in changes if c.var == "HARNESS_MODE"]
            if mode_changes:
                assert mode_changes[0].is_valid is False

    def test_additional_vars(self):
        """Test monitoring additional variables."""
        monitor = EnvMonitor(additional_vars={"CUSTOM_VAR"})
        assert "CUSTOM_VAR" in monitor.monitored_vars


class TestEnvironmentState:
    """Tests for EnvironmentState."""

    def test_empty_state(self):
        """Test empty state."""
        state = EnvironmentState()
        assert not state.has_changes
        assert not state.has_urgent_instructions
        assert not state.should_pause
        assert not state.should_stop

    def test_has_changes(self):
        """Test has_changes property."""
        state = EnvironmentState(
            file_events=[FileEvent(event_type=FileEventType.MODIFIED, path="/test.py")]
        )
        assert state.has_changes

    def test_has_urgent_instructions(self):
        """Test detecting urgent instructions."""
        state = EnvironmentState(
            instructions=[
                Instruction(id="001", content="Urgent!", priority=InstructionPriority.URGENT)
            ]
        )
        assert state.has_urgent_instructions

    def test_should_pause(self):
        """Test should_pause from env change."""
        state = EnvironmentState(
            env_changes=[
                EnvChange(var="HARNESS_PAUSE", old_value="false", new_value="true")
            ]
        )
        assert state.should_pause

    def test_should_stop(self):
        """Test should_stop from env change."""
        state = EnvironmentState(
            env_changes=[
                EnvChange(var="HARNESS_STOP", old_value=None, new_value="true")
            ]
        )
        assert state.should_stop

    def test_get_prompt_section_empty(self):
        """Test empty prompt section."""
        state = EnvironmentState()
        assert state.get_prompt_section() == ""

    def test_get_prompt_section_with_instructions(self):
        """Test prompt section with instructions."""
        state = EnvironmentState(
            instructions=[
                Instruction(id="001", content="Focus on tests", priority=InstructionPriority.NORMAL)
            ]
        )
        section = state.get_prompt_section()
        assert "New Instructions" in section
        assert "Focus on tests" in section

    def test_get_prompt_section_with_env_changes(self):
        """Test prompt section with env changes."""
        state = EnvironmentState(
            env_changes=[
                EnvChange(var="HARNESS_MODE", old_value="normal", new_value="turbo")
            ]
        )
        section = state.get_prompt_section()
        assert "Environment Changes" in section
        assert "HARNESS_MODE" in section

    def test_serialization(self):
        """Test state serialization."""
        state = EnvironmentState(
            file_events=[FileEvent(event_type=FileEventType.MODIFIED, path="/test.py")],
            instructions=[Instruction(id="001", content="Test")],
        )
        data = state.to_dict()
        assert len(data["file_events"]) == 1
        assert len(data["instructions"]) == 1

        restored = EnvironmentState.from_dict(data)
        assert len(restored.file_events) == 1
        assert len(restored.instructions) == 1


class TestEnvironmentManager:
    """Tests for EnvironmentManager."""

    def test_basic_init(self):
        """Test basic initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EnvironmentManager(
                workspace_path=Path(tmpdir),
                enable_file_watcher=False,  # Disable for simpler tests
            )
            assert manager.workspace_path == Path(tmpdir).resolve()

    def test_poll_empty(self):
        """Test polling with no changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EnvironmentManager(
                workspace_path=Path(tmpdir),
                enable_file_watcher=False,
            )
            state = manager.poll()
            assert not state.has_changes

    def test_poll_with_instruction(self):
        """Test polling with pending instruction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EnvironmentManager(
                workspace_path=Path(tmpdir),
                enable_file_watcher=False,
            )
            manager.add_instruction("Test instruction")

            state = manager.poll()
            assert state.has_changes
            assert len(state.instructions) == 1

    def test_check_stop_file(self):
        """Test checking stop file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness_path = Path(tmpdir) / ".harness"
            harness_path.mkdir()

            manager = EnvironmentManager(
                workspace_path=Path(tmpdir),
                harness_path=harness_path,
                enable_file_watcher=False,
            )

            # No stop file
            assert manager.check_stop_file() is None

            # Create stop file
            stop_file = harness_path / "stop"
            stop_file.write_text("stop")

            assert manager.check_stop_file() == "stop"
            # File should be removed after reading
            assert not stop_file.exists()

    def test_check_stop_file_abort(self):
        """Test abort stop file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness_path = Path(tmpdir) / ".harness"
            harness_path.mkdir()

            manager = EnvironmentManager(
                workspace_path=Path(tmpdir),
                harness_path=harness_path,
                enable_file_watcher=False,
            )

            stop_file = harness_path / "stop"
            stop_file.write_text("abort")

            assert manager.check_stop_file() == "abort"

    def test_is_paused(self):
        """Test is_paused check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"HARNESS_PAUSE": "true"}, clear=False):
                manager = EnvironmentManager(
                    workspace_path=Path(tmpdir),
                    enable_file_watcher=False,
                )
                assert manager.is_paused()

    def test_get_config_overrides(self):
        """Test getting config overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "HARNESS_MODE": "turbo",
                    "HARNESS_CLI": "gemini",
                },
                clear=False,
            ):
                manager = EnvironmentManager(
                    workspace_path=Path(tmpdir),
                    enable_file_watcher=False,
                )
                overrides = manager.get_config_overrides()
                assert overrides.get("mode") == "turbo"
                assert overrides.get("cli") == "gemini"

    def test_get_status(self):
        """Test getting status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EnvironmentManager(
                workspace_path=Path(tmpdir),
                enable_file_watcher=False,
            )
            status = manager.get_status()
            assert status["started"] is False
            assert "workspace_path" in status
            assert "instruction_queue" in status

    def test_context_manager(self):
        """Test context manager protocol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with EnvironmentManager(
                workspace_path=Path(tmpdir),
                enable_file_watcher=False,
            ) as manager:
                assert manager._started is True
            assert manager._started is False

    def test_poll_urgent_only(self):
        """Test polling urgent items only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EnvironmentManager(
                workspace_path=Path(tmpdir),
                enable_file_watcher=False,
            )
            manager.add_instruction("Normal", InstructionPriority.NORMAL)
            manager.add_instruction("Urgent!", InstructionPriority.URGENT)

            state = manager.poll_urgent_only()
            assert len(state.instructions) == 1
            assert state.instructions[0].content == "Urgent!"


class TestEnvironmentManagerIntegration:
    """Integration tests for EnvironmentManager."""

    def test_full_workflow(self):
        """Test complete environment workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EnvironmentManager(
                workspace_path=Path(tmpdir),
                enable_file_watcher=False,
            )
            manager.start()

            # Add some instructions
            manager.add_instruction("First task")
            manager.add_instruction("Urgent!", InstructionPriority.URGENT)

            # Poll and check
            state = manager.poll()
            assert len(state.instructions) == 2
            assert state.has_urgent_instructions

            # Second poll should be empty (instructions processed)
            state2 = manager.poll()
            assert len(state2.instructions) == 0

            manager.stop()

    def test_with_file_watcher_polling(self):
        """Test with polling file watcher."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EnvironmentManager(
                workspace_path=Path(tmpdir),
                enable_file_watcher=True,
                use_polling=True,
            )
            manager.start()

            # Create a file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            # Poll to detect
            state = manager.poll()
            assert len(state.file_events) >= 1

            manager.stop()
