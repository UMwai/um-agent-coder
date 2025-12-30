"""
Unit tests for the harness module.

Tests:
1. Roadmap parsing - verify it correctly parses specs/roadmap.md
2. Task dependency resolution - verify tasks respect dependencies
3. CLI selection - verify per-task CLI override works
4. Dry-run mode - verify it marks tasks complete without executing
"""

import unittest
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime
from pathlib import Path
import tempfile
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from um_agent_coder.harness.roadmap_parser import RoadmapParser
from um_agent_coder.harness.models import (
    Roadmap, Phase, Task, TaskStatus, ExecutionResult, HarnessState
)
from um_agent_coder.harness.state import StateManager
from um_agent_coder.harness.main import Harness


# Sample roadmap content for testing
SAMPLE_ROADMAP = """# Roadmap: Test Project

## Objective
Test objective for unit tests

## Constraints
- Max time per task: 15 min
- Max retries per task: 2
- Working directory: ./test_dir

## Success Criteria
- [ ] All tests pass
- [ ] Code deployed

## Tasks

### Phase 1: Setup
- [ ] **task-001**: Initialize project
  - timeout: 10min
  - depends: none
  - success: Project exists
  - cwd: ./test_dir
  - cli: codex
  - model: gpt-5.2

- [x] **task-002**: Already completed task
  - timeout: 5min
  - depends: task-001
  - success: Completed
  - cli: gemini

### Phase 2: Implementation
- [ ] **task-003**: Implement feature
  - timeout: 20min
  - depends: task-001, task-002
  - success: Feature works
  - cli: claude
  - model: claude-opus-4.5

- [ ] **task-004**: Test feature
  - timeout: 15min
  - depends: task-003
  - success: Tests pass

### Phase 3: Deployment
- [ ] **task-005**: Deploy application
  - timeout: 10min
  - depends: task-004
  - success: App is live
  - cli: codex

## Growth Mode
1. Analyze performance metrics
2. Optimize bottlenecks
3. Add new features
"""


class TestRoadmapParser(unittest.TestCase):
    """Test roadmap parsing functionality."""

    def setUp(self):
        """Set up temporary roadmap file."""
        self.temp_dir = tempfile.mkdtemp()
        self.roadmap_path = os.path.join(self.temp_dir, "roadmap.md")
        with open(self.roadmap_path, 'w') as f:
            f.write(SAMPLE_ROADMAP)
        self.parser = RoadmapParser(self.roadmap_path)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_parse_project_name(self):
        """Test that project name is extracted correctly."""
        roadmap = self.parser.parse()
        self.assertEqual(roadmap.name, "Test Project")

    def test_parse_objective(self):
        """Test that objective is extracted correctly."""
        roadmap = self.parser.parse()
        self.assertEqual(roadmap.objective, "Test objective for unit tests")

    def test_parse_constraints(self):
        """Test that constraints are parsed correctly."""
        roadmap = self.parser.parse()
        self.assertEqual(roadmap.max_time_per_task, 15)
        self.assertEqual(roadmap.max_retries, 2)
        self.assertEqual(roadmap.working_directory, "./test_dir")

    def test_parse_success_criteria(self):
        """Test that success criteria are parsed correctly."""
        roadmap = self.parser.parse()
        self.assertEqual(len(roadmap.success_criteria), 2)
        self.assertIn("All tests pass", roadmap.success_criteria)
        self.assertIn("Code deployed", roadmap.success_criteria)

    def test_parse_phases(self):
        """Test that phases are parsed correctly."""
        roadmap = self.parser.parse()
        self.assertEqual(len(roadmap.phases), 3)
        self.assertEqual(roadmap.phases[0].name, "Setup")
        self.assertEqual(roadmap.phases[1].name, "Implementation")
        self.assertEqual(roadmap.phases[2].name, "Deployment")

    def test_parse_tasks(self):
        """Test that tasks are parsed correctly."""
        roadmap = self.parser.parse()
        all_tasks = roadmap.all_tasks

        self.assertEqual(len(all_tasks), 5)

        # Check first task
        task1 = all_tasks[0]
        self.assertEqual(task1.id, "task-001")
        self.assertEqual(task1.description, "Initialize project")
        self.assertEqual(task1.phase, "Setup")
        self.assertEqual(task1.timeout_minutes, 10)
        self.assertEqual(task1.success_criteria, "Project exists")
        self.assertEqual(task1.cwd, "./test_dir")
        self.assertEqual(task1.cli, "codex")
        self.assertEqual(task1.model, "gpt-5.2")
        self.assertEqual(task1.depends, [])
        self.assertEqual(task1.status, TaskStatus.PENDING)

    def test_parse_completed_task(self):
        """Test that already completed tasks are marked correctly."""
        roadmap = self.parser.parse()
        task2 = roadmap.get_task("task-002")

        self.assertIsNotNone(task2)
        self.assertEqual(task2.status, TaskStatus.COMPLETED)

    def test_parse_task_dependencies(self):
        """Test that task dependencies are parsed correctly."""
        roadmap = self.parser.parse()

        task3 = roadmap.get_task("task-003")
        self.assertEqual(task3.depends, ["task-001", "task-002"])

        task4 = roadmap.get_task("task-004")
        self.assertEqual(task4.depends, ["task-003"])

    def test_parse_cli_override(self):
        """Test that per-task CLI override is parsed correctly."""
        roadmap = self.parser.parse()

        task1 = roadmap.get_task("task-001")
        self.assertEqual(task1.cli, "codex")

        task2 = roadmap.get_task("task-002")
        self.assertEqual(task2.cli, "gemini")

        task3 = roadmap.get_task("task-003")
        self.assertEqual(task3.cli, "claude")

        # Task without CLI override should have empty string
        task4 = roadmap.get_task("task-004")
        self.assertEqual(task4.cli, "")

    def test_parse_model_override(self):
        """Test that per-task model override is parsed correctly."""
        roadmap = self.parser.parse()

        task1 = roadmap.get_task("task-001")
        self.assertEqual(task1.model, "gpt-5.2")

        task3 = roadmap.get_task("task-003")
        self.assertEqual(task3.model, "claude-opus-4.5")

        # Task without model override should have empty string
        task4 = roadmap.get_task("task-004")
        self.assertEqual(task4.model, "")

    def test_parse_growth_instructions(self):
        """Test that growth mode instructions are parsed correctly."""
        roadmap = self.parser.parse()
        self.assertEqual(len(roadmap.growth_instructions), 3)
        self.assertIn("Analyze performance metrics", roadmap.growth_instructions)
        self.assertIn("Optimize bottlenecks", roadmap.growth_instructions)

    def test_update_task_status(self):
        """Test updating task status in roadmap file."""
        roadmap = self.parser.parse()

        # Update task-001 to completed
        self.parser.update_task_status("task-001", completed=True)

        # Re-parse and check
        roadmap = self.parser.parse()
        task1 = roadmap.get_task("task-001")
        self.assertEqual(task1.status, TaskStatus.COMPLETED)


class TestTaskDependencyResolution(unittest.TestCase):
    """Test task dependency resolution."""

    def test_can_execute_no_dependencies(self):
        """Test that task with no dependencies can execute."""
        task = Task(
            id="task-001",
            description="Test task",
            phase="Test",
            depends=[]
        )

        completed_tasks = set()
        self.assertTrue(task.can_execute(completed_tasks))

    def test_can_execute_with_satisfied_dependencies(self):
        """Test that task can execute when all dependencies are satisfied."""
        task = Task(
            id="task-003",
            description="Test task",
            phase="Test",
            depends=["task-001", "task-002"]
        )

        completed_tasks = {"task-001", "task-002"}
        self.assertTrue(task.can_execute(completed_tasks))

    def test_cannot_execute_with_unsatisfied_dependencies(self):
        """Test that task cannot execute when dependencies are not satisfied."""
        task = Task(
            id="task-003",
            description="Test task",
            phase="Test",
            depends=["task-001", "task-002"]
        )

        # Only one dependency satisfied
        completed_tasks = {"task-001"}
        self.assertFalse(task.can_execute(completed_tasks))

    def test_dependency_chain_ordering(self):
        """Test that dependency chain is respected."""
        tasks = [
            Task(id="task-001", description="First", phase="Test", depends=[]),
            Task(id="task-002", description="Second", phase="Test", depends=["task-001"]),
            Task(id="task-003", description="Third", phase="Test", depends=["task-002"]),
        ]

        completed = set()

        # Only task-001 can execute initially
        self.assertTrue(tasks[0].can_execute(completed))
        self.assertFalse(tasks[1].can_execute(completed))
        self.assertFalse(tasks[2].can_execute(completed))

        # After task-001 completes, task-002 can execute
        completed.add("task-001")
        self.assertTrue(tasks[1].can_execute(completed))
        self.assertFalse(tasks[2].can_execute(completed))

        # After task-002 completes, task-003 can execute
        completed.add("task-002")
        self.assertTrue(tasks[2].can_execute(completed))


class TestStateManager(unittest.TestCase):
    """Test state persistence."""

    def setUp(self):
        """Set up temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_state.db")
        self.state_manager = StateManager(self.db_path)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_init_harness(self):
        """Test harness state initialization."""
        state = self.state_manager.init_harness("test_roadmap.md")

        self.assertEqual(state.roadmap_path, "test_roadmap.md")
        self.assertEqual(state.tasks_completed, 0)
        self.assertEqual(state.tasks_failed, 0)
        self.assertFalse(state.in_growth_mode)

    def test_save_and_load_task(self):
        """Test saving and loading tasks."""
        task = Task(
            id="task-001",
            description="Test task",
            phase="Test",
            depends=["task-000"],
            timeout_minutes=15,
            success_criteria="Test passes",
            cwd="./test",
            cli="codex",
            model="gpt-5.2",
        )

        self.state_manager.save_task(task)
        loaded_task = self.state_manager.load_task("task-001")

        self.assertIsNotNone(loaded_task)
        self.assertEqual(loaded_task.id, task.id)
        self.assertEqual(loaded_task.description, task.description)
        self.assertEqual(loaded_task.depends, task.depends)
        self.assertEqual(loaded_task.cli, task.cli)
        self.assertEqual(loaded_task.model, task.model)

    def test_get_completed_task_ids(self):
        """Test getting completed task IDs."""
        tasks = [
            Task(id="task-001", description="Test 1", phase="Test", status=TaskStatus.COMPLETED),
            Task(id="task-002", description="Test 2", phase="Test", status=TaskStatus.PENDING),
            Task(id="task-003", description="Test 3", phase="Test", status=TaskStatus.COMPLETED),
        ]

        for task in tasks:
            self.state_manager.save_task(task)

        completed_ids = self.state_manager.get_completed_task_ids()
        self.assertEqual(completed_ids, {"task-001", "task-003"})

    def test_log_execution(self):
        """Test logging task execution."""
        task = Task(id="task-001", description="Test", phase="Test")
        self.state_manager.save_task(task)

        result = ExecutionResult(
            success=True,
            output="Task completed",
            error="",
            duration_seconds=10.5
        )

        self.state_manager.log_execution("task-001", 1, result)

        history = self.state_manager.get_execution_history("task-001")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["attempt"], 1)
        self.assertEqual(history[0]["success"], 1)


class TestCLISelection(unittest.TestCase):
    """Test CLI backend selection."""

    def setUp(self):
        """Set up temporary roadmap file."""
        self.temp_dir = tempfile.mkdtemp()
        self.roadmap_path = os.path.join(self.temp_dir, "roadmap.md")
        with open(self.roadmap_path, 'w') as f:
            f.write(SAMPLE_ROADMAP)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('um_agent_coder.harness.main.CodexExecutor')
    @patch('um_agent_coder.harness.main.GeminiExecutor')
    @patch('um_agent_coder.harness.main.ClaudeExecutor')
    @patch('um_agent_coder.harness.main.StateManager')
    @patch('um_agent_coder.harness.main.GrowthLoop')
    def test_default_cli_selection(self, mock_growth, mock_state, mock_claude, mock_gemini, mock_codex):
        """Test default CLI backend is used when no override specified."""
        # Mock executors
        mock_codex_instance = MagicMock()
        mock_codex.return_value = mock_codex_instance

        # Create harness with codex as default
        harness = Harness(
            roadmap_path=self.roadmap_path,
            cli="codex",
            dry_run=True
        )

        # Verify codex executor was created
        mock_codex.assert_called_once()

    @patch('um_agent_coder.harness.main.CodexExecutor')
    @patch('um_agent_coder.harness.main.GeminiExecutor')
    @patch('um_agent_coder.harness.main.ClaudeExecutor')
    @patch('um_agent_coder.harness.main.StateManager')
    @patch('um_agent_coder.harness.main.GrowthLoop')
    def test_per_task_cli_override(self, mock_growth, mock_state, mock_claude, mock_gemini, mock_codex):
        """Test per-task CLI override is respected."""
        # Mock executors
        mock_codex_instance = MagicMock()
        mock_gemini_instance = MagicMock()
        mock_claude_instance = MagicMock()

        mock_codex.return_value = mock_codex_instance
        mock_gemini.return_value = mock_gemini_instance
        mock_claude.return_value = mock_claude_instance

        # Create harness with codex as default
        harness = Harness(
            roadmap_path=self.roadmap_path,
            cli="codex",
            dry_run=True
        )

        # Create tasks with different CLI overrides
        task_codex = Task(id="task-001", description="Codex task", phase="Test", cli="codex")
        task_gemini = Task(id="task-002", description="Gemini task", phase="Test", cli="gemini")
        task_claude = Task(id="task-003", description="Claude task", phase="Test", cli="claude")
        task_default = Task(id="task-004", description="Default task", phase="Test", cli="")

        # Get executors for each task
        exec_codex = harness._get_executor_for_task(task_codex)
        exec_gemini = harness._get_executor_for_task(task_gemini)
        exec_claude = harness._get_executor_for_task(task_claude)
        exec_default = harness._get_executor_for_task(task_default)

        # Verify correct executors are created
        self.assertEqual(exec_codex, mock_codex_instance)
        self.assertEqual(exec_gemini, mock_gemini_instance)
        self.assertEqual(exec_claude, mock_claude_instance)
        self.assertEqual(exec_default, mock_codex_instance)  # Should use default

    @patch('um_agent_coder.harness.main.CodexExecutor')
    @patch('um_agent_coder.harness.main.StateManager')
    @patch('um_agent_coder.harness.main.GrowthLoop')
    def test_model_override(self, mock_growth, mock_state, mock_codex):
        """Test per-task model override is respected."""
        mock_codex_instance = MagicMock()
        mock_codex.return_value = mock_codex_instance

        harness = Harness(
            roadmap_path=self.roadmap_path,
            cli="codex",
            dry_run=True
        )

        # Create task with model override
        task = Task(
            id="task-001",
            description="Test",
            phase="Test",
            cli="codex",
            model="gpt-5.3"  # Custom model
        )

        # Get executor for task
        executor = harness._get_executor_for_task(task)

        # Verify CodexExecutor was called with custom model
        # (In the second call, as the first is for default executor)
        self.assertEqual(mock_codex.call_count, 2)


class TestDryRunMode(unittest.TestCase):
    """Test dry-run mode functionality."""

    def setUp(self):
        """Set up temporary roadmap file."""
        self.temp_dir = tempfile.mkdtemp()
        self.roadmap_path = os.path.join(self.temp_dir, "roadmap.md")
        with open(self.roadmap_path, 'w') as f:
            f.write(SAMPLE_ROADMAP)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('um_agent_coder.harness.main.StateManager')
    @patch('um_agent_coder.harness.main.CodexExecutor')
    @patch('um_agent_coder.harness.main.GrowthLoop')
    def test_dry_run_marks_complete_without_execution(self, mock_growth, mock_codex, mock_state_cls):
        """Test that dry-run mode marks tasks complete without executing."""
        # Mock executor
        mock_executor = MagicMock()
        mock_codex.return_value = mock_executor

        # Use a real StateManager with temp db
        temp_harness_dir = os.path.join(self.temp_dir, ".harness")
        os.makedirs(temp_harness_dir, exist_ok=True)
        real_state = StateManager(os.path.join(temp_harness_dir, "state.db"))
        mock_state_cls.return_value = real_state

        # Create harness in dry-run mode
        harness = Harness(
            roadmap_path=self.roadmap_path,
            cli="codex",
            dry_run=True
        )

        # Initialize harness
        harness._initialize()

        # Find a pending task
        pending_tasks = [t for t in harness.roadmap.all_tasks if t.status == TaskStatus.PENDING]
        self.assertGreater(len(pending_tasks), 0, "Should have pending tasks")

        # Get the first pending task that can execute
        task = harness._get_next_task()
        self.assertIsNotNone(task, "Should have a task ready to execute")

        # Execute task iteration (should not actually execute)
        harness._task_iteration()

        # Verify task was marked complete
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(task.completed_at)

        # Verify executor was NOT called
        mock_executor.execute.assert_not_called()

    @patch('um_agent_coder.harness.main.StateManager')
    @patch('um_agent_coder.harness.main.CodexExecutor')
    @patch('um_agent_coder.harness.main.GrowthLoop')
    def test_dry_run_processes_all_tasks(self, mock_growth, mock_codex, mock_state_cls):
        """Test that dry-run mode processes all tasks in order."""
        mock_executor = MagicMock()
        mock_codex.return_value = mock_executor

        # Use a real StateManager with temp db
        temp_harness_dir = os.path.join(self.temp_dir, ".harness")
        os.makedirs(temp_harness_dir, exist_ok=True)
        real_state = StateManager(os.path.join(temp_harness_dir, "state2.db"))
        mock_state_cls.return_value = real_state

        harness = Harness(
            roadmap_path=self.roadmap_path,
            cli="codex",
            dry_run=True
        )

        harness._initialize()

        # Count initially pending tasks
        pending_count = sum(1 for task in harness.roadmap.all_tasks
                           if task.status == TaskStatus.PENDING)

        # Run iterations until no more tasks
        iterations = 0
        max_iterations = pending_count + 5  # Safety limit

        while iterations < max_iterations:
            task = harness._get_next_task()
            if not task:
                break
            harness._task_iteration()
            iterations += 1

        # Verify all tasks are now complete
        completed_count = sum(1 for task in harness.roadmap.all_tasks
                             if task.status == TaskStatus.COMPLETED)

        self.assertEqual(completed_count, len(harness.roadmap.all_tasks))

        # Verify executor was never called
        mock_executor.execute.assert_not_called()

    @patch('um_agent_coder.harness.main.StateManager')
    @patch('um_agent_coder.harness.main.CodexExecutor')
    @patch('um_agent_coder.harness.main.GrowthLoop')
    def test_dry_run_respects_dependencies(self, mock_growth, mock_codex, mock_state_cls):
        """Test that dry-run mode respects task dependencies."""
        mock_executor = MagicMock()
        mock_codex.return_value = mock_executor

        # Use a real StateManager with temp db
        temp_harness_dir = os.path.join(self.temp_dir, ".harness")
        os.makedirs(temp_harness_dir, exist_ok=True)
        real_state = StateManager(os.path.join(temp_harness_dir, "state3.db"))
        mock_state_cls.return_value = real_state

        harness = Harness(
            roadmap_path=self.roadmap_path,
            cli="codex",
            dry_run=True
        )

        harness._initialize()

        # Track execution order
        execution_order = []

        # Run iterations and track order
        max_iterations = 10
        for _ in range(max_iterations):
            task = harness._get_next_task()
            if not task:
                break
            execution_order.append(task.id)
            harness._task_iteration()

        # Verify dependencies are satisfied in execution order
        # task-003 depends on task-001 and task-002
        # task-004 depends on task-003
        # task-005 depends on task-004

        if "task-003" in execution_order:
            task3_index = execution_order.index("task-003")
            task1_index = execution_order.index("task-001")

            # task-001 must come before task-003
            self.assertLess(task1_index, task3_index)

        if "task-004" in execution_order and "task-003" in execution_order:
            task4_index = execution_order.index("task-004")
            task3_index = execution_order.index("task-003")

            # task-003 must come before task-004
            self.assertLess(task3_index, task4_index)


class TestHarnessIntegration(unittest.TestCase):
    """Integration tests for the harness."""

    def setUp(self):
        """Set up temporary files."""
        self.temp_dir = tempfile.mkdtemp()
        self.roadmap_path = os.path.join(self.temp_dir, "roadmap.md")
        with open(self.roadmap_path, 'w') as f:
            f.write(SAMPLE_ROADMAP)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('um_agent_coder.harness.main.CodexExecutor')
    @patch('um_agent_coder.harness.main.GrowthLoop')
    def test_harness_initialization(self, mock_growth, mock_codex):
        """Test harness initialization."""
        mock_executor = MagicMock()
        mock_codex.return_value = mock_executor

        harness = Harness(
            roadmap_path=self.roadmap_path,
            cli="codex",
            dry_run=True
        )

        harness._initialize()

        # Verify roadmap loaded
        self.assertIsNotNone(harness.roadmap)
        self.assertEqual(harness.roadmap.name, "Test Project")

        # Verify state initialized
        self.assertIsNotNone(harness.harness_state)

        # Verify tasks loaded
        self.assertEqual(len(harness.roadmap.all_tasks), 5)

    @patch('um_agent_coder.harness.main.CodexExecutor')
    @patch('um_agent_coder.harness.main.GrowthLoop')
    def test_build_task_context(self, mock_growth, mock_codex):
        """Test building task context from dependencies."""
        mock_executor = MagicMock()
        mock_codex.return_value = mock_executor

        harness = Harness(
            roadmap_path=self.roadmap_path,
            cli="codex",
            dry_run=True
        )

        harness._initialize()

        # Get task-003 which depends on task-001 and task-002
        task3 = harness.roadmap.get_task("task-003")

        # Mark dependencies as complete with output
        task1 = harness.roadmap.get_task("task-001")
        task1.output = "Project initialized successfully"
        task1.status = TaskStatus.COMPLETED

        task2 = harness.roadmap.get_task("task-002")
        task2.output = "Task 2 completed"
        task2.status = TaskStatus.COMPLETED

        # Build context
        context = harness._build_task_context(task3)

        # Verify context includes project info and dependencies
        self.assertIn("Test Project", context)
        self.assertIn("Test objective for unit tests", context)
        self.assertIn("task-001", context)
        self.assertIn("task-002", context)
        self.assertIn("Project initialized successfully", context)


if __name__ == "__main__":
    unittest.main()
