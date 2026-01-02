"""
Integration tests for RalphExecutor.

Tests:
1. Multi-iteration completion
2. Early completion (iteration 1)
3. Late completion (iteration N)
4. Max iterations exceeded
5. Error handling
6. Resumption from interruption
"""

import os
import shutil
import sys
import tempfile
import unittest
from datetime import timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from um_agent_coder.harness.models import ExecutionResult, Task
from um_agent_coder.harness.ralph.executor import RalphExecutor, RalphResult
from um_agent_coder.harness.ralph.persistence import RalphPersistence


class MockExecutor:
    """Mock executor for testing."""

    def __init__(self, responses: list):
        """Initialize with list of responses to return in sequence.

        Args:
            responses: List of (success, output, error) tuples
        """
        self.responses = responses
        self.call_count = 0
        self.prompts_received = []

    def execute(self, task: Task, context: str = "") -> ExecutionResult:
        """Return the next response in sequence."""
        self.prompts_received.append(context)

        if self.call_count >= len(self.responses):
            # Default response if we run out
            return ExecutionResult(success=True, output="No more responses")

        resp = self.responses[self.call_count]
        self.call_count += 1

        if isinstance(resp, tuple):
            success, output, error = resp
            return ExecutionResult(success=success, output=output, error=error)
        else:
            return ExecutionResult(success=True, output=resp)


class TestRalphExecutorBasic(unittest.TestCase):
    """Basic tests for RalphExecutor."""

    def setUp(self):
        """Set up temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "ralph.db")
        self.persistence = RalphPersistence(self.db_path)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_early_completion_iteration_1(self):
        """Test completion on first iteration."""
        mock_exec = MockExecutor([
            "Task done! <promise>COMPLETE</promise>"
        ])

        ralph = RalphExecutor(
            base_executor=mock_exec,
            max_iterations=10,
            completion_promise="COMPLETE",
            persistence=self.persistence,
        )

        task = Task(id="early-001", description="Test task", phase="Test")
        result = ralph.execute(task)

        self.assertTrue(result.success)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.reason, "promise_found")
        self.assertEqual(result.promise_text, "COMPLETE")

    def test_late_completion(self):
        """Test completion after multiple iterations."""
        mock_exec = MockExecutor([
            "Working on it...",
            "Still working...",
            "Almost there...",
            "Done! <promise>COMPLETE</promise>",
        ])

        ralph = RalphExecutor(
            base_executor=mock_exec,
            max_iterations=10,
            completion_promise="COMPLETE",
            persistence=self.persistence,
        )

        task = Task(id="late-001", description="Test task", phase="Test")
        result = ralph.execute(task)

        self.assertTrue(result.success)
        self.assertEqual(result.iterations, 4)
        self.assertEqual(result.reason, "promise_found")

    def test_max_iterations_exceeded(self):
        """Test failure when max iterations exceeded."""
        mock_exec = MockExecutor([
            "Working..." for _ in range(10)
        ])

        ralph = RalphExecutor(
            base_executor=mock_exec,
            max_iterations=5,
            completion_promise="COMPLETE",
            persistence=self.persistence,
        )

        task = Task(id="max-001", description="Test task", phase="Test")
        result = ralph.execute(task)

        self.assertFalse(result.success)
        self.assertEqual(result.iterations, 5)
        self.assertEqual(result.reason, "max_iterations_exceeded")

    def test_promise_case_insensitive(self):
        """Test that promise detection is case-insensitive."""
        mock_exec = MockExecutor([
            "Done! <PROMISE>complete</PROMISE>"
        ])

        ralph = RalphExecutor(
            base_executor=mock_exec,
            max_iterations=5,
            completion_promise="COMPLETE",
            persistence=self.persistence,
        )

        task = Task(id="case-001", description="Test task", phase="Test")
        result = ralph.execute(task)

        self.assertTrue(result.success)

    def test_custom_promise_text(self):
        """Test with custom promise text."""
        mock_exec = MockExecutor([
            "<promise>RALPH_LOOP_IMPLEMENTATION_COMPLETE</promise>"
        ])

        ralph = RalphExecutor(
            base_executor=mock_exec,
            max_iterations=5,
            completion_promise="RALPH_LOOP_IMPLEMENTATION_COMPLETE",
            persistence=self.persistence,
        )

        task = Task(id="custom-001", description="Test task", phase="Test")
        result = ralph.execute(task)

        self.assertTrue(result.success)
        self.assertEqual(result.promise_text, "RALPH_LOOP_IMPLEMENTATION_COMPLETE")


class TestRalphExecutorPersistence(unittest.TestCase):
    """Test persistence and resumption."""

    def setUp(self):
        """Set up temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "ralph.db")
        self.persistence = RalphPersistence(self.db_path)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_state_persisted_after_each_iteration(self):
        """Test that state is saved after each iteration."""
        mock_exec = MockExecutor([
            "Iteration 1",
            "Iteration 2",
            "Done! <promise>COMPLETE</promise>",
        ])

        ralph = RalphExecutor(
            base_executor=mock_exec,
            max_iterations=10,
            completion_promise="COMPLETE",
            persistence=self.persistence,
        )

        task = Task(id="persist-001", description="Test task", phase="Test")
        ralph.execute(task)

        # Verify state was saved
        tracker = self.persistence.load_tracker("persist-001")
        self.assertIsNotNone(tracker)
        self.assertEqual(tracker.current_iteration, 3)
        self.assertEqual(len(tracker.iteration_history), 3)

    def test_resume_from_interruption(self):
        """Test resuming after simulated interruption."""
        # First execution - simulate stopping at iteration 2
        mock_exec1 = MockExecutor([
            "Iteration 1",
            "Iteration 2",
        ])

        ralph1 = RalphExecutor(
            base_executor=mock_exec1,
            max_iterations=3,  # Will stop at max
            completion_promise="COMPLETE",
            persistence=self.persistence,
        )

        task = Task(id="resume-001", description="Test task", phase="Test")
        result1 = ralph1.execute(task)

        self.assertFalse(result1.success)
        self.assertEqual(result1.iterations, 3)

        # Reset the tracker to allow continuation (simulating partial run)
        tracker = self.persistence.load_tracker("resume-001")
        # Manually revert to simulate stopping mid-way
        tracker._completed = False
        tracker._completion_reason = None
        tracker.current_iteration = 2
        tracker.iteration_history = tracker.iteration_history[:2]
        self.persistence.save_tracker(tracker)

        # Second execution - should resume and complete
        mock_exec2 = MockExecutor([
            "Done! <promise>COMPLETE</promise>",
        ])

        ralph2 = RalphExecutor(
            base_executor=mock_exec2,
            max_iterations=10,
            completion_promise="COMPLETE",
            persistence=self.persistence,
        )

        result2 = ralph2.execute(task, resume=True)

        self.assertTrue(result2.success)
        self.assertEqual(result2.iterations, 3)  # Continued from 2

    def test_no_resume_creates_fresh_tracker(self):
        """Test that resume=False creates a fresh tracker."""
        # First execution
        mock_exec1 = MockExecutor([
            "Iteration 1",
        ])

        ralph1 = RalphExecutor(
            base_executor=mock_exec1,
            max_iterations=1,
            completion_promise="COMPLETE",
            persistence=self.persistence,
        )

        task = Task(id="noresume-001", description="Test task", phase="Test")
        ralph1.execute(task)

        # Second execution without resume
        mock_exec2 = MockExecutor([
            "Fresh start! <promise>COMPLETE</promise>",
        ])

        ralph2 = RalphExecutor(
            base_executor=mock_exec2,
            max_iterations=10,
            completion_promise="COMPLETE",
            persistence=self.persistence,
        )

        result = ralph2.execute(task, resume=False)

        self.assertTrue(result.success)
        self.assertEqual(result.iterations, 1)  # Fresh start


class TestRalphExecutorErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""

    def setUp(self):
        """Set up temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "ralph.db")
        self.persistence = RalphPersistence(self.db_path)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_executor_error_continues_loop(self):
        """Test that non-fatal executor errors don't stop the loop."""
        mock_exec = MockExecutor([
            (False, "Error occurred", "Some error"),
            (True, "Recovered! <promise>COMPLETE</promise>", ""),
        ])

        ralph = RalphExecutor(
            base_executor=mock_exec,
            max_iterations=5,
            completion_promise="COMPLETE",
            persistence=self.persistence,
        )

        task = Task(id="error-001", description="Test task", phase="Test")
        result = ralph.execute(task)

        self.assertTrue(result.success)
        self.assertEqual(result.iterations, 2)

    def test_iteration_history_records_errors(self):
        """Test that errors are recorded in iteration history."""
        mock_exec = MockExecutor([
            (False, "Failed", "Error message"),
            (True, "<promise>COMPLETE</promise>", ""),
        ])

        ralph = RalphExecutor(
            base_executor=mock_exec,
            max_iterations=5,
            completion_promise="COMPLETE",
            persistence=self.persistence,
        )

        task = Task(id="errorhist-001", description="Test task", phase="Test")
        ralph.execute(task)

        tracker = self.persistence.load_tracker("errorhist-001")
        self.assertEqual(tracker.iteration_history[0].error, "Error message")
        self.assertIsNone(tracker.iteration_history[1].error)


class TestRalphExecutorPromptBuilding(unittest.TestCase):
    """Test prompt building functionality."""

    def setUp(self):
        """Set up temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "ralph.db")
        self.persistence = RalphPersistence(self.db_path)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_prompt_includes_task_info(self):
        """Test that prompt includes task information."""
        mock_exec = MockExecutor([
            "<promise>COMPLETE</promise>"
        ])

        ralph = RalphExecutor(
            base_executor=mock_exec,
            max_iterations=5,
            completion_promise="COMPLETE",
            persistence=self.persistence,
        )

        task = Task(
            id="prompt-001",
            description="Implement feature X",
            phase="Development",
            success_criteria="Tests pass",
        )
        ralph.execute(task)

        # Check the prompt that was sent
        prompt = mock_exec.prompts_received[0]
        self.assertIn("prompt-001", prompt)
        self.assertIn("Implement feature X", prompt)
        self.assertIn("Tests pass", prompt)
        self.assertIn("<promise>COMPLETE</promise>", prompt)

    def test_prompt_includes_context(self):
        """Test that prompt includes provided context."""
        mock_exec = MockExecutor([
            "<promise>COMPLETE</promise>"
        ])

        ralph = RalphExecutor(
            base_executor=mock_exec,
            max_iterations=5,
            completion_promise="COMPLETE",
            persistence=self.persistence,
        )

        task = Task(id="ctx-001", description="Test", phase="Test")
        ralph.execute(task, context="Previous task output here")

        prompt = mock_exec.prompts_received[0]
        self.assertIn("Previous task output here", prompt)


class TestRalphResult(unittest.TestCase):
    """Test RalphResult dataclass."""

    def test_summary_success(self):
        """Test summary for successful result."""
        result = RalphResult(
            success=True,
            iterations=5,
            total_duration=timedelta(seconds=120),
            final_output="Done!",
            reason="promise_found",
        )

        summary = result.summary
        self.assertIn("SUCCESS", summary)
        self.assertIn("5", summary)
        self.assertIn("promise_found", summary)

    def test_summary_failure(self):
        """Test summary for failed result."""
        result = RalphResult(
            success=False,
            iterations=10,
            total_duration=timedelta(seconds=300),
            final_output="Still working...",
            reason="max_iterations_exceeded",
        )

        summary = result.summary
        self.assertIn("FAILED", summary)
        self.assertIn("10", summary)
        self.assertIn("max_iterations_exceeded", summary)


class TestRalphExecutorHelperMethods(unittest.TestCase):
    """Test helper methods."""

    def setUp(self):
        """Set up temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "ralph.db")
        self.persistence = RalphPersistence(self.db_path)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_get_tracker(self):
        """Test get_tracker method."""
        mock_exec = MockExecutor([
            "Working...",
        ])

        ralph = RalphExecutor(
            base_executor=mock_exec,
            max_iterations=1,
            completion_promise="COMPLETE",
            persistence=self.persistence,
        )

        task = Task(id="gettrack-001", description="Test", phase="Test")
        ralph.execute(task)

        tracker = ralph.get_tracker("gettrack-001")
        self.assertIsNotNone(tracker)
        self.assertEqual(tracker.task_id, "gettrack-001")

    def test_reset_tracker(self):
        """Test reset_tracker method."""
        mock_exec = MockExecutor([
            "Working...",
        ])

        ralph = RalphExecutor(
            base_executor=mock_exec,
            max_iterations=1,
            completion_promise="COMPLETE",
            persistence=self.persistence,
        )

        task = Task(id="reset-001", description="Test", phase="Test")
        ralph.execute(task)

        # Verify it exists
        self.assertIsNotNone(ralph.get_tracker("reset-001"))

        # Reset it
        result = ralph.reset_tracker("reset-001")
        self.assertTrue(result)

        # Verify it's gone
        self.assertIsNone(ralph.get_tracker("reset-001"))


if __name__ == "__main__":
    unittest.main()
