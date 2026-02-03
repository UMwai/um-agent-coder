"""
Tests for Ralph executor QA validation integration.

Tests:
1. QA validation enabled behavior
2. Tests fail - promise rejected
3. Tests pass - promise accepted
4. Test failure prompt generation
5. Iteration tracking with test info
"""

import os
import shutil
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from um_agent_coder.harness.models import ExecutionResult, RalphConfig, Task
from um_agent_coder.harness.ralph.executor import RalphExecutor, RalphResult
from um_agent_coder.harness.ralph.persistence import RalphPersistence
from um_agent_coder.harness.test_runner import TestResult, TestRunner


class MockExecutor:
    """Mock executor for testing."""

    def __init__(self, responses: list):
        """Initialize with list of responses to return in sequence."""
        self.responses = responses
        self.call_count = 0
        self.prompts_received = []

    def execute(self, task: Task, context: str = "") -> ExecutionResult:
        """Return the next response in sequence."""
        self.prompts_received.append(context)

        if self.call_count >= len(self.responses):
            return ExecutionResult(success=True, output="No more responses")

        resp = self.responses[self.call_count]
        self.call_count += 1

        if isinstance(resp, tuple):
            success, output, error = resp
            return ExecutionResult(success=success, output=output, error=error)
        else:
            return ExecutionResult(success=True, output=resp)


class MockTestRunner:
    """Mock test runner for testing."""

    def __init__(self, results: list):
        """Initialize with list of TestResults to return."""
        self.results = results
        self.call_count = 0

    def run_tests(self, test_path: str = "tests", cwd: str = "./") -> TestResult:
        """Return the next result in sequence."""
        if self.call_count >= len(self.results):
            return TestResult(success=True, total_tests=1, passed=1)

        result = self.results[self.call_count]
        self.call_count += 1
        return result

    def format_failure_prompt(self, result: TestResult, include_full_output: bool = False) -> str:
        """Return a mock failure prompt."""
        return f"## Tests Failed\n\n{result.error_summary}"


class TestRalphQAValidationEnabled(unittest.TestCase):
    """Tests for QA validation when enabled."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.persistence = RalphPersistence(
            db_path=os.path.join(self.temp_dir, "ralph_state.db")
        )
        self.task = Task(
            id="test-qa-001",
            description="Test task with QA",
            phase="Test",
            cwd=self.temp_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_promise_accepted_when_tests_pass(self):
        """Test that promise is accepted when tests pass."""
        # Mock executor returns promise on first try
        mock_executor = MockExecutor([
            "<promise>COMPLETE</promise>"
        ])

        # Mock test runner that returns success
        with patch.object(TestRunner, 'run_tests') as mock_run:
            mock_run.return_value = TestResult(
                success=True,
                total_tests=5,
                passed=5,
                failed=0,
            )

            executor = RalphExecutor(
                base_executor=mock_executor,
                max_iterations=10,
                completion_promise="COMPLETE",
                persistence=self.persistence,
                require_tests_passing=True,
                test_path="tests",
            )

            result = executor.execute(self.task, resume=False)

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "promise_found")
        self.assertEqual(result.iterations, 1)

    def test_promise_rejected_when_tests_fail(self):
        """Test that promise is rejected when tests fail."""
        # Mock executor returns promise on iterations 1 and 2
        mock_executor = MockExecutor([
            "<promise>COMPLETE</promise>",  # First try - tests will fail
            "<promise>COMPLETE</promise>",  # Second try - tests will pass
        ])

        test_results = [
            TestResult(success=False, total_tests=5, passed=3, failed=2, error_summary="2 tests failed"),
            TestResult(success=True, total_tests=5, passed=5, failed=0),
        ]
        result_index = [0]

        def mock_run_tests(*args, **kwargs):
            idx = result_index[0]
            result_index[0] += 1
            return test_results[idx] if idx < len(test_results) else test_results[-1]

        with patch.object(TestRunner, 'run_tests', side_effect=mock_run_tests):
            executor = RalphExecutor(
                base_executor=mock_executor,
                max_iterations=10,
                completion_promise="COMPLETE",
                persistence=self.persistence,
                require_tests_passing=True,
                test_path="tests",
            )

            result = executor.execute(self.task, resume=False)

        self.assertTrue(result.success)
        self.assertEqual(result.iterations, 2)  # Should take 2 iterations
        # Second prompt should contain test failure info
        self.assertIn("Test", mock_executor.prompts_received[1])

    def test_qa_from_ralph_config(self):
        """Test QA settings from task's RalphConfig."""
        self.task.ralph_config = RalphConfig(
            enabled=True,
            max_iterations=30,
            completion_promise="DONE",
            require_tests_passing=True,
            test_path="tests/unit",
        )

        mock_executor = MockExecutor(["<promise>DONE</promise>"])

        with patch.object(TestRunner, 'run_tests') as mock_run:
            mock_run.return_value = TestResult(success=True, total_tests=3, passed=3)

            executor = RalphExecutor(
                base_executor=mock_executor,
                max_iterations=30,
                completion_promise="DONE",
                persistence=self.persistence,
                require_tests_passing=False,  # Disabled at executor level
            )

            result = executor.execute(self.task, resume=False)

        # QA should be enabled from task config
        self.assertTrue(result.success)
        # Test runner should have been called with task's test path
        mock_run.assert_called_once()


class TestRalphQAIterationTracking(unittest.TestCase):
    """Tests for iteration tracking with QA info."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.persistence = RalphPersistence(
            db_path=os.path.join(self.temp_dir, "ralph_state.db")
        )
        self.task = Task(
            id="test-qa-track-001",
            description="Test tracking",
            phase="Test",
            cwd=self.temp_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_iteration_records_test_info(self):
        """Test that iteration records include test info."""
        mock_executor = MockExecutor([
            "<promise>COMPLETE</promise>",  # Will trigger test run
        ])

        test_result = TestResult(
            success=True,
            total_tests=10,
            passed=10,
            failed=0,
        )

        # Create executor first
        executor = RalphExecutor(
            base_executor=mock_executor,
            max_iterations=10,
            completion_promise="COMPLETE",
            persistence=self.persistence,
            require_tests_passing=True,
        )

        # Now mock the test_runner instance that was created
        executor.test_runner = MagicMock()
        executor.test_runner.run_tests.return_value = test_result

        result = executor.execute(self.task, resume=False)

        # Get tracker and check iteration records
        tracker = executor.get_tracker(self.task.id)
        self.assertIsNotNone(tracker)
        self.assertEqual(len(tracker.iteration_history), 1)

        record = tracker.iteration_history[0]
        # test_passed is set to True when tests pass
        self.assertEqual(record.test_passed, True)
        # test_summary should contain the TestResult.summary
        self.assertIn("PASSED", record.test_summary)


class TestRalphQADisabled(unittest.TestCase):
    """Tests for behavior when QA validation is disabled."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.persistence = RalphPersistence(
            db_path=os.path.join(self.temp_dir, "ralph_state.db")
        )
        self.task = Task(
            id="test-no-qa-001",
            description="Test without QA",
            phase="Test",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_promise_accepted_without_tests(self):
        """Test that promise is accepted without running tests when disabled."""
        mock_executor = MockExecutor(["<promise>COMPLETE</promise>"])

        with patch.object(TestRunner, 'run_tests') as mock_run:
            executor = RalphExecutor(
                base_executor=mock_executor,
                max_iterations=10,
                completion_promise="COMPLETE",
                persistence=self.persistence,
                require_tests_passing=False,  # Disabled
            )

            result = executor.execute(self.task, resume=False)

        self.assertTrue(result.success)
        mock_run.assert_not_called()  # Tests should not run


class TestRalphQATestFailurePrompt(unittest.TestCase):
    """Tests for test failure prompt generation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.persistence = RalphPersistence(
            db_path=os.path.join(self.temp_dir, "ralph_state.db")
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_failure_prompt_contains_task_info(self):
        """Test that failure prompt includes original task info."""
        task = Task(
            id="test-prompt-001",
            description="Implement feature X",
            phase="Test",
            success_criteria="All tests pass",
        )

        mock_executor = MockExecutor([
            "<promise>COMPLETE</promise>",  # First - tests fail
            "Working on fixes...",  # Second - no promise
        ])

        test_results = [
            TestResult(success=False, total_tests=5, failed=2, error_summary="Error details"),
            TestResult(success=True, total_tests=5, passed=5),
        ]
        call_idx = [0]

        def mock_run(*args, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            return test_results[min(idx, len(test_results) - 1)]

        with patch.object(TestRunner, 'run_tests', side_effect=mock_run):
            executor = RalphExecutor(
                base_executor=mock_executor,
                max_iterations=3,
                completion_promise="COMPLETE",
                persistence=self.persistence,
                require_tests_passing=True,
            )

            # Don't complete - we just want to check the prompt
            result = executor.execute(task, resume=False)

        # Check that the second prompt contains task info
        self.assertTrue(len(mock_executor.prompts_received) >= 2)
        second_prompt = mock_executor.prompts_received[1]
        self.assertIn("Test Validation Failed", second_prompt)
        self.assertIn("test-prompt-001", second_prompt)
        self.assertIn("Implement feature X", second_prompt)


if __name__ == "__main__":
    unittest.main()
