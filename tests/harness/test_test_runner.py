"""
Tests for TestRunner QA validation.

Tests:
1. Test result parsing from JSON report
2. Fallback parsing from stdout
3. Test failure prompt generation
4. Test path validation
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from um_agent_coder.harness.test_runner import TestResult, TestRunner


class TestTestResult(unittest.TestCase):
    """Tests for TestResult dataclass."""

    def test_success_result(self):
        """Test successful test result."""
        result = TestResult(
            success=True,
            total_tests=10,
            passed=10,
            failed=0,
            skipped=0,
        )
        self.assertTrue(result.success)
        self.assertIn("PASSED", result.summary)
        self.assertIn("10/10", result.summary)

    def test_failure_result(self):
        """Test failed test result."""
        result = TestResult(
            success=False,
            total_tests=10,
            passed=7,
            failed=3,
            skipped=0,
            failed_tests=["test_a", "test_b", "test_c"],
        )
        self.assertFalse(result.success)
        self.assertIn("FAILED", result.summary)
        self.assertEqual(len(result.failed_tests), 3)

    def test_empty_result(self):
        """Test result with no tests."""
        result = TestResult(success=False, total_tests=0)
        self.assertFalse(result.success)


class TestTestRunnerParsing(unittest.TestCase):
    """Tests for TestRunner output parsing."""

    def setUp(self):
        self.runner = TestRunner()

    def test_parse_stdout_success(self):
        """Test parsing success from stdout."""
        output = """
        ============================= test session starts ==============================
        collected 5 items

        test_example.py::test_one PASSED
        test_example.py::test_two PASSED

        ============================== 5 passed in 0.12s ===============================
        """
        result = self.runner._parse_stdout_fallback(output, returncode=0)
        self.assertTrue(result.success)
        self.assertEqual(result.passed, 5)

    def test_parse_stdout_failure(self):
        """Test parsing failure from stdout."""
        output = """
        ============================= test session starts ==============================
        collected 5 items

        test_example.py::test_one PASSED
        test_example.py::test_two FAILED

        ============================== 3 passed, 2 failed in 0.15s ===============================
        """
        result = self.runner._parse_stdout_fallback(output, returncode=1)
        self.assertFalse(result.success)
        self.assertEqual(result.passed, 3)
        self.assertEqual(result.failed, 2)

    def test_parse_stdout_with_skipped(self):
        """Test parsing with skipped tests."""
        output = """
        ============================== 8 passed, 1 failed, 3 skipped in 1.23s ===============================
        """
        result = self.runner._parse_stdout_fallback(output, returncode=1)
        self.assertFalse(result.success)
        self.assertEqual(result.passed, 8)
        self.assertEqual(result.failed, 1)
        self.assertEqual(result.skipped, 3)
        self.assertAlmostEqual(result.duration_seconds, 1.23, places=2)


class TestTestRunnerPromptGeneration(unittest.TestCase):
    """Tests for failure prompt generation."""

    def setUp(self):
        self.runner = TestRunner()

    def test_format_failure_prompt_basic(self):
        """Test basic failure prompt format."""
        result = TestResult(
            success=False,
            total_tests=10,
            passed=7,
            failed=3,
            error_summary="AssertionError in test_foo",
            failed_tests=["test_foo", "test_bar"],
        )
        prompt = self.runner.format_failure_prompt(result)

        self.assertIn("Test Failures Detected", prompt)
        self.assertIn("3", prompt)  # Failed count
        self.assertIn("test_foo", prompt)
        self.assertIn("AssertionError", prompt)
        self.assertIn("DO NOT", prompt)  # Warning about promise

    def test_format_failure_prompt_many_failures(self):
        """Test prompt with many failures (truncated)."""
        failed_tests = [f"test_{i}" for i in range(15)]
        result = TestResult(
            success=False,
            total_tests=20,
            passed=5,
            failed=15,
            failed_tests=failed_tests,
        )
        prompt = self.runner.format_failure_prompt(result)

        # Should truncate to 10 and show "... and X more"
        self.assertIn("and 5 more", prompt)


class TestTestRunnerExecution(unittest.TestCase):
    """Tests for actual test execution."""

    def setUp(self):
        self.runner = TestRunner(timeout_seconds=10)
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_check_tests_exist_file(self):
        """Test checking if test file exists."""
        # Create a test file
        test_file = Path(self.test_dir) / "test_example.py"
        test_file.write_text("def test_example(): pass")

        self.assertTrue(self.runner.check_tests_exist("test_example.py", self.test_dir))
        self.assertFalse(self.runner.check_tests_exist("test_nonexistent.py", self.test_dir))

    def test_check_tests_exist_directory(self):
        """Test checking if test directory exists."""
        # Create a tests directory with test files
        tests_dir = Path(self.test_dir) / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_one.py").write_text("def test_one(): pass")

        self.assertTrue(self.runner.check_tests_exist("tests", self.test_dir))

    @patch("subprocess.run")
    def test_run_tests_timeout(self, mock_run):
        """Test handling of test timeout."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("pytest", 10)

        result = self.runner.run_tests(test_path="tests", cwd=self.test_dir)

        self.assertFalse(result.success)
        self.assertIn("timed out", result.error_summary.lower())


class TestTestRunnerIntegration(unittest.TestCase):
    """Integration tests for TestRunner."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.runner = TestRunner(timeout_seconds=30)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_run_passing_tests(self):
        """Test running tests that pass."""
        # Create a passing test file
        tests_dir = Path(self.test_dir) / "tests"
        tests_dir.mkdir()
        (tests_dir / "__init__.py").touch()
        (tests_dir / "test_pass.py").write_text("""
def test_passes():
    assert True

def test_also_passes():
    assert 1 + 1 == 2
""")

        result = self.runner.run_tests(test_path="tests", cwd=self.test_dir)

        # May fail if pytest not installed, but structure should be valid
        self.assertIsInstance(result, TestResult)
        self.assertIsInstance(result.success, bool)

    def test_run_failing_tests(self):
        """Test running tests that fail."""
        # Create a failing test file
        tests_dir = Path(self.test_dir) / "tests"
        tests_dir.mkdir()
        (tests_dir / "__init__.py").touch()
        (tests_dir / "test_fail.py").write_text("""
def test_fails():
    assert False, "This test fails"

def test_passes():
    assert True
""")

        result = self.runner.run_tests(test_path="tests", cwd=self.test_dir)

        self.assertIsInstance(result, TestResult)
        # If pytest available, should have failed
        if result.total_tests > 0:
            self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
