"""
Test runner for QA validation loops.

Runs pytest and parses results for integration with the ralph loop.
When require_tests_passing is enabled, tests must pass before
the completion promise is accepted.
"""

import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result from running pytest."""

    success: bool
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    error_summary: str = ""
    full_output: str = ""
    failed_tests: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def summary(self) -> str:
        """Brief summary for logging."""
        status = "PASSED" if self.success else "FAILED"
        return (
            f"[{status}] {self.passed}/{self.total_tests} passed, "
            f"{self.failed} failed, {self.skipped} skipped"
        )


class TestRunner:
    """Run pytest and parse results for QA validation.

    Integrates with the ralph loop to ensure tests pass before
    accepting a completion promise.

    Example:
        runner = TestRunner()
        result = runner.run_tests("tests/api")
        if not result.success:
            prompt = runner.format_failure_prompt(result)
    """

    def __init__(
        self,
        pytest_args: Optional[List[str]] = None,
        timeout_seconds: int = 300,
    ):
        """Initialize the test runner.

        Args:
            pytest_args: Additional arguments to pass to pytest
            timeout_seconds: Maximum time for test execution
        """
        self.pytest_args = pytest_args or []
        self.timeout_seconds = timeout_seconds

    def run_tests(
        self,
        test_path: str = "tests",
        cwd: str = "./",
        markers: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
    ) -> TestResult:
        """Run pytest and return parsed results.

        Args:
            test_path: Path to test files/directory
            cwd: Working directory for test execution
            markers: Pytest marker expression (e.g., "not slow")
            extra_args: Additional pytest arguments

        Returns:
            TestResult with parsed test outcomes
        """
        # Create temp file for JSON report
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as report_file:
            report_path = report_file.name

        try:
            # Build pytest command
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                test_path,
                f"--json-report-file={report_path}",
                "--json-report",
                "-v",
                "--tb=short",
            ]

            # Add marker filter if specified
            if markers:
                cmd.extend(["-m", markers])

            # Add configured args
            cmd.extend(self.pytest_args)

            # Add extra args
            if extra_args:
                cmd.extend(extra_args)

            logger.info(f"Running tests: {' '.join(cmd)}")

            # Run pytest
            process = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )

            full_output = process.stdout + "\n" + process.stderr

            # Parse JSON report
            return self._parse_json_report(report_path, full_output)

        except subprocess.TimeoutExpired:
            logger.error(f"Test execution timed out after {self.timeout_seconds}s")
            return TestResult(
                success=False,
                error_summary=f"Test execution timed out after {self.timeout_seconds} seconds",
                full_output="TIMEOUT",
            )

        except FileNotFoundError:
            # pytest-json-report not installed, fall back to basic parsing
            logger.warning("pytest-json-report not available, using basic parsing")
            return self._run_basic_tests(test_path, cwd, markers, extra_args)

        except Exception as e:
            logger.exception(f"Error running tests: {e}")
            return TestResult(
                success=False,
                error_summary=str(e),
                full_output=str(e),
            )

        finally:
            # Clean up temp file
            try:
                Path(report_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _parse_json_report(self, report_path: str, full_output: str) -> TestResult:
        """Parse pytest-json-report output.

        Args:
            report_path: Path to JSON report file
            full_output: Full stdout/stderr from pytest

        Returns:
            Parsed TestResult
        """
        try:
            with open(report_path) as f:
                report = json.load(f)

            summary = report.get("summary", {})
            tests = report.get("tests", [])

            total = summary.get("total", 0)
            passed = summary.get("passed", 0)
            failed = summary.get("failed", 0)
            skipped = summary.get("skipped", 0)
            duration = report.get("duration", 0.0)

            # Extract failed test names and messages
            failed_tests = []
            error_parts = []

            for test in tests:
                if test.get("outcome") == "failed":
                    test_name = test.get("nodeid", "unknown")
                    failed_tests.append(test_name)

                    # Get failure message
                    call_info = test.get("call", {})
                    longrepr = call_info.get("longrepr", "")
                    if longrepr:
                        error_parts.append(f"{test_name}:\n{longrepr[:500]}")

            error_summary = "\n\n".join(error_parts) if error_parts else ""

            return TestResult(
                success=(failed == 0 and total > 0),
                total_tests=total,
                passed=passed,
                failed=failed,
                skipped=skipped,
                error_summary=error_summary,
                full_output=full_output,
                failed_tests=failed_tests,
                duration_seconds=duration,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse JSON report: {e}")
            return self._parse_stdout_fallback(full_output)

        except FileNotFoundError:
            logger.warning("JSON report file not found, using stdout parsing")
            return self._parse_stdout_fallback(full_output)

    def _run_basic_tests(
        self,
        test_path: str,
        cwd: str,
        markers: Optional[str],
        extra_args: Optional[List[str]],
    ) -> TestResult:
        """Run pytest without JSON report (fallback).

        Args:
            test_path: Path to test files/directory
            cwd: Working directory
            markers: Pytest marker expression
            extra_args: Additional arguments

        Returns:
            TestResult parsed from stdout
        """
        cmd = [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"]

        if markers:
            cmd.extend(["-m", markers])

        cmd.extend(self.pytest_args)

        if extra_args:
            cmd.extend(extra_args)

        try:
            process = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )

            full_output = process.stdout + "\n" + process.stderr
            return self._parse_stdout_fallback(full_output, process.returncode)

        except subprocess.TimeoutExpired:
            return TestResult(
                success=False,
                error_summary=f"Test execution timed out after {self.timeout_seconds} seconds",
                full_output="TIMEOUT",
            )

    def _parse_stdout_fallback(
        self,
        output: str,
        returncode: int = 1,
    ) -> TestResult:
        """Parse test results from pytest stdout when JSON is unavailable.

        Args:
            output: Pytest stdout/stderr
            returncode: Process return code

        Returns:
            Parsed TestResult (best effort)
        """
        import re

        # Parse individual result counts from anywhere in the output
        # Look for patterns like "5 passed", "2 failed", "1 skipped"
        passed = 0
        failed = 0
        skipped = 0
        duration = 0.0

        # Extract counts from patterns like "5 passed", "2 failed", etc.
        passed_match = re.search(r"(\d+)\s+passed", output, re.IGNORECASE)
        failed_match = re.search(r"(\d+)\s+failed", output, re.IGNORECASE)
        skipped_match = re.search(r"(\d+)\s+skipped", output, re.IGNORECASE)
        duration_match = re.search(r"in\s+([\d.]+)s", output, re.IGNORECASE)

        if passed_match:
            passed = int(passed_match.group(1))
        if failed_match:
            failed = int(failed_match.group(1))
        if skipped_match:
            skipped = int(skipped_match.group(1))
        if duration_match:
            duration = float(duration_match.group(1))

        total = passed + failed + skipped

        if total > 0:

            # Extract failed test names
            failed_tests = []
            failed_pattern = r"FAILED\s+(\S+)"
            for m in re.finditer(failed_pattern, output):
                failed_tests.append(m.group(1))

            # Extract error summary (short traceback)
            error_lines = []
            in_failure = False
            for line in output.split("\n"):
                if line.startswith("FAILED") or line.startswith("E "):
                    in_failure = True
                if in_failure and line.strip():
                    error_lines.append(line)
                    if len(error_lines) > 20:  # Limit error summary
                        break

            return TestResult(
                success=(failed == 0 and total > 0),
                total_tests=total,
                passed=passed,
                failed=failed,
                skipped=skipped,
                error_summary="\n".join(error_lines),
                full_output=output,
                failed_tests=failed_tests,
                duration_seconds=duration,
            )

        # Couldn't parse - use return code
        return TestResult(
            success=(returncode == 0),
            error_summary="Could not parse test output",
            full_output=output,
        )

    def format_failure_prompt(
        self,
        result: TestResult,
        include_full_output: bool = False,
    ) -> str:
        """Format a prompt describing test failures for the ralph loop.

        Args:
            result: TestResult from run_tests
            include_full_output: Whether to include full pytest output

        Returns:
            Formatted prompt string for the next iteration
        """
        parts = [
            "## Test Failures Detected",
            "",
            f"Tests ran but **{result.failed}** test(s) failed.",
            "",
            "### Summary",
            f"- Total: {result.total_tests}",
            f"- Passed: {result.passed}",
            f"- Failed: {result.failed}",
            f"- Skipped: {result.skipped}",
            "",
        ]

        if result.failed_tests:
            parts.extend(
                [
                    "### Failed Tests",
                    "",
                ]
            )
            for test in result.failed_tests[:10]:  # Limit to 10
                parts.append(f"- `{test}`")
            if len(result.failed_tests) > 10:
                parts.append(f"- ... and {len(result.failed_tests) - 10} more")
            parts.append("")

        if result.error_summary:
            parts.extend(
                [
                    "### Error Details",
                    "```",
                    result.error_summary[:2000],  # Limit size
                    "```",
                    "",
                ]
            )

        parts.extend(
            [
                "### Next Steps",
                "",
                "Please fix the failing tests before outputting the completion promise.",
                "Review the errors above and make the necessary code changes.",
                "",
                "**DO NOT** output the completion promise until all tests pass.",
            ]
        )

        if include_full_output and result.full_output:
            parts.extend(
                [
                    "",
                    "### Full Test Output",
                    "```",
                    result.full_output[:5000],
                    "```",
                ]
            )

        return "\n".join(parts)

    def check_tests_exist(self, test_path: str, cwd: str = "./") -> bool:
        """Check if test files exist at the specified path.

        Args:
            test_path: Path to check for tests
            cwd: Working directory

        Returns:
            True if tests exist
        """
        full_path = Path(cwd) / test_path

        if full_path.is_file():
            return full_path.suffix == ".py"

        if full_path.is_dir():
            # Check for test files
            return bool(list(full_path.glob("test_*.py")) or list(full_path.glob("*_test.py")))

        return False
