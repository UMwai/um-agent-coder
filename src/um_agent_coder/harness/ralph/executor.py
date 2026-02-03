"""
Ralph loop executor wrapper.

Wraps base CLI executors to implement the iterative re-feed loop
that continues until a completion promise is detected or max
iterations is exceeded.

QA Validation:
    When require_tests_passing is enabled in RalphConfig, tests are
    run after each iteration where a promise is detected. If tests fail,
    the loop continues with a test failure prompt instead of accepting
    the promise.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from ..executors import BaseCLIExecutor
from ..models import Task
from ..test_runner import TestResult, TestRunner
from .iteration_tracker import IterationTracker
from .persistence import RalphPersistence
from .promise_detector import PromiseDetector

logger = logging.getLogger(__name__)


@dataclass
class RalphResult:
    """Result from a ralph loop execution."""

    success: bool
    iterations: int
    total_duration: timedelta
    final_output: str
    reason: Optional[str] = None  # "promise_found", "max_iterations_exceeded", "error"
    promise_text: Optional[str] = None
    error: Optional[str] = None

    @property
    def summary(self) -> str:
        """Brief summary for logging."""
        status = "SUCCESS" if self.success else "FAILED"
        duration = f"{self.total_duration.total_seconds():.1f}s"
        return f"[{status}] {self.iterations} iterations in {duration}: {self.reason}"


class RalphExecutor:
    """Executor wrapper that implements the ralph re-feed loop.

    Wraps a base executor (Codex, Gemini, Claude) and repeatedly
    executes the task until:
    1. A completion promise is detected in the output
    2. Max iterations is exceeded
    3. A fatal error occurs

    Example:
        base_executor = CodexExecutor()
        ralph = RalphExecutor(
            base_executor=base_executor,
            max_iterations=30,
            completion_promise="TASK_COMPLETE"
        )
        result = ralph.execute(task)
        if result.success:
            print(f"Completed in {result.iterations} iterations")
    """

    def __init__(
        self,
        base_executor: BaseCLIExecutor,
        max_iterations: int = 30,
        completion_promise: str = "COMPLETE",
        persistence: Optional[RalphPersistence] = None,
        require_xml_format: bool = True,
        require_tests_passing: bool = False,
        test_path: str = "tests",
    ):
        """Initialize the ralph executor.

        Args:
            base_executor: The CLI executor to wrap
            max_iterations: Maximum iterations before giving up
            completion_promise: The promise text to detect
            persistence: Optional persistence layer for resumption
            require_xml_format: If True, only match XML-style promises
            require_tests_passing: If True, run tests before accepting promise
            test_path: Path to test files when require_tests_passing is True
        """
        self.base_executor = base_executor
        self.max_iterations = max_iterations
        self.completion_promise = completion_promise
        self.persistence = persistence or RalphPersistence()
        self.require_xml_format = require_xml_format
        self.require_tests_passing = require_tests_passing
        self.test_path = test_path

        self.promise_detector = PromiseDetector(
            promise=completion_promise,
            require_xml_format=require_xml_format,
        )

        # Initialize test runner if QA validation is enabled
        self.test_runner: Optional[TestRunner] = None
        if require_tests_passing:
            self.test_runner = TestRunner()

    def execute(
        self,
        task: Task,
        context: str = "",
        resume: bool = True,
    ) -> RalphResult:
        """Execute a task using the ralph loop.

        Args:
            task: The task to execute
            context: Additional context to provide
            resume: Whether to resume from previous state if available

        Returns:
            RalphResult with execution outcome
        """
        # Check if task has QA validation config
        require_tests = self.require_tests_passing
        test_path = self.test_path
        if task.ralph_config:
            require_tests = require_tests or task.ralph_config.require_tests_passing
            test_path = task.ralph_config.test_path or test_path

        # Initialize test runner if needed
        test_runner = self.test_runner
        if require_tests and test_runner is None:
            test_runner = TestRunner()

        # Try to resume existing tracker
        tracker = None
        if resume:
            tracker = self.persistence.load_tracker(task.id)
            if tracker:
                logger.info(
                    f"Resuming ralph loop for {task.id} at iteration "
                    f"{tracker.current_iteration}/{tracker.max_iterations}"
                )

        # Create new tracker if not resuming
        if tracker is None:
            tracker = IterationTracker(
                task_id=task.id,
                max_iterations=self.max_iterations,
            )

        # Build initial prompt with ralph instructions
        ralph_prompt = self._build_ralph_prompt(task, context)

        # Main loop
        last_output = ""
        last_test_result: Optional[TestResult] = None

        while tracker.can_continue():
            iteration_num = tracker.current_iteration + 1
            logger.info(
                f"Ralph loop iteration {iteration_num}/{tracker.max_iterations} "
                f"for task {task.id}"
            )

            # Start iteration tracking
            tracker.start_iteration()

            try:
                # Execute via base executor
                result = self.base_executor.execute(task, ralph_prompt)
                last_output = result.output

                # Check for promise
                detection = self.promise_detector.detect(result.output)

                # Initialize test tracking for this iteration
                test_passed: Optional[bool] = None
                test_summary = ""

                # If promise found and QA validation enabled, run tests
                if detection.found and require_tests and test_runner:
                    logger.info(f"Promise detected, running tests at {test_path}...")

                    test_result = test_runner.run_tests(
                        test_path=test_path,
                        cwd=task.cwd,
                    )
                    last_test_result = test_result
                    test_passed = test_result.success
                    test_summary = test_result.summary

                    if not test_result.success:
                        # Tests failed - don't accept the promise
                        logger.warning(
                            f"Promise found but tests failed: {test_result.summary}"
                        )
                        detection.found = False  # Override promise detection

                        # Build test failure prompt for next iteration
                        ralph_prompt = self._build_test_failure_prompt(
                            task, context, test_result
                        )

                # End iteration tracking with test info
                tracker.end_iteration(
                    output=result.output,
                    promise_found=detection.found,
                    error=result.error if not result.success else None,
                )
                # Update the iteration record with test info
                if tracker.iteration_history:
                    tracker.iteration_history[-1].test_passed = test_passed
                    tracker.iteration_history[-1].test_summary = test_summary

                # Save state after each iteration
                self.persistence.save_tracker(tracker)

                if detection.found:
                    logger.info(
                        f"Promise detected on iteration {tracker.current_iteration}: "
                        f"{detection.promise_text}"
                    )
                    if require_tests:
                        logger.info("Tests passed - accepting promise")
                    return RalphResult(
                        success=True,
                        iterations=tracker.current_iteration,
                        total_duration=tracker.total_duration,
                        final_output=result.output,
                        reason="promise_found",
                        promise_text=detection.promise_text,
                    )

                # Log iteration summary
                logger.debug(
                    f"Iteration {tracker.current_iteration} complete, "
                    f"promise not found. Output preview: {result.output[:200]}"
                )

            except Exception as e:
                # Handle execution errors
                logger.exception(f"Error in ralph loop iteration: {e}")
                tracker.end_iteration(
                    output=last_output,
                    promise_found=False,
                    error=str(e),
                )
                self.persistence.save_tracker(tracker)

                # Continue loop unless it's a fatal error
                if self._is_fatal_error(e):
                    tracker.mark_complete("fatal_error")
                    self.persistence.save_tracker(tracker)
                    return RalphResult(
                        success=False,
                        iterations=tracker.current_iteration,
                        total_duration=tracker.total_duration,
                        final_output=last_output,
                        reason="error",
                        error=str(e),
                    )

        # Max iterations exceeded
        logger.warning(
            f"Ralph loop exceeded max iterations ({tracker.max_iterations}) " f"for task {task.id}"
        )
        tracker.mark_exceeded()
        self.persistence.save_tracker(tracker)

        return RalphResult(
            success=False,
            iterations=tracker.current_iteration,
            total_duration=tracker.total_duration,
            final_output=last_output,
            reason="max_iterations_exceeded",
        )

    def _build_ralph_prompt(self, task: Task, context: str = "") -> str:
        """Build the prompt with ralph loop instructions.

        Args:
            task: The task being executed
            context: Additional context

        Returns:
            Formatted prompt string
        """
        promise_format = f"<promise>{self.completion_promise}</promise>"

        prompt_parts = [
            "## Ralph Loop Task",
            "",
            f"**Task ID:** {task.id}",
            f"**Description:** {task.description}",
            "",
        ]

        if task.success_criteria:
            prompt_parts.extend(
                [
                    "## Success Criteria",
                    task.success_criteria,
                    "",
                ]
            )

        if context:
            prompt_parts.extend(
                [
                    "## Context",
                    context,
                    "",
                ]
            )

        # Add ralph-specific instructions
        prompt_parts.extend(
            [
                "## Completion Instructions",
                "",
                "This task is running in a ralph loop. When you have FULLY completed",
                "all requirements and verified they work correctly, output:",
                "",
                f"    {promise_format}",
                "",
                "**CRITICAL**: Only output this promise when the task is TRULY complete.",
                "The loop will continue until this promise is detected.",
                "",
                "If you cannot complete the task, explain the blocker clearly.",
                "Do NOT output the promise if the work is incomplete.",
            ]
        )

        return "\n".join(prompt_parts)

    def _build_test_failure_prompt(
        self,
        task: Task,
        context: str,
        test_result: TestResult,
    ) -> str:
        """Build a prompt when tests fail after promise detection.

        Args:
            task: The task being executed
            context: Original context
            test_result: The failed test result

        Returns:
            Formatted prompt string for the next iteration
        """
        prompt_parts = [
            "## Test Validation Failed",
            "",
            "You output the completion promise, but **tests failed**.",
            "The task is NOT complete until all tests pass.",
            "",
        ]

        # Add test failure details
        if self.test_runner:
            prompt_parts.append(self.test_runner.format_failure_prompt(test_result))
        else:
            prompt_parts.extend([
                "### Test Results",
                f"- Total: {test_result.total_tests}",
                f"- Passed: {test_result.passed}",
                f"- Failed: {test_result.failed}",
                "",
                "### Error Summary",
                "```",
                test_result.error_summary[:1500] if test_result.error_summary else "No details available",
                "```",
            ])

        prompt_parts.extend([
            "",
            "---",
            "",
            "## Original Task",
            "",
            f"**Task ID:** {task.id}",
            f"**Description:** {task.description}",
            "",
        ])

        if task.success_criteria:
            prompt_parts.extend([
                "## Success Criteria",
                task.success_criteria,
                "",
            ])

        if context:
            prompt_parts.extend([
                "## Context",
                context,
                "",
            ])

        # Remind about completion
        promise_format = f"<promise>{self.completion_promise}</promise>"
        prompt_parts.extend([
            "## Next Steps",
            "",
            "1. Fix the failing tests",
            "2. Verify your changes work correctly",
            "3. Only output the completion promise when ALL tests pass:",
            "",
            f"    {promise_format}",
            "",
            "**DO NOT** output the promise until tests pass.",
        ])

        return "\n".join(prompt_parts)

    def _is_fatal_error(self, error: Exception) -> bool:
        """Determine if an error should stop the loop.

        Args:
            error: The exception that occurred

        Returns:
            True if the loop should stop
        """
        # These error types indicate unrecoverable issues
        fatal_types = (
            KeyboardInterrupt,
            SystemExit,
            MemoryError,
        )
        return isinstance(error, fatal_types)

    def get_tracker(self, task_id: str) -> Optional[IterationTracker]:
        """Get the iteration tracker for a task.

        Args:
            task_id: The task ID

        Returns:
            IterationTracker if exists, None otherwise
        """
        return self.persistence.load_tracker(task_id)

    def reset_tracker(self, task_id: str) -> bool:
        """Reset/delete the tracker for a task.

        Args:
            task_id: The task ID

        Returns:
            True if tracker was deleted
        """
        return self.persistence.delete_tracker(task_id)
