"""
Ralph loop executor wrapper.

Wraps base CLI executors to implement the iterative re-feed loop
that continues until a completion promise is detected or max
iterations is exceeded.

Intelligent Loop Features (ported from Gemini Intelligence Layer):
    1. Per-iteration scoring: Evaluate output every N iterations for trajectory tracking
    2. Strategic retry prompts: Targeted fix guidance based on failing criteria
    3. Oscillation detection: Detect stuck loops and suggest recovery actions
    4. Pre-task checklist: Inject goal criteria into initial prompt
    5. Accuracy-first cascade: Quick syntax check before full test suite
"""

import logging
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Optional

from ..executors import BaseCLIExecutor
from ..models import Task
from ..test_runner import TestResult, TestRunner
from .goal_validator import GoalValidationResult, GoalValidator
from .iteration_tracker import IterationTracker
from .persistence import RalphPersistence
from .promise_detector import PromiseDetector
from .strategies import build_strategic_prompt, select_strategies

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
    best_score: Optional[float] = None  # Best eval score achieved

    @property
    def summary(self) -> str:
        """Brief summary for logging."""
        status = "SUCCESS" if self.success else "FAILED"
        duration = f"{self.total_duration.total_seconds():.1f}s"
        score_info = f" best_score={self.best_score:.2f}" if self.best_score is not None else ""
        return f"[{status}] {self.iterations} iterations in {duration}: {self.reason}{score_info}"


class RalphExecutor:
    """Executor wrapper that implements the ralph re-feed loop.

    Wraps a base executor (Codex, Gemini, Claude) and repeatedly
    executes the task until:
    1. A completion promise is detected in the output
    2. Max iterations is exceeded
    3. A fatal error occurs

    Intelligent loop features (from Gemini Intelligence Layer):
    - Per-iteration scoring tracks trajectory between promises
    - Strategic retry prompts give targeted fix guidance
    - Oscillation detection prevents infinite stuck loops
    - Pre-task checklist injection gives the agent grading criteria upfront
    - Accuracy-first cascade avoids expensive test runs on broken code

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
        goal_validator: Optional[GoalValidator] = None,
        # Intelligent loop config
        scoring_interval: int = 3,
        inject_checklist: bool = True,
        enable_oscillation_detection: bool = True,
        oscillation_window: int = 4,
        oscillation_spread: float = 0.03,
    ):
        self.base_executor = base_executor
        self.max_iterations = max_iterations
        self.completion_promise = completion_promise
        self.persistence = persistence or RalphPersistence()
        self.require_xml_format = require_xml_format
        self.require_tests_passing = require_tests_passing
        self.test_path = test_path
        self.goal_validator = goal_validator

        # Intelligent loop settings
        self.scoring_interval = scoring_interval
        self.inject_checklist = inject_checklist
        self.enable_oscillation_detection = enable_oscillation_detection
        self.oscillation_window = oscillation_window
        self.oscillation_spread = oscillation_spread

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
        """Execute a task using the ralph loop with intelligent evaluation."""
        # Check if task has QA validation config
        require_tests = self.require_tests_passing
        test_path = self.test_path
        scoring_interval = self.scoring_interval
        if task.ralph_config:
            require_tests = require_tests or task.ralph_config.require_tests_passing
            test_path = task.ralph_config.test_path or test_path
            scoring_interval = task.ralph_config.scoring_interval

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
                    "Resuming ralph loop for %s at iteration %d/%d",
                    task.id,
                    tracker.current_iteration,
                    tracker.max_iterations,
                )

        # Create new tracker if not resuming
        if tracker is None:
            tracker = IterationTracker(
                task_id=task.id,
                max_iterations=self.max_iterations,
            )

        # Build initial prompt with ralph instructions + checklist injection
        ralph_prompt = self._build_ralph_prompt(task, context)

        # Main loop
        last_output = ""

        while tracker.can_continue():
            iteration_num = tracker.current_iteration + 1
            logger.info(
                "Ralph loop iteration %d/%d for task %s",
                iteration_num,
                tracker.max_iterations,
                task.id,
            )

            # Start iteration tracking
            tracker.start_iteration()

            try:
                # Execute via base executor
                result = self.base_executor.execute(task, ralph_prompt)
                last_output = result.output

                # Check for promise
                detection = self.promise_detector.detect(result.output)

                # Initialize tracking for this iteration
                test_passed: Optional[bool] = None
                test_summary = ""
                goal_score: Optional[float] = None
                goal_passed: Optional[bool] = None
                eval_score: Optional[float] = None

                # --- Per-iteration scoring (Improvement #1) ---
                # Score periodically even without a promise to track trajectory
                should_score = self.goal_validator and tracker.should_score_this_iteration(
                    scoring_interval
                )
                if should_score and not detection.found:
                    logger.info(
                        "Running periodic scoring (iteration %d, interval=%d)",
                        tracker.current_iteration,
                        scoring_interval,
                    )
                    try:
                        periodic_result = self.goal_validator.validate(
                            result.output,
                            iteration=tracker.current_iteration,
                        )
                        eval_score = periodic_result.score
                        goal_score = periodic_result.score
                        goal_passed = periodic_result.passed
                        logger.info(
                            "Periodic score: %.2f (passed=%s, %d/%d criteria)",
                            periodic_result.score,
                            periodic_result.passed,
                            len(periodic_result.criteria_results)
                            - len(periodic_result.failing_criteria),
                            len(periodic_result.criteria_results),
                        )

                        # Use periodic score to build a better prompt
                        if not periodic_result.passed:
                            ralph_prompt = self._build_strategic_retry(
                                task,
                                context,
                                periodic_result,
                                tracker,
                                last_output,
                            )
                    except Exception as e:
                        logger.warning("Periodic scoring failed: %s", e)

                # --- Accuracy-first cascade (Improvement #5) ---
                # On promise detection, do a quick syntax check before tests
                if detection.found and require_tests and test_runner:
                    syntax_ok = self._quick_syntax_check(task.cwd, test_path)
                    if not syntax_ok:
                        logger.warning(
                            "Promise found but code has syntax errors, "
                            "skipping expensive test run"
                        )
                        detection.found = False
                        ralph_prompt = self._build_syntax_failure_prompt(task, context)
                        # Skip to end_iteration
                        tracker.end_iteration(
                            output=result.output,
                            promise_found=False,
                            error="syntax_check_failed",
                        )
                        if tracker.iteration_history:
                            tracker.iteration_history[-1].eval_score = eval_score
                        self.persistence.save_tracker(tracker)
                        continue

                # If promise found and QA validation enabled, run tests
                if detection.found and require_tests and test_runner:
                    logger.info("Promise detected, running tests at %s...", test_path)

                    test_result = test_runner.run_tests(
                        test_path=test_path,
                        cwd=task.cwd,
                    )
                    test_passed = test_result.success
                    test_summary = test_result.summary

                    if not test_result.success:
                        logger.warning(
                            "Promise found but tests failed: %s",
                            test_result.summary,
                        )
                        detection.found = False
                        ralph_prompt = self._build_test_failure_prompt(task, context, test_result)

                # Goal validation gate (runs after test gate)
                if detection.found and self.goal_validator:
                    logger.info("Promise detected, running goal validation...")

                    goal_result = self.goal_validator.validate(
                        result.output,
                        iteration=tracker.current_iteration,
                    )
                    goal_score = goal_result.score
                    goal_passed = goal_result.passed
                    eval_score = goal_result.score

                    if not goal_result.passed:
                        logger.warning(
                            "Promise found but goal validation failed: "
                            "score=%.2f (%d failing criteria)",
                            goal_result.score,
                            len(goal_result.failing_criteria),
                        )
                        detection.found = False

                        # Use strategic retry (Improvement #2)
                        ralph_prompt = self._build_strategic_retry(
                            task,
                            context,
                            goal_result,
                            tracker,
                            last_output,
                        )
                    else:
                        logger.info(
                            "Goal validation passed: score=%.2f",
                            goal_result.score,
                        )

                # End iteration tracking
                tracker.end_iteration(
                    output=result.output,
                    promise_found=detection.found,
                    error=result.error if not result.success else None,
                )
                # Update the iteration record with all tracking info
                if tracker.iteration_history:
                    record = tracker.iteration_history[-1]
                    record.test_passed = test_passed
                    record.test_summary = test_summary
                    record.goal_score = goal_score
                    record.goal_passed = goal_passed
                    record.eval_score = eval_score

                # Save state after each iteration
                self.persistence.save_tracker(tracker)

                if detection.found:
                    logger.info(
                        "Promise detected on iteration %d: %s",
                        tracker.current_iteration,
                        detection.promise_text,
                    )
                    if require_tests:
                        logger.info("Tests passed - accepting promise")

                    trajectory = tracker.get_score_trajectory()
                    return RalphResult(
                        success=True,
                        iterations=tracker.current_iteration,
                        total_duration=tracker.total_duration,
                        final_output=result.output,
                        reason="promise_found",
                        promise_text=detection.promise_text,
                        best_score=trajectory.get("best_score"),
                    )

                # --- Oscillation detection (Improvement #3) ---
                if self.enable_oscillation_detection and eval_score is not None:
                    osc = tracker.detect_oscillation(
                        window=self.oscillation_window,
                        spread=self.oscillation_spread,
                    )
                    if osc["oscillating"]:
                        logger.warning(
                            "Oscillation detected: scores=%s spread=%.3f " "suggestion=%s",
                            osc["scores"],
                            osc["spread"],
                            osc["suggestion"],
                        )
                        ralph_prompt = self._inject_oscillation_guidance(ralph_prompt, osc)

                # Log iteration summary
                score_info = f" score={eval_score:.2f}" if eval_score is not None else ""
                logger.debug(
                    "Iteration %d complete, promise not found.%s",
                    tracker.current_iteration,
                    score_info,
                )

            except Exception as e:
                logger.exception("Error in ralph loop iteration: %s", e)
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
                    trajectory = tracker.get_score_trajectory()
                    return RalphResult(
                        success=False,
                        iterations=tracker.current_iteration,
                        total_duration=tracker.total_duration,
                        final_output=last_output,
                        reason="error",
                        error=str(e),
                        best_score=trajectory.get("best_score"),
                    )

        # Max iterations exceeded
        logger.warning(
            "Ralph loop exceeded max iterations (%d) for task %s",
            tracker.max_iterations,
            task.id,
        )
        tracker.mark_exceeded()
        self.persistence.save_tracker(tracker)

        trajectory = tracker.get_score_trajectory()
        return RalphResult(
            success=False,
            iterations=tracker.current_iteration,
            total_duration=tracker.total_duration,
            final_output=last_output,
            reason="max_iterations_exceeded",
            best_score=trajectory.get("best_score"),
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_ralph_prompt(self, task: Task, context: str = "") -> str:
        """Build the initial prompt with ralph instructions.

        Includes pre-task checklist injection (Improvement #4) when
        goal_validator has criteria loaded.
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

        # --- Pre-task checklist injection (Improvement #4) ---
        if self.inject_checklist and self.goal_validator and self.goal_validator.criteria:
            prompt_parts.extend(
                [
                    "## Evaluation Checklist",
                    "",
                    "Your output will be graded against these criteria.",
                    "Address ALL items marked as breaking priority:",
                    "",
                ]
            )
            for c in self.goal_validator.criteria:
                severity_tag = c.severity.upper()
                prompt_parts.append(f"- [{severity_tag}] ({c.dimension}) {c.description}")
            prompt_parts.append("")

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

    def _build_strategic_retry(
        self,
        task: Task,
        context: str,
        goal_result: GoalValidationResult,
        tracker: IterationTracker,
        previous_output: str,
    ) -> str:
        """Build a strategic retry prompt using the strategy engine (Improvement #2).

        Instead of a generic "goal validation failed" message, this provides
        dimension-specific fix guidance with severity prioritization.
        """
        # Select strategies based on failing criteria
        strategies = select_strategies(
            result=goal_result,
            criteria=self.goal_validator.criteria if self.goal_validator else None,
            max_strategies=3,
        )

        # Get trend info from goal validator and tracker
        trend_info = None
        if self.goal_validator:
            trend_info = self.goal_validator.detect_trend()
        if trend_info is None or trend_info.get("trend") == "unknown":
            trajectory = tracker.get_score_trajectory()
            trend_info = {
                "trend": (
                    trajectory["trend"] if trajectory["trend"] != "insufficient_data" else "unknown"
                ),
                "best_score": trajectory["best_score"],
                "current_score": trajectory["current_score"],
            }

        prompt = build_strategic_prompt(
            task_description=task.description,
            success_criteria=task.success_criteria or "",
            context=context,
            result=goal_result,
            strategies=strategies,
            previous_output=previous_output,
            completion_promise=self.completion_promise,
            trend_info=trend_info,
        )

        logger.info(
            "Built strategic retry: %d strategies, trend=%s",
            len(strategies),
            trend_info.get("trend", "unknown") if trend_info else "unknown",
        )

        return prompt

    def _build_test_failure_prompt(
        self,
        task: Task,
        context: str,
        test_result: TestResult,
    ) -> str:
        """Build a prompt when tests fail after promise detection."""
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
            prompt_parts.extend(
                [
                    "### Test Results",
                    f"- Total: {test_result.total_tests}",
                    f"- Passed: {test_result.passed}",
                    f"- Failed: {test_result.failed}",
                    "",
                    "### Error Summary",
                    "```",
                    (
                        test_result.error_summary[:1500]
                        if test_result.error_summary
                        else "No details available"
                    ),
                    "```",
                ]
            )

        prompt_parts.extend(
            [
                "",
                "---",
                "",
                "## Original Task",
                "",
                f"**Task ID:** {task.id}",
                f"**Description:** {task.description}",
                "",
            ]
        )

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

        # Remind about completion
        promise_format = f"<promise>{self.completion_promise}</promise>"
        prompt_parts.extend(
            [
                "## Next Steps",
                "",
                "1. Fix the failing tests",
                "2. Verify your changes work correctly",
                "3. Only output the completion promise when ALL tests pass:",
                "",
                f"    {promise_format}",
                "",
                "**DO NOT** output the promise until tests pass.",
            ]
        )

        return "\n".join(prompt_parts)

    def _build_syntax_failure_prompt(self, task: Task, context: str) -> str:
        """Build a prompt when code has syntax errors (Improvement #5)."""
        promise_format = f"<promise>{self.completion_promise}</promise>"

        parts = [
            "## Syntax Check Failed",
            "",
            "Your code has **syntax errors** that prevent it from being imported.",
            "Fix all syntax errors before outputting the completion promise.",
            "",
            "Run `python -c 'import py_compile; py_compile.compile(\"<file>\")'`",
            "on each modified file to verify syntax.",
            "",
            "---",
            "",
            "## Original Task",
            "",
            f"**Description:** {task.description}",
            "",
        ]

        if task.success_criteria:
            parts.extend(["## Success Criteria", task.success_criteria, ""])

        if context:
            parts.extend(["## Context", context, ""])

        parts.extend(
            [
                "## Completion",
                "",
                f"Output {promise_format} only when syntax is clean and all work is done.",
            ]
        )

        return "\n".join(parts)

    def _build_goal_failure_prompt(
        self,
        task: Task,
        context: str,
        goal_result: GoalValidationResult,
    ) -> str:
        """Build a prompt when goal validation fails after promise detection."""
        if self.goal_validator:
            return self.goal_validator.build_failure_prompt(
                result=goal_result,
                task_description=task.description,
                context=context,
            )
        return self._build_ralph_prompt(task, context)

    # ------------------------------------------------------------------
    # Intelligent loop helpers
    # ------------------------------------------------------------------

    def _quick_syntax_check(self, cwd: str, test_path: str) -> bool:
        """Quick syntax check on Python files (Improvement #5).

        Uses ``python -c 'import ast; ast.parse(open(f).read())'`` on
        individual .py files under *test_path* to catch obvious syntax
        errors before running the expensive full test suite.

        If *test_path* is a directory, all ``*.py`` files under it are
        checked.  If it is a single file, only that file is checked.
        Returns True (pass) if no syntax errors are found **or** if
        the check cannot run for any reason.
        """
        import glob as _glob
        from pathlib import Path as _Path

        try:
            base = _Path(cwd or ".") / test_path
            if base.is_file():
                files = [str(base)]
            elif base.is_dir():
                files = _glob.glob(str(base / "**" / "*.py"), recursive=True)
            else:
                return True  # path doesn't exist yet — don't block

            for fpath in files[:20]:  # cap to avoid long scans
                result = subprocess.run(
                    ["python3", "-c", f"import ast; ast.parse(open({fpath!r}).read())"],
                    cwd=cwd or ".",
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    logger.debug("Syntax error in %s: %s", fpath, result.stderr[:200])
                    return False
            return True
        except Exception as e:
            logger.debug("Syntax check skipped: %s", e)
            return True  # Don't block on check failure

    def _inject_oscillation_guidance(self, prompt: str, osc: Dict) -> str:
        """Inject oscillation recovery guidance into the prompt (Improvement #3)."""
        suggestion = osc.get("suggestion", "continue")
        if suggestion == "continue":
            return prompt

        guidance_parts = [
            "",
            "## Loop Stuck Warning",
            "",
            f"Your last {len(osc.get('scores', []))} iterations produced nearly "
            f"identical scores (spread={osc.get('spread', 0):.3f}).",
            "",
        ]

        if suggestion == "mutate_prompt":
            guidance_parts.extend(
                [
                    "You are close but stuck. Try a fundamentally different approach:",
                    "- Restructure the code architecture",
                    "- Use different libraries or patterns",
                    "- Break the problem down differently",
                    "- Re-read the requirements and check if you missed something",
                ]
            )
        elif suggestion == "escalate_model":
            guidance_parts.extend(
                [
                    "Your approach may be fundamentally wrong. Consider:",
                    "- Starting over with a fresh design",
                    "- Reading the existing codebase more carefully",
                    "- Simplifying the solution dramatically",
                    "- Focusing on getting ONE thing right before expanding",
                ]
            )

        guidance_parts.append("")

        return prompt + "\n".join(guidance_parts)

    # ------------------------------------------------------------------
    # Error handling and utilities
    # ------------------------------------------------------------------

    def _is_fatal_error(self, error: Exception) -> bool:
        """Determine if an error should stop the loop."""
        fatal_types = (
            KeyboardInterrupt,
            SystemExit,
            MemoryError,
        )
        return isinstance(error, fatal_types)

    def get_tracker(self, task_id: str) -> Optional[IterationTracker]:
        """Get the iteration tracker for a task."""
        return self.persistence.load_tracker(task_id)

    def reset_tracker(self, task_id: str) -> bool:
        """Reset/delete the tracker for a task."""
        return self.persistence.delete_tracker(task_id)
