"""
Ralph loop executor wrapper.

Wraps base CLI executors to implement the iterative re-feed loop
that continues until a completion promise is detected or max
iterations is exceeded.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from ..executors import BaseCLIExecutor
from ..models import Task
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
    ):
        """Initialize the ralph executor.

        Args:
            base_executor: The CLI executor to wrap
            max_iterations: Maximum iterations before giving up
            completion_promise: The promise text to detect
            persistence: Optional persistence layer for resumption
            require_xml_format: If True, only match XML-style promises
        """
        self.base_executor = base_executor
        self.max_iterations = max_iterations
        self.completion_promise = completion_promise
        self.persistence = persistence or RalphPersistence()
        self.require_xml_format = require_xml_format

        self.promise_detector = PromiseDetector(
            promise=completion_promise,
            require_xml_format=require_xml_format,
        )

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

                # End iteration tracking
                tracker.end_iteration(
                    output=result.output,
                    promise_found=detection.found,
                    error=result.error if not result.success else None,
                )

                # Save state after each iteration
                self.persistence.save_tracker(tracker)

                if detection.found:
                    logger.info(
                        f"Promise detected on iteration {tracker.current_iteration}: "
                        f"{detection.promise_text}"
                    )
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
            f"Ralph loop exceeded max iterations ({tracker.max_iterations}) "
            f"for task {task.id}"
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

        # Add ralph-specific instructions
        prompt_parts.extend([
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
