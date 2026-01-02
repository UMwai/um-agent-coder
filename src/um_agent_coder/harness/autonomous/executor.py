"""Autonomous loop executor.

Integrates all autonomous loop features:
- Progress detection
- Stuck recovery
- Context management
- CLI routing
- Environmental awareness
- Alerts
- Time-based termination

Reference: specs/autonomous-loop-spec.md
"""

import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..executors import BaseCLIExecutor
from ..models import Task
from ..ralph.persistence import RalphPersistence
from ..ralph.promise_detector import PromiseDetector
from .alerts import (
    AlertConfig,
    AlertManager,
    PauseRequestedError,
    RunawayConfig,
    RunawayDetector,
)
from .cli_router import CLIRouter
from .context_manager import ContextManager, IterationContext, LoopContext
from .environment import EnvironmentManager
from .progress_detector import ProgressDetector, calculate_progress_score
from .progress_markers import extract_progress_markers
from .recovery import RecoveryManager, RecoveryResult, RecoveryStrategy, StuckDetector

logger = logging.getLogger(__name__)


class TerminationReason(Enum):
    """Reasons for loop termination."""

    GOAL_COMPLETE = "goal_complete"
    TIME_LIMIT = "time_limit"
    ITERATION_LIMIT = "iteration_limit"
    MANUAL_STOP = "manual_stop"
    ALERT_PAUSE = "alert_pause"
    FATAL_ERROR = "fatal_error"
    RECOVERY_FAILED = "recovery_failed"


@dataclass
class AutonomousConfig:
    """Configuration for autonomous executor."""

    # Limits
    max_iterations: int = 1000
    max_time_seconds: Optional[float] = None  # None = no limit

    # Progress detection
    progress_threshold: float = 0.15
    stuck_after: int = 3
    recovery_budget: int = 20

    # Context management
    raw_window_size: int = 5
    summarize_every: int = 10

    # CLI routing
    cli_spec: str = "auto"
    opus_daily_limit: int = 50
    prefer_cheap: bool = True

    # Environment
    enable_file_watcher: bool = True
    check_env_interval: int = 1  # Check every N iterations

    # Alerts
    alert_milestone_interval: int = 10
    pause_on_critical: bool = True

    # Promise detection
    completion_promise: str = "COMPLETE"
    require_xml_format: bool = True


@dataclass
class AutonomousResult:
    """Result from autonomous loop execution."""

    success: bool
    iterations: int
    total_duration: timedelta
    final_output: str
    termination_reason: TerminationReason
    promise_text: Optional[str] = None
    error: Optional[str] = None
    recovery_attempts: int = 0
    models_used: list[str] = field(default_factory=list)
    alerts_issued: int = 0

    @property
    def summary(self) -> str:
        """Brief summary for logging."""
        status = "SUCCESS" if self.success else "FAILED"
        duration = f"{self.total_duration.total_seconds():.1f}s"
        return (
            f"[{status}] {self.iterations} iterations in {duration}: "
            f"{self.termination_reason.value}"
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "iterations": self.iterations,
            "total_duration_seconds": self.total_duration.total_seconds(),
            "final_output": self.final_output,
            "termination_reason": self.termination_reason.value,
            "promise_text": self.promise_text,
            "error": self.error,
            "recovery_attempts": self.recovery_attempts,
            "models_used": self.models_used,
            "alerts_issued": self.alerts_issued,
        }


class AutonomousExecutor:
    """Executor that implements the full autonomous loop.

    Integrates:
    - Progress detection (multi-signal)
    - Stuck recovery (mutation, escalation, branching)
    - Context management (rolling window + summarization)
    - CLI routing (auto or explicit)
    - Environmental awareness (file watcher, instruction queue, env vars)
    - Alerts (CLI notifications, file logging)
    - Time-based termination

    Example:
        executor = AutonomousExecutor(
            executors={"codex": codex_exec, "gemini": gemini_exec},
            config=AutonomousConfig(max_time_seconds=3600),
        )
        result = executor.execute(task)
    """

    def __init__(
        self,
        executors: dict[str, BaseCLIExecutor],
        config: Optional[AutonomousConfig] = None,
        workspace_path: Optional[Path] = None,
        harness_path: Optional[Path] = None,
        persistence: Optional[RalphPersistence] = None,
    ):
        """Initialize autonomous executor.

        Args:
            executors: Map of CLI name to executor instance.
            config: Autonomous configuration.
            workspace_path: Root workspace path.
            harness_path: Path to .harness directory.
            persistence: Optional persistence layer.
        """
        self.executors = executors
        self.config = config or AutonomousConfig()
        self.workspace_path = workspace_path or Path.cwd()
        self.harness_path = harness_path or self.workspace_path / ".harness"
        self.persistence = persistence or RalphPersistence()

        # Initialize components
        self._init_components()

        # State
        self._stop_requested = False
        self._pause_requested = False

    def _init_components(self) -> None:
        """Initialize all component systems."""
        # Promise detection
        self.promise_detector = PromiseDetector(
            promise=self.config.completion_promise,
            require_xml_format=self.config.require_xml_format,
        )

        # Progress detection
        self.progress_detector = ProgressDetector(
            workspace=self.workspace_path,
            no_progress_threshold=self.config.progress_threshold,
        )

        # Stuck detection
        self.stuck_detector = StuckDetector(
            stuck_threshold=self.config.stuck_after,
            recovery_budget=self.config.recovery_budget,
        )

        # Context management
        self.context_manager = ContextManager(
            raw_window_size=self.config.raw_window_size,
            summarize_every=self.config.summarize_every,
        )

        # CLI router (must be before recovery_manager)
        self.cli_router = CLIRouter(
            cli_spec=self.config.cli_spec,
            opus_daily_limit=self.config.opus_daily_limit,
            prefer_cheap=self.config.prefer_cheap,
        )

        # Recovery manager
        from .recovery import ModelEscalator

        model_escalator = ModelEscalator(enabled_clis=set(self._get_enabled_clis()))
        self.recovery_manager = RecoveryManager(
            stuck_detector=self.stuck_detector,
            model_escalator=model_escalator,
        )

        # Environment manager
        self.environment_manager = EnvironmentManager(
            workspace_path=self.workspace_path,
            harness_path=self.harness_path,
            enable_file_watcher=self.config.enable_file_watcher,
        )

        # Alert manager
        alert_config = AlertConfig(
            alert_log_path=self.harness_path / "alerts.log",
            milestone_interval=self.config.alert_milestone_interval,
            pause_on_critical=self.config.pause_on_critical,
        )
        self.alert_manager = AlertManager(alert_config)

        # Runaway detector
        runaway_config = RunawayConfig(
            has_time_limit=self.config.max_time_seconds is not None,
        )
        self.runaway_detector = RunawayDetector(runaway_config)

    def _get_enabled_clis(self) -> list[str]:
        """Get list of enabled CLIs from config."""
        return self.cli_router.get_enabled_clis()

    def execute(
        self,
        task: Task,
        context: str = "",
        resume: bool = True,
    ) -> AutonomousResult:
        """Execute a task using the autonomous loop.

        Args:
            task: The task to execute.
            context: Additional context to provide.
            resume: Whether to resume from previous state.

        Returns:
            AutonomousResult with execution outcome.
        """
        # Setup signal handlers
        self._setup_signal_handlers()

        # Initialize loop context
        loop_context = self._init_loop_context(task, context, resume)

        # Start environment monitoring
        self.environment_manager.start()

        start_time = datetime.now()
        last_output = ""
        models_used = set()
        recovery_attempts = 0

        try:
            while self._can_continue(loop_context, start_time):
                iteration_start = time.time()

                # Check for stop/pause signals
                if self._stop_requested:
                    return self._create_result(
                        success=False,
                        iterations=loop_context.total_iterations,
                        start_time=start_time,
                        final_output=last_output,
                        reason=TerminationReason.MANUAL_STOP,
                        models_used=list(models_used),
                        recovery_attempts=recovery_attempts,
                    )

                if self._pause_requested:
                    return self._create_result(
                        success=False,
                        iterations=loop_context.total_iterations,
                        start_time=start_time,
                        final_output=last_output,
                        reason=TerminationReason.ALERT_PAUSE,
                        models_used=list(models_used),
                        recovery_attempts=recovery_attempts,
                    )

                # Check environment
                if loop_context.total_iterations % self.config.check_env_interval == 0:
                    env_state = self.environment_manager.poll()
                    if env_state.should_stop:
                        return self._create_result(
                            success=False,
                            iterations=loop_context.total_iterations,
                            start_time=start_time,
                            final_output=last_output,
                            reason=TerminationReason.MANUAL_STOP,
                            models_used=list(models_used),
                            recovery_attempts=recovery_attempts,
                        )

                # Route to CLI/model
                cli_name, model = self.cli_router.route(
                    goal=task.description,
                    context_size=len(loop_context.summary),
                    consecutive_no_progress=self.stuck_detector.consecutive_no_progress,
                    total_iterations=loop_context.total_iterations,
                )
                models_used.add(model)
                loop_context.current_cli = cli_name
                loop_context.current_model = model

                # Get executor
                executor = self.executors.get(cli_name)
                if not executor:
                    logger.warning(f"No executor for CLI: {cli_name}, using first available")
                    cli_name = list(self.executors.keys())[0]
                    executor = self.executors[cli_name]

                # Build prompt
                prompt = self.context_manager.build_prompt(loop_context)

                # Execute iteration
                logger.info(
                    f"Autonomous loop iteration {loop_context.total_iterations + 1} "
                    f"using {cli_name}/{model}"
                )

                try:
                    result = executor.execute(task, prompt)
                    last_output = result.output or ""

                    # Check for promise
                    detection = self.promise_detector.detect(last_output)
                    if detection.found:
                        logger.info(f"Promise detected: {detection.promise_text}")
                        self.alert_manager.goal_complete(
                            iteration=loop_context.total_iterations + 1,
                            promise=detection.promise_text,
                        )
                        return self._create_result(
                            success=True,
                            iterations=loop_context.total_iterations + 1,
                            start_time=start_time,
                            final_output=last_output,
                            reason=TerminationReason.GOAL_COMPLETE,
                            promise_text=detection.promise_text,
                            models_used=list(models_used),
                            recovery_attempts=recovery_attempts,
                        )

                    # Calculate progress
                    prev_output = ""
                    if loop_context.iterations:
                        prev_output = loop_context.iterations[-1].output

                    progress_signal = self.progress_detector.detect(
                        curr_output=last_output,
                        prev_output=prev_output,
                    )
                    progress_score = calculate_progress_score(progress_signal)

                    # Extract markers
                    markers = extract_progress_markers(last_output)

                    # Record iteration
                    iteration_duration = time.time() - iteration_start
                    iteration_ctx = IterationContext(
                        iteration_number=loop_context.total_iterations + 1,
                        timestamp=datetime.now(),
                        cli_used=cli_name,
                        model_used=model,
                        prompt=prompt[:500],  # Truncate for storage
                        output=last_output[:5000],  # Truncate for storage
                        progress_score=progress_score,
                        progress_markers=markers,
                        duration_seconds=iteration_duration,
                    )
                    self.context_manager.add_iteration(loop_context, iteration_ctx)

                    # Update stuck detector
                    had_progress = progress_score >= self.config.progress_threshold
                    self.stuck_detector.record_iteration(
                        iteration=loop_context.total_iterations,
                        progress_score=progress_score,
                        had_progress=had_progress,
                        output_snippet=last_output[:200] if last_output else "",
                    )

                    # Check runaway
                    runaway_alert = self.runaway_detector.check(
                        iteration=loop_context.total_iterations,
                        duration=iteration_duration,
                        output=last_output,
                    )
                    if runaway_alert:
                        try:
                            self.alert_manager.alert(
                                runaway_alert.alert_type,
                                runaway_alert.message,
                                runaway_alert.severity,
                                iteration=loop_context.total_iterations,
                            )
                        except PauseRequestedError:
                            self._pause_requested = True
                            continue

                    # Check milestone
                    self.alert_manager.iteration_milestone(loop_context.total_iterations)

                    # Check if stuck and attempt recovery
                    if self.stuck_detector.is_stuck():
                        logger.warning(
                            f"Stuck detected after {self.stuck_detector.consecutive_no_progress} "
                            "iterations without progress"
                        )
                        self.alert_manager.no_progress(
                            self.stuck_detector.consecutive_no_progress,
                            iteration=loop_context.total_iterations,
                        )

                        if self.recovery_manager.needs_recovery():
                            recovery_result = self._attempt_recovery(task, loop_context)
                            recovery_attempts += 1

                            if not recovery_result.success:
                                logger.error("Recovery failed, terminating")
                                self.alert_manager.fatal_error(
                                    "Recovery exhausted",
                                    iteration=loop_context.total_iterations,
                                )
                                return self._create_result(
                                    success=False,
                                    iterations=loop_context.total_iterations,
                                    start_time=start_time,
                                    final_output=last_output,
                                    reason=TerminationReason.RECOVERY_FAILED,
                                    models_used=list(models_used),
                                    recovery_attempts=recovery_attempts,
                                )

                except Exception as e:
                    logger.exception(f"Error in iteration: {e}")
                    if self._is_fatal_error(e):
                        return self._create_result(
                            success=False,
                            iterations=loop_context.total_iterations,
                            start_time=start_time,
                            final_output=last_output,
                            reason=TerminationReason.FATAL_ERROR,
                            error=str(e),
                            models_used=list(models_used),
                            recovery_attempts=recovery_attempts,
                        )

            # Determine termination reason
            reason = self._get_termination_reason(loop_context, start_time)
            return self._create_result(
                success=False,
                iterations=loop_context.total_iterations,
                start_time=start_time,
                final_output=last_output,
                reason=reason,
                models_used=list(models_used),
                recovery_attempts=recovery_attempts,
            )

        finally:
            # Cleanup
            self.environment_manager.stop()
            self._restore_signal_handlers()

    def _init_loop_context(
        self,
        task: Task,
        context: str,
        resume: bool,
    ) -> LoopContext:
        """Initialize or resume loop context.

        Args:
            task: The task to execute.
            context: Additional context string.
            resume: Whether to resume from previous state.

        Returns:
            Initialized or resumed LoopContext.
        """
        if resume and self.persistence:
            # Try to load previous state
            tracker = self.persistence.load_tracker(task.id)
            if tracker and tracker.iteration_history:
                # Resume from previous state
                loop_context = LoopContext(
                    task_id=task.id,
                    goal=task.description,
                    start_time=datetime.now(),
                )
                loop_context.total_iterations = tracker.current_iteration
                return loop_context

        # Start fresh
        return LoopContext(
            task_id=task.id,
            goal=task.description,
            start_time=datetime.now(),
        )

    def _can_continue(self, context: LoopContext, start_time: datetime) -> bool:
        """Check if loop should continue."""
        # Check iteration limit
        if context.total_iterations >= self.config.max_iterations:
            return False

        # Check time limit
        if self.config.max_time_seconds:
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= self.config.max_time_seconds:
                return False

        return True

    def _get_termination_reason(
        self,
        context: LoopContext,
        start_time: datetime,
    ) -> TerminationReason:
        """Determine why loop terminated."""
        if context.total_iterations >= self.config.max_iterations:
            return TerminationReason.ITERATION_LIMIT

        if self.config.max_time_seconds:
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= self.config.max_time_seconds:
                return TerminationReason.TIME_LIMIT

        return TerminationReason.MANUAL_STOP

    def _attempt_recovery(
        self,
        task: Task,
        context: LoopContext,
    ) -> RecoveryResult:
        """Attempt stuck recovery."""
        self.alert_manager.stuck_recovery(
            "recovery_attempt",
            iteration=context.total_iterations,
        )

        # Build recovery context
        recovery_context = self.context_manager.get_summary_for_recovery(context)

        # Use recovery manager
        result = self.recovery_manager.recover(
            goal=task.description,
            context=recovery_context,
        )

        if result.strategy == RecoveryStrategy.MODEL_ESCALATION and result.escalated_to:
            self.alert_manager.model_escalation(
                from_model=context.current_model,
                to_model=result.escalated_to[1],
                iteration=context.total_iterations,
            )

        return result

    def _create_result(
        self,
        success: bool,
        iterations: int,
        start_time: datetime,
        final_output: str,
        reason: TerminationReason,
        promise_text: Optional[str] = None,
        error: Optional[str] = None,
        models_used: Optional[list[str]] = None,
        recovery_attempts: int = 0,
    ) -> AutonomousResult:
        """Create result object."""
        return AutonomousResult(
            success=success,
            iterations=iterations,
            total_duration=datetime.now() - start_time,
            final_output=final_output,
            termination_reason=reason,
            promise_text=promise_text,
            error=error,
            recovery_attempts=recovery_attempts,
            models_used=models_used or [],
            alerts_issued=len(self.alert_manager.alerts),
        )

    def _is_fatal_error(self, error: Exception) -> bool:
        """Determine if error should stop loop."""
        fatal_types = (
            KeyboardInterrupt,
            SystemExit,
            MemoryError,
        )
        return isinstance(error, fatal_types)

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful termination."""
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if hasattr(self, "_original_sigint"):
            signal.signal(signal.SIGINT, self._original_sigint)
        if hasattr(self, "_original_sigterm"):
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def _handle_interrupt(self, signum, frame) -> None:
        """Handle interrupt signal."""
        logger.info("Interrupt received, stopping after current iteration")
        self._stop_requested = True

    def request_stop(self) -> None:
        """Request graceful stop."""
        self._stop_requested = True

    def request_pause(self) -> None:
        """Request pause."""
        self._pause_requested = True

    def get_status(self) -> dict[str, Any]:
        """Get current executor status."""
        return {
            "stop_requested": self._stop_requested,
            "pause_requested": self._pause_requested,
            "enabled_clis": self._get_enabled_clis(),
            "opus_remaining": self.cli_router.get_opus_remaining(),
            "stuck_state": self.stuck_detector.current_state.value,
            "alerts_issued": len(self.alert_manager.alerts),
            "runaway_detected": self.runaway_detector.has_detected_runaway(),
        }
