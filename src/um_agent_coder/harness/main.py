"""
Main harness loop for 24/7 CLI execution.

Supports multiple CLI backends:
- codex: OpenAI Codex CLI (ChatGPT Pro) - gpt-5.2
- gemini: Google Gemini CLI - gemini-3-pro, gemini-3-flash
- claude: Anthropic Claude CLI - claude-opus-4.5

Usage:
    python -m src.um_agent_coder.harness --roadmap specs/roadmap.md
    python -m src.um_agent_coder.harness --roadmap specs/roadmap.md --cli gemini
    python -m src.um_agent_coder.harness --roadmap specs/roadmap.md --cli claude --daemon
"""

import argparse
import logging
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .executors import (
    BaseCLIExecutor,
    ClaudeExecutor,
    CodexExecutor,
    GeminiExecutor,
)
from .growth import GrowthLoop
from .models import HarnessState, Roadmap, Task, TaskStatus
from .ralph import RalphExecutor, RalphPersistence
from .roadmap_parser import RoadmapParser
from .state import StateManager

# Ensure .harness directory exists for logging
Path(".harness").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(".harness/harness.log"),
    ],
)
logger = logging.getLogger(__name__)


class Harness:
    """Main 24/7 CLI execution harness."""

    # Default models per CLI
    DEFAULT_MODELS = {
        "codex": "gpt-5.2",
        "gemini": "gemini-3-pro",
        "claude": "claude-opus-4.5",
    }

    def __init__(
        self,
        roadmap_path: str,
        cli: str = "codex",
        model: Optional[str] = None,
        reasoning_effort: str = "high",
        cooldown_seconds: int = 10,
        dry_run: bool = False,
        parallel: bool = False,
        max_parallel_tasks: int = 4,
        ralph_default_iterations: int = 30,
        ralph_default_promise: str = "COMPLETE",
    ):
        self.roadmap_path = roadmap_path
        self.dry_run = dry_run
        self.cooldown_seconds = cooldown_seconds
        self.default_cli = cli.lower()
        self.default_model = model or self.DEFAULT_MODELS.get(self.default_cli, "")
        self.parallel = parallel
        self.max_parallel_tasks = max_parallel_tasks

        # Ralph-specific defaults
        self.ralph_default_iterations = ralph_default_iterations
        self.ralph_default_promise = ralph_default_promise

        # Initialize components
        self.parser = RoadmapParser(roadmap_path)

        # Create default executor
        self.default_executor = self._create_executor(
            cli=self.default_cli,
            model=self.default_model,
            reasoning_effort=reasoning_effort,
        )

        # Cache for task-specific executors
        self._executors: dict[str, BaseCLIExecutor] = {self.default_cli: self.default_executor}

        self.state = StateManager()
        self.growth = GrowthLoop(self.default_executor)
        self.reasoning_effort = reasoning_effort

        # Ralph persistence (shared across ralph executors)
        self.ralph_persistence = RalphPersistence()

        # Runtime state
        self.roadmap: Optional[Roadmap] = None
        self.harness_state: Optional[HarnessState] = None
        self._running = False
        self._shutdown_requested = False

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True

    def _create_executor(
        self,
        cli: str,
        model: Optional[str] = None,
        reasoning_effort: str = "high",
    ) -> BaseCLIExecutor:
        """Create an executor for the specified CLI."""
        cli = cli.lower()

        if cli == "codex":
            return CodexExecutor(
                model=model or "gpt-5.2",
                reasoning_effort=reasoning_effort,
            )
        elif cli == "gemini":
            return GeminiExecutor(
                model=model or "gemini-3-pro",
            )
        elif cli == "claude":
            return ClaudeExecutor(
                model=model or "claude-opus-4.5",
            )
        else:
            raise ValueError(f"Unknown CLI backend: {cli}")

    def _get_executor_for_task(self, task: Task) -> BaseCLIExecutor:
        """Get the appropriate executor for a task (with per-task CLI support)."""
        # Use task's CLI if specified, otherwise use default
        cli = task.cli or self.default_cli
        model = task.model or None

        # Check cache first
        cache_key = f"{cli}:{model or 'default'}"
        if cache_key in self._executors:
            return self._executors[cache_key]

        # Create new executor
        executor = self._create_executor(
            cli=cli,
            model=model,
            reasoning_effort=self.reasoning_effort,
        )
        self._executors[cache_key] = executor

        return executor

    def run(self) -> None:
        """Main execution loop."""
        logger.info(f"Starting harness with roadmap: {self.roadmap_path}")
        logger.info(f"Default CLI: {self.default_cli} (model: {self.default_model})")

        # Initialize
        self._initialize()

        self._running = True
        while self._running and not self._shutdown_requested:
            try:
                self._iteration()
            except Exception as e:
                logger.exception(f"Error in main loop: {e}")
                time.sleep(self.cooldown_seconds * 3)  # Longer cooldown on error

            if self._running:
                time.sleep(self.cooldown_seconds)

        logger.info("Harness shutdown complete")
        self._print_summary()

    def _initialize(self) -> None:
        """Initialize harness state and load roadmap."""
        # Parse roadmap
        self.roadmap = self.parser.parse()
        logger.info(f"Loaded roadmap: {self.roadmap.name}")
        logger.info(f"Objective: {self.roadmap.objective}")
        logger.info(
            f"Tasks: {len(self.roadmap.all_tasks)} across {len(self.roadmap.phases)} phases"
        )

        # Initialize state
        self.harness_state = self.state.init_harness(self.roadmap_path)

        # Sync tasks to state
        for task in self.roadmap.all_tasks:
            existing = self.state.load_task(task.id)
            if existing:
                # Restore state from database
                task.status = existing.status
                task.attempts = existing.attempts
                task.output = existing.output
                task.error = existing.error
                task.conversation_id = existing.conversation_id
            else:
                # Save new task
                self.state.save_task(task)

    def _iteration(self) -> None:
        """Execute one iteration of the main loop."""
        # Check if in growth mode
        if self.roadmap.is_complete and not self.harness_state.in_growth_mode:
            logger.info("All tasks complete! Entering growth mode...")
            self.harness_state.in_growth_mode = True
            self.state.update_harness_state(in_growth_mode=True)

        if self.harness_state.in_growth_mode:
            self._growth_iteration()
        else:
            self._task_iteration()

    def _task_iteration(self) -> None:
        """Execute next pending task."""
        # Get next executable task
        task = self._get_next_task()

        if not task:
            logger.debug("No tasks ready to execute")
            return

        logger.info(f"Executing task: {task.id} - {task.description}")

        if self.dry_run:
            cli_info = f"{task.cli or self.default_cli}"
            if task.model:
                cli_info += f" ({task.model})"
            logger.info(f"[DRY RUN] Would execute via {cli_info}: {task.description}")
            # Mark as completed to progress through all tasks
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            self.state.save_task(task)
            return

        # Execute the task
        self._execute_task(task)

    def _growth_iteration(self) -> None:
        """Generate and execute a growth task."""
        completed_tasks = [t for t in self.roadmap.all_tasks if t.status == TaskStatus.COMPLETED]

        logger.info(f"Growth iteration {self.harness_state.growth_iterations + 1}")

        # Generate new growth task
        growth_task = self.growth.generate_growth_task(self.roadmap, completed_tasks)

        if not growth_task:
            logger.warning("Could not generate growth task, waiting...")
            return

        if not self.growth.validate_growth_task(growth_task, self.roadmap):
            logger.warning("Generated growth task was invalid")
            return

        logger.info(f"Generated growth task: {growth_task.id} - {growth_task.description}")

        if self.dry_run:
            logger.info("[DRY RUN] Would execute growth task")
            return

        # Add to roadmap and state
        self.parser.append_growth_task(growth_task)
        self.roadmap.phases[-1].tasks.append(growth_task)  # Add to last phase
        self.state.save_task(growth_task)

        # Execute the growth task
        self._execute_task(growth_task)

        # Update growth counter
        self.harness_state.growth_iterations += 1
        self.state.update_harness_state(growth_iterations=self.harness_state.growth_iterations)

    def _get_next_task(self) -> Optional[Task]:
        """Get the next task that is ready to execute."""
        completed_ids = self.state.get_completed_task_ids()

        for task in self.roadmap.all_tasks:
            if task.status == TaskStatus.PENDING:
                if task.can_execute(completed_ids):
                    return task
            elif task.status == TaskStatus.FAILED and task.attempts < task.max_retries:
                if task.can_execute(completed_ids):
                    return task

        return None

    def _execute_task(self, task: Task) -> None:
        """Execute a single task.

        If the task has ralph_config enabled, it will be executed via
        RalphExecutor which loops until promise is detected.
        """
        task.status = TaskStatus.IN_PROGRESS
        task.attempts += 1
        task.started_at = datetime.utcnow()
        self.state.save_task(task)

        # Get executor for this task (supports per-task CLI override)
        executor = self._get_executor_for_task(task)
        cli_info = f"{task.cli or self.default_cli}"
        if task.model:
            cli_info += f" ({task.model})"

        # Check if this is a ralph task
        if task.is_ralph_task:
            self._execute_ralph_task(task, executor, cli_info)
        else:
            self._execute_regular_task(task, executor, cli_info)

    def _execute_ralph_task(
        self,
        task: Task,
        base_executor: BaseCLIExecutor,
        cli_info: str,
    ) -> None:
        """Execute a task using the ralph loop."""
        config = task.ralph_config

        logger.info(
            f"Starting ralph loop execution via {cli_info} "
            f"(max {config.max_iterations} iterations, promise: {config.completion_promise})"
        )

        # Create ralph executor wrapping the base executor
        ralph_executor = RalphExecutor(
            base_executor=base_executor,
            max_iterations=config.max_iterations,
            completion_promise=config.completion_promise,
            persistence=self.ralph_persistence,
            require_xml_format=config.require_xml_format,
        )

        # Build context from dependencies
        context = self._build_task_context(task)

        # Execute via ralph loop
        result = ralph_executor.execute(task, context, resume=True)

        # Update task based on result
        task.output = result.final_output
        task.ralph_iterations = result.iterations

        if result.success:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            self.harness_state.tasks_completed += 1
            logger.info(
                f"Task {task.id} completed via ralph loop in {result.iterations} iterations"
            )

            # Update roadmap file
            self.parser.update_task_status(task.id, completed=True)
        else:
            task.error = result.error or result.reason
            if task.attempts >= task.max_retries:
                task.status = TaskStatus.BLOCKED
                self.harness_state.tasks_failed += 1
                logger.error(
                    f"Task {task.id} failed ralph loop after {result.iterations} iterations: "
                    f"{result.reason}"
                )
            else:
                task.status = TaskStatus.FAILED
                logger.warning(f"Task {task.id} failed ralph loop ({result.reason}), will retry")

        # Save state
        self.state.save_task(task)
        self.state.update_harness_state(
            tasks_completed=self.harness_state.tasks_completed,
            tasks_failed=self.harness_state.tasks_failed,
            execution_time=result.total_duration.total_seconds(),
        )

    def _execute_regular_task(
        self,
        task: Task,
        executor: BaseCLIExecutor,
        cli_info: str,
    ) -> None:
        """Execute a regular (non-ralph) task."""
        logger.info(
            f"Starting execution (attempt {task.attempts}/{task.max_retries}) via {cli_info}"
        )

        # Build context from dependencies
        context = self._build_task_context(task)

        # Execute via selected CLI
        result = executor.execute(task, context)

        # Log execution
        self.state.log_execution(task.id, task.attempts, result)

        # Update task based on result
        task.output = result.output
        task.error = result.error
        task.conversation_id = result.conversation_id

        if result.success:
            # Verify success criteria if defined
            if task.success_criteria and not executor.verify_success(task, result):
                logger.warning("Task output did not meet success criteria")
                result.success = False

        if result.success:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            self.harness_state.tasks_completed += 1
            logger.info(f"Task {task.id} completed successfully")

            # Update roadmap file
            self.parser.update_task_status(task.id, completed=True)
        else:
            if task.attempts >= task.max_retries:
                task.status = TaskStatus.BLOCKED
                self.harness_state.tasks_failed += 1
                logger.error(
                    f"Task {task.id} failed after {task.attempts} attempts: {result.error}"
                )
            else:
                task.status = TaskStatus.FAILED
                logger.warning(f"Task {task.id} failed, will retry. Error: {result.error}")

        # Save state
        self.state.save_task(task)
        self.state.update_harness_state(
            tasks_completed=self.harness_state.tasks_completed,
            tasks_failed=self.harness_state.tasks_failed,
            execution_time=result.duration_seconds,
        )

    def _build_task_context(self, task: Task) -> str:
        """Build context for a task from its dependencies."""
        context_parts = []

        # Add objective context
        context_parts.append(f"Project: {self.roadmap.name}")
        context_parts.append(f"Objective: {self.roadmap.objective}")

        # Add dependency outputs
        if task.depends:
            context_parts.append("\n## Previous Task Outputs:")
            for dep_id in task.depends:
                dep_task = self.roadmap.get_task(dep_id)
                if dep_task and dep_task.output:
                    context_parts.append(f"\n### {dep_id}:")
                    # Truncate long outputs
                    output = dep_task.output[:2000]
                    context_parts.append(output)

        return "\n".join(context_parts)

    def _print_summary(self) -> None:
        """Print execution summary."""
        stats = self.state.get_statistics()

        print("\n" + "=" * 50)
        print("HARNESS EXECUTION SUMMARY")
        print("=" * 50)
        print(f"Started: {stats['started_at']}")
        print(f"Last Activity: {stats['last_activity']}")
        print(f"Tasks Completed: {stats['tasks_completed']}")
        print(f"Tasks Failed: {stats['tasks_failed']}")
        print(f"Total Execution Time: {stats['total_execution_time']:.1f}s")
        print(f"Growth Mode: {stats['in_growth_mode']}")
        print(f"Growth Iterations: {stats['growth_iterations']}")
        print(f"Total Executions: {stats['total_executions']}")
        print("=" * 50)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="24/7 CLI Harness - Autonomous task execution via Codex, Gemini, or Claude"
    )
    parser.add_argument("--roadmap", "-r", required=True, help="Path to roadmap.md file")
    parser.add_argument(
        "--cli",
        "-c",
        default="codex",
        choices=["codex", "gemini", "claude"],
        help="CLI backend to use (default: codex)",
    )
    parser.add_argument(
        "--model", "-m", default=None, help="Model override (default: auto-selected based on CLI)"
    )
    parser.add_argument(
        "--reasoning",
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning effort level for Codex (default: high)",
    )
    parser.add_argument(
        "--cooldown", type=int, default=10, help="Seconds between iterations (default: 10)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be done without executing"
    )
    parser.add_argument("--daemon", action="store_true", help="Run as daemon (implies 24/7 mode)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--reset", action="store_true", help="Reset all state and start fresh")
    parser.add_argument("--status", action="store_true", help="Print current status and exit")
    parser.add_argument(
        "--parallel", action="store_true", help="Enable parallel execution of independent tasks"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="Maximum number of tasks to run in parallel (default: 4)",
    )
    parser.add_argument(
        "--ralph-default-iterations",
        type=int,
        default=30,
        help="Default max iterations for ralph loop tasks (default: 30)",
    )
    parser.add_argument(
        "--ralph-default-promise",
        type=str,
        default="COMPLETE",
        help="Default completion promise for ralph loop tasks (default: COMPLETE)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure .harness directory exists
    Path(".harness").mkdir(exist_ok=True)

    if args.status:
        state = StateManager()
        stats = state.get_statistics()
        import json

        print(json.dumps(stats, indent=2, default=str))
        return

    if args.reset:
        state = StateManager()
        state.reset()
        print("State reset complete")
        return

    harness = Harness(
        roadmap_path=args.roadmap,
        cli=args.cli,
        model=args.model,
        reasoning_effort=args.reasoning,
        cooldown_seconds=args.cooldown,
        dry_run=args.dry_run,
        parallel=args.parallel,
        max_parallel_tasks=args.max_parallel,
        ralph_default_iterations=args.ralph_default_iterations,
        ralph_default_promise=args.ralph_default_promise,
    )

    harness.run()


if __name__ == "__main__":
    main()
