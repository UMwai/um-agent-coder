"""
Main harness loop for 24/7 Codex execution.

Usage:
    python -m src.um_agent_coder.harness --roadmap specs/roadmap.md
    python -m src.um_agent_coder.harness --roadmap specs/roadmap.md --daemon
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import Task, TaskStatus, Roadmap, HarnessState
from .roadmap_parser import RoadmapParser
from .codex_executor import CodexExecutor
from .state import StateManager
from .growth import GrowthLoop

# Ensure .harness directory exists for logging
Path(".harness").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(".harness/harness.log"),
    ]
)
logger = logging.getLogger(__name__)


class Harness:
    """Main 24/7 Codex execution harness."""

    def __init__(
        self,
        roadmap_path: str,
        model: str = "gpt-5.2",
        reasoning_effort: str = "high",
        cooldown_seconds: int = 10,
        dry_run: bool = False,
    ):
        self.roadmap_path = roadmap_path
        self.dry_run = dry_run
        self.cooldown_seconds = cooldown_seconds

        # Initialize components
        self.parser = RoadmapParser(roadmap_path)
        self.executor = CodexExecutor(
            model=model,
            reasoning_effort=reasoning_effort,
        )
        self.state = StateManager()
        self.growth = GrowthLoop(self.executor)

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

    def run(self) -> None:
        """Main execution loop."""
        logger.info(f"Starting harness with roadmap: {self.roadmap_path}")

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
        logger.info(f"Tasks: {len(self.roadmap.all_tasks)} across {len(self.roadmap.phases)} phases")

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
            logger.info("[DRY RUN] Would execute task")
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
        self.state.update_harness_state(
            growth_iterations=self.harness_state.growth_iterations
        )

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
        """Execute a single task."""
        task.status = TaskStatus.IN_PROGRESS
        task.attempts += 1
        task.started_at = datetime.utcnow()
        self.state.save_task(task)

        logger.info(f"Starting execution (attempt {task.attempts}/{task.max_retries})")

        # Build context from dependencies
        context = self._build_task_context(task)

        # Execute via Codex
        result = self.executor.execute(task, context)

        # Log execution
        self.state.log_execution(task.id, task.attempts, result)

        # Update task based on result
        task.output = result.output
        task.error = result.error
        task.conversation_id = result.conversation_id

        if result.success:
            # Verify success criteria if defined
            if task.success_criteria and not self.executor.verify_success(task, result):
                logger.warning(f"Task output did not meet success criteria")
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
                logger.error(f"Task {task.id} failed after {task.attempts} attempts: {result.error}")
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
        description="24/7 Codex Harness - Autonomous task execution"
    )
    parser.add_argument(
        "--roadmap", "-r",
        required=True,
        help="Path to roadmap.md file"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-5.2",
        help="Codex model to use (default: gpt-5.2)"
    )
    parser.add_argument(
        "--reasoning",
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning effort level (default: high)"
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=10,
        help="Seconds between iterations (default: 10)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon (implies 24/7 mode)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all state and start fresh"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current status and exit"
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
        model=args.model,
        reasoning_effort=args.reasoning,
        cooldown_seconds=args.cooldown,
        dry_run=args.dry_run,
    )

    harness.run()


if __name__ == "__main__":
    main()
