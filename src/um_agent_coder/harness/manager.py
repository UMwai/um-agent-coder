"""
HarnessManager for spawning and coordinating sub-harnesses.

The core of the meta-harness system - manages sub-harness lifecycle,
coordination, and result aggregation.
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .handle import HarnessHandle
from .meta_state import MetaStateManager
from .result import AggregatedResult, HarnessResult, HarnessStatus
from .worktree_manager import MergeStatus, WorktreeManager

logger = logging.getLogger(__name__)


@dataclass
class MetaHarnessConfig:
    """Configuration for meta-harness."""

    max_concurrent: int = 10
    poll_interval_seconds: float = 5.0
    fail_fast: bool = False
    retry_failed: bool = True
    max_retries: int = 2
    default_timeout_minutes: int = 60
    # Worktree isolation settings
    use_worktrees: bool = False
    worktree_base_branch: str = "main"
    auto_merge: bool = True
    run_tests_before_merge: bool = True


class HarnessManager:
    """
    Spawns, monitors, and coordinates sub-harnesses.

    Example usage:
        manager = HarnessManager()
        handle = manager.spawn_harness(
            harness_id="auth",
            roadmap=Path("auth/roadmap.md"),
            working_dir=Path("./auth"),
            cli="codex"
        )
        results = manager.wait_for([handle])
    """

    def __init__(
        self,
        config: Optional[MetaHarnessConfig] = None,
        harness_dir: Path = Path(".harness"),
    ):
        self.config = config or MetaHarnessConfig()
        self.harness_dir = harness_dir
        self.handles: Dict[str, HarnessHandle] = {}
        self._lock = threading.Lock()
        self._stop_requested = False

        # Ensure harness directory exists
        self.harness_dir.mkdir(parents=True, exist_ok=True)

        # Initialize meta state manager for tracking all sub-harnesses
        self.meta_state = MetaStateManager(
            db_path=str(self.harness_dir / "meta_state.db")
        )

        # Initialize worktree manager if worktree isolation is enabled
        self.worktree_manager: Optional[WorktreeManager] = None
        if self.config.use_worktrees:
            try:
                self.worktree_manager = WorktreeManager()
                logger.info("Worktree isolation enabled")
            except ValueError as e:
                logger.warning(f"Could not initialize worktree manager: {e}")

    def spawn_harness(
        self,
        harness_id: str,
        roadmap: Path,
        working_dir: Optional[Path] = None,
        cli: str = "auto",
        model: str = "",
        parent_context: Optional[Dict[str, Any]] = None,
        env: Optional[Dict[str, str]] = None,
        use_worktree: Optional[bool] = None,
        worktree_base_branch: Optional[str] = None,
        auto_merge: Optional[bool] = None,
    ) -> HarnessHandle:
        """
        Spawn an independent sub-harness.

        Args:
            harness_id: Unique identifier for this sub-harness
            roadmap: Path to the roadmap.md file
            working_dir: Working directory for execution (defaults to current)
            cli: CLI backend (codex, gemini, claude, auto)
            model: Model override
            parent_context: Context passed from parent
            env: Additional environment variables
            use_worktree: Create isolated git worktree (overrides config)
            worktree_base_branch: Base branch for worktree (overrides config)
            auto_merge: Auto-merge to main on completion (overrides config)

        Returns:
            HarnessHandle for async control
        """
        working_dir = working_dir or Path.cwd()
        working_dir = working_dir.resolve()

        # Determine worktree settings
        should_use_worktree = use_worktree if use_worktree is not None else self.config.use_worktrees
        base_branch = worktree_base_branch or self.config.worktree_base_branch
        should_auto_merge = auto_merge if auto_merge is not None else self.config.auto_merge

        # Create worktree if enabled
        worktree_info = None
        if should_use_worktree and self.worktree_manager:
            try:
                worktree_info = self.worktree_manager.create_worktree(
                    harness_id=harness_id,
                    base_branch=base_branch,
                )
                working_dir = worktree_info.path
                logger.info(f"Created worktree for {harness_id} at {working_dir}")

                # Register worktree in meta state
                self.meta_state.register_worktree(
                    harness_id=harness_id,
                    branch_name=worktree_info.branch_name,
                    worktree_path=str(worktree_info.path),
                    base_branch=base_branch,
                )

                # Add worktree info to parent context
                if parent_context is None:
                    parent_context = {}
                parent_context["worktree"] = {
                    "branch": worktree_info.branch_name,
                    "path": str(worktree_info.path),
                    "base_branch": base_branch,
                    "auto_merge": should_auto_merge,
                }
            except Exception as e:
                logger.error(f"Failed to create worktree for {harness_id}: {e}")
                # Continue without worktree

        # Setup isolated state directory
        state_dir = self.harness_dir / harness_id
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "inbox").mkdir(exist_ok=True)

        # Write parent context if provided
        if parent_context:
            context_file = state_dir / "parent_context.json"
            context_file.write_text(json.dumps(parent_context, indent=2))

        # Build environment
        spawn_env = os.environ.copy()
        spawn_env["HARNESS_STATE_DIR"] = str(state_dir)
        spawn_env["HARNESS_ID"] = harness_id
        spawn_env["HARNESS_SUBPROCESS"] = "1"
        if env:
            spawn_env.update(env)

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "src.um_agent_coder.harness",
            "--roadmap",
            str(roadmap.resolve()),
            "--subprocess",
            "--harness-id",
            harness_id,
        ]

        if cli and cli != "auto":
            cmd.extend(["--cli", cli])
        if model:
            cmd.extend(["--model", model])

        logger.info(f"Spawning sub-harness {harness_id}: {' '.join(cmd)}")

        # Spawn subprocess
        log_file = state_dir / "harness.log"
        with open(log_file, "w") as log_f:
            process = subprocess.Popen(
                cmd,
                cwd=working_dir,
                env=spawn_env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        # Create handle
        handle = HarnessHandle(
            harness_id=harness_id,
            pid=process.pid,
            working_dir=working_dir,
            roadmap_path=roadmap,
            cli=cli,
            model=model,
            state_dir=state_dir,
            state_db=state_dir / "state.db",
            log_file=log_file,
            inbox_dir=state_dir / "inbox",
            status=HarnessStatus.RUNNING,
            started_at=datetime.now(),
        )
        handle._process = process

        # Track handle
        with self._lock:
            self.handles[harness_id] = handle

        # Register with meta state
        self.meta_state.register_sub_harness(
            harness_id=harness_id,
            roadmap_path=str(roadmap.resolve()),
            working_dir=str(working_dir),
            cli=cli,
            model=model,
            pid=process.pid,
            parent_context=parent_context,
        )
        self.meta_state.update_harness_started(harness_id, process.pid)

        logger.info(f"Sub-harness {harness_id} spawned with PID {process.pid}")
        return handle

    def wait_for(
        self,
        handles: List[HarnessHandle],
        timeout: Optional[timedelta] = None,
        callback: Optional[Callable[[HarnessHandle], None]] = None,
    ) -> List[HarnessResult]:
        """
        Wait for specified harnesses to complete.

        Args:
            handles: List of HarnessHandles to wait for
            timeout: Optional timeout (None = wait forever)
            callback: Optional callback when each harness completes

        Returns:
            List of HarnessResult for each handle
        """
        start_time = datetime.now()
        results: Dict[str, HarnessResult] = {}
        pending = set(h.harness_id for h in handles)

        while pending and not self._stop_requested:
            # Check timeout
            if timeout and (datetime.now() - start_time) > timeout:
                logger.warning(f"Timeout waiting for harnesses: {pending}")
                break

            # Check each pending harness
            for harness_id in list(pending):
                handle = self.handles.get(harness_id)
                if handle and handle.is_complete():
                    result = handle.get_result()
                    results[harness_id] = result
                    pending.remove(harness_id)
                    logger.info(
                        f"Harness {harness_id} completed with status {result.status.value}"
                    )

                    # Update meta state
                    self.meta_state.update_harness_completed(
                        harness_id=harness_id,
                        success=result.success,
                        error=result.error,
                    )

                    # Handle worktree merge on success
                    if result.success and self.worktree_manager:
                        worktree_record = self.meta_state.get_worktree(harness_id)
                        if worktree_record:
                            should_merge = self.config.auto_merge
                            # Check if parent context has auto_merge override
                            parent_ctx = self.meta_state.get_harness(harness_id)
                            if parent_ctx:
                                try:
                                    ctx = json.loads(parent_ctx.get("parent_context", "{}"))
                                    if "worktree" in ctx:
                                        should_merge = ctx["worktree"].get("auto_merge", should_merge)
                                except json.JSONDecodeError:
                                    pass

                            if should_merge:
                                self._merge_worktree(harness_id)

                    if callback:
                        callback(handle)

            if pending:
                time.sleep(self.config.poll_interval_seconds)

        # Build final results in handle order
        return [
            results.get(h.harness_id, h.get_result())
            for h in handles
        ]

    def wait_for_any(
        self,
        handles: List[HarnessHandle],
        timeout: Optional[timedelta] = None,
    ) -> Tuple[HarnessHandle, HarnessResult]:
        """
        Wait for first harness to complete.

        Returns:
            Tuple of (HarnessHandle, HarnessResult) for first to complete
        """
        start_time = datetime.now()

        while not self._stop_requested:
            # Check timeout
            if timeout and (datetime.now() - start_time) > timeout:
                raise TimeoutError("Timeout waiting for any harness")

            # Check each harness
            for handle in handles:
                if handle.is_complete():
                    return handle, handle.get_result()

            time.sleep(self.config.poll_interval_seconds)

        raise InterruptedError("Stop requested")

    def coordinate(
        self,
        handles: List[HarnessHandle],
        strategy: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> AggregatedResult:
        """
        Coordinate harnesses using specified strategy.

        Args:
            handles: List of HarnessHandles to coordinate
            strategy: Strategy name (parallel, pipeline, race, voting)
            config: Strategy-specific configuration

        Returns:
            AggregatedResult with all results
        """
        config = config or {}
        started_at = datetime.now()

        if strategy == "parallel":
            results = self._coordinate_parallel(handles, config)
        elif strategy == "pipeline":
            results = self._coordinate_pipeline(handles, config)
        elif strategy == "race":
            results = self._coordinate_race(handles, config)
        elif strategy == "voting":
            results = self._coordinate_voting(handles, config)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        completed_at = datetime.now()

        return AggregatedResult(
            strategy=strategy,
            success=all(r.success for r in results) if strategy != "race" else any(r.success for r in results),
            results=results,
            winner=results[0] if strategy in ("race", "voting") and results else None,
            started_at=started_at,
            completed_at=completed_at,
        )

    def _coordinate_parallel(
        self,
        handles: List[HarnessHandle],
        config: Dict[str, Any],
    ) -> List[HarnessResult]:
        """Run all harnesses in parallel, wait for all."""
        fail_fast = config.get("fail_fast", self.config.fail_fast)
        results: List[HarnessResult] = []

        def on_complete(handle: HarnessHandle):
            result = handle.get_result()
            if fail_fast and not result.success:
                logger.warning(f"Fail-fast: {handle.harness_id} failed, stopping others")
                self.request_stop_all()

        return self.wait_for(handles, callback=on_complete)

    def _coordinate_pipeline(
        self,
        handles: List[HarnessHandle],
        config: Dict[str, Any],
    ) -> List[HarnessResult]:
        """Execute sequentially, pass context between stages."""
        results: List[HarnessResult] = []
        context: Dict[str, Any] = config.get("initial_context", {})

        for i, handle in enumerate(handles):
            # Wait for this stage
            stage_results = self.wait_for([handle])
            result = stage_results[0] if stage_results else handle.get_result()
            results.append(result)

            if not result.success:
                logger.warning(f"Pipeline stage {i} ({handle.harness_id}) failed, stopping")
                break

            # Pass context to next stage
            if i < len(handles) - 1:
                next_handle = handles[i + 1]
                context["previous_stage"] = {
                    "harness_id": handle.harness_id,
                    "output": result.final_output,
                    "metrics": {
                        "tasks_completed": result.tasks_completed,
                        "tasks_failed": result.tasks_failed,
                    },
                }
                next_handle.send_instruction(
                    f"Context from previous stage:\n{json.dumps(context, indent=2)}"
                )

        return results

    def _coordinate_race(
        self,
        handles: List[HarnessHandle],
        config: Dict[str, Any],
    ) -> List[HarnessResult]:
        """First to complete wins, terminate others."""
        min_progress = config.get("min_progress_to_win", 0.8)
        terminate_losers = config.get("terminate_losers", True)

        # Wait for first to complete
        winner, winner_result = self.wait_for_any(handles)

        # Check if winner meets criteria
        if winner_result.progress < min_progress and winner_result.success:
            logger.warning(
                f"Winner {winner.harness_id} has progress {winner_result.progress} < {min_progress}"
            )

        results = [winner_result]

        # Terminate losers
        if terminate_losers:
            for handle in handles:
                if handle.harness_id != winner.harness_id:
                    logger.info(f"Terminating loser: {handle.harness_id}")
                    handle.request_stop()
                    results.append(handle.get_result())

        return results

    def _coordinate_voting(
        self,
        handles: List[HarnessHandle],
        config: Dict[str, Any],
    ) -> List[HarnessResult]:
        """Multiple complete, pick best by criteria."""
        min_votes = config.get("min_votes", 2)
        selection_criteria = config.get("selection_criteria", "first")

        # Wait for minimum number to complete
        completed: List[HarnessResult] = []
        pending = list(handles)

        while len(completed) < min_votes and pending:
            for handle in list(pending):
                if handle.is_complete():
                    result = handle.get_result()
                    if result.success:
                        completed.append(result)
                    pending.remove(handle)

            if len(completed) < min_votes and pending:
                time.sleep(self.config.poll_interval_seconds)

        if not completed:
            # All failed
            return [h.get_result() for h in handles]

        # Select winner based on criteria
        if selection_criteria == "first":
            winner = completed[0]
        elif selection_criteria == "best_progress":
            winner = max(completed, key=lambda r: r.progress)
        elif selection_criteria == "best_tests":
            winner = max(completed, key=lambda r: r.metrics.tests_passed)
        else:
            winner = completed[0]

        # Put winner first
        results = [winner] + [r for r in completed if r != winner]
        return results

    def broadcast_instruction(self, instruction: str) -> None:
        """Send instruction to all running sub-harnesses."""
        with self._lock:
            for handle in self.handles.values():
                if handle.status == HarnessStatus.RUNNING:
                    handle.send_instruction(instruction)

    def request_stop_all(self) -> None:
        """Request graceful stop of all sub-harnesses."""
        self._stop_requested = True
        with self._lock:
            for handle in self.handles.values():
                if handle.status == HarnessStatus.RUNNING:
                    handle.request_stop()

    def get_status(self) -> Dict[str, Any]:
        """Get status of all harnesses."""
        with self._lock:
            statuses = {}
            for harness_id, handle in self.handles.items():
                handle.refresh()
                statuses[harness_id] = handle.to_dict()

                # Update meta state with latest progress
                self.meta_state.update_progress(
                    harness_id=harness_id,
                    progress=handle.progress,
                    current_task=handle.current_task,
                    current_iteration=handle.current_iteration,
                    tasks_completed=handle.tasks_completed,
                    tasks_failed=handle.tasks_failed,
                )

            return statuses

    def get_meta_status(self) -> Dict[str, Any]:
        """Get meta-harness overall status from database."""
        meta = self.meta_state.get_meta_state()
        harnesses = self.meta_state.get_all_harnesses()
        return {
            "meta": meta,
            "harnesses": harnesses,
            "running_count": len(self.meta_state.get_running_harnesses()),
            "pending_count": len(self.meta_state.get_pending_harnesses()),
        }

    def get_handle(self, harness_id: str) -> Optional[HarnessHandle]:
        """Get handle by ID."""
        with self._lock:
            return self.handles.get(harness_id)

    def _merge_worktree(self, harness_id: str) -> bool:
        """Merge a worktree back to main branch.

        Args:
            harness_id: The harness identifier

        Returns:
            True if merge was successful
        """
        if not self.worktree_manager:
            return False

        logger.info(f"Merging worktree for {harness_id}...")

        merge_result = self.worktree_manager.merge_to_main(
            harness_id=harness_id,
            run_tests=self.config.run_tests_before_merge,
        )

        if merge_result.success:
            # Update meta state with merge info
            self.meta_state.update_worktree_merged(
                harness_id=harness_id,
                merge_commit=merge_result.commit_sha or "",
            )
            logger.info(f"Successfully merged {harness_id}: {merge_result.commit_sha}")
            return True
        else:
            logger.error(
                f"Failed to merge {harness_id}: {merge_result.status.value} - {merge_result.error}"
            )
            return False

    def cleanup_worktree(self, harness_id: str, force: bool = False) -> bool:
        """Cleanup a worktree after completion.

        Args:
            harness_id: The harness identifier
            force: Force cleanup even with uncommitted changes

        Returns:
            True if cleanup was successful
        """
        if not self.worktree_manager:
            return True

        success = self.worktree_manager.cleanup_worktree(harness_id, force=force)
        if success:
            self.meta_state.delete_worktree(harness_id)
        return success

    def cleanup(self) -> None:
        """Stop all harnesses and cleanup."""
        self.request_stop_all()
        time.sleep(1)

        # Force kill any still running
        with self._lock:
            for handle in self.handles.values():
                if not handle.is_complete():
                    handle.force_kill()

        # Cleanup worktrees
        if self.worktree_manager:
            for worktree in self.meta_state.get_active_worktrees():
                harness_id = worktree.get("harness_id")
                if harness_id:
                    logger.info(f"Cleaning up worktree for {harness_id}")
                    self.cleanup_worktree(harness_id, force=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
