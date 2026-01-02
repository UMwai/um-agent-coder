"""
Parallel Executor - Spawns subagents for concurrent task execution.

This module enables:
1. Parallel execution of independent subtasks
2. Subagent spawning (each model runs in its own process/thread)
3. Dependency-aware scheduling
4. Result aggregation and flow
"""

import asyncio
import concurrent.futures
import json
import os
import subprocess
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Optional

from .claude_subagent import ClaudeCodeSubagentSpawner, SubagentType
from .task_decomposer import DecomposedTask, ModelRole, SubTask


class ExecutionMode(Enum):
    """How to execute subtasks."""

    SEQUENTIAL = "sequential"  # One at a time
    PARALLEL_THREADS = "threads"  # ThreadPoolExecutor
    PARALLEL_ASYNC = "async"  # asyncio
    SUBAGENT_SPAWN = "subagent"  # Spawn separate processes (legacy)
    CLAUDE_CODE_SPAWN = "claude_code"  # Spawn using ClaudeCodeSubagentSpawner


@dataclass
class SubagentResult:
    """Result from a subagent execution."""

    subtask_id: str
    success: bool
    output: Any
    error: Optional[str] = None
    started_at: str = ""
    completed_at: str = ""
    model_used: str = ""
    tokens_used: int = 0


@dataclass
class ExecutionGraph:
    """
    Dependency graph for parallel execution.

    Tracks which tasks can run in parallel vs. which must wait.
    """

    subtasks: dict[str, SubTask]
    dependencies: dict[str, set[str]]  # task_id -> set of task_ids it depends on
    dependents: dict[str, set[str]]  # task_id -> set of task_ids that depend on it

    @classmethod
    def from_decomposed_task(cls, task: DecomposedTask) -> "ExecutionGraph":
        """Build execution graph from decomposed task."""
        subtasks = {st.id: st for st in task.subtasks}
        dependencies = {st.id: set(st.depends_on) for st in task.subtasks}

        # Build reverse dependency map
        dependents: dict[str, set[str]] = {st.id: set() for st in task.subtasks}
        for task_id, deps in dependencies.items():
            for dep in deps:
                if dep in dependents:
                    dependents[dep].add(task_id)

        return cls(subtasks=subtasks, dependencies=dependencies, dependents=dependents)

    def get_ready_tasks(self, completed: set[str]) -> list[str]:
        """Get tasks that are ready to execute (all dependencies met)."""
        ready = []
        for task_id, deps in self.dependencies.items():
            if task_id not in completed and deps.issubset(completed):
                ready.append(task_id)
        return ready

    def get_parallel_groups(self) -> list[list[str]]:
        """
        Get tasks grouped by execution level.

        Level 0: Tasks with no dependencies (can all run in parallel)
        Level 1: Tasks that depend only on level 0 (can run in parallel after level 0)
        etc.
        """
        completed: set[str] = set()
        groups: list[list[str]] = []
        remaining = set(self.subtasks.keys())

        while remaining:
            ready = self.get_ready_tasks(completed)
            if not ready:
                # Circular dependency or error
                break

            groups.append(ready)
            completed.update(ready)
            remaining -= set(ready)

        return groups


class ParallelExecutor:
    """
    Executes subtasks in parallel using subagents.

    Features:
    - Spawns independent subagents for each model type
    - Respects dependencies (waits for upstream tasks)
    - Aggregates results and passes between tasks
    - Supports human-in-the-loop pausing

    Usage:
        executor = ParallelExecutor(
            gemini_llm=gemini,
            codex_llm=codex,
            claude_llm=claude
        )

        results = executor.execute(decomposed_task)
    """

    def __init__(
        self,
        gemini_llm=None,
        codex_llm=None,
        claude_llm=None,
        max_workers: int = 4,
        execution_mode: ExecutionMode = ExecutionMode.PARALLEL_THREADS,
        checkpoint_dir: str = ".parallel_checkpoints",
        human_approval_callback: Optional[Callable] = None,
        verbose: bool = True,
        use_claude_code_spawner: bool = True,
        claude_spawner_fallback: bool = True,
    ):
        self.models = {
            ModelRole.GEMINI: gemini_llm,
            ModelRole.CODEX: codex_llm,
            ModelRole.CLAUDE: claude_llm,
        }

        self.max_workers = max_workers
        self.execution_mode = execution_mode
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.human_approval_callback = human_approval_callback
        self.verbose = verbose

        # Claude Code subagent spawner
        self.use_claude_code_spawner = use_claude_code_spawner
        self.claude_spawner = (
            ClaudeCodeSubagentSpawner(
                use_task_tool=True,
                fallback_to_subprocess=claude_spawner_fallback,
                verbose=verbose,
                checkpoint_dir=str(self.checkpoint_dir / "claude_subagents"),
            )
            if use_claude_code_spawner
            else None
        )

        # Execution state
        self.results: dict[str, SubagentResult] = {}
        self.lock = threading.Lock()

        # Human-in-the-loop control
        self.pause_requested = False
        self.approval_queue: Queue = Queue()

    def execute(
        self,
        task: DecomposedTask,
        task_id: Optional[str] = None,
        require_approval_at: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Execute all subtasks with parallelization.

        Args:
            task: Decomposed task with subtasks
            task_id: Optional task ID for tracking
            require_approval_at: List of subtask IDs that require human approval

        Returns:
            Dict with all results and metadata
        """
        task_id = task_id or str(uuid.uuid4())[:8]
        require_approval_at = require_approval_at or []

        # Build execution graph
        graph = ExecutionGraph.from_decomposed_task(task)
        parallel_groups = graph.get_parallel_groups()

        if self.verbose:
            print(f"\n{'='*60}")
            print("PARALLEL EXECUTOR")
            print(f"{'='*60}")
            print(f"Task ID: {task_id}")
            print(f"Total subtasks: {len(task.subtasks)}")
            print(f"Parallel groups: {len(parallel_groups)}")
            for i, group in enumerate(parallel_groups):
                print(f"  Level {i}: {group} (parallel)")
            print(f"{'='*60}\n")

        # Execute by level
        completed: set[str] = set()
        all_results: dict[str, SubagentResult] = {}

        for level, group in enumerate(parallel_groups):
            if self.verbose:
                print(f"\n[Level {level}] Executing {len(group)} tasks in parallel...")

            # Check for human approval requirements
            approval_needed = [t for t in group if t in require_approval_at]
            if approval_needed:
                if not self._request_human_approval(task_id, approval_needed, all_results):
                    return {
                        "success": False,
                        "task_id": task_id,
                        "error": "Human approval denied",
                        "completed_tasks": list(completed),
                        "results": {k: v.output for k, v in all_results.items()},
                    }

            # Execute group in parallel
            group_results = self._execute_parallel_group(
                group, graph.subtasks, all_results, task_id
            )

            # Collect results
            for subtask_id, result in group_results.items():
                all_results[subtask_id] = result
                if result.success:
                    completed.add(subtask_id)
                else:
                    # Fail fast on error
                    return {
                        "success": False,
                        "task_id": task_id,
                        "error": f"Subtask {subtask_id} failed: {result.error}",
                        "completed_tasks": list(completed),
                        "results": {k: v.output for k, v in all_results.items()},
                    }

            # Checkpoint after each level
            self._save_checkpoint(task_id, all_results, completed)

        # Get final output (from last task in execution order)
        final_task_id = task.execution_order[-1] if task.execution_order else None
        final_output = all_results.get(
            final_task_id, SubagentResult(subtask_id="", success=False, output=None)
        ).output

        return {
            "success": True,
            "task_id": task_id,
            "output": final_output,
            "all_results": {k: v.output for k, v in all_results.items()},
            "execution_summary": {
                "total_tasks": len(task.subtasks),
                "completed": len(completed),
                "parallel_levels": len(parallel_groups),
            },
        }

    def _execute_parallel_group(
        self,
        group: list[str],
        subtasks: dict[str, SubTask],
        prior_results: dict[str, SubagentResult],
        task_id: str,
    ) -> dict[str, SubagentResult]:
        """Execute a group of independent tasks in parallel."""

        if self.execution_mode == ExecutionMode.SEQUENTIAL:
            return self._execute_sequential(group, subtasks, prior_results)

        elif self.execution_mode == ExecutionMode.PARALLEL_THREADS:
            return self._execute_threaded(group, subtasks, prior_results)

        elif self.execution_mode == ExecutionMode.PARALLEL_ASYNC:
            return asyncio.run(self._execute_async(group, subtasks, prior_results))

        elif self.execution_mode == ExecutionMode.CLAUDE_CODE_SPAWN:
            return self._execute_claude_code_subagents(group, subtasks, prior_results, task_id)

        elif self.execution_mode == ExecutionMode.SUBAGENT_SPAWN:
            return self._execute_subagents(group, subtasks, prior_results, task_id)

        else:
            return self._execute_sequential(group, subtasks, prior_results)

    def _execute_sequential(
        self,
        group: list[str],
        subtasks: dict[str, SubTask],
        prior_results: dict[str, SubagentResult],
    ) -> dict[str, SubagentResult]:
        """Execute tasks one at a time."""
        results = {}
        for task_id in group:
            subtask = subtasks[task_id]
            results[task_id] = self._execute_single_task(subtask, prior_results)
        return results

    def _execute_threaded(
        self,
        group: list[str],
        subtasks: dict[str, SubTask],
        prior_results: dict[str, SubagentResult],
    ) -> dict[str, SubagentResult]:
        """Execute tasks in parallel using threads."""
        results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self._execute_single_task, subtasks[task_id], prior_results
                ): task_id
                for task_id in group
            }

            for future in concurrent.futures.as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    results[task_id] = future.result()
                except Exception as e:
                    results[task_id] = SubagentResult(
                        subtask_id=task_id, success=False, output=None, error=str(e)
                    )

        return results

    async def _execute_async(
        self,
        group: list[str],
        subtasks: dict[str, SubTask],
        prior_results: dict[str, SubagentResult],
    ) -> dict[str, SubagentResult]:
        """Execute tasks in parallel using asyncio."""

        async def run_task(task_id: str) -> tuple:
            # Run in thread pool since LLM calls are blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._execute_single_task, subtasks[task_id], prior_results
            )
            return task_id, result

        tasks = [run_task(task_id) for task_id in group]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for item in completed:
            if isinstance(item, Exception):
                continue
            task_id, result = item
            results[task_id] = result

        return results

    def _execute_claude_code_subagents(
        self,
        group: list[str],
        subtasks: dict[str, SubTask],
        prior_results: dict[str, SubagentResult],
        parent_task_id: str,
    ) -> dict[str, SubagentResult]:
        """
        Spawn subagents using ClaudeCodeSubagentSpawner.

        This method uses the enhanced Claude Code spawner which:
        1. Tries to use Claude Code's Task tool when available
        2. Falls back to subprocess execution
        3. Provides better prompt formatting for different agent types
        """
        if not self.claude_spawner:
            # Fallback to legacy subprocess method
            return self._execute_subagents(group, subtasks, prior_results, parent_task_id)

        results = {}

        # Map ModelRole to SubagentType
        model_to_subagent_type = {
            ModelRole.GEMINI: SubagentType.EXPLORE,
            ModelRole.CODEX: SubagentType.GENERIC,
            ModelRole.CLAUDE: SubagentType.ARCHITECT,
        }

        # Spawn all subagents in parallel using threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {}

            for task_id in group:
                subtask = subtasks[task_id]

                # Prepare input data
                input_data = self._gather_inputs(subtask, prior_results)

                # Determine subagent type
                subagent_type = model_to_subagent_type.get(subtask.model, SubagentType.GENERIC)

                # Build context
                context = {
                    "subtask_id": task_id,
                    "description": subtask.description,
                    "task_type": subtask.type.value,
                    "inputs": input_data,
                }

                if self.verbose:
                    print(f"    Spawning {subagent_type.value} agent for {task_id}...")

                # Submit task to thread pool
                future = executor.submit(
                    self.claude_spawner.spawn_task,
                    prompt=subtask.prompt,
                    subagent_type=subagent_type,
                    context=context,
                    timeout=600,
                )
                future_to_task[future] = task_id

            # Collect results
            for future in concurrent.futures.as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    claude_result = future.result()

                    # Convert ClaudeCodeSubagentSpawner result to ParallelExecutor result
                    results[task_id] = SubagentResult(
                        subtask_id=task_id,
                        success=claude_result.success,
                        output=claude_result.output,
                        error=claude_result.error,
                        started_at=claude_result.started_at,
                        completed_at=claude_result.completed_at,
                        model_used=subtasks[task_id].model.value,
                    )

                    if self.verbose:
                        status = "✓" if claude_result.success else "✗"
                        print(
                            f"    {status} {task_id} completed in {claude_result.duration_seconds:.2f}s"
                        )

                except Exception as e:
                    results[task_id] = SubagentResult(
                        subtask_id=task_id,
                        success=False,
                        output=None,
                        error=str(e),
                        started_at=datetime.now().isoformat(),
                        completed_at=datetime.now().isoformat(),
                    )

                    if self.verbose:
                        print(f"    ✗ {task_id} failed: {e}")

        return results

    def _execute_subagents(
        self,
        group: list[str],
        subtasks: dict[str, SubTask],
        prior_results: dict[str, SubagentResult],
        parent_task_id: str,
    ) -> dict[str, SubagentResult]:
        """
        Spawn separate subagent processes for each task (legacy method).

        This is the most isolated approach - each model runs in its own process.
        """
        results = {}
        processes = {}

        # Spawn subagents
        for task_id in group:
            subtask = subtasks[task_id]

            # Prepare input data
            input_data = self._gather_inputs(subtask, prior_results)

            # Create subagent script
            script_path = self._create_subagent_script(parent_task_id, subtask, input_data)

            if self.verbose:
                print(f"    Spawning subagent for {task_id} [{subtask.model.value}]...")

            # Spawn process
            proc = subprocess.Popen(
                ["python3", str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd(),
            )
            processes[task_id] = (proc, script_path, datetime.now().isoformat())

        # Wait for all to complete
        for task_id, (proc, script_path, started_at) in processes.items():
            try:
                stdout, stderr = proc.communicate(timeout=600)
                completed_at = datetime.now().isoformat()

                if proc.returncode == 0:
                    # Parse output
                    output = stdout.decode("utf-8").strip()
                    results[task_id] = SubagentResult(
                        subtask_id=task_id,
                        success=True,
                        output=output,
                        started_at=started_at,
                        completed_at=completed_at,
                        model_used=subtasks[task_id].model.value,
                    )
                    if self.verbose:
                        print(f"    ✓ {task_id} completed")
                else:
                    results[task_id] = SubagentResult(
                        subtask_id=task_id,
                        success=False,
                        output=None,
                        error=stderr.decode("utf-8"),
                        started_at=started_at,
                        completed_at=completed_at,
                    )
                    if self.verbose:
                        print(f"    ✗ {task_id} failed")

            except subprocess.TimeoutExpired:
                proc.kill()
                results[task_id] = SubagentResult(
                    subtask_id=task_id, success=False, output=None, error="Timeout expired"
                )

            # Cleanup script
            try:
                script_path.unlink()
            except Exception:
                pass

        return results

    def _execute_single_task(
        self, subtask: SubTask, prior_results: dict[str, SubagentResult]
    ) -> SubagentResult:
        """Execute a single subtask."""
        started_at = datetime.now().isoformat()

        try:
            # Get the right model
            model = self.models.get(subtask.model)
            if not model:
                raise ValueError(f"No model configured for {subtask.model.value}")

            # Gather inputs from dependencies
            input_data = self._gather_inputs(subtask, prior_results)

            # Build prompt with context
            prompt = subtask.prompt
            if input_data:
                context = "\n\n".join(
                    [f"=== Input from {k} ===\n{v}" for k, v in input_data.items()]
                )
                prompt = f"{prompt}\n\n--- CONTEXT FROM PREVIOUS STEPS ---\n{context}"

            if self.verbose:
                print(f"    → {subtask.id} [{subtask.model.value}]: {subtask.description[:50]}...")

            # Execute
            output = model.chat(prompt)

            if self.verbose:
                preview = str(output)[:60] + "..." if len(str(output)) > 60 else str(output)
                print(f"      ✓ {preview}")

            return SubagentResult(
                subtask_id=subtask.id,
                success=True,
                output=output,
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
                model_used=subtask.model.value,
            )

        except Exception as e:
            if self.verbose:
                print(f"      ✗ Error: {e}")

            return SubagentResult(
                subtask_id=subtask.id,
                success=False,
                output=None,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
            )

    def _gather_inputs(
        self, subtask: SubTask, prior_results: dict[str, SubagentResult]
    ) -> dict[str, Any]:
        """Gather inputs from upstream tasks."""
        inputs = {}
        for dep_id in subtask.input_from:
            if dep_id in prior_results and prior_results[dep_id].success:
                inputs[dep_id] = prior_results[dep_id].output
        return inputs

    def _create_subagent_script(
        self, parent_task_id: str, subtask: SubTask, input_data: dict[str, Any]
    ) -> Path:
        """Create a temporary Python script for subagent execution."""
        script_content = f'''#!/usr/bin/env python3
"""Subagent script for {subtask.id}"""
import sys
sys.path.insert(0, "src")

from um_agent_coder.llm.providers.mcp_local import MCPLocalLLM

# Configure model
model = MCPLocalLLM(backend="{subtask.model.value}")

# Build prompt
prompt = """{subtask.prompt}"""

# Add context from prior steps
context = """{json.dumps(input_data, indent=2)}"""
if context and context != "{{}}":
    prompt += f"\\n\\n--- CONTEXT FROM PREVIOUS STEPS ---\\n{{context}}"

# Execute
try:
    result = model.chat(prompt)
    print(result)
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
'''

        script_path = self.checkpoint_dir / f"subagent_{parent_task_id}_{subtask.id}.py"
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        return script_path

    def _request_human_approval(
        self, task_id: str, subtask_ids: list[str], current_results: dict[str, SubagentResult]
    ) -> bool:
        """Request human approval before proceeding."""
        if self.human_approval_callback:
            return self.human_approval_callback(task_id, subtask_ids, current_results)

        # Default: interactive console approval
        print(f"\n{'='*60}")
        print("HUMAN APPROVAL REQUIRED")
        print(f"{'='*60}")
        print(f"Task: {task_id}")
        print(f"Pending subtasks: {subtask_ids}")

        if current_results:
            print("\nCompleted so far:")
            for st_id, result in current_results.items():
                status = "✓" if result.success else "✗"
                print(f"  {status} {st_id}")

        response = input("\nProceed? (y/n): ").strip().lower()
        return response == "y"

    def _save_checkpoint(
        self, task_id: str, results: dict[str, SubagentResult], completed: set[str]
    ):
        """Save execution checkpoint."""
        checkpoint = {
            "task_id": task_id,
            "completed": list(completed),
            "results": {
                k: {
                    "success": v.success,
                    "output": v.output,
                    "error": v.error,
                    "model": v.model_used,
                }
                for k, v in results.items()
            },
            "timestamp": datetime.now().isoformat(),
        }

        path = self.checkpoint_dir / f"{task_id}_checkpoint.json"
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

    def request_pause(self):
        """Request pause at next checkpoint."""
        self.pause_requested = True

    def resume_from_checkpoint(self, task_id: str) -> Optional[dict]:
        """Load checkpoint for resuming."""
        path = self.checkpoint_dir / f"{task_id}_checkpoint.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
