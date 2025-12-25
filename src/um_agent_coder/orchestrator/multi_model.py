"""
Multi-Model Orchestrator - Coordinates Gemini, Codex, and Claude for complex tasks.

This orchestrator:
1. Takes a decomposed task with subtasks assigned to different models
2. Executes subtasks respecting dependencies
3. Passes outputs between models
4. Checkpoints progress for resumability
5. Handles failures and retries
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

from .task_decomposer import (
    DecomposedTask, SubTask, SubTaskType, ModelRole,
    TaskDecomposer
)


class PipelineStatus(Enum):
    """Status of the overall pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineStep:
    """A single step in the execution pipeline."""
    subtask: SubTask
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtask": self.subtask.to_dict(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "retries": self.retries
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineStep':
        return cls(
            subtask=SubTask.from_dict(data["subtask"]),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            input_data=data.get("input_data", {}),
            output_data=data.get("output_data"),
            error=data.get("error"),
            retries=data.get("retries", 0)
        )


@dataclass
class TaskPipeline:
    """Complete pipeline state for a decomposed task."""
    task_id: str
    decomposed_task: DecomposedTask
    steps: List[PipelineStep]
    status: PipelineStatus
    current_step_index: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    final_output: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "current_step_index": self.current_step_index,
            "total_steps": len(self.steps),
            "decomposed_task": self.decomposed_task.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "final_output": self.final_output
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskPipeline':
        return cls(
            task_id=data["task_id"],
            status=PipelineStatus(data["status"]),
            decomposed_task=DecomposedTask.from_dict(data["decomposed_task"]),
            steps=[PipelineStep.from_dict(step) for step in data["steps"]],
            current_step_index=data.get("current_step_index", 0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            final_output=data.get("final_output")
        )


class MultiModelOrchestrator:
    """
    Orchestrates multi-model workflows for complex tasks.

    Usage:
        from um_agent_coder.llm.providers.mcp_local import MCPLocalLLM

        # Create model instances
        gemini = MCPLocalLLM(backend="gemini", model="gemini-3-pro-preview")
        codex = MCPLocalLLM(backend="codex", model="o4-mini")
        claude = MCPLocalLLM(backend="claude", model="claude-sonnet")

        # Create orchestrator
        orchestrator = MultiModelOrchestrator(
            gemini=gemini,
            codex=codex,
            claude=claude
        )

        # Run a complex task
        result = orchestrator.run("identify biotech M&A opportunities")
    """

    def __init__(
        self,
        gemini=None,
        codex=None,
        claude=None,
        checkpoint_dir: str = ".pipeline_checkpoints",
        verbose: bool = True
    ):
        """
        Initialize orchestrator with model instances.

        Args:
            gemini: LLM instance for Gemini (research, large context)
            codex: LLM instance for Codex (code generation)
            claude: LLM instance for Claude (reasoning, synthesis)
            checkpoint_dir: Directory for pipeline checkpoints
            verbose: Print progress updates
        """
        self.models = {
            ModelRole.GEMINI: gemini,
            ModelRole.CODEX: codex,
            ModelRole.CLAUDE: claude
        }

        # Use first available model as fallback decomposer
        decomposer_llm = claude or gemini or codex
        self.decomposer = TaskDecomposer(decomposer_llm)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.verbose = verbose

        # Callbacks for external integrations (n8n, webhooks, etc.)
        self.on_step_start: Optional[Callable] = None
        self.on_step_complete: Optional[Callable] = None
        self.on_pipeline_complete: Optional[Callable] = None

    def run(
        self,
        prompt: str,
        task_id: Optional[str] = None,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Run a complex task through the multi-model pipeline.

        Args:
            prompt: The original task prompt
            task_id: Optional ID for tracking (auto-generated if not provided)
            max_retries: Max retries per subtask on failure

        Returns:
            Dict with final output and execution metadata
        """
        import uuid
        task_id = task_id or str(uuid.uuid4())[:8]

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"MULTI-MODEL ORCHESTRATOR")
            print(f"Task ID: {task_id}")
            print(f"{'='*60}")

        # Step 1: Decompose the task
        if self.verbose:
            print("\n[1/3] Decomposing task...")

        decomposed = self.decomposer.decompose(prompt, use_llm=True)

        if self.verbose:
            print(self.decomposer.visualize(decomposed))

        # Step 2: Build pipeline
        pipeline = self._build_pipeline(task_id, decomposed)
        self._save_checkpoint(pipeline)

        # Step 3: Execute pipeline
        if self.verbose:
            print(f"\n[2/3] Executing {len(pipeline.steps)} steps...")

        result = self._execute_pipeline(pipeline, max_retries)

        # Step 4: Return results
        if self.verbose:
            print(f"\n[3/3] Pipeline {'completed' if result['success'] else 'failed'}")
            print(f"{'='*60}\n")

        return result

    def resume(self, task_id: str, max_retries: int = 2) -> Dict[str, Any]:
        """Resume a paused or failed pipeline from checkpoint."""
        pipeline = self._load_checkpoint(task_id)
        if not pipeline:
            return {
                "success": False,
                "error": f"No checkpoint found for task {task_id}",
                "task_id": task_id
            }

        if self.verbose:
            completed = sum(1 for s in pipeline.steps if s.subtask.status == "completed")
            print(f"Resuming task {task_id} from step {completed + 1}/{len(pipeline.steps)}")

        return self._execute_pipeline(pipeline, max_retries)

    def _build_pipeline(self, task_id: str, decomposed: DecomposedTask) -> TaskPipeline:
        """Build execution pipeline from decomposed task."""
        # Create steps in execution order
        steps = []
        subtask_map = {st.id: st for st in decomposed.subtasks}

        for subtask_id in decomposed.execution_order:
            subtask = subtask_map.get(subtask_id)
            if subtask:
                steps.append(PipelineStep(subtask=subtask))

        return TaskPipeline(
            task_id=task_id,
            decomposed_task=decomposed,
            steps=steps,
            status=PipelineStatus.PENDING
        )

    def _execute_pipeline(
        self,
        pipeline: TaskPipeline,
        max_retries: int
    ) -> Dict[str, Any]:
        """Execute the pipeline steps."""
        pipeline.status = PipelineStatus.RUNNING
        outputs = {}  # Store outputs by subtask ID

        for i, step in enumerate(pipeline.steps):
            # Skip completed steps
            if step.subtask.status == "completed":
                if step.output_data:
                    outputs[step.subtask.id] = step.output_data
                continue

            pipeline.current_step_index = i

            if self.verbose:
                model_name = step.subtask.model.value.upper()
                print(f"\n  [{i+1}/{len(pipeline.steps)}] {model_name}: {step.subtask.description}")

            # Gather inputs from dependencies
            input_data = {}
            for dep_id in step.subtask.input_from:
                if dep_id in outputs:
                    input_data[dep_id] = outputs[dep_id]

            step.input_data = input_data
            step.started_at = datetime.now().isoformat()

            # Execute with retries
            success = False
            for attempt in range(max_retries + 1):
                try:
                    # Fire callback
                    if self.on_step_start:
                        self.on_step_start(pipeline.task_id, step.subtask.id)

                    # Execute the step
                    output = self._execute_step(step, input_data)

                    step.output_data = output
                    step.subtask.status = "completed"
                    step.completed_at = datetime.now().isoformat()
                    outputs[step.subtask.id] = output
                    success = True

                    if self.verbose:
                        preview = str(output)[:100] + "..." if len(str(output)) > 100 else str(output)
                        print(f"      -> {preview}")

                    # Fire callback
                    if self.on_step_complete:
                        self.on_step_complete(pipeline.task_id, step.subtask.id, output)

                    break

                except Exception as e:
                    step.retries += 1
                    step.error = str(e)

                    if attempt < max_retries:
                        if self.verbose:
                            print(f"      Retry {attempt + 1}/{max_retries}: {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        step.subtask.status = "failed"
                        if self.verbose:
                            print(f"      FAILED: {e}")

            # Save checkpoint after each step
            self._save_checkpoint(pipeline)

            if not success:
                pipeline.status = PipelineStatus.FAILED
                return {
                    "success": False,
                    "task_id": pipeline.task_id,
                    "error": f"Step {step.subtask.id} failed: {step.error}",
                    "completed_steps": i,
                    "total_steps": len(pipeline.steps),
                    "partial_outputs": outputs
                }

        # Pipeline completed successfully
        pipeline.status = PipelineStatus.COMPLETED
        pipeline.final_output = outputs.get(pipeline.steps[-1].subtask.id)
        self._save_checkpoint(pipeline)

        if self.on_pipeline_complete:
            self.on_pipeline_complete(pipeline.task_id, pipeline.final_output)

        return {
            "success": True,
            "task_id": pipeline.task_id,
            "output": pipeline.final_output,
            "all_outputs": outputs,
            "steps_completed": len(pipeline.steps),
            "pipeline": pipeline.to_dict()
        }

    def _execute_step(self, step: PipelineStep, input_data: Dict[str, Any]) -> Any:
        """Execute a single pipeline step."""
        model = self.models.get(step.subtask.model)

        if not model:
            raise ValueError(f"No model configured for {step.subtask.model.value}")

        # Build prompt with input context
        prompt = step.subtask.prompt

        if input_data:
            context = "\n\n".join([
                f"=== Input from {k} ===\n{v}"
                for k, v in input_data.items()
            ])
            prompt = f"{prompt}\n\n--- CONTEXT FROM PREVIOUS STEPS ---\n{context}"

        # Execute
        response = model.chat(prompt)
        return response

    def _save_checkpoint(self, pipeline: TaskPipeline):
        """Save pipeline checkpoint to disk."""
        pipeline.updated_at = datetime.now().isoformat()
        checkpoint_path = self.checkpoint_dir / f"{pipeline.task_id}.json"

        with open(checkpoint_path, 'w') as f:
            json.dump(pipeline.to_dict(), f, indent=2, default=str)

    def _load_checkpoint(self, task_id: str) -> Optional[TaskPipeline]:
        """Load pipeline checkpoint from disk."""
        checkpoint_path = self.checkpoint_dir / f"{task_id}.json"

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)

            return TaskPipeline.from_dict(data)

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None

    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all pipeline checkpoints."""
        pipelines = []

        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)

                pipelines.append({
                    "task_id": data["task_id"],
                    "status": data["status"],
                    "progress": f"{data.get('current_step_index', 0)}/{data.get('total_steps', 0)}",
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at")
                })
            except Exception:
                continue

        return sorted(pipelines, key=lambda x: x.get("updated_at", ""), reverse=True)


def create_orchestrator_from_config(config: Dict[str, Any]) -> MultiModelOrchestrator:
    """
    Factory function to create orchestrator from config.

    Config example:
        orchestrator:
          gemini:
            model: gemini-3-pro-preview
          codex:
            model: o4-mini
            sandbox: workspace-write
          claude:
            model: claude-sonnet
          checkpoint_dir: .pipeline_checkpoints
          verbose: true
    """
    from um_agent_coder.llm.providers.mcp_local import MCPLocalLLM

    orch_config = config.get("orchestrator", {})

    gemini = None
    codex = None
    claude = None

    if orch_config.get("gemini"):
        gemini = MCPLocalLLM(
            backend="gemini",
            **orch_config["gemini"]
        )

    if orch_config.get("codex"):
        codex = MCPLocalLLM(
            backend="codex",
            **orch_config["codex"]
        )

    if orch_config.get("claude"):
        claude = MCPLocalLLM(
            backend="claude",
            **orch_config["claude"]
        )

    return MultiModelOrchestrator(
        gemini=gemini,
        codex=codex,
        claude=claude,
        checkpoint_dir=orch_config.get("checkpoint_dir", ".pipeline_checkpoints"),
        verbose=orch_config.get("verbose", True)
    )
