import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from um_agent_coder.llm.base import LLM
from um_agent_coder.tools import ToolRegistry
from um_agent_coder.context import ContextManager, ContextType
from um_agent_coder.models import ModelRegistry
from um_agent_coder.persistence import TaskCheckpointer, TaskState, TaskStatus
from um_agent_coder.persistence.checkpointer import StepState
from um_agent_coder.utils.spinner import Spinner
from um_agent_coder.utils.colors import ANSI
from .planner import TaskPlanner, TaskAnalysis, ExecutionPlan, TaskType
from .cost_tracker import CostTracker


class EnhancedAgent:
    """
    Enhanced agent with planning, context management, and cost tracking.
    """
    
    def __init__(self, llm: LLM, config: Dict[str, Any]):
        self.llm = llm
        self.config = config

        # Initialize components
        self.tool_registry = ToolRegistry()
        self.context_manager = ContextManager(
            max_tokens=config.get("max_context_tokens", 100000)
        )
        self.task_planner = TaskPlanner()
        self.cost_tracker = CostTracker()
        self.model_registry = ModelRegistry()

        # Initialize checkpointer for long-running tasks
        checkpoint_dir = config.get("checkpoint_dir", ".task_checkpoints")
        self.checkpointer = TaskCheckpointer(checkpoint_dir)

        # Register default tools
        self._register_tools()

        # Agent settings
        self.verbose = config.get("verbose", True)
        self.auto_summarize = config.get("auto_summarize", True)
        self.require_approval = config.get("require_approval", False)
        self.enable_checkpointing = config.get("enable_checkpointing", True)
    
    def run(self, prompt: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the agent with planning and execution.

        Args:
            prompt: The task prompt
            task_id: Optional task ID for tracking (auto-generated if not provided)

        Returns:
            Dictionary with response, metrics, and execution details
        """
        task_id = task_id or str(uuid.uuid4())[:8]

        try:
            # 1. Planning Stage
            with Spinner("Analyzing task...", verbose=self.verbose):
                task_analysis = self.task_planner.analyze_task(prompt)
                execution_plan = self.task_planner.create_execution_plan(task_analysis, prompt)

            # Start tracking
            self.cost_tracker.start_task(task_id, prompt, len(execution_plan.steps))

            # Initialize checkpoint state
            if self.enable_checkpointing:
                task_state = self._create_task_state(task_id, prompt, task_analysis, execution_plan)
                self.checkpointer.save(task_state)
                if self.verbose:
                    print(f"Task {task_id} - checkpoint enabled")

            # Add task info to context
            self.context_manager.add(
                content=f"Task: {prompt}\nType: {task_analysis.task_type.value}\nComplexity: {task_analysis.complexity}/10",
                type=ContextType.PROJECT_INFO,
                source="task_analysis",
                priority=9
            )

            # 2. Approval Stage (if required)
            if self.require_approval:
                approval = self._get_approval(task_analysis, execution_plan)
                if not approval:
                    if self.enable_checkpointing:
                        task_state.status = TaskStatus.PAUSED
                        self.checkpointer.save(task_state)
                    return {
                        "response": "Task cancelled by user",
                        "success": False,
                        "task_id": task_id
                    }

            # 3. Context Loading Stage
            with Spinner("Loading context...", verbose=self.verbose):
                self._load_initial_context(task_analysis)

            # 4. Execution Stage
            if self.verbose:
                print(f"Executing {len(execution_plan.steps)} steps...")

            results = []
            for i, step in enumerate(execution_plan.steps):
                description = step.description
                if len(description) > 50:
                    description = description[:47] + "..."

                step_msg = f"Step {i+1}/{len(execution_plan.steps)}: {description}"

                # Update checkpoint before executing step
                if self.enable_checkpointing:
                    task_state.current_step = i
                    task_state.steps[i].status = "running"
                    task_state.steps[i].started_at = datetime.now().isoformat()
                    self.checkpointer.save(task_state)

                with Spinner(step_msg, verbose=self.verbose):
                    result = self._execute_step(step)

                results.append(result)

                # Update checkpoint after step completion
                if self.enable_checkpointing:
                    task_state.steps[i].status = "completed" if result["success"] else "failed"
                    task_state.steps[i].result = result
                    task_state.steps[i].completed_at = datetime.now().isoformat()
                    if not result["success"]:
                        task_state.steps[i].error = result.get("error")
                    # Save context and cost state
                    task_state.context_items = self.context_manager.export_state()
                    task_state.cost_state = self.cost_tracker.export_state()
                    self.checkpointer.save(task_state)

                # Track progress
                self.cost_tracker.track_step(
                    tokens=step.estimated_tokens,
                    cost=self._calculate_step_cost(step.estimated_tokens)
                )

                # Auto-summarize if needed
                if self.auto_summarize:
                    self.context_manager.summarize_if_needed(self.llm)

            # 5. Response Generation
            with Spinner("Generating response...", verbose=self.verbose):
                response = self._generate_response(prompt, results)

            # Complete tracking
            self.cost_tracker.complete_task(success=True)

            # Mark task as completed in checkpoint
            if self.enable_checkpointing:
                task_state.status = TaskStatus.COMPLETED
                task_state.completed_at = datetime.now().isoformat()
                task_state.cost_state = self.cost_tracker.export_state()
                self.checkpointer.save(task_state)

            # 6. Return comprehensive result
            return {
                "response": response,
                "success": True,
                "task_id": task_id,
                "task_analysis": {
                    "type": task_analysis.task_type.value,
                    "complexity": task_analysis.complexity,
                    "estimated_tokens": task_analysis.estimated_tokens
                },
                "execution_plan": {
                    "steps": len(execution_plan.steps),
                    "estimated_cost": execution_plan.estimated_cost,
                    "estimated_time": execution_plan.estimated_time_minutes
                },
                "metrics": self.cost_tracker.get_statistics(),
                "context_usage": self.context_manager.get_usage()
            }

        except Exception as e:
            # Handle errors
            self.cost_tracker.complete_task(success=False, error=str(e))

            # Save failed state to checkpoint
            if self.enable_checkpointing:
                try:
                    task_state = self.checkpointer.load(task_id)
                    if task_state:
                        task_state.status = TaskStatus.FAILED
                        task_state.metadata["error"] = str(e)
                        self.checkpointer.save(task_state)
                except Exception:
                    pass

            return {
                "response": f"Error executing task: {str(e)}",
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "metrics": self.cost_tracker.get_statistics()
            }

    def resume(self, task_id: str) -> Dict[str, Any]:
        """
        Resume a previously paused or failed task from its checkpoint.

        Args:
            task_id: The task ID to resume

        Returns:
            Dictionary with response, metrics, and execution details
        """
        # Load checkpoint
        task_state = self.checkpointer.load(task_id)
        if not task_state:
            return {
                "response": f"No checkpoint found for task {task_id}",
                "success": False,
                "task_id": task_id,
                "error": "Checkpoint not found"
            }

        if task_state.status == TaskStatus.COMPLETED:
            return {
                "response": "Task already completed",
                "success": True,
                "task_id": task_id
            }

        if self.verbose:
            completed = sum(1 for s in task_state.steps if s.status == "completed")
            print(f"Resuming task {task_id} from step {completed + 1}/{len(task_state.steps)}")

        # Restore state
        self.context_manager.import_state(task_state.context_items)
        self.cost_tracker.import_state(task_state.cost_state)

        # Mark as running
        task_state.status = TaskStatus.RUNNING
        self.checkpointer.save(task_state)

        # Find first incomplete step
        start_step = 0
        for i, step in enumerate(task_state.steps):
            if step.status != "completed":
                start_step = i
                break

        # Execute remaining steps
        results = []
        for i in range(start_step, len(task_state.steps)):
            step = task_state.steps[i]

            if self.verbose:
                print(f"  Step {i+1}/{len(task_state.steps)}: {step.description}")

            # Update checkpoint
            task_state.current_step = i
            task_state.steps[i].status = "running"
            task_state.steps[i].started_at = datetime.now().isoformat()
            self.checkpointer.save(task_state)

            # Execute step
            tool = self.tool_registry.get(step.action)
            if not tool:
                result = {
                    "success": False,
                    "error": f"Tool {step.action} not found",
                    "data": None,
                    "tool": step.action,
                    "description": step.description
                }
            else:
                tool_result = tool.execute(**step.parameters)
                result = {
                    "success": tool_result.success,
                    "error": tool_result.error,
                    "data": tool_result.data,
                    "tool": step.action,
                    "description": step.description
                }

                # Add result to context
                if tool_result.success and tool_result.data:
                    self.context_manager.add(
                        content=str(tool_result.data)[:2000],
                        type=ContextType.TOOL_RESULT,
                        source=f"{step.action}_result",
                        priority=5
                    )

            results.append(result)

            # Update checkpoint
            task_state.steps[i].status = "completed" if result["success"] else "failed"
            task_state.steps[i].result = result
            task_state.steps[i].completed_at = datetime.now().isoformat()
            task_state.context_items = self.context_manager.export_state()
            task_state.cost_state = self.cost_tracker.export_state()
            self.checkpointer.save(task_state)

            # Track progress
            self.cost_tracker.track_step(tokens=200, cost=0.002)

        # Generate response
        with Spinner("Generating response...", verbose=self.verbose):
            # Collect all results (including previously completed)
            all_results = []
            for step in task_state.steps:
                if step.result:
                    all_results.append(step.result)

            response = self._generate_response(task_state.prompt, all_results)

        # Mark complete
        task_state.status = TaskStatus.COMPLETED
        task_state.completed_at = datetime.now().isoformat()
        self.checkpointer.save(task_state)

        self.cost_tracker.complete_task(success=True)

        return {
            "response": response,
            "success": True,
            "task_id": task_id,
            "resumed": True,
            "metrics": self.cost_tracker.get_statistics(),
            "context_usage": self.context_manager.get_usage()
        }

    def pause(self, task_id: str) -> bool:
        """
        Pause a running task (marks checkpoint as paused).

        Args:
            task_id: The task ID to pause

        Returns:
            True if paused successfully
        """
        task_state = self.checkpointer.load(task_id)
        if not task_state:
            return False

        task_state.status = TaskStatus.PAUSED
        task_state.context_items = self.context_manager.export_state()
        task_state.cost_state = self.cost_tracker.export_state()
        return self.checkpointer.save(task_state)

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """
        List all checkpointed tasks.

        Args:
            status: Optional filter by status

        Returns:
            List of task summaries
        """
        return self.checkpointer.list_tasks(status)

    def get_task_state(self, task_id: str) -> Optional[TaskState]:
        """
        Get the full state of a task.

        Args:
            task_id: The task ID

        Returns:
            TaskState if found
        """
        return self.checkpointer.load(task_id)

    def _create_task_state(
        self,
        task_id: str,
        prompt: str,
        analysis: TaskAnalysis,
        plan: ExecutionPlan
    ) -> TaskState:
        """Create initial task state from analysis and plan."""
        steps = []
        for i, step in enumerate(plan.steps):
            steps.append(StepState(
                step_index=i,
                description=step.description,
                action=step.action,
                parameters=step.parameters,
                status="pending"
            ))

        return TaskState(
            task_id=task_id,
            prompt=prompt,
            status=TaskStatus.RUNNING,
            current_step=0,
            steps=steps,
            task_analysis={
                "type": analysis.task_type.value,
                "complexity": analysis.complexity,
                "estimated_tokens": analysis.estimated_tokens,
                "required_tools": analysis.required_tools,
                "files_to_analyze": analysis.files_to_analyze,
                "potential_risks": analysis.potential_risks
            }
        )
    
    def _register_tools(self):
        """Register available tools."""
        from um_agent_coder.tools import (
            FileReader, FileWriter, FileSearcher,
            CodeSearcher, ProjectAnalyzer, CommandExecutor, ArchitectTool
        )
        
        tools = [
            FileReader(),
            FileWriter(),
            FileSearcher(),
            CodeSearcher(),
            ProjectAnalyzer(),
            CommandExecutor(),
            ArchitectTool(self.llm)
        ]
        
        for tool in tools:
            self.tool_registry.register(tool)
    
    def _load_initial_context(self, task_analysis: TaskAnalysis):
        """Load initial context based on task analysis."""
        # Load project structure if complex task
        if task_analysis.complexity > 6:
            analyzer = self.tool_registry.get("ProjectAnalyzer")
            if analyzer:
                result = analyzer.execute(directory=".")
                if result.success:
                    self.context_manager.add(
                        content=json.dumps(result.data, indent=2),
                        type=ContextType.PROJECT_INFO,
                        source="project_structure",
                        priority=7
                    )
        
        # Load mentioned files
        file_reader = self.tool_registry.get("FileReader")
        if file_reader:
            for file_path in task_analysis.files_to_analyze[:5]:  # Limit to 5 files
                result = file_reader.execute(file_path=file_path)
                if result.success:
                    self.context_manager.add(
                        content=result.data[:5000],  # Limit content size
                        type=ContextType.FILE,
                        source=file_path,
                        priority=8
                    )
    
    def _execute_step(self, step) -> Dict[str, Any]:
        """Execute a single step of the plan."""
        tool = self.tool_registry.get(step.action)
        
        if not tool:
            return {
                "success": False,
                "error": f"Tool {step.action} not found",
                "data": None
            }
        
        # Execute tool
        result = tool.execute(**step.parameters)
        
        # Add result to context if successful
        if result.success and result.data:
            self.context_manager.add(
                content=str(result.data)[:2000],  # Limit size
                type=ContextType.TOOL_RESULT,
                source=f"{step.action}_result",
                priority=step.priority
            )
        
        return {
            "success": result.success,
            "error": result.error,
            "data": result.data,
            "tool": step.action,
            "description": step.description
        }
    
    def _generate_response(self, original_prompt: str, results: List[Dict[str, Any]]) -> str:
        """Generate final response based on execution results."""
        # Build prompt for response generation
        context = self.context_manager.get_context()
        
        results_summary = "\n".join([
            f"- {r['description']}: {'✓' if r['success'] else '✗'}"
            for r in results
        ])
        
        response_prompt = f"""
Based on the following context and execution results, provide a comprehensive response to the user's request.

Original Request: {original_prompt}

Execution Results:
{results_summary}

Context:
{context}

Please provide a clear, helpful response that addresses the user's request.
"""
        
        # Generate response
        response = self.llm.chat(response_prompt)
        
        # Add response to context for future reference
        self.context_manager.add(
            content=f"User: {original_prompt}\nAssistant: {response}",
            type=ContextType.CONVERSATION,
            source="conversation",
            priority=6
        )
        
        return response
    
    def _calculate_step_cost(self, tokens: int) -> float:
        """Calculate cost for a step based on token usage."""
        model_info = self.llm.get_model_info()
        if isinstance(model_info, dict) and "cost_per_1k_input" in model_info:
            # Assume 50/50 input/output split
            input_cost = (tokens / 2 / 1000) * model_info["cost_per_1k_input"]
            output_cost = (tokens / 2 / 1000) * model_info["cost_per_1k_output"]
            return input_cost + output_cost
        return 0.0
    
    def _get_approval(self, task_analysis: TaskAnalysis, plan: ExecutionPlan) -> bool:
        """Get user approval for task execution."""
        print("\n" + ANSI.style("="*60, ANSI.BLUE))
        print(ANSI.style("📋 TASK ANALYSIS & PLAN", ANSI.BOLD))
        print(ANSI.style("="*60, ANSI.BLUE))

        # Key Metrics Grid
        print(f"Type: {ANSI.style(task_analysis.task_type.value.replace('_', ' ').title(), ANSI.CYAN)}")

        complexity_color = ANSI.GREEN if task_analysis.complexity < 4 else (ANSI.WARNING if task_analysis.complexity < 7 else ANSI.FAIL)
        print(f"Complexity: {ANSI.style(f'{task_analysis.complexity}/10', complexity_color)}")

        print(f"Est. Tokens: {ANSI.style(f'{task_analysis.estimated_tokens:,}', ANSI.CYAN)}")
        print(f"Est. Cost: {ANSI.style(f'${plan.estimated_cost:.4f}', ANSI.GREEN if plan.estimated_cost < 0.1 else ANSI.WARNING)}")
        print(f"Est. Time: {ANSI.style(f'{plan.estimated_time_minutes:.1f} min', ANSI.CYAN)}")
        
        if task_analysis.potential_risks:
            print(f"\n{ANSI.style('⚠️  POTENTIAL RISKS:', ANSI.WARNING)}")
            for risk in task_analysis.potential_risks:
                print(f"  • {risk}")
        
        print(f"\n{ANSI.style(f'🚀 EXECUTION PLAN ({len(plan.steps)} steps):', ANSI.BOLD)}")
        for i, step in enumerate(plan.steps, 1):
            print(f"  {ANSI.style(str(i), ANSI.BLUE)}. {step.description}")
        
        print(ANSI.style("-" * 60, ANSI.BLUE))
        response = input(f"{ANSI.style('Proceed with execution? (y/n):', ANSI.BOLD)} ")
        return response.lower() == 'y'
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics and statistics."""
        return self.cost_tracker.get_statistics()
    
    def export_metrics(self, filepath: str):
        """Export metrics to file."""
        self.cost_tracker.export_metrics(filepath)