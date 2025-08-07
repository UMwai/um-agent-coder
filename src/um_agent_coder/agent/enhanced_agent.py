import uuid
from typing import Dict, Any, List, Optional
import json

from um_agent_coder.llm.base import LLM
from um_agent_coder.tools import ToolRegistry
from um_agent_coder.context import ContextManager, ContextType
from um_agent_coder.models import ModelRegistry
from .planner import TaskPlanner, TaskAnalysis, ExecutionPlan
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
        
        # Register default tools
        self._register_tools()
        
        # Agent settings
        self.verbose = config.get("verbose", True)
        self.auto_summarize = config.get("auto_summarize", True)
        self.require_approval = config.get("require_approval", False)
    
    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Run the agent with planning and execution.
        
        Returns:
            Dictionary with response, metrics, and execution details
        """
        task_id = str(uuid.uuid4())[:8]
        
        try:
            # 1. Planning Stage
            if self.verbose:
                print(f"🔍 Analyzing task...")
            
            task_analysis = self.task_planner.analyze_task(prompt)
            execution_plan = self.task_planner.create_execution_plan(task_analysis, prompt)
            
            # Start tracking
            self.cost_tracker.start_task(task_id, prompt, len(execution_plan.steps))
            
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
                    return {
                        "response": "Task cancelled by user",
                        "success": False,
                        "task_id": task_id
                    }
            
            # 3. Context Loading Stage
            if self.verbose:
                print(f"📚 Loading context...")
            
            self._load_initial_context(task_analysis)
            
            # 4. Execution Stage
            if self.verbose:
                print(f"🚀 Executing {len(execution_plan.steps)} steps...")
            
            results = []
            for i, step in enumerate(execution_plan.steps):
                if self.verbose:
                    print(f"  Step {i+1}/{len(execution_plan.steps)}: {step.description}")
                
                result = self._execute_step(step)
                results.append(result)
                
                # Track progress
                self.cost_tracker.track_step(
                    tokens=step.estimated_tokens,
                    cost=self._calculate_step_cost(step.estimated_tokens)
                )
                
                # Auto-summarize if needed
                if self.auto_summarize:
                    self.context_manager.summarize_if_needed(self.llm)
            
            # 5. Response Generation
            if self.verbose:
                print(f"✍️ Generating response...")
            
            response = self._generate_response(prompt, results)
            
            # Complete tracking
            self.cost_tracker.complete_task(success=True)
            
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
            
            return {
                "response": f"Error executing task: {str(e)}",
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "metrics": self.cost_tracker.get_statistics()
            }
    
    def _register_tools(self):
        """Register available tools."""
        from um_agent_coder.tools import (
            FileReader, FileWriter, FileSearcher,
            CodeSearcher, ProjectAnalyzer, CommandExecutor
        )
        
        tools = [
            FileReader(),
            FileWriter(),
            FileSearcher(),
            CodeSearcher(),
            ProjectAnalyzer(),
            CommandExecutor()
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
        print("\n" + "="*50)
        print("TASK ANALYSIS")
        print("="*50)
        print(f"Type: {task_analysis.task_type.value}")
        print(f"Complexity: {task_analysis.complexity}/10")
        print(f"Estimated tokens: {task_analysis.estimated_tokens}")
        print(f"Estimated cost: ${plan.estimated_cost:.4f}")
        print(f"Estimated time: {plan.estimated_time_minutes:.1f} minutes")
        
        if task_analysis.potential_risks:
            print(f"\n⚠️ Risks:")
            for risk in task_analysis.potential_risks:
                print(f"  - {risk}")
        
        print(f"\nExecution plan ({len(plan.steps)} steps):")
        for i, step in enumerate(plan.steps, 1):
            print(f"  {i}. {step.description}")
        
        response = input("\nProceed with execution? (y/n): ")
        return response.lower() == 'y'
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics and statistics."""
        return self.cost_tracker.get_statistics()
    
    def export_metrics(self, filepath: str):
        """Export metrics to file."""
        self.cost_tracker.export_metrics(filepath)