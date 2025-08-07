"""
Roo-inspired Agent: A multi-mode, customizable coding agent.
Integrates ideas from Roo-Code with the existing enhanced agent architecture.
"""

import uuid
from typing import Dict, Any, List, Optional, Tuple
import json
import re

from um_agent_coder.llm.base import LLM
from um_agent_coder.tools import ToolRegistry
from um_agent_coder.context import ContextManager, ContextType
from um_agent_coder.models import ModelRegistry
from .planner import TaskPlanner, TaskAnalysis, ExecutionPlan
from .cost_tracker import CostTracker
from .modes import ModeManager, AgentMode, ModeConfig


class RooAgent:
    """
    Multi-mode agent inspired by Roo-Code architecture.
    Features:
    - Multiple specialized modes (Code, Architect, Debug, etc.)
    - Customizable instructions and personas
    - Smart tool selection based on mode
    - Interactive approval system
    - Extended context management
    """
    
    def __init__(self, llm: LLM, config: Dict[str, Any]):
        self.llm = llm
        self.config = config
        
        # Core components
        self.tool_registry = ToolRegistry()
        self.context_manager = ContextManager(
            max_tokens=config.get("max_context_tokens", 100000)
        )
        self.task_planner = TaskPlanner()
        self.cost_tracker = CostTracker()
        self.model_registry = ModelRegistry()
        self.mode_manager = ModeManager()
        
        # Register tools
        self._register_tools()
        
        # Load custom instructions if provided
        self.custom_instructions = config.get("custom_instructions", "")
        
        # Settings
        self.verbose = config.get("verbose", True)
        self.auto_mode = config.get("auto_mode", True)  # Auto-detect mode from prompt
        self.require_approval = config.get("require_approval", False)
        self.auto_summarize = config.get("auto_summarize", True)
        self.interactive = config.get("interactive", False)
        
        # Memory for conversation context
        self.conversation_history = []
        
        # Load custom modes if defined
        self._load_custom_modes()
    
    def run(self, prompt: str, mode: Optional[AgentMode] = None) -> Dict[str, Any]:
        """
        Run the agent with the given prompt and optional mode.
        
        Args:
            prompt: User's request
            mode: Optional specific mode to use
            
        Returns:
            Comprehensive result dictionary
        """
        task_id = str(uuid.uuid4())[:8]
        
        try:
            # 1. Mode Selection
            if mode is None and self.auto_mode:
                mode = self.mode_manager.detect_mode_from_prompt(prompt)
                if self.verbose:
                    print(f"🎭 Auto-detected mode: {mode.value}")
            elif mode is None:
                mode = AgentMode.CODE
            
            # Set the mode
            mode_config = self.mode_manager.set_mode(mode)
            
            # 2. Enhanced Planning with Mode Context
            if self.verbose:
                print(f"🔍 Analyzing task in {mode_config.name}...")
            
            # Add mode context to prompt
            enhanced_prompt = self._enhance_prompt_with_mode(prompt, mode_config)
            
            # Analyze and plan
            task_analysis = self.task_planner.analyze_task(prompt)
            execution_plan = self._create_mode_aware_plan(task_analysis, prompt, mode_config)
            
            # Start tracking
            self.cost_tracker.start_task(task_id, prompt, len(execution_plan.steps))
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": prompt,
                "mode": mode.value,
                "timestamp": uuid.uuid4().hex[:8]
            })
            
            # 3. Context Preparation
            self._prepare_mode_context(task_analysis, mode_config)
            
            # 4. Interactive Approval (if required)
            if self.require_approval and not self._should_auto_approve(execution_plan, mode_config):
                approval = self._get_interactive_approval(task_analysis, execution_plan, mode_config)
                if not approval:
                    return {
                        "response": "Task cancelled by user",
                        "success": False,
                        "task_id": task_id,
                        "mode": mode.value
                    }
            
            # 5. Execution with Mode-Specific Handling
            if self.verbose:
                print(f"🚀 Executing {len(execution_plan.steps)} steps in {mode_config.name}...")
            
            results = self._execute_with_mode(execution_plan, mode_config)
            
            # 6. Response Generation with Mode Context
            response = self._generate_mode_aware_response(
                enhanced_prompt, results, mode_config
            )
            
            # Update conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "mode": mode.value,
                "timestamp": uuid.uuid4().hex[:8]
            })
            
            # Complete tracking
            self.cost_tracker.complete_task(success=True)
            
            # 7. Return comprehensive result
            return {
                "response": response,
                "success": True,
                "task_id": task_id,
                "mode": mode.value,
                "mode_config": {
                    "name": mode_config.name,
                    "description": mode_config.description
                },
                "task_analysis": {
                    "type": task_analysis.task_type.value,
                    "complexity": task_analysis.complexity,
                    "estimated_tokens": task_analysis.estimated_tokens
                },
                "execution": {
                    "steps_executed": len(results),
                    "successful_steps": sum(1 for r in results if r["success"]),
                    "tools_used": list(set(r.get("tool", "") for r in results if r.get("tool")))
                },
                "metrics": self.cost_tracker.get_statistics(),
                "context_usage": self.context_manager.get_usage()
            }
            
        except Exception as e:
            self.cost_tracker.complete_task(success=False, error=str(e))
            return {
                "response": f"Error: {str(e)}",
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "mode": mode.value if mode else "unknown"
            }
    
    def _enhance_prompt_with_mode(self, prompt: str, mode_config: ModeConfig) -> str:
        """Enhance the prompt with mode-specific context."""
        enhanced = f"{mode_config.system_prompt}\n\n"
        
        if self.custom_instructions:
            enhanced += f"Custom Instructions:\n{self.custom_instructions}\n\n"
        
        enhanced += f"User Request:\n{prompt}"
        
        return enhanced
    
    def _create_mode_aware_plan(
        self, 
        analysis: TaskAnalysis, 
        prompt: str, 
        mode_config: ModeConfig
    ) -> ExecutionPlan:
        """Create an execution plan aware of the current mode."""
        base_plan = self.task_planner.create_execution_plan(analysis, prompt)
        
        # Filter and prioritize steps based on mode
        mode_steps = []
        for step in base_plan.steps:
            # Check if tool is preferred in this mode
            if step.action in mode_config.preferred_tools:
                step.priority += 2  # Boost priority
            
            # Add mode-specific parameters
            step.parameters["mode"] = mode_config.name
            mode_steps.append(step)
        
        # Reorder by priority
        mode_steps.sort(key=lambda s: s.priority, reverse=True)
        
        base_plan.steps = mode_steps
        return base_plan
    
    def _prepare_mode_context(self, analysis: TaskAnalysis, mode_config: ModeConfig):
        """Prepare context based on mode priorities."""
        # Load context based on mode priorities
        for context_type, priority in mode_config.context_priorities.items():
            if priority >= 8:  # High priority contexts
                self._load_context_type(context_type, analysis)
    
    def _load_context_type(self, context_type: str, analysis: TaskAnalysis):
        """Load specific context type."""
        if context_type == "project_structure":
            analyzer = self.tool_registry.get("ProjectAnalyzer")
            if analyzer:
                result = analyzer.execute(directory=".")
                if result.success:
                    self.context_manager.add(
                        content=json.dumps(result.data, indent=2)[:5000],
                        type=ContextType.PROJECT_INFO,
                        source="project_structure",
                        priority=9
                    )
        
        elif context_type == "related_code":
            searcher = self.tool_registry.get("CodeSearcher")
            if searcher and analysis.files_to_analyze:
                for file in analysis.files_to_analyze[:3]:
                    result = searcher.execute(pattern=file)
                    if result.success:
                        self.context_manager.add(
                            content=str(result.data)[:2000],
                            type=ContextType.FILE,
                            source=f"related_{file}",
                            priority=8
                        )
    
    def _should_auto_approve(self, plan: ExecutionPlan, mode_config: ModeConfig) -> bool:
        """Check if all actions in the plan are auto-approved."""
        for step in plan.steps:
            if not self.mode_manager.should_auto_approve(step.action):
                return False
        return True
    
    def _execute_with_mode(
        self, 
        plan: ExecutionPlan, 
        mode_config: ModeConfig
    ) -> List[Dict[str, Any]]:
        """Execute plan with mode-specific handling."""
        results = []
        
        for i, step in enumerate(plan.steps):
            if self.verbose:
                print(f"  [{mode_config.name}] Step {i+1}: {step.description}")
            
            # Check for mode-specific approval
            if not self.mode_manager.should_auto_approve(step.action):
                if self.interactive:
                    if not self._get_step_approval(step):
                        results.append({
                            "success": False,
                            "error": "Step skipped by user",
                            "tool": step.action,
                            "description": step.description
                        })
                        continue
            
            # Execute step
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
        
        return results
    
    def _execute_step(self, step) -> Dict[str, Any]:
        """Execute a single step."""
        tool = self.tool_registry.get(step.action)
        
        if not tool:
            return {
                "success": False,
                "error": f"Tool {step.action} not found",
                "tool": step.action,
                "description": step.description
            }
        
        # Execute tool
        result = tool.execute(**step.parameters)
        
        # Add result to context
        if result.success and result.data:
            self.context_manager.add(
                content=str(result.data)[:2000],
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
    
    def _generate_mode_aware_response(
        self, 
        prompt: str, 
        results: List[Dict[str, Any]], 
        mode_config: ModeConfig
    ) -> str:
        """Generate response aware of the current mode."""
        context = self.context_manager.get_context()
        
        # Build mode-specific response prompt
        response_prompt = f"""
Operating in {mode_config.name}:
{mode_config.description}

Original Request: {prompt}

Execution Results:
{self._format_results(results)}

Available Context:
{context[:10000]}  # Limit context size

Based on the above, provide a response that:
1. Addresses the user's request completely
2. Follows the mode's approach and style
3. References specific code/files when relevant
4. Suggests next steps if applicable

Response:"""
        
        # Set temperature based on mode
        original_temp = getattr(self.llm, 'temperature', 0.7)
        if hasattr(self.llm, 'temperature'):
            self.llm.temperature = mode_config.temperature
        
        response = self.llm.chat(response_prompt)
        
        # Restore original temperature
        if hasattr(self.llm, 'temperature'):
            self.llm.temperature = original_temp
        
        return response
    
    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format execution results for display."""
        formatted = []
        for i, result in enumerate(results, 1):
            status = "✓" if result["success"] else "✗"
            formatted.append(
                f"{i}. [{status}] {result['description']}"
            )
            if not result["success"] and result.get("error"):
                formatted.append(f"   Error: {result['error']}")
        return "\n".join(formatted)
    
    def _get_interactive_approval(
        self, 
        analysis: TaskAnalysis, 
        plan: ExecutionPlan, 
        mode_config: ModeConfig
    ) -> bool:
        """Get user approval for task execution."""
        print("\n" + "="*60)
        print(f"TASK APPROVAL - {mode_config.name}")
        print("="*60)
        print(f"Task Type: {analysis.task_type.value}")
        print(f"Complexity: {analysis.complexity}/10")
        print(f"Estimated Cost: ${plan.estimated_cost:.4f}")
        print(f"Mode Tools: {', '.join(mode_config.preferred_tools)}")
        
        if analysis.potential_risks:
            print(f"\n⚠️ Potential Risks:")
            for risk in analysis.potential_risks:
                print(f"  - {risk}")
        
        print(f"\nExecution Plan ({len(plan.steps)} steps):")
        for i, step in enumerate(plan.steps[:10], 1):  # Show first 10 steps
            auto = " [auto]" if self.mode_manager.should_auto_approve(step.action) else ""
            print(f"  {i}. {step.description}{auto}")
        
        if len(plan.steps) > 10:
            print(f"  ... and {len(plan.steps) - 10} more steps")
        
        response = input("\nProceed? (y/n/m to modify): ").lower()
        
        if response == 'm':
            # Allow mode switching
            print("\nAvailable modes: code, architect, ask, debug, review")
            new_mode = input("Enter new mode: ").lower()
            try:
                self.mode_manager.set_mode(AgentMode(new_mode))
                print(f"Switched to {new_mode} mode")
                return self._get_interactive_approval(analysis, plan, self.mode_manager.get_current_mode())
            except:
                print("Invalid mode, keeping current")
                return False
        
        return response == 'y'
    
    def _get_step_approval(self, step) -> bool:
        """Get approval for a single step."""
        print(f"\n🔸 Approve step: {step.description}")
        print(f"   Tool: {step.action}")
        response = input("   Execute? (y/n): ").lower()
        return response == 'y'
    
    def _calculate_step_cost(self, tokens: int) -> float:
        """Calculate cost for a step."""
        model_info = self.llm.get_model_info() if hasattr(self.llm, 'get_model_info') else {}
        if isinstance(model_info, dict) and "cost_per_1k_input" in model_info:
            input_cost = (tokens / 2 / 1000) * model_info["cost_per_1k_input"]
            output_cost = (tokens / 2 / 1000) * model_info.get("cost_per_1k_output", 0)
            return input_cost + output_cost
        return 0.0
    
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
    
    def _load_custom_modes(self):
        """Load custom modes from configuration."""
        custom_modes = self.config.get("custom_modes", {})
        for mode_name, mode_data in custom_modes.items():
            config = ModeConfig(
                name=mode_data.get("name", mode_name),
                description=mode_data.get("description", ""),
                system_prompt=mode_data.get("system_prompt", ""),
                temperature=mode_data.get("temperature", 0.7),
                preferred_tools=mode_data.get("preferred_tools", []),
                auto_approve_actions=mode_data.get("auto_approve_actions", []),
                context_priorities=mode_data.get("context_priorities", {})
            )
            self.mode_manager.add_custom_mode(mode_name, config)
    
    def switch_mode(self, mode: AgentMode) -> str:
        """Switch to a different mode."""
        mode_config = self.mode_manager.set_mode(mode)
        return f"Switched to {mode_config.name}: {mode_config.description}"
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_context(self):
        """Clear the context manager."""
        self.context_manager = ContextManager(
            max_tokens=self.config.get("max_context_tokens", 100000)
        )
    
    def export_session(self, filepath: str):
        """Export the current session data."""
        session_data = {
            "conversation_history": self.conversation_history,
            "metrics": self.cost_tracker.get_statistics(),
            "context_usage": self.context_manager.get_usage(),
            "current_mode": self.mode_manager.current_mode.value
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)