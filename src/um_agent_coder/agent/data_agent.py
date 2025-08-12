"""
Data-focused Agent: Specialized for data engineering, science, and ML tasks.
Combines data modes, tools, and optimized models for data workflows.
"""

import uuid
from typing import Dict, Any, List, Optional, Tuple
import json
import yaml
from pathlib import Path

from um_agent_coder.llm.base import LLM
from um_agent_coder.context import ContextManager, ContextType
from um_agent_coder.models import ModelRegistry
from .planner import TaskPlanner, TaskAnalysis, ExecutionPlan
from .cost_tracker import CostTracker
from .data_modes import DataAgentMode, DataModeManager, DataModeConfig


class DataAgent:
    """
    Specialized agent for data engineering, data science, and ML tasks.
    
    Features:
    - Multiple data-focused modes (Data Engineer, Data Scientist, ML Engineer, etc.)
    - Specialized data tools (SQL, profiling, schema analysis, pipeline building)
    - Optimized model selection for data tasks
    - Data lineage tracking
    - Pipeline orchestration
    """
    
    def __init__(self, llm: LLM, config: Dict[str, Any]):
        self.llm = llm
        self.config = config
        
        # Core components
        self.context_manager = ContextManager(
            max_tokens=config.get("max_context_tokens", 150000)  # Larger context for data
        )
        self.task_planner = TaskPlanner()
        self.cost_tracker = CostTracker()
        self.model_registry = ModelRegistry()
        self.mode_manager = DataModeManager()
        
        # Data-specific components
        self.data_tools = self._initialize_data_tools()
        self.pipeline_registry = {}
        self.data_lineage = {}
        
        # Settings
        self.verbose = config.get("verbose", True)
        self.auto_profile = config.get("auto_profile", True)
        self.validate_data = config.get("validate_data", True)
        self.track_lineage = config.get("track_lineage", True)
        
        # Conversation history for context
        self.conversation_history = []
        
        # Load data configurations
        self._load_data_configs()
    
    def run(
        self,
        prompt: str,
        mode: Optional[DataAgentMode] = None,
        data_source: Optional[str] = None,
        output_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the data agent with the given prompt.
        
        Args:
            prompt: User's request
            mode: Optional specific data mode to use
            data_source: Optional data source (file path, connection string, etc.)
            output_format: Optional output format (sql, python, dbt, etc.)
            
        Returns:
            Comprehensive result with data insights and artifacts
        """
        task_id = str(uuid.uuid4())[:8]
        
        try:
            # 1. Mode Detection/Selection
            if mode is None:
                mode = self.mode_manager.detect_mode_from_prompt(prompt)
                if self.verbose:
                    print(f"📊 Auto-detected mode: {mode.value}")
            
            mode_config = self.mode_manager.set_mode(mode)
            
            # 2. Data Context Loading
            if data_source:
                self._load_data_context(data_source)
            
            # 3. Task Analysis with Data Focus
            task_analysis = self._analyze_data_task(prompt, mode_config)
            
            # 4. Execution Planning
            execution_plan = self._create_data_pipeline_plan(
                task_analysis, prompt, mode_config, output_format
            )
            
            # Start tracking
            self.cost_tracker.start_task(task_id, prompt, len(execution_plan.steps))
            
            # 5. Execute Data Pipeline
            if self.verbose:
                print(f"🚀 Executing {len(execution_plan.steps)} steps in {mode_config.name}...")
            
            results = self._execute_data_pipeline(execution_plan, mode_config)
            
            # 6. Generate Response with Data Insights
            response, artifacts = self._generate_data_response(
                prompt, results, mode_config, output_format
            )
            
            # 7. Track Data Lineage
            if self.track_lineage:
                self._update_data_lineage(task_id, prompt, results)
            
            # Complete tracking
            self.cost_tracker.complete_task(success=True)
            
            # 8. Return comprehensive data result
            return {
                "response": response,
                "success": True,
                "task_id": task_id,
                "mode": mode.value,
                "artifacts": artifacts,
                "data_insights": self._extract_insights(results),
                "execution_details": {
                    "steps_executed": len(results),
                    "tools_used": list(set(r.get("tool", "") for r in results if r.get("tool"))),
                    "data_processed": self._calculate_data_volume(results)
                },
                "metrics": self.cost_tracker.get_statistics(),
                "lineage": self.data_lineage.get(task_id, {})
            }
            
        except Exception as e:
            self.cost_tracker.complete_task(success=False, error=str(e))
            return {
                "response": f"Error: {str(e)}",
                "success": False,
                "task_id": task_id,
                "error": str(e)
            }
    
    def _initialize_data_tools(self) -> Dict[str, Any]:
        """Initialize data-specific tools."""
        from um_agent_coder.tools.data_tools import (
            SQLExecutor, SchemaAnalyzer, DataProfiler,
            PipelineBuilder, DataValidator, DimensionalModeler
        )
        
        tools = {
            "sql_executor": SQLExecutor(),
            "schema_analyzer": SchemaAnalyzer(),
            "data_profiler": DataProfiler(),
            "pipeline_builder": PipelineBuilder(),
            "data_validator": DataValidator(),
            "dimensional_modeler": DimensionalModeler()
        }
        
        return tools
    
    def _load_data_configs(self):
        """Load data-specific configurations."""
        config_path = Path(self.config.get("data_config_path", "config/data_config.yaml"))
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                data_config = yaml.safe_load(f)
                
                # Load connection strings
                self.connections = data_config.get("connections", {})
                
                # Load validation rules
                self.validation_rules = data_config.get("validation_rules", {})
                
                # Load pipeline templates
                self.pipeline_templates = data_config.get("pipeline_templates", {})
    
    def _load_data_context(self, data_source: str):
        """Load data source into context."""
        if self.verbose:
            print(f"📁 Loading data context from: {data_source}")
        
        # Profile the data if enabled
        if self.auto_profile and Path(data_source).exists():
            profiler = self.data_tools["data_profiler"]
            profile_result = profiler.execute(data_source, sample_size=10000)
            
            if profile_result.success:
                self.context_manager.add(
                    content=json.dumps(profile_result.data, indent=2),
                    type=ContextType.PROJECT_INFO,
                    source="data_profile",
                    priority=9
                )
        
        # Analyze schema if it's a database connection
        if data_source.startswith(("sqlite://", "postgresql://", "mysql://")):
            analyzer = self.data_tools["schema_analyzer"]
            schema_result = analyzer.execute(connection_string=data_source)
            
            if schema_result.success:
                self.context_manager.add(
                    content=json.dumps(schema_result.data, indent=2),
                    type=ContextType.PROJECT_INFO,
                    source="database_schema",
                    priority=9
                )
    
    def _analyze_data_task(
        self,
        prompt: str,
        mode_config: DataModeConfig
    ) -> TaskAnalysis:
        """Analyze data-specific task requirements."""
        # Use base task analysis
        base_analysis = self.task_planner.analyze_task(prompt)
        
        # Enhance with data-specific analysis
        data_keywords = {
            "etl": ["extract", "transform", "load", "pipeline", "airflow"],
            "analysis": ["analyze", "explore", "eda", "statistics", "distribution"],
            "modeling": ["model", "predict", "train", "feature", "algorithm"],
            "sql": ["query", "select", "join", "aggregate", "database"],
            "visualization": ["plot", "chart", "dashboard", "visualize", "report"]
        }
        
        # Detect data task types
        detected_types = []
        prompt_lower = prompt.lower()
        
        for task_type, keywords in data_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                detected_types.append(task_type)
        
        # Add to analysis
        base_analysis.detected_data_types = detected_types
        base_analysis.requires_data_validation = self.validate_data
        base_analysis.mode_capabilities = mode_config.specialized_capabilities
        
        return base_analysis
    
    def _create_data_pipeline_plan(
        self,
        analysis: TaskAnalysis,
        prompt: str,
        mode_config: DataModeConfig,
        output_format: Optional[str]
    ) -> ExecutionPlan:
        """Create execution plan for data pipeline."""
        base_plan = self.task_planner.create_execution_plan(analysis, prompt)
        
        # Enhance with data-specific steps
        data_steps = []
        
        # Add data validation step if needed
        if self.validate_data and hasattr(analysis, 'detected_data_types'):
            if 'etl' in analysis.detected_data_types or 'modeling' in analysis.detected_data_types:
                data_steps.append({
                    "action": "DataValidator",
                    "description": "Validate data quality",
                    "parameters": {"rules": self.validation_rules},
                    "priority": 9,
                    "estimated_tokens": 500
                })
        
        # Add profiling step for analysis tasks
        if hasattr(analysis, 'detected_data_types') and 'analysis' in analysis.detected_data_types:
            data_steps.append({
                "action": "DataProfiler",
                "description": "Profile dataset",
                "parameters": {"profile_all": True},
                "priority": 8,
                "estimated_tokens": 1000
            })
        
        # Insert data steps at appropriate positions
        enhanced_steps = base_plan.steps[:1] + data_steps + base_plan.steps[1:]
        base_plan.steps = enhanced_steps
        
        # Adjust for output format
        if output_format:
            base_plan.output_format = output_format
        
        return base_plan
    
    def _execute_data_pipeline(
        self,
        plan: ExecutionPlan,
        mode_config: DataModeConfig
    ) -> List[Dict[str, Any]]:
        """Execute data pipeline with specialized handling."""
        results = []
        
        for step in plan.steps:
            if self.verbose:
                print(f"  ▶ {step.description}")
            
            # Route to appropriate tool
            tool_name = step.action.lower().replace(" ", "_")
            
            if tool_name in self.data_tools:
                tool = self.data_tools[tool_name]
                result = tool.execute(**step.parameters)
            else:
                # Fall back to LLM for non-tool steps
                result = self._execute_llm_step(step, mode_config)
            
            results.append({
                "tool": step.action,
                "description": step.description,
                "success": result.success if hasattr(result, 'success') else True,
                "data": result.data if hasattr(result, 'data') else result
            })
            
            # Update context with results
            if hasattr(result, 'data') and result.data:
                self.context_manager.add(
                    content=str(result.data)[:3000],
                    type=ContextType.TOOL_RESULT,
                    source=f"{step.action}_result",
                    priority=step.priority
                )
        
        return results
    
    def _execute_llm_step(self, step, mode_config: DataModeConfig) -> Any:
        """Execute a step using the LLM."""
        prompt = f"""
{mode_config.system_prompt}

Task: {step.description}
Parameters: {json.dumps(step.parameters, indent=2)}

Please complete this task and provide the result.
"""
        
        response = self.llm.chat(prompt)
        
        return type('Result', (), {
            'success': True,
            'data': response
        })()
    
    def _generate_data_response(
        self,
        prompt: str,
        results: List[Dict[str, Any]],
        mode_config: DataModeConfig,
        output_format: Optional[str]
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response with data insights and artifacts."""
        # Collect artifacts
        artifacts = {}
        
        for result in results:
            if result.get("tool") == "PipelineBuilder" and result.get("data"):
                artifacts["pipeline_config"] = result["data"]
            elif result.get("tool") == "DimensionalModeler" and result.get("data"):
                artifacts["data_model"] = result["data"]
            elif result.get("tool") == "DataProfiler" and result.get("data"):
                artifacts["data_profile"] = result["data"]
        
        # Build response prompt
        context = self.context_manager.get_context()
        
        response_prompt = f"""
{mode_config.system_prompt}

Based on the following execution results, provide a comprehensive response to the user's request.

Original Request: {prompt}
Output Format: {output_format or 'default'}

Execution Results:
{json.dumps(results, indent=2)[:5000]}

Context:
{context[:5000]}

Please provide:
1. A clear summary of what was accomplished
2. Key insights or findings from the data
3. Any recommendations or next steps
4. Code or configuration if requested

Format the response appropriately for the requested output format.
"""
        
        response = self.llm.chat(response_prompt)
        
        return response, artifacts
    
    def _extract_insights(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key insights from execution results."""
        insights = {
            "data_quality_issues": [],
            "statistics": {},
            "recommendations": []
        }
        
        for result in results:
            if result.get("tool") == "DataProfiler" and result.get("data"):
                data = result["data"]
                if "quality_issues" in data:
                    insights["data_quality_issues"].extend(data["quality_issues"])
                if "dataset_info" in data:
                    insights["statistics"] = data["dataset_info"]
            
            elif result.get("tool") == "DataValidator" and result.get("data"):
                data = result["data"]
                if "validation_results" in data:
                    failed = [r for r in data["validation_results"] if not r["passed"]]
                    insights["data_quality_issues"].extend(failed)
        
        return insights
    
    def _calculate_data_volume(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate data volume processed."""
        volume = {
            "rows_processed": 0,
            "tables_analyzed": 0,
            "pipelines_created": 0
        }
        
        for result in results:
            if result.get("data"):
                data = result["data"]
                if isinstance(data, dict):
                    if "dataset_info" in data and "rows" in data["dataset_info"]:
                        volume["rows_processed"] += data["dataset_info"]["rows"]
                    if "schemas" in data:
                        volume["tables_analyzed"] += len(data["schemas"])
                    if "pipeline_type" in data:
                        volume["pipelines_created"] += 1
        
        return volume
    
    def _update_data_lineage(self, task_id: str, prompt: str, results: List[Dict[str, Any]]):
        """Track data lineage for the task."""
        lineage = {
            "task_id": task_id,
            "prompt": prompt,
            "timestamp": str(uuid.uuid4())[:8],
            "data_sources": [],
            "transformations": [],
            "outputs": []
        }
        
        for result in results:
            if result.get("tool") in ["SQLExecutor", "DataProfiler"]:
                lineage["data_sources"].append({
                    "tool": result["tool"],
                    "description": result["description"]
                })
            elif result.get("tool") in ["PipelineBuilder", "DimensionalModeler"]:
                lineage["outputs"].append({
                    "tool": result["tool"],
                    "description": result["description"],
                    "artifacts": list(result.get("data", {}).keys()) if result.get("data") else []
                })
        
        self.data_lineage[task_id] = lineage
    
    def get_recommended_model(self, mode: DataAgentMode) -> str:
        """Get the recommended model for a data mode."""
        recommendations = self.mode_manager.get_recommended_models(mode)
        
        # Return the first available model
        for model_name in recommendations:
            if self.model_registry.get(model_name):
                return model_name
        
        # Default to Claude 3.5 Sonnet
        return "claude-3.5-sonnet-20241022"
    
    def export_pipeline(self, task_id: str, format: str = "yaml") -> str:
        """Export pipeline configuration from a task."""
        if task_id not in self.data_lineage:
            return None
        
        lineage = self.data_lineage[task_id]
        
        if format == "yaml":
            return yaml.dump(lineage, default_flow_style=False)
        elif format == "json":
            return json.dumps(lineage, indent=2)
        else:
            return str(lineage)