from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    CODE_MODIFICATION = "code_modification"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    ANALYSIS = "analysis"
    TESTING = "testing"
    GENERAL = "general"


@dataclass
class TaskStep:
    description: str
    action: str  # tool to use
    parameters: Dict[str, Any]
    estimated_tokens: int
    priority: int = 5


@dataclass
class TaskAnalysis:
    task_type: TaskType
    complexity: int  # 1-10
    estimated_tokens: int
    required_tools: List[str]
    files_to_analyze: List[str]
    potential_risks: List[str]


@dataclass
class ExecutionPlan:
    steps: List[TaskStep]
    total_estimated_tokens: int
    estimated_cost: float
    estimated_time_minutes: float


class TaskPlanner:
    """Plans and decomposes tasks for efficient execution."""
    
    def __init__(self):
        self.task_patterns = {
            "implement": TaskType.CODE_GENERATION,
            "create": TaskType.CODE_GENERATION,
            "add": TaskType.CODE_MODIFICATION,
            "fix": TaskType.DEBUGGING,
            "debug": TaskType.DEBUGGING,
            "refactor": TaskType.REFACTORING,
            "optimize": TaskType.REFACTORING,
            "document": TaskType.DOCUMENTATION,
            "analyze": TaskType.ANALYSIS,
            "test": TaskType.TESTING
        }
    
    def analyze_task(self, prompt: str) -> TaskAnalysis:
        """
        Analyze a user prompt to understand the task.
        """
        # Determine task type
        task_type = self._determine_task_type(prompt)
        
        # Estimate complexity
        complexity = self._estimate_complexity(prompt, task_type)
        
        # Identify required tools
        required_tools = self._identify_required_tools(prompt, task_type)
        
        # Find files to analyze
        files_to_analyze = self._extract_file_references(prompt)
        
        # Identify risks
        risks = self._identify_risks(prompt, task_type)
        
        # Estimate tokens
        estimated_tokens = complexity * 1000  # Rough estimate
        
        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity,
            estimated_tokens=estimated_tokens,
            required_tools=required_tools,
            files_to_analyze=files_to_analyze,
            potential_risks=risks
        )
    
    def create_execution_plan(self, analysis: TaskAnalysis, prompt: str) -> ExecutionPlan:
        """
        Create a step-by-step execution plan.
        """
        steps = []
        
        # Step 1: Analyze project structure if needed
        if analysis.complexity > 5:
            steps.append(TaskStep(
                description="Analyze project structure",
                action="ProjectAnalyzer",
                parameters={"directory": "."},
                estimated_tokens=500,
                priority=9
            ))
        
        # Step 2: Read relevant files
        for file in analysis.files_to_analyze:
            steps.append(TaskStep(
                description=f"Read {file}",
                action="FileReader",
                parameters={"file_path": file},
                estimated_tokens=200,
                priority=8
            ))
        
        # Step 3: Search for related code if needed
        if analysis.task_type in [TaskType.CODE_MODIFICATION, TaskType.DEBUGGING]:
            steps.append(TaskStep(
                description="Search for related code",
                action="CodeSearcher",
                parameters={"pattern": self._extract_search_pattern(prompt)},
                estimated_tokens=300,
                priority=7
            ))
        
        # Step 4: Main task execution
        main_step = self._create_main_task_step(analysis, prompt)
        steps.append(main_step)
        
        # Step 5: Validation
        if analysis.task_type in [TaskType.CODE_GENERATION, TaskType.CODE_MODIFICATION]:
            steps.append(TaskStep(
                description="Validate changes",
                action="CommandExecutor",
                parameters={"command": "python -m py_compile"},
                estimated_tokens=100,
                priority=6
            ))
        
        # Calculate totals
        total_tokens = sum(step.estimated_tokens for step in steps)
        estimated_cost = self._estimate_cost(total_tokens)
        estimated_time = len(steps) * 0.5  # 30 seconds per step
        
        return ExecutionPlan(
            steps=steps,
            total_estimated_tokens=total_tokens,
            estimated_cost=estimated_cost,
            estimated_time_minutes=estimated_time
        )
    
    def _determine_task_type(self, prompt: str) -> TaskType:
        """Determine task type from prompt."""
        prompt_lower = prompt.lower()
        
        for keyword, task_type in self.task_patterns.items():
            if keyword in prompt_lower:
                return task_type
        
        return TaskType.GENERAL
    
    def _estimate_complexity(self, prompt: str, task_type: TaskType) -> int:
        """Estimate task complexity (1-10)."""
        # Base complexity by task type
        base_complexity = {
            TaskType.CODE_GENERATION: 6,
            TaskType.CODE_MODIFICATION: 5,
            TaskType.DEBUGGING: 7,
            TaskType.REFACTORING: 8,
            TaskType.DOCUMENTATION: 3,
            TaskType.ANALYSIS: 4,
            TaskType.TESTING: 5,
            TaskType.GENERAL: 4
        }
        
        complexity = base_complexity.get(task_type, 5)
        
        # Adjust based on prompt indicators
        if "complex" in prompt.lower() or "entire" in prompt.lower():
            complexity += 2
        if "simple" in prompt.lower() or "basic" in prompt.lower():
            complexity -= 1
        if len(prompt) > 500:  # Long detailed prompt
            complexity += 1
        
        return max(1, min(10, complexity))
    
    def _identify_required_tools(self, prompt: str, task_type: TaskType) -> List[str]:
        """Identify tools needed for the task."""
        tools = []
        
        # Base tools by task type
        if task_type == TaskType.CODE_GENERATION:
            tools.extend(["FileWriter", "FileReader"])
        elif task_type == TaskType.CODE_MODIFICATION:
            tools.extend(["FileReader", "FileWriter", "CodeSearcher"])
        elif task_type == TaskType.DEBUGGING:
            tools.extend(["FileReader", "CodeSearcher", "CommandExecutor"])
        elif task_type == TaskType.ANALYSIS:
            tools.extend(["ProjectAnalyzer", "CodeSearcher"])
        
        # Add tools based on prompt content
        if "test" in prompt.lower():
            tools.append("CommandExecutor")
        if "search" in prompt.lower() or "find" in prompt.lower():
            tools.append("CodeSearcher")
        
        return list(set(tools))  # Remove duplicates
    
    def _extract_file_references(self, prompt: str) -> List[str]:
        """Extract file paths mentioned in prompt."""
        import re
        
        # Common file patterns
        patterns = [
            r'[\'"`]([^\'"`]+\.[a-zA-Z]+)[\'"`]',  # Quoted filenames
            r'(\S+\.py)',  # Python files
            r'(\S+\.js)',  # JavaScript files
            r'(\S+\.md)',  # Markdown files
        ]
        
        files = []
        for pattern in patterns:
            matches = re.findall(pattern, prompt)
            files.extend(matches)
        
        return list(set(files))
    
    def _identify_risks(self, prompt: str, task_type: TaskType) -> List[str]:
        """Identify potential risks in the task."""
        risks = []
        
        if task_type == TaskType.REFACTORING:
            risks.append("May break existing functionality")
        if "delete" in prompt.lower() or "remove" in prompt.lower():
            risks.append("Destructive operation")
        if "production" in prompt.lower():
            risks.append("Affects production code")
        if task_type == TaskType.DEBUGGING and "critical" in prompt.lower():
            risks.append("Critical bug - needs immediate attention")
        
        return risks
    
    def _extract_search_pattern(self, prompt: str) -> str:
        """Extract search pattern from prompt."""
        # Simple heuristic - find quoted strings or function names
        import re
        
        # Look for quoted strings
        quoted = re.findall(r'[\'"`]([^\'"`]+)[\'"`]', prompt)
        if quoted:
            return quoted[0]
        
        # Look for function/class names (CamelCase or snake_case)
        identifiers = re.findall(r'\b([A-Z][a-zA-Z0-9_]*|[a-z]+_[a-z_]+)\b', prompt)
        if identifiers:
            return identifiers[0]
        
        return "TODO"  # Default search pattern
    
    def _create_main_task_step(self, analysis: TaskAnalysis, prompt: str) -> TaskStep:
        """Create the main execution step."""
        action_map = {
            TaskType.CODE_GENERATION: "FileWriter",
            TaskType.CODE_MODIFICATION: "FileWriter",
            TaskType.DEBUGGING: "CodeSearcher",
            TaskType.REFACTORING: "FileWriter",
            TaskType.DOCUMENTATION: "FileWriter",
            TaskType.ANALYSIS: "ProjectAnalyzer",
            TaskType.TESTING: "CommandExecutor",
            TaskType.GENERAL: "FileReader"
        }
        
        return TaskStep(
            description=f"Execute {analysis.task_type.value}",
            action=action_map.get(analysis.task_type, "FileReader"),
            parameters={"prompt": prompt},
            estimated_tokens=analysis.estimated_tokens,
            priority=10
        )
    
    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on tokens."""
        # Assuming average model cost
        cost_per_1k = 0.01  # $0.01 per 1K tokens
        return (tokens / 1000) * cost_per_1k