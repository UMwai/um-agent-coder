from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    success: bool
    data: Any
    error: Optional[str] = None
    tokens_used: int = 0
    cost: float = 0.0


class Tool(ABC):
    """Base class for all tools."""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.description = self.__doc__ or "No description"
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return parameter schema for this tool."""
        pass


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def get_all(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self.tools.values())
    
    def get_tools_for_task(self, task_type: str) -> List[Tool]:
        """Get relevant tools for a specific task type."""
        # TODO: Implement task-based tool selection
        return self.get_all()