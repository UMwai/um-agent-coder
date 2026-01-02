from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ToolResult:
    success: bool
    data: Any
    error: Optional[str] = None
    tokens_used: int = 0
    cost: float = 0.0


class Tool(ABC):
    """Base class for all tools."""

    TASK_TYPES: list[str] = []

    def __init__(self):
        self.name = self.__class__.__name__
        self.description = self.__doc__ or "No description"

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return parameter schema for this tool."""
        pass


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def get_all(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self.tools.values())

    def get_tools_for_task(self, task_type: str) -> list[Tool]:
        """Get relevant tools for a specific task type."""
        # Filter tools based on task type or include if 'general'
        return [
            tool
            for tool in self.tools.values()
            if task_type in tool.TASK_TYPES or "general" in tool.TASK_TYPES
        ]
