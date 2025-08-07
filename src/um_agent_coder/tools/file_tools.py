import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import glob

from .base import Tool, ToolResult


class FileReader(Tool):
    """Read contents of a file."""
    
    def execute(self, file_path: str, start_line: Optional[int] = None, 
                end_line: Optional[int] = None) -> ToolResult:
        try:
            path = Path(file_path)
            if not path.exists():
                return ToolResult(False, None, f"File not found: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if start_line is not None or end_line is not None:
                start = (start_line - 1) if start_line else 0
                end = end_line if end_line else len(lines)
                lines = lines[start:end]
            
            content = ''.join(lines)
            return ToolResult(True, content)
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "file_path": {"type": "string", "required": True},
            "start_line": {"type": "integer", "required": False},
            "end_line": {"type": "integer", "required": False}
        }


class FileWriter(Tool):
    """Write or create a file."""
    
    def execute(self, file_path: str, content: str, mode: str = 'w') -> ToolResult:
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, mode, encoding='utf-8') as f:
                f.write(content)
            
            return ToolResult(True, f"Successfully wrote to {file_path}")
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "file_path": {"type": "string", "required": True},
            "content": {"type": "string", "required": True},
            "mode": {"type": "string", "required": False, "default": "w"}
        }


class FileSearcher(Tool):
    """Search for files matching a pattern."""
    
    def execute(self, pattern: str, directory: str = ".") -> ToolResult:
        try:
            matches = []
            search_pattern = os.path.join(directory, pattern)
            
            for file_path in glob.glob(search_pattern, recursive=True):
                if os.path.isfile(file_path):
                    matches.append(file_path)
            
            return ToolResult(True, matches)
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "pattern": {"type": "string", "required": True},
            "directory": {"type": "string", "required": False, "default": "."}
        }