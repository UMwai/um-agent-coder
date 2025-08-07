import os
import re
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult


class CodeSearcher(Tool):
    """Search for code patterns in files."""
    
    def execute(self, pattern: str, file_pattern: str = "*.py", 
                directory: str = ".", case_sensitive: bool = True) -> ToolResult:
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)
            results = []
            
            for root, _, files in os.walk(directory):
                for file in files:
                    if self._matches_pattern(file, file_pattern):
                        file_path = os.path.join(root, file)
                        matches = self._search_file(file_path, regex)
                        if matches:
                            results.append({
                                "file": file_path,
                                "matches": matches
                            })
            
            return ToolResult(True, results)
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)
    
    def _search_file(self, file_path: str, regex: re.Pattern) -> List[Dict[str, Any]]:
        matches = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if regex.search(line):
                        matches.append({
                            "line_number": line_num,
                            "line": line.strip(),
                            "match": regex.findall(line)
                        })
        except:
            pass
        return matches
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "pattern": {"type": "string", "required": True},
            "file_pattern": {"type": "string", "required": False, "default": "*.py"},
            "directory": {"type": "string", "required": False, "default": "."},
            "case_sensitive": {"type": "boolean", "required": False, "default": True}
        }


class ProjectAnalyzer(Tool):
    """Analyze project structure and dependencies."""
    
    def execute(self, directory: str = ".") -> ToolResult:
        try:
            analysis = {
                "structure": self._analyze_structure(directory),
                "python_files": self._find_python_files(directory),
                "imports": self._analyze_imports(directory),
                "classes": self._find_classes(directory),
                "functions": self._find_functions(directory)
            }
            
            return ToolResult(True, analysis)
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    def _analyze_structure(self, directory: str) -> Dict[str, Any]:
        structure = {"dirs": 0, "files": 0, "py_files": 0}
        
        for root, dirs, files in os.walk(directory):
            structure["dirs"] += len(dirs)
            structure["files"] += len(files)
            structure["py_files"] += sum(1 for f in files if f.endswith('.py'))
        
        return structure
    
    def _find_python_files(self, directory: str) -> List[str]:
        py_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    py_files.append(os.path.join(root, file))
        return py_files
    
    def _analyze_imports(self, directory: str) -> List[str]:
        imports = set()
        for py_file in self._find_python_files(directory):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.add(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.add(node.module)
            except:
                pass
        return sorted(list(imports))
    
    def _find_classes(self, directory: str) -> List[Dict[str, str]]:
        classes = []
        for py_file in self._find_python_files(directory):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            classes.append({
                                "name": node.name,
                                "file": py_file,
                                "line": node.lineno
                            })
            except:
                pass
        return classes
    
    def _find_functions(self, directory: str) -> List[Dict[str, str]]:
        functions = []
        for py_file in self._find_python_files(directory):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            functions.append({
                                "name": node.name,
                                "file": py_file,
                                "line": node.lineno
                            })
            except:
                pass
        return functions
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "directory": {"type": "string", "required": False, "default": "."}
        }