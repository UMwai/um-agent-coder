"""
Smart tools inspired by Roo-Code's tool system.
Enhanced tools with better error handling, validation, and context awareness.
"""

import os
import subprocess
import json
import ast
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from .base import Tool, ToolResult


@dataclass
class CodeContext:
    """Context information for code operations."""
    language: str
    framework: Optional[str]
    dependencies: List[str]
    style_guide: Optional[str]


class SmartFileReader(Tool):
    """Enhanced file reader with intelligent content parsing."""
    
    TASK_TYPES = ["code", "development", "smart"]

    def __init__(self):
        super().__init__()
        self.name = "SmartFileReader"
        self.description = "Read files with intelligent parsing and context extraction"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "file_path": {"type": "string", "required": True}
        }

    def execute(self, file_path: str, **kwargs) -> ToolResult:
        """Read file with enhanced parsing."""
        try:
            path = Path(file_path)
            if not path.exists():
                return ToolResult(
                    success=False,
                    error=f"File not found: {file_path}"
                )
            
            content = path.read_text(encoding='utf-8')
            
            # Extract metadata based on file type
            metadata = self._extract_metadata(file_path, content)
            
            # Parse structured content if applicable
            parsed_content = self._parse_content(file_path, content)
            
            return ToolResult(
                success=True,
                data={
                    "content": content,
                    "metadata": metadata,
                    "parsed": parsed_content,
                    "stats": {
                        "lines": len(content.splitlines()),
                        "size": len(content),
                        "type": path.suffix
                    }
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error reading file: {str(e)}"
            )
    
    def _extract_metadata(self, file_path: str, content: str) -> Dict[str, Any]:
        """Extract metadata from file content."""
        metadata = {}
        
        # Language detection
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust'
        }
        metadata['language'] = language_map.get(ext, 'unknown')
        
        # Extract imports/dependencies
        if metadata['language'] == 'python':
            imports = re.findall(r'^(?:from|import)\s+(\S+)', content, re.MULTILINE)
            metadata['imports'] = list(set(imports))
        elif metadata['language'] in ['javascript', 'typescript']:
            imports = re.findall(r'(?:import|require)\s*\(?\s*[\'"]([^\'"]+)[\'"]', content)
            metadata['imports'] = list(set(imports))
        
        # Extract function/class definitions
        if metadata['language'] == 'python':
            functions = re.findall(r'^def\s+(\w+)\s*\(', content, re.MULTILINE)
            classes = re.findall(r'^class\s+(\w+)\s*[\(:]', content, re.MULTILINE)
            metadata['functions'] = functions
            metadata['classes'] = classes
        
        return metadata
    
    def _parse_content(self, file_path: str, content: str) -> Optional[Any]:
        """Parse structured content."""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.json':
            try:
                return json.loads(content)
            except:
                return None
        elif ext in ['.yaml', '.yml']:
            try:
                import yaml
                return yaml.safe_load(content)
            except:
                return None
        elif ext == '.py':
            try:
                return ast.parse(content)
            except:
                return None
        
        return None


class SmartFileWriter(Tool):
    """Enhanced file writer with validation and backup."""
    
    TASK_TYPES = ["code", "development", "smart"]

    def __init__(self):
        super().__init__()
        self.name = "SmartFileWriter"
        self.description = "Write files with validation, formatting, and backup"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "file_path": {"type": "string", "required": True},
            "content": {"type": "string", "required": True},
            "validate": {"type": "boolean", "required": False, "default": True},
            "backup": {"type": "boolean", "required": False, "default": True},
            "format_code": {"type": "boolean", "required": False, "default": True}
        }
    
    def execute(
        self, 
        file_path: str, 
        content: str, 
        validate: bool = True,
        backup: bool = True,
        format_code: bool = True,
        **kwargs
    ) -> ToolResult:
        """Write file with enhanced features."""
        try:
            path = Path(file_path)
            
            # Create backup if file exists
            if backup and path.exists():
                backup_path = path.with_suffix(path.suffix + '.bak')
                backup_path.write_text(path.read_text())
            
            # Format code if requested
            if format_code:
                content = self._format_code(file_path, content)
            
            # Validate before writing
            if validate:
                validation = self._validate_content(file_path, content)
                if not validation['valid']:
                    return ToolResult(
                        success=False,
                        error=f"Validation failed: {validation['error']}"
                    )
            
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the file
            path.write_text(content, encoding='utf-8')
            
            return ToolResult(
                success=True,
                data={
                    "file_path": str(path),
                    "size": len(content),
                    "lines": len(content.splitlines()),
                    "backup_created": backup and path.exists()
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error writing file: {str(e)}"
            )
    
    def _validate_content(self, file_path: str, content: str) -> Dict[str, Any]:
        """Validate file content."""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.json':
            try:
                json.loads(content)
                return {"valid": True}
            except json.JSONDecodeError as e:
                return {"valid": False, "error": str(e)}
        
        elif ext == '.py':
            try:
                compile(content, file_path, 'exec')
                return {"valid": True}
            except SyntaxError as e:
                return {"valid": False, "error": str(e)}
        
        elif ext in ['.yaml', '.yml']:
            try:
                import yaml
                yaml.safe_load(content)
                return {"valid": True}
            except Exception as e:
                return {"valid": False, "error": str(e)}
        
        return {"valid": True}  # No validation for other file types
    
    def _format_code(self, file_path: str, content: str) -> str:
        """Format code based on language."""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.py':
            try:
                import black
                return black.format_str(content, mode=black.Mode())
            except:
                pass
        
        elif ext == '.json':
            try:
                obj = json.loads(content)
                return json.dumps(obj, indent=2)
            except:
                pass
        
        return content


class SmartCodeSearcher(Tool):
    """Enhanced code searcher with semantic understanding."""
    
    TASK_TYPES = ["code", "development", "smart", "research"]

    def __init__(self):
        super().__init__()
        self.name = "SmartCodeSearcher"
        self.description = "Search code with semantic understanding and context"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "pattern": {"type": "string", "required": True},
            "search_type": {"type": "string", "required": False, "default": "regex", "enum": ["regex", "semantic", "ast"]},
            "file_types": {"type": "array", "items": {"type": "string"}, "required": False},
            "directory": {"type": "string", "required": False, "default": "."},
            "context_lines": {"type": "integer", "required": False, "default": 3}
        }
    
    def execute(
        self, 
        pattern: str,
        search_type: str = "regex",  # regex, semantic, ast
        file_types: Optional[List[str]] = None,
        directory: str = ".",
        context_lines: int = 3,
        **kwargs
    ) -> ToolResult:
        """Search code with enhanced capabilities."""
        try:
            results = []
            
            if search_type == "regex":
                results = self._regex_search(pattern, directory, file_types, context_lines)
            elif search_type == "semantic":
                results = self._semantic_search(pattern, directory, file_types)
            elif search_type == "ast":
                results = self._ast_search(pattern, directory, file_types)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown search type: {search_type}"
                )
            
            return ToolResult(
                success=True,
                data={
                    "pattern": pattern,
                    "search_type": search_type,
                    "matches": results,
                    "total_matches": len(results)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Search error: {str(e)}"
            )
    
    def _regex_search(
        self, 
        pattern: str, 
        directory: str, 
        file_types: Optional[List[str]],
        context_lines: int
    ) -> List[Dict[str, Any]]:
        """Perform regex search."""
        results = []
        path = Path(directory)
        
        # Determine file patterns
        if file_types:
            patterns = [f"*.{ft}" for ft in file_types]
        else:
            patterns = ["*.py", "*.js", "*.ts", "*.java", "*.cpp", "*.go"]
        
        for pattern_glob in patterns:
            for file_path in path.rglob(pattern_glob):
                if file_path.is_file():
                    matches = self._search_file(file_path, pattern, context_lines)
                    if matches:
                        results.extend(matches)
        
        return results
    
    def _search_file(
        self, 
        file_path: Path, 
        pattern: str, 
        context_lines: int
    ) -> List[Dict[str, Any]]:
        """Search within a single file."""
        matches = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.splitlines()
            
            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    # Get context
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    context = lines[start:end]
                    
                    matches.append({
                        "file": str(file_path),
                        "line_number": i + 1,
                        "line": line,
                        "context": context,
                        "context_start": start + 1,
                        "context_end": end
                    })
        except:
            pass  # Skip files that can't be read
        
        return matches
    
    def _semantic_search(
        self, 
        query: str, 
        directory: str, 
        file_types: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Perform semantic search (simplified version)."""
        # This would use embeddings in a real implementation
        # For now, fallback to keyword search
        keywords = query.lower().split()
        results = []
        
        path = Path(directory)
        for file_path in path.rglob("*"):
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding='utf-8').lower()
                    score = sum(1 for kw in keywords if kw in content)
                    if score > 0:
                        results.append({
                            "file": str(file_path),
                            "relevance_score": score,
                            "preview": content[:200]
                        })
                except:
                    pass
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:20]  # Top 20 results
    
    def _ast_search(
        self, 
        pattern: str, 
        directory: str, 
        file_types: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Search using AST patterns (Python only for now)."""
        results = []
        path = Path(directory)
        
        for file_path in path.rglob("*.py"):
            try:
                content = file_path.read_text(encoding='utf-8')
                tree = ast.parse(content)
                
                # Find matching nodes
                for node in ast.walk(tree):
                    if self._match_ast_pattern(node, pattern):
                        results.append({
                            "file": str(file_path),
                            "node_type": node.__class__.__name__,
                            "line_number": getattr(node, 'lineno', 0),
                            "pattern": pattern
                        })
            except:
                pass
        
        return results
    
    def _match_ast_pattern(self, node: ast.AST, pattern: str) -> bool:
        """Match AST node against pattern."""
        # Simple pattern matching for demonstration
        if pattern == "function":
            return isinstance(node, ast.FunctionDef)
        elif pattern == "class":
            return isinstance(node, ast.ClassDef)
        elif pattern == "import":
            return isinstance(node, (ast.Import, ast.ImportFrom))
        return False


class SmartCommandExecutor(Tool):
    """Enhanced command executor with safety and environment management."""
    
    TASK_TYPES = ["system", "general", "smart"]

    def __init__(self):
        super().__init__()
        self.name = "SmartCommandExecutor"
        self.description = "Execute commands with safety checks and environment management"
        self.safe_commands = [
            "ls", "pwd", "echo", "cat", "grep", "find",
            "python", "node", "npm", "pip", "git status",
            "git diff", "git log", "pytest", "jest"
        ]
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "command": {"type": "string", "required": True},
            "cwd": {"type": "string", "required": False},
            "env": {"type": "object", "required": False},
            "timeout": {"type": "integer", "required": False, "default": 30},
            "safe_mode": {"type": "boolean", "required": False, "default": True}
        }

    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        safe_mode: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute command with enhanced features."""
        try:
            # Safety check
            if safe_mode and not self._is_safe_command(command):
                return ToolResult(
                    success=False,
                    error=f"Command blocked by safe mode: {command}"
                )
            
            # Prepare environment
            exec_env = os.environ.copy()
            if env:
                exec_env.update(env)
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                env=exec_env,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return ToolResult(
                success=result.returncode == 0,
                data={
                    "command": command,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "cwd": cwd or os.getcwd()
                },
                error=result.stderr if result.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Command execution error: {str(e)}"
            )
    
    def _is_safe_command(self, command: str) -> bool:
        """Check if command is safe to execute."""
        # Check against safe command list
        for safe_cmd in self.safe_commands:
            if command.startswith(safe_cmd):
                return True
        
        # Block dangerous commands
        dangerous = ["rm", "del", "format", "shutdown", "reboot", "kill"]
        for danger in dangerous:
            if danger in command.lower():
                return False
        
        return False


class SmartProjectAnalyzer(Tool):
    """Enhanced project analyzer with deep insights."""
    
    TASK_TYPES = ["code", "development", "smart", "research"]

    def __init__(self):
        super().__init__()
        self.name = "SmartProjectAnalyzer"
        self.description = "Analyze project structure with deep insights"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "directory": {"type": "string", "required": False, "default": "."}
        }
    
    def execute(self, directory: str = ".", **kwargs) -> ToolResult:
        """Analyze project with enhanced capabilities."""
        try:
            path = Path(directory)
            
            analysis = {
                "structure": self._analyze_structure(path),
                "languages": self._detect_languages(path),
                "frameworks": self._detect_frameworks(path),
                "dependencies": self._analyze_dependencies(path),
                "patterns": self._detect_patterns(path),
                "metrics": self._calculate_metrics(path)
            }
            
            return ToolResult(
                success=True,
                data=analysis
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Analysis error: {str(e)}"
            )
    
    def _analyze_structure(self, path: Path) -> Dict[str, Any]:
        """Analyze directory structure."""
        structure = {
            "directories": [],
            "files": [],
            "total_files": 0,
            "total_dirs": 0
        }
        
        for item in path.rglob("*"):
            if item.is_dir():
                structure["directories"].append(str(item.relative_to(path)))
                structure["total_dirs"] += 1
            else:
                structure["files"].append(str(item.relative_to(path)))
                structure["total_files"] += 1
        
        return structure
    
    def _detect_languages(self, path: Path) -> Dict[str, int]:
        """Detect programming languages used."""
        languages = {}
        
        for file_path in path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in ['.py', '.js', '.ts', '.java', '.cpp', '.go', '.rs']:
                    lang = {
                        '.py': 'Python',
                        '.js': 'JavaScript',
                        '.ts': 'TypeScript',
                        '.java': 'Java',
                        '.cpp': 'C++',
                        '.go': 'Go',
                        '.rs': 'Rust'
                    }.get(ext, 'Unknown')
                    languages[lang] = languages.get(lang, 0) + 1
        
        return languages
    
    def _detect_frameworks(self, path: Path) -> List[str]:
        """Detect frameworks and libraries."""
        frameworks = []
        
        # Check for common framework indicators
        if (path / "package.json").exists():
            frameworks.append("Node.js")
            try:
                with open(path / "package.json") as f:
                    data = json.load(f)
                    deps = data.get("dependencies", {})
                    if "react" in deps:
                        frameworks.append("React")
                    if "vue" in deps:
                        frameworks.append("Vue")
                    if "express" in deps:
                        frameworks.append("Express")
            except:
                pass
        
        if (path / "requirements.txt").exists():
            frameworks.append("Python")
            try:
                content = (path / "requirements.txt").read_text()
                if "django" in content.lower():
                    frameworks.append("Django")
                if "flask" in content.lower():
                    frameworks.append("Flask")
                if "fastapi" in content.lower():
                    frameworks.append("FastAPI")
            except:
                pass
        
        return frameworks
    
    def _analyze_dependencies(self, path: Path) -> Dict[str, List[str]]:
        """Analyze project dependencies."""
        deps = {}
        
        # Python dependencies
        if (path / "requirements.txt").exists():
            try:
                content = (path / "requirements.txt").read_text()
                deps["python"] = [
                    line.strip() for line in content.splitlines()
                    if line.strip() and not line.startswith("#")
                ]
            except:
                pass
        
        # Node dependencies
        if (path / "package.json").exists():
            try:
                with open(path / "package.json") as f:
                    data = json.load(f)
                    deps["npm"] = list(data.get("dependencies", {}).keys())
            except:
                pass
        
        return deps
    
    def _detect_patterns(self, path: Path) -> Dict[str, Any]:
        """Detect design patterns and architecture."""
        patterns = {
            "has_tests": any(path.rglob("test_*.py")) or any(path.rglob("*.test.js")),
            "has_ci": (path / ".github/workflows").exists() or (path / ".gitlab-ci.yml").exists(),
            "has_docker": (path / "Dockerfile").exists(),
            "has_docs": (path / "docs").exists() or (path / "documentation").exists(),
            "has_config": (path / "config").exists() or (path / ".env").exists()
        }
        
        return patterns
    
    def _calculate_metrics(self, path: Path) -> Dict[str, Any]:
        """Calculate project metrics."""
        metrics = {
            "total_lines": 0,
            "code_files": 0,
            "avg_file_size": 0
        }
        
        total_size = 0
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.py', '.js', '.ts', '.java']:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    metrics["total_lines"] += len(content.splitlines())
                    metrics["code_files"] += 1
                    total_size += len(content)
                except:
                    pass
        
        if metrics["code_files"] > 0:
            metrics["avg_file_size"] = total_size // metrics["code_files"]
        
        return metrics