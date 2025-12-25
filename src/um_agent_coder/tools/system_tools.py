import subprocess
import os
import shlex
from typing import Any, Dict, Optional
import time

from .base import Tool, ToolResult


class CommandExecutor(Tool):
    """Execute shell commands safely."""
    
    TASK_TYPES = ["system", "general", "code"]

    def __init__(self, timeout: int = 30, safe_mode: bool = True):
        super().__init__()
        self.timeout = timeout
        self.safe_mode = safe_mode
        self.forbidden_commands = [
            "rm -rf /",
            "format",
            "dd if=/dev/zero",
            ":(){ :|:& };:",  # Fork bomb
        ]
    
    def execute(self, command: str, cwd: Optional[str] = None, 
                env: Optional[Dict[str, str]] = None) -> ToolResult:
        """
        Execute a shell command.
        
        Args:
            command: Command to execute
            cwd: Working directory
            env: Environment variables
            
        Returns:
            ToolResult with stdout/stderr
        """
        # Safety check
        if self.safe_mode and self._is_dangerous_command(command):
            return ToolResult(
                False, 
                None, 
                f"Command blocked for safety: {command}"
            )
        
        try:
            # Parse command
            if os.name == 'nt':  # Windows
                shell = True
                cmd = command
            else:  # Unix
                shell = False
                cmd = shlex.split(command)
            
            # Set up environment
            exec_env = os.environ.copy()
            if env:
                exec_env.update(env)
            
            # Execute command
            start_time = time.time()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=exec_env,
                shell=shell,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                execution_time = time.time() - start_time
                
                result = {
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_code": process.returncode,
                    "execution_time": execution_time
                }
                
                # Determine success
                success = process.returncode == 0
                
                return ToolResult(
                    success=success,
                    data=result,
                    error=stderr if not success else None
                )
                
            except subprocess.TimeoutExpired:
                process.kill()
                return ToolResult(
                    False,
                    None,
                    f"Command timed out after {self.timeout} seconds"
                )
                
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    def _is_dangerous_command(self, command: str) -> bool:
        """Check if command is potentially dangerous."""
        command_lower = command.lower()
        
        # Check forbidden commands
        for forbidden in self.forbidden_commands:
            if forbidden in command:
                return True
        
        # Check other dangerous patterns
        dangerous_patterns = [
            "rm -rf",
            "del /f /s /q",
            "> /dev/sda",
            "mkfs",
            "format c:",
        ]
        
        for pattern in dangerous_patterns:
            if pattern in command_lower:
                return True
        
        return False
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "command": {"type": "string", "required": True},
            "cwd": {"type": "string", "required": False},
            "env": {"type": "object", "required": False}
        }