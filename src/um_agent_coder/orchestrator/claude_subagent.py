"""
Claude Code Subagent Spawner - Spawn subagents using Claude Code's Task tool.

This module enables programmatic spawning of Claude Code subagents with:
1. Native Task tool integration (when running inside Claude Code)
2. Formatted prompts for different agent types (Explore, code-reviewer, etc.)
3. Result collection and parsing from spawned agents
4. Fallback to subprocess mode when Task tool is unavailable

Usage:
    spawner = ClaudeCodeSubagentSpawner()

    # Spawn an exploration agent
    result = spawner.spawn_explore_agent(
        query="Find all uses of the OpenAI API in this codebase",
        files_pattern="**/*.py"
    )

    # Spawn a code reviewer
    review = spawner.spawn_code_reviewer(
        files=["src/agent/agent.py"],
        focus="Check for proper error handling"
    )

    # Generic task spawning
    result = spawner.spawn_task(
        prompt="Analyze the test coverage",
        subagent_type="Explore"
    )
"""

import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class SubagentType(Enum):
    """Types of Claude Code subagents that can be spawned."""
    EXPLORE = "Explore"          # Multi-round search and analysis
    CODE_REVIEWER = "code-reviewer"  # Code review specialist
    ARCHITECT = "Architect"      # System design and architecture
    DEBUGGER = "Debugger"       # Debugging specialist
    OPTIMIZER = "Optimizer"     # Performance optimization
    TESTER = "Tester"           # Test generation and validation
    DOCUMENTER = "Documenter"   # Documentation generation
    GENERIC = "Generic"         # Generic task execution


@dataclass
class SubagentConfig:
    """Configuration for a subagent."""
    subagent_type: SubagentType
    prompt: str
    working_directory: Optional[str] = None
    timeout: int = 600  # seconds
    max_iterations: int = 10  # for explore agents
    focus_files: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.subagent_type.value,
            "prompt": self.prompt,
            "working_directory": self.working_directory,
            "timeout": self.timeout,
            "max_iterations": self.max_iterations,
            "focus_files": self.focus_files,
            "context": self.context,
        }


@dataclass
class SubagentResult:
    """Result from a spawned subagent."""
    success: bool
    output: Any
    error: Optional[str] = None
    subagent_type: str = ""
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "subagent_type": self.subagent_type,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


class ClaudeCodeSubagentSpawner:
    """
    Spawns Claude Code subagents using the Task tool pattern.

    This class provides methods to spawn different types of Claude Code
    subagents programmatically, either through:
    1. Claude Code's native Task tool (when available)
    2. Subprocess execution as fallback

    The spawner handles:
    - Prompt formatting for different agent types
    - Context passing between agents
    - Result collection and parsing
    - Error handling and timeouts

    Usage:
        spawner = ClaudeCodeSubagentSpawner(
            use_task_tool=True,  # Try Task tool first
            fallback_to_subprocess=True,  # Fall back if Task unavailable
            verbose=True
        )

        result = spawner.spawn_task(
            prompt="Find all database queries",
            subagent_type=SubagentType.EXPLORE
        )
    """

    def __init__(
        self,
        use_task_tool: bool = True,
        fallback_to_subprocess: bool = True,
        working_directory: Optional[str] = None,
        verbose: bool = False,
        checkpoint_dir: str = ".subagent_results"
    ):
        """
        Initialize the subagent spawner.

        Args:
            use_task_tool: Whether to try using Claude Code's Task tool
            fallback_to_subprocess: Whether to fall back to subprocess if Task unavailable
            working_directory: Default working directory for subagents
            verbose: Whether to print verbose output
            checkpoint_dir: Directory to store subagent results
        """
        self.use_task_tool = use_task_tool
        self.fallback_to_subprocess = fallback_to_subprocess
        self.working_directory = working_directory or os.getcwd()
        self.verbose = verbose
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track spawned agents
        self.active_agents: Dict[str, Dict[str, Any]] = {}

    def spawn_task(
        self,
        prompt: str,
        subagent_type: Union[SubagentType, str] = SubagentType.GENERIC,
        context: Optional[Dict[str, Any]] = None,
        timeout: int = 600,
        max_iterations: int = 10,
        focus_files: Optional[List[str]] = None,
        working_directory: Optional[str] = None,
    ) -> SubagentResult:
        """
        Spawn a generic task with the specified agent type.

        Args:
            prompt: The task prompt
            subagent_type: Type of subagent to spawn
            context: Additional context to pass to the agent
            timeout: Maximum execution time in seconds
            max_iterations: Maximum iterations for explore agents
            focus_files: Specific files to focus on
            working_directory: Working directory override

        Returns:
            SubagentResult with the task outcome
        """
        if isinstance(subagent_type, str):
            try:
                subagent_type = SubagentType(subagent_type)
            except ValueError:
                subagent_type = SubagentType.GENERIC

        config = SubagentConfig(
            subagent_type=subagent_type,
            prompt=prompt,
            working_directory=working_directory or self.working_directory,
            timeout=timeout,
            max_iterations=max_iterations,
            focus_files=focus_files,
            context=context,
        )

        return self._execute_subagent(config)

    def spawn_explore_agent(
        self,
        query: str,
        files_pattern: Optional[str] = None,
        max_iterations: int = 10,
        context: Optional[Dict[str, Any]] = None,
    ) -> SubagentResult:
        """
        Spawn an Explore agent for multi-round search and analysis.

        Explore agents are good for:
        - Finding patterns across a large codebase
        - Investigating complex questions requiring multiple searches
        - Analyzing system architecture

        Args:
            query: The exploration query
            files_pattern: Optional glob pattern to limit search scope
            max_iterations: Maximum search iterations
            context: Additional context

        Returns:
            SubagentResult with exploration findings
        """
        prompt = self._format_explore_prompt(query, files_pattern, context)

        return self.spawn_task(
            prompt=prompt,
            subagent_type=SubagentType.EXPLORE,
            context=context,
            max_iterations=max_iterations,
        )

    def spawn_code_reviewer(
        self,
        files: List[str],
        focus: Optional[str] = None,
        review_criteria: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SubagentResult:
        """
        Spawn a code-reviewer agent for code review.

        Args:
            files: List of file paths to review
            focus: Specific focus area for the review
            review_criteria: List of criteria to check
            context: Additional context

        Returns:
            SubagentResult with review findings
        """
        prompt = self._format_code_review_prompt(files, focus, review_criteria, context)

        return self.spawn_task(
            prompt=prompt,
            subagent_type=SubagentType.CODE_REVIEWER,
            context=context,
            focus_files=files,
        )

    def spawn_architect_agent(
        self,
        task: str,
        existing_architecture: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SubagentResult:
        """
        Spawn an Architect agent for system design.

        Args:
            task: The architectural task
            existing_architecture: Description of current architecture
            constraints: Design constraints
            context: Additional context

        Returns:
            SubagentResult with architectural design
        """
        prompt = self._format_architect_prompt(task, existing_architecture, constraints, context)

        return self.spawn_task(
            prompt=prompt,
            subagent_type=SubagentType.ARCHITECT,
            context=context,
        )

    def spawn_debugger_agent(
        self,
        issue_description: str,
        files_to_investigate: Optional[List[str]] = None,
        error_logs: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SubagentResult:
        """
        Spawn a Debugger agent to investigate and fix issues.

        Args:
            issue_description: Description of the issue
            files_to_investigate: Files that may contain the issue
            error_logs: Error logs or stack traces
            context: Additional context

        Returns:
            SubagentResult with debugging findings and fixes
        """
        prompt = self._format_debugger_prompt(
            issue_description,
            files_to_investigate,
            error_logs,
            context
        )

        return self.spawn_task(
            prompt=prompt,
            subagent_type=SubagentType.DEBUGGER,
            context=context,
            focus_files=files_to_investigate,
        )

    def spawn_tester_agent(
        self,
        code_to_test: str,
        test_type: str = "unit",
        coverage_requirements: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SubagentResult:
        """
        Spawn a Tester agent to generate tests.

        Args:
            code_to_test: Path to code or description
            test_type: Type of tests (unit, integration, e2e)
            coverage_requirements: Coverage requirements
            context: Additional context

        Returns:
            SubagentResult with generated tests
        """
        prompt = self._format_tester_prompt(
            code_to_test,
            test_type,
            coverage_requirements,
            context
        )

        return self.spawn_task(
            prompt=prompt,
            subagent_type=SubagentType.TESTER,
            context=context,
        )

    def spawn_documenter_agent(
        self,
        target: str,
        doc_type: str = "api",
        audience: str = "developers",
        context: Optional[Dict[str, Any]] = None,
    ) -> SubagentResult:
        """
        Spawn a Documenter agent to generate documentation.

        Args:
            target: What to document (file path, module, etc.)
            doc_type: Type of documentation (api, guide, tutorial)
            audience: Target audience
            context: Additional context

        Returns:
            SubagentResult with generated documentation
        """
        prompt = self._format_documenter_prompt(target, doc_type, audience, context)

        return self.spawn_task(
            prompt=prompt,
            subagent_type=SubagentType.DOCUMENTER,
            context=context,
        )

    # Prompt formatting methods

    def _format_explore_prompt(
        self,
        query: str,
        files_pattern: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Format a prompt for an Explore agent."""
        prompt_parts = [
            f"Explore the codebase to answer: {query}",
            "",
        ]

        if files_pattern:
            prompt_parts.append(f"Focus on files matching: {files_pattern}")
            prompt_parts.append("")

        if context:
            prompt_parts.append("Context from previous steps:")
            prompt_parts.append(json.dumps(context, indent=2))
            prompt_parts.append("")

        prompt_parts.extend([
            "Please:",
            "1. Search for relevant files and code patterns",
            "2. Analyze findings across multiple files",
            "3. Provide a comprehensive answer with specific code references",
        ])

        return "\n".join(prompt_parts)

    def _format_code_review_prompt(
        self,
        files: List[str],
        focus: Optional[str],
        review_criteria: Optional[List[str]],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Format a prompt for a code-reviewer agent."""
        prompt_parts = [
            "Review the following files:",
            "",
        ]

        for file in files:
            prompt_parts.append(f"- {file}")
        prompt_parts.append("")

        if focus:
            prompt_parts.append(f"Focus area: {focus}")
            prompt_parts.append("")

        if review_criteria:
            prompt_parts.append("Review criteria:")
            for criterion in review_criteria:
                prompt_parts.append(f"- {criterion}")
            prompt_parts.append("")
        else:
            prompt_parts.extend([
                "Check for:",
                "- Code quality and best practices",
                "- Potential bugs and edge cases",
                "- Security vulnerabilities",
                "- Performance issues",
                "- Documentation completeness",
                "",
            ])

        if context:
            prompt_parts.append("Additional context:")
            prompt_parts.append(json.dumps(context, indent=2))
            prompt_parts.append("")

        prompt_parts.append("Provide specific findings with file locations and code snippets.")

        return "\n".join(prompt_parts)

    def _format_architect_prompt(
        self,
        task: str,
        existing_architecture: Optional[str],
        constraints: Optional[List[str]],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Format a prompt for an Architect agent."""
        prompt_parts = [
            f"Architectural task: {task}",
            "",
        ]

        if existing_architecture:
            prompt_parts.append("Current architecture:")
            prompt_parts.append(existing_architecture)
            prompt_parts.append("")

        if constraints:
            prompt_parts.append("Constraints:")
            for constraint in constraints:
                prompt_parts.append(f"- {constraint}")
            prompt_parts.append("")

        if context:
            prompt_parts.append("Context:")
            prompt_parts.append(json.dumps(context, indent=2))
            prompt_parts.append("")

        prompt_parts.extend([
            "Please provide:",
            "1. High-level architecture design",
            "2. Component breakdown and responsibilities",
            "3. Data flow and interactions",
            "4. Technology recommendations",
            "5. Implementation considerations",
        ])

        return "\n".join(prompt_parts)

    def _format_debugger_prompt(
        self,
        issue_description: str,
        files_to_investigate: Optional[List[str]],
        error_logs: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Format a prompt for a Debugger agent."""
        prompt_parts = [
            f"Debug issue: {issue_description}",
            "",
        ]

        if error_logs:
            prompt_parts.append("Error logs:")
            prompt_parts.append("```")
            prompt_parts.append(error_logs)
            prompt_parts.append("```")
            prompt_parts.append("")

        if files_to_investigate:
            prompt_parts.append("Files to investigate:")
            for file in files_to_investigate:
                prompt_parts.append(f"- {file}")
            prompt_parts.append("")

        if context:
            prompt_parts.append("Context:")
            prompt_parts.append(json.dumps(context, indent=2))
            prompt_parts.append("")

        prompt_parts.extend([
            "Please:",
            "1. Investigate the issue",
            "2. Identify root cause",
            "3. Propose a fix with code changes",
            "4. Explain why the fix works",
        ])

        return "\n".join(prompt_parts)

    def _format_tester_prompt(
        self,
        code_to_test: str,
        test_type: str,
        coverage_requirements: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Format a prompt for a Tester agent."""
        prompt_parts = [
            f"Generate {test_type} tests for: {code_to_test}",
            "",
        ]

        if coverage_requirements:
            prompt_parts.append(f"Coverage requirements: {coverage_requirements}")
            prompt_parts.append("")

        if context:
            prompt_parts.append("Context:")
            prompt_parts.append(json.dumps(context, indent=2))
            prompt_parts.append("")

        prompt_parts.extend([
            "Please provide:",
            "1. Comprehensive test cases",
            "2. Edge case coverage",
            "3. Test setup and teardown code",
            "4. Assertions for expected behavior",
            "5. Mock/stub configurations if needed",
        ])

        return "\n".join(prompt_parts)

    def _format_documenter_prompt(
        self,
        target: str,
        doc_type: str,
        audience: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Format a prompt for a Documenter agent."""
        prompt_parts = [
            f"Generate {doc_type} documentation for: {target}",
            f"Target audience: {audience}",
            "",
        ]

        if context:
            prompt_parts.append("Context:")
            prompt_parts.append(json.dumps(context, indent=2))
            prompt_parts.append("")

        doc_type_requirements = {
            "api": [
                "1. Function/class signatures",
                "2. Parameter descriptions",
                "3. Return values",
                "4. Usage examples",
                "5. Error conditions",
            ],
            "guide": [
                "1. Overview and purpose",
                "2. Key concepts",
                "3. Step-by-step instructions",
                "4. Code examples",
                "5. Common pitfalls",
            ],
            "tutorial": [
                "1. Learning objectives",
                "2. Prerequisites",
                "3. Step-by-step walkthrough",
                "4. Working code examples",
                "5. Exercises",
            ],
        }

        requirements = doc_type_requirements.get(doc_type, [
            "1. Clear explanation",
            "2. Code examples",
            "3. Usage instructions",
        ])

        prompt_parts.append("Include:")
        prompt_parts.extend(requirements)

        return "\n".join(prompt_parts)

    # Execution methods

    def _execute_subagent(self, config: SubagentConfig) -> SubagentResult:
        """
        Execute a subagent with the given configuration.

        Tries Task tool first if enabled, falls back to subprocess.
        """
        agent_id = str(uuid.uuid4())[:8]
        started_at = datetime.now().isoformat()

        if self.verbose:
            print(f"\n[Subagent {agent_id}] Spawning {config.subagent_type.value} agent...")

        self.active_agents[agent_id] = {
            "config": config.to_dict(),
            "started_at": started_at,
            "status": "running",
        }

        result = None

        # Try Task tool first
        if self.use_task_tool:
            try:
                result = self._execute_via_task_tool(config, agent_id)
            except Exception as e:
                if self.verbose:
                    print(f"[Subagent {agent_id}] Task tool failed: {e}")
                if not self.fallback_to_subprocess:
                    completed_at = datetime.now().isoformat()
                    return SubagentResult(
                        success=False,
                        output=None,
                        error=f"Task tool failed: {e}",
                        subagent_type=config.subagent_type.value,
                        started_at=started_at,
                        completed_at=completed_at,
                    )

        # Fallback to subprocess
        if result is None and self.fallback_to_subprocess:
            try:
                result = self._execute_via_subprocess(config, agent_id)
            except Exception as e:
                if self.verbose:
                    print(f"[Subagent {agent_id}] Subprocess failed: {e}")
                completed_at = datetime.now().isoformat()
                return SubagentResult(
                    success=False,
                    output=None,
                    error=f"Subprocess failed: {e}",
                    subagent_type=config.subagent_type.value,
                    started_at=started_at,
                    completed_at=completed_at,
                )

        if result is None:
            completed_at = datetime.now().isoformat()
            return SubagentResult(
                success=False,
                output=None,
                error="No execution method available",
                subagent_type=config.subagent_type.value,
                started_at=started_at,
                completed_at=completed_at,
            )

        # Update tracking
        self.active_agents[agent_id]["status"] = "completed" if result.success else "failed"
        self.active_agents[agent_id]["completed_at"] = result.completed_at

        # Save result checkpoint
        self._save_result_checkpoint(agent_id, result)

        if self.verbose:
            status = "✓" if result.success else "✗"
            print(f"[Subagent {agent_id}] {status} Completed in {result.duration_seconds:.2f}s")

        return result

    def _execute_via_task_tool(
        self,
        config: SubagentConfig,
        agent_id: str
    ) -> Optional[SubagentResult]:
        """
        Execute via Claude Code's Task tool.

        This method is designed to be called when running inside Claude Code.
        It formats the request in a way that Claude Code can intercept and
        execute using the Task tool.

        NOTE: This is a placeholder that returns None - actual Task tool
        integration would require Claude Code to intercept these calls.
        """
        # In a real Claude Code environment, this would trigger the Task tool
        # For now, we return None to trigger subprocess fallback

        # The actual implementation would look something like:
        # task_request = {
        #     "subagent_type": config.subagent_type.value,
        #     "prompt": config.prompt,
        #     "working_directory": config.working_directory,
        #     "timeout": config.timeout,
        #     "context": config.context,
        # }
        #
        # # This would be intercepted by Claude Code
        # result = CLAUDE_CODE_TASK_TOOL.spawn(task_request)
        # return result

        return None

    def _execute_via_subprocess(
        self,
        config: SubagentConfig,
        agent_id: str
    ) -> SubagentResult:
        """
        Execute via subprocess as fallback.

        Creates a temporary script that runs the task.
        """
        started_at = datetime.now()

        # Create a temporary script
        script_content = self._create_subagent_script(config, agent_id)

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            dir=self.checkpoint_dir
        ) as script_file:
            script_file.write(script_content)
            script_path = script_file.name

        try:
            # Make executable
            os.chmod(script_path, 0o755)

            # Execute
            proc = subprocess.Popen(
                ["python3", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=config.working_directory or self.working_directory,
            )

            try:
                stdout, stderr = proc.communicate(timeout=config.timeout)
                completed_at = datetime.now()
                duration = (completed_at - started_at).total_seconds()

                if proc.returncode == 0:
                    output = stdout.decode('utf-8').strip()
                    return SubagentResult(
                        success=True,
                        output=output,
                        subagent_type=config.subagent_type.value,
                        started_at=started_at.isoformat(),
                        completed_at=completed_at.isoformat(),
                        duration_seconds=duration,
                    )
                else:
                    return SubagentResult(
                        success=False,
                        output=None,
                        error=stderr.decode('utf-8'),
                        subagent_type=config.subagent_type.value,
                        started_at=started_at.isoformat(),
                        completed_at=completed_at.isoformat(),
                        duration_seconds=duration,
                    )

            except subprocess.TimeoutExpired:
                proc.kill()
                completed_at = datetime.now()
                duration = (completed_at - started_at).total_seconds()

                return SubagentResult(
                    success=False,
                    output=None,
                    error=f"Timeout after {config.timeout}s",
                    subagent_type=config.subagent_type.value,
                    started_at=started_at.isoformat(),
                    completed_at=completed_at.isoformat(),
                    duration_seconds=duration,
                )

        finally:
            # Cleanup
            try:
                os.unlink(script_path)
            except Exception:
                pass

    def _create_subagent_script(
        self,
        config: SubagentConfig,
        agent_id: str
    ) -> str:
        """Create a Python script for subagent execution."""
        context_json = json.dumps(config.context or {}, indent=2)

        script = f'''#!/usr/bin/env python3
"""Subagent script for {config.subagent_type.value} - {agent_id}"""
import sys
import os

# Add project to path
sys.path.insert(0, "src")
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from um_agent_coder.llm.providers.mcp_local import MCPLocalLLM

# Determine which backend to use based on agent type
agent_type = "{config.subagent_type.value}"
backend_map = {{
    "Explore": "gemini",      # Large context for exploration
    "code-reviewer": "codex",  # Code analysis
    "Architect": "claude",    # Design and orchestration
    "Debugger": "codex",      # Code analysis and fixes
    "Tester": "codex",        # Test generation
    "Documenter": "gemini",   # Documentation generation
    "Generic": "claude",      # Default
}}

backend = backend_map.get(agent_type, "claude")

# Initialize model
model = MCPLocalLLM(backend=backend)

# Build prompt
prompt = """{config.prompt}"""

# Add context if provided
context = {context_json}
if context:
    import json
    context_str = json.dumps(context, indent=2)
    prompt += f"\\n\\n--- CONTEXT ---\\n{{context_str}}"

# Execute
try:
    result = model.chat(prompt)
    print(result)
    sys.exit(0)
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''
        return script

    def _save_result_checkpoint(self, agent_id: str, result: SubagentResult):
        """Save subagent result to checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{agent_id}_result.json"

        with open(checkpoint_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def get_active_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active agents."""
        return self.active_agents.copy()

    def load_result(self, agent_id: str) -> Optional[SubagentResult]:
        """Load a saved result by agent ID."""
        checkpoint_path = self.checkpoint_dir / f"{agent_id}_result.json"

        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path, 'r') as f:
            data = json.load(f)

        return SubagentResult(**data)
