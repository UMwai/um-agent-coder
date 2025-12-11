#!/usr/bin/env python3
"""
Example usage of ClaudeCodeSubagentSpawner.

This script demonstrates how to programmatically spawn Claude Code subagents
for different tasks using the ClaudeCodeSubagentSpawner class.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from um_agent_coder.orchestrator import ClaudeCodeSubagentSpawner, SubagentType


def example_explore_agent():
    """Example: Spawn an Explore agent to search the codebase."""
    print("\n" + "="*60)
    print("Example 1: Explore Agent")
    print("="*60)

    spawner = ClaudeCodeSubagentSpawner(
        use_task_tool=True,
        fallback_to_subprocess=True,
        verbose=True
    )

    result = spawner.spawn_explore_agent(
        query="Find all usages of the LLM base class in the codebase",
        files_pattern="**/*.py",
        max_iterations=5
    )

    print(f"\nSuccess: {result.success}")
    if result.success:
        print(f"Output:\n{result.output}")
    else:
        print(f"Error: {result.error}")


def example_code_reviewer():
    """Example: Spawn a code-reviewer agent."""
    print("\n" + "="*60)
    print("Example 2: Code Reviewer Agent")
    print("="*60)

    spawner = ClaudeCodeSubagentSpawner(verbose=True)

    result = spawner.spawn_code_reviewer(
        files=["src/um_agent_coder/agent/agent.py"],
        focus="Check for proper error handling and logging",
        review_criteria=[
            "Exception handling completeness",
            "Logging coverage",
            "Resource cleanup",
            "Type hints",
        ]
    )

    print(f"\nSuccess: {result.success}")
    if result.success:
        print(f"Review:\n{result.output}")
    else:
        print(f"Error: {result.error}")


def example_architect_agent():
    """Example: Spawn an Architect agent for system design."""
    print("\n" + "="*60)
    print("Example 3: Architect Agent")
    print("="*60)

    spawner = ClaudeCodeSubagentSpawner(verbose=True)

    result = spawner.spawn_architect_agent(
        task="Design a plugin system for adding new LLM providers",
        existing_architecture="Current system has a base LLM class with provider implementations",
        constraints=[
            "Must support hot-reloading of plugins",
            "Should not break existing providers",
            "Must be backward compatible",
        ]
    )

    print(f"\nSuccess: {result.success}")
    if result.success:
        print(f"Architecture:\n{result.output}")
    else:
        print(f"Error: {result.error}")


def example_generic_task():
    """Example: Spawn a generic task."""
    print("\n" + "="*60)
    print("Example 4: Generic Task")
    print("="*60)

    spawner = ClaudeCodeSubagentSpawner(verbose=True)

    result = spawner.spawn_task(
        prompt="""
        Analyze the project structure and suggest improvements for:
        1. Code organization
        2. Module dependencies
        3. Test coverage
        4. Documentation completeness
        """,
        subagent_type=SubagentType.EXPLORE,
        context={
            "project_root": "/home/umwai/um-agent-coder",
            "language": "Python",
            "framework": "None (custom agent framework)"
        }
    )

    print(f"\nSuccess: {result.success}")
    if result.success:
        print(f"Analysis:\n{result.output}")
    else:
        print(f"Error: {result.error}")


def example_parallel_execution():
    """Example: Execute multiple subagents in parallel using ParallelExecutor."""
    print("\n" + "="*60)
    print("Example 5: Parallel Execution with Claude Code Spawner")
    print("="*60)

    from um_agent_coder.orchestrator import (
        ParallelExecutor,
        ExecutionMode,
        SubTask,
        DecomposedTask,
        ModelRole,
    )
    from um_agent_coder.orchestrator.task_decomposer import SubTaskType

    # Create a decomposed task with multiple subtasks
    subtasks = [
        SubTask(
            id="explore_llm",
            description="Explore LLM implementations",
            type=SubTaskType.RESEARCH,
            model=ModelRole.GEMINI,
            prompt="Find and analyze all LLM provider implementations",
            depends_on=[],
        ),
        SubTask(
            id="review_agent",
            description="Review agent code",
            type=SubTaskType.VALIDATION,
            model=ModelRole.CODEX,
            prompt="Review the main agent code for best practices",
            depends_on=[],
        ),
        SubTask(
            id="synthesize",
            description="Synthesize findings",
            type=SubTaskType.SYNTHESIS,
            model=ModelRole.CLAUDE,
            prompt="Synthesize findings from exploration and review",
            depends_on=["explore_llm", "review_agent"],
            input_from=["explore_llm", "review_agent"],
        ),
    ]

    task = DecomposedTask(
        original_prompt="Analyze the codebase comprehensively",
        clarified_goal="Explore, review, and synthesize findings about the codebase",
        subtasks=subtasks,
        execution_order=["explore_llm", "review_agent", "synthesize"],
        data_sources=[],
        estimated_total_tokens=10000,
    )

    # Create executor with Claude Code spawner
    executor = ParallelExecutor(
        execution_mode=ExecutionMode.CLAUDE_CODE_SPAWN,
        use_claude_code_spawner=True,
        verbose=True,
    )

    # Execute
    result = executor.execute(task)

    print(f"\nSuccess: {result['success']}")
    if result['success']:
        print(f"Final output: {result['output']}")
        print(f"\nExecution summary: {result['execution_summary']}")
    else:
        print(f"Error: {result.get('error')}")


if __name__ == "__main__":
    # Run examples
    print("\n" + "="*60)
    print("Claude Code Subagent Spawner Examples")
    print("="*60)

    # Choose which examples to run
    examples = {
        "1": ("Explore Agent", example_explore_agent),
        "2": ("Code Reviewer", example_code_reviewer),
        "3": ("Architect Agent", example_architect_agent),
        "4": ("Generic Task", example_generic_task),
        "5": ("Parallel Execution", example_parallel_execution),
    }

    if len(sys.argv) > 1:
        # Run specific example
        example_num = sys.argv[1]
        if example_num in examples:
            name, func = examples[example_num]
            print(f"\nRunning: {name}\n")
            func()
        else:
            print(f"Invalid example number. Choose from: {', '.join(examples.keys())}")
    else:
        # Run all examples
        print("\nRunning all examples...\n")
        for name, func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"\nExample failed with error: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
