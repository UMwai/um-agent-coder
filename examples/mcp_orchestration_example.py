#!/usr/bin/env python3
"""
Example: Using MCP Tool Orchestration

This example demonstrates how to use the enhanced MCP tool integration
to orchestrate multi-model workflows.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from um_agent_coder.llm.providers.mcp_local import MCPLocalLLM, MCPOrchestrator


def example_individual_tools():
    """Example 1: Using individual MCP tools directly."""
    print("=" * 80)
    print("Example 1: Direct MCP Tool Calls")
    print("=" * 80)

    # Initialize backends
    gemini = MCPLocalLLM(backend="gemini")
    codex = MCPLocalLLM(backend="codex")

    # Use Gemini to analyze files
    print("\n1. Using Gemini to analyze codebase...")
    context = gemini.mcp_gemini_ask(
        "Summarize the architecture and key components",
        files=["src/um_agent_coder/agent/agent.py", "src/um_agent_coder/llm/base.py"]
    )
    print(f"Context gathered: {context[:200]}...")

    # Use Codex to create a plan
    print("\n2. Using Codex to create implementation plan...")
    plan = codex.mcp_codex_plan(
        task="Add support for streaming responses",
        context=context
    )
    print(f"Plan created: {plan[:200]}...")

    # Use Codex to implement
    print("\n3. Using Codex to implement...")
    implementation = codex.mcp_codex_invoke(
        prompt=f"Implement this plan:\n{plan}",
        sandbox="workspace-write",
        approval_policy="never"
    )
    print(f"Implementation: {implementation[:200]}...")


def example_orchestrator_basic():
    """Example 2: Using the orchestrator for basic workflows."""
    print("\n" + "=" * 80)
    print("Example 2: Basic Orchestrator Usage")
    print("=" * 80)

    orchestrator = MCPOrchestrator()

    # Gather context
    print("\n1. Gathering context with Gemini...")
    context = orchestrator.gather_context(
        "What are the main components of this agent system?",
        files=["src/um_agent_coder/agent/agent.py"]
    )
    print(f"Context: {context[:200]}...")

    # Brainstorm improvements
    print("\n2. Brainstorming with Gemini...")
    ideas = orchestrator.brainstorm(
        topic="improvements to the agent's capabilities",
        context="Focus on planning and multi-step execution"
    )
    print(f"Ideas: {ideas[:200]}...")

    # Create plan with Codex
    print("\n3. Planning with Codex...")
    plan = orchestrator.plan(
        task="Implement the most promising improvement",
        context=f"Context: {context}\n\nIdeas: {ideas}"
    )
    print(f"Plan: {plan[:200]}...")


def example_full_workflow():
    """Example 3: Complete multi-model workflow."""
    print("\n" + "=" * 80)
    print("Example 3: Full Multi-Model Workflow")
    print("=" * 80)

    orchestrator = MCPOrchestrator()

    print("\nExecuting 4-phase workflow:")
    print("  Phase 1: Gemini gathers context")
    print("  Phase 2: Codex creates plan")
    print("  Phase 3: Codex implements")
    print("  Phase 4: Claude reviews")

    results = orchestrator.full_workflow(
        user_request="Add better error handling to the LLM providers",
        files_to_analyze=[
            "src/um_agent_coder/llm/providers/openai.py",
            "src/um_agent_coder/llm/providers/mcp_local.py"
        ]
    )

    print("\n--- Phase 1: Context (Gemini) ---")
    print(results["context"][:200] + "...")

    print("\n--- Phase 2: Plan (Codex) ---")
    print(results["plan"][:200] + "...")

    print("\n--- Phase 3: Implementation (Codex) ---")
    print(results["implementation"][:200] + "...")

    print("\n--- Phase 4: Review (Claude) ---")
    print(results["review"][:200] + "...")


def example_automatic_routing():
    """Example 4: Automatic task routing."""
    print("\n" + "=" * 80)
    print("Example 4: Automatic Task Routing")
    print("=" * 80)

    orchestrator = MCPOrchestrator()

    tasks = [
        ("analyze the codebase structure", "context"),
        ("plan how to add streaming support", "plan"),
        ("implement the streaming feature", "implement"),
        ("review and test the implementation", "execute"),
    ]

    for task, expected_type in tasks:
        print(f"\nTask: '{task}'")
        print(f"Expected routing: {expected_type}")

        # Auto-routing will classify and route to appropriate backend
        result = orchestrator.route(task, task_type="auto")
        print(f"Result: {result[:100]}...")


def example_gemini_file_references():
    """Example 5: Using Gemini with file references."""
    print("\n" + "=" * 80)
    print("Example 5: Gemini with File References (@syntax)")
    print("=" * 80)

    gemini = MCPLocalLLM(backend="gemini")

    # Method 1: Files in prompt with @ syntax
    print("\n1. Using @ syntax in prompt:")
    result1 = gemini.mcp_gemini_ask(
        "@src/um_agent_coder/agent/agent.py explain the Agent class structure"
    )
    print(f"Result: {result1[:200]}...")

    # Method 2: Files as parameter
    print("\n2. Using files parameter:")
    result2 = gemini.mcp_gemini_ask(
        "Compare the architecture of these two files",
        files=[
            "src/um_agent_coder/agent/agent.py",
            "src/um_agent_coder/llm/providers/mcp_local.py"
        ]
    )
    print(f"Result: {result2[:200]}...")


def example_codex_sandbox_modes():
    """Example 6: Different Codex sandbox modes."""
    print("\n" + "=" * 80)
    print("Example 6: Codex Sandbox Modes")
    print("=" * 80)

    codex = MCPLocalLLM(backend="codex")

    # Read-only: For analysis only
    print("\n1. Read-only sandbox (analysis):")
    analysis = codex.mcp_codex_invoke(
        "Analyze the code quality and suggest improvements",
        sandbox="read-only"
    )
    print(f"Analysis: {analysis[:200]}...")

    # Workspace-write: Can modify files in workspace
    print("\n2. Workspace-write sandbox (implementation):")
    impl = codex.mcp_codex_invoke(
        "Add docstrings to all functions",
        sandbox="workspace-write"
    )
    print(f"Implementation: {impl[:200]}...")

    # Danger-full-access: Full system access (use with caution!)
    print("\n3. Full-access sandbox (system operations):")
    print("   (Not running - requires explicit user approval)")
    # ops = codex.mcp_codex_invoke(
    #     "Install dependencies and run tests",
    #     sandbox="danger-full-access",
    #     approval_policy="always"  # Require approval for dangerous ops
    # )


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "MCP Tool Orchestration Examples" + " " * 25 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    print("This script demonstrates the enhanced MCP tool integration.")
    print("Note: Actual MCP tool execution requires Claude Code environment.")
    print("The examples show the API usage patterns.\n")

    try:
        # Run examples
        example_individual_tools()
        example_orchestrator_basic()
        example_full_workflow()
        example_automatic_routing()
        example_gemini_file_references()
        example_codex_sandbox_modes()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n\nNote: Examples may not execute fully outside Claude Code environment.")
        print(f"Error: {e}")
        print("\nThis is expected - the examples demonstrate API patterns.")


if __name__ == "__main__":
    main()
