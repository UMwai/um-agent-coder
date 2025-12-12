#!/usr/bin/env python3
"""Subagent script for Explore - aa6f60b9"""
import sys
import os

# Add project to path
sys.path.insert(0, "src")
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from um_agent_coder.llm.providers.mcp_local import MCPLocalLLM

# Determine which backend to use based on agent type
agent_type = "Explore"
backend_map = {
    "Explore": "gemini",      # Large context for exploration
    "code-reviewer": "codex",  # Code analysis
    "Architect": "claude",    # Design and orchestration
    "Debugger": "codex",      # Code analysis and fixes
    "Tester": "codex",        # Test generation
    "Documenter": "gemini",   # Documentation generation
    "Generic": "claude",      # Default
}

backend = backend_map.get(agent_type, "claude")

# Initialize model
model = MCPLocalLLM(backend=backend)

# Build prompt
prompt = """Explore the codebase to answer: Find all API endpoints

Focus on files matching: **/*.py

Please:
1. Search for relevant files and code patterns
2. Analyze findings across multiple files
3. Provide a comprehensive answer with specific code references"""

# Add context if provided
context = {}
if context:
    import json
    context_str = json.dumps(context, indent=2)
    prompt += f"\n\n--- CONTEXT ---\n{context_str}"

# Execute
try:
    result = model.chat(prompt)
    print(result)
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
