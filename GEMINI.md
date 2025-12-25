# um-agent-coder

`um-agent-coder` is a sophisticated AI coding agent designed to assist developers by automating coding tasks, analyzing codebases, and executing complex plans. It offers two modes of operation: a basic agent for simple queries and an enhanced agent with advanced capabilities like planning, cost tracking, and multi-provider support.

## Project Overview

This project implements an AI agent capable of interacting with various LLM providers (OpenAI, Anthropic, Google) to perform software engineering tasks.

### Key Features
*   **Multi-Provider Support:** Seamlessly switch between OpenAI (GPT-4o), Anthropic (Claude 3.5 Sonnet), and Google (Gemini 2.5/1.5) models.
*   **Enhanced Agent Architecture:**
    *   **Task Planner:** Analyzes requests to create a step-by-step execution plan.
    *   **Context Manager:** Optimizes token usage by managing relevant context and summarizing conversation history.
    *   **Cost Tracker:** Monitors token consumption and calculates costs in real-time.
    *   **Tool Registry:** Extensible system for file operations, code search, and shell command execution.
*   **Safety:** Optional user approval workflow for potentially risky operations.

## Project Structure

```
src/um_agent_coder/
├── __init__.py
├── main.py                 # Entry point for the basic agent
├── main_enhanced.py        # Entry point for the enhanced agent
├── config.py               # Configuration loader
├── agent/
│   ├── agent.py            # Basic agent implementation
│   └── enhanced_agent.py   # Enhanced agent with planning and tools
├── llm/                    # LLM provider abstractions
├── tools/                  # Tool implementations (FileReader, FileWriter, etc.)
├── context/                # Context management logic
├── models/                 # Model registry and definitions
└── planner/                # Task planning logic
```

## Setup and Configuration

1.  **Installation:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration:**
    Copy the example config and update it with your API keys:
    ```bash
    cp config/config.yaml.example config/config.yaml
    ```

    Edit `config/config.yaml`:
    ```yaml
    llm:
      provider: openai  # Default provider
      openai:
        api_key: "YOUR_OPENAI_API_KEY"
        model: "gpt-4o"
      anthropic:
        api_key: "YOUR_ANTHROPIC_API_KEY"
        model: "claude-3.5-sonnet-20241022"
      google:
        api_key: "YOUR_GOOGLE_API_KEY"
        model: "gemini-1.5-flash"
    
    agent:
      max_context_tokens: 100000
      auto_summarize: true
      require_approval: false
    ```
    *Note: You can also set API keys via environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`.*

## Usage

### Enhanced Agent (Recommended)
The enhanced agent supports planning, tools, and detailed metrics.

```bash
# Basic usage
python -m src.um_agent_coder.main_enhanced "Refactor the database connection logic"

# Specify a different model/provider
python -m src.um_agent_coder.main_enhanced "Fix bug in auth.py" --provider anthropic --model claude-3.5-sonnet-20241022

# Enable verbose output to see the plan and tool usage
python -m src.um_agent_coder.main_enhanced "Analyze the project structure" --verbose

# List available models
python -m src.um_agent_coder.main_enhanced --list-models
```

**CLI Options:**
*   `--provider`: Override the LLM provider (openai, anthropic, google).
*   `--model`: Override the specific model.
*   `--verbose`: Show detailed execution logs (planning, tool calls).
*   `--require-approval`: Pause for user confirmation before executing the plan.
*   `--export-metrics <file>`: Save execution metrics to a JSON file.
*   `--simple`: Run in simple mode (bypassing the planner).

### Basic Agent
A simpler version for direct Q&A without complex tool usage.

```bash
python -m src.um_agent_coder.main "Write a Python function to calculate Fibonacci numbers"
```

## Architecture & Flow (Enhanced Agent)

1.  **Analysis & Planning:** The agent analyzes the user's prompt, estimates complexity, and creates an `ExecutionPlan`.
2.  **Approval (Optional):** If configured, the agent presents the plan and estimated cost for user approval.
3.  **Context Loading:** Relevant files and project structure are loaded into the context window.
4.  **Execution:** The agent executes the plan step-by-step using tools (reading files, writing code, running commands).
5.  **Refinement:** The agent tracks progress and can auto-summarize context to stay within token limits.
6.  **Response:** A final response is generated based on the execution results.

## Development

*   **Adding Tools:** Extend `um_agent_coder.tools.base.Tool` and register it in `EnhancedAgent._register_tools`.
*   **Adding Providers:** Extend `um_agent_coder.llm.base.LLM` and update `LLMFactory`.
