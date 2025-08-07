# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Agent
```bash
python -m src.um_agent_coder "YOUR_PROMPT"
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Configuration
The agent expects a configuration file at `config/config.yaml`. If it doesn't exist, the agent will create a default one. You can also set the `OPENAI_API_KEY` environment variable instead of adding it to the config file.

## Architecture

This is a simple AI coding agent built with a modular architecture:

1. **Entry Point**: `src/um_agent_coder/__main__.py` -> `main.py`
   - Handles argument parsing and configuration loading
   - Creates LLM provider and Agent instances
   - Executes the agent with the user's prompt

2. **Core Components**:
   - **Agent** (`agent/agent.py`): Main agent class that orchestrates interactions with the LLM
   - **LLM Base** (`llm/base.py`): Abstract base class defining the LLM interface
   - **Config** (`config.py`): Handles YAML configuration loading with dot notation support

3. **LLM Providers**:
   - Located in `llm/providers/`
   - Currently supports OpenAI (`openai.py`) - NOTE: Implementation is incomplete (returns mock response)
   - New providers should inherit from the `LLM` base class

4. **Extension Points**:
   - Add new LLM providers by creating a new class in `llm/providers/` that inherits from `LLM`
   - The provider selection is controlled by the `llm.provider` config value

## Key Implementation Notes

- The OpenAI provider currently returns a mock response - the actual API integration needs to be implemented
- Configuration uses dot notation for nested values (e.g., `llm.openai.api_key`)
- The project currently has no tests or linting configuration