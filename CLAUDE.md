# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Agent
```bash
python -m src.um_agent_coder "YOUR_PROMPT"
```

### Running the 24/7 Codex Harness
```bash
# Dry run - preview execution without running Codex
python -m src.um_agent_coder.harness --roadmap specs/roadmap.md --dry-run

# Run 24/7 daemon mode
python -m src.um_agent_coder.harness --roadmap specs/roadmap.md --daemon

# Check current status
python -m src.um_agent_coder.harness --status

# Reset state and start fresh
python -m src.um_agent_coder.harness --reset
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Configuration
The agent expects a configuration file at `config/config.yaml`. If it doesn't exist, the agent will create a default one. You can also set the `OPENAI_API_KEY` environment variable instead of adding it to the config file.

## Architecture

This is an AI coding agent built with a modular architecture:

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
   - Supports: OpenAI, Anthropic, Google, Claude CLI, Google ADC
   - New providers should inherit from the `LLM` base class

4. **24/7 Codex Harness** (`harness/`):
   - Autonomous task execution via Codex CLI
   - Roadmap-driven planning from `specs/roadmap.md`
   - SQLite state persistence for resume capability
   - Growth mode for continuous improvement after completion

5. **Extension Points**:
   - Add new LLM providers by creating a new class in `llm/providers/` that inherits from `LLM`
   - The provider selection is controlled by the `llm.provider` config value

## 24/7 Codex Harness

The harness enables autonomous 24/7 task execution using the Codex CLI (ChatGPT Pro subscription).

### How It Works

1. **Parse** `specs/roadmap.md` for objectives, tasks, and dependencies
2. **Execute** tasks in order via Codex CLI (gpt-5.2, reasoning=high)
3. **Track** state in SQLite (`.harness/state.db`)
4. **Update** roadmap checkboxes as tasks complete
5. **Enter Growth Mode** when all tasks done - generates improvement tasks
6. **Loop 24/7** until stopped with Ctrl+C

### Roadmap Format

Edit `specs/roadmap.md` to define your project:

```markdown
# Roadmap: My Project

## Objective
High-level goal description

## Constraints
- Max time per task: 30 min
- Max retries per task: 3

## Success Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Tasks

### Phase 1: Setup
- [ ] **task-001**: Task description
  - timeout: 15min
  - depends: none
  - success: How to verify completion

### Phase 2: Build
- [ ] **task-002**: Another task
  - timeout: 30min
  - depends: task-001
  - success: Verification method

## Growth Mode
Instructions for continuous improvement after completion
```

### Harness Files

```
src/um_agent_coder/harness/
├── main.py              # Main daemon loop
├── models.py            # Data classes
├── roadmap_parser.py    # Parse specs/roadmap.md
├── codex_executor.py    # Codex CLI wrapper
├── state.py             # SQLite persistence
└── growth.py            # Improvement loop
```

## Key Implementation Notes

- Configuration uses dot notation for nested values (e.g., `llm.openai.api_key`)
- The harness requires Codex CLI authenticated via OAuth (ChatGPT Pro)
- State persists in `.harness/` directory
- Logs written to `.harness/harness.log`