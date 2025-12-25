# Run Agent

Run the AI coding agent with a specific task.

## Task: $ARGUMENTS

## Commands
```bash
cd /Users/waiyang/Desktop/repo/um-agent-coder
python -m src.um_agent_coder.main "$ARGUMENTS"
```

## Agent Architecture
- **Main Loop**: Task decomposition and execution
- **LLM Providers**: Multiple provider support
- **Tools**: File operations, shell commands, web search

## Key Files
- `src/um_agent_coder/main.py` - Entry point
- `src/um_agent_coder/llm/base.py` - LLM interface
- `src/um_agent_coder/llm/providers/` - Provider implementations
- `src/um_agent_coder/tools/base.py` - Tool interface

## Configuration
Set LLM provider via environment:
```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-...
```

## Example Usage
- `/run-agent "Write a function to parse CSV files"`
- `/run-agent "Debug the authentication issue in auth.py"`
- `/run-agent "Add unit tests for the user module"`
