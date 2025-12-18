# UM Agent Coder - Enhanced AI Coding Assistant

An advanced AI coding agent with planning, context management, and cost tracking capabilities. Inspired by leading AI coding assistants like Cline and OpenCode.

## Features

### 🚀 Core Capabilities
- **Multi-Provider Support**: OpenAI (GPT-5.2), Anthropic (Claude Opus 4.5), Google (Gemini 3)
- **Intelligent Planning**: Analyzes tasks before execution with complexity estimation
- **Architecture & Planning**: Converts abstract requirements into detailed technical specifications
- **Context Management**: Smart context window optimization with auto-summarization
- **Cost Tracking**: Real-time cost monitoring and effectiveness metrics
- **Tool System**: Extensible tools for file operations, code search, and more

### 🎯 Recommended Models

| Provider | Model | Use Case |
|----------|-------|----------|
| OpenAI | **gpt-5.2** | Best reasoning, complex coding |
| Anthropic | **claude-opus-4.5** | Deep analysis, synthesis |
| Google | **gemini-3-pro** | Large context (1M+), research |
| Google | **gemini-3-flash** | Fast, cost-effective |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/um-agent-coder.git
cd um-agent-coder

# Install dependencies
pip install -r requirements.txt

# Set up API keys (choose one or more)
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

## Usage

### Basic Usage

```bash
# Use the enhanced agent (default)
python -m src.um_agent_coder "implement a fibonacci function"

# Use specific model
python -m src.um_agent_coder "fix the bug in auth.py" --model claude-opus-4.5

# Use specific provider
python -m src.um_agent_coder "refactor database.py" --provider anthropic

# List available models
python -m src.um_agent_coder --list-models
```

### Advanced Options

```bash
# Enable verbose output
python -m src.um_agent_coder "analyze codebase" --verbose

# Require approval before execution
python -m src.um_agent_coder "refactor entire module" --require-approval

# Export metrics after execution
python -m src.um_agent_coder "implement feature X" --export-metrics metrics.json

# Use simple agent (no planning)
python -m src.um_agent_coder "quick fix" --simple
```

## Configuration

Create `config/config.yaml`:

```yaml
llm:
  provider: openai  # or anthropic, google
  openai:
    api_key: "YOUR_API_KEY"  # or use env var
    model: "gpt-5.2"
  anthropic:
    api_key: "YOUR_API_KEY"
    model: "claude-opus-4.5"
  google:
    api_key: "YOUR_API_KEY"
    model: "gemini-3-flash"

agent:
  max_context_tokens: 100000
  auto_summarize: true
  require_approval: false
```

## Architecture

### Enhanced Agent Flow

1. **Planning Stage (15-20%)**
   - Task analysis and complexity estimation
   - Tool requirement identification
   - Risk assessment

2. **Context Loading (10%)**
   - Smart file loading based on task
   - Project structure analysis for complex tasks

3. **Execution Stage (60-70%)**
   - Step-by-step plan execution
   - Real-time progress tracking
   - Automatic context optimization

4. **Validation Stage (10%)**
   - Result validation
   - Cost effectiveness calculation

### Cost-Effectiveness Algorithm

```
Effectiveness = (Success Rate × Completion Rate) / (Avg Cost × Avg Time)
```

### Key Components

- **LLM Providers**: Modular provider system with factory pattern
- **Context Manager**: Intelligent context window management
- **Task Planner**: Decomposes complex tasks into steps
- **Cost Tracker**: Monitors token usage and effectiveness
- **Tool Registry**: Extensible tool system

## Tools Available

1. **FileReader**: Read file contents with line limits
2. **FileWriter**: Create/modify files
3. **FileSearcher**: Search files by pattern
4. **CodeSearcher**: Search code with regex
5. **ProjectAnalyzer**: Analyze project structure
6. **CommandExecutor**: Run shell commands safely
7. **ArchitectTool**: Generate technical specs from abstract prompts

## Performance Metrics

The agent tracks:
- Task success rate
- Token usage and costs
- Execution time
- Context utilization
- Cost-effectiveness score

## Extending the Agent

### Adding a New LLM Provider

```python
from um_agent_coder.llm.base import LLM

class CustomLLM(LLM):
    def chat(self, prompt: str) -> str:
        # Implement your provider
        pass
```

### Adding a New Tool

```python
from um_agent_coder.tools.base import Tool, ToolResult

class CustomTool(Tool):
    def execute(self, **kwargs) -> ToolResult:
        # Implement your tool
        pass
```

## Comparison with Other AI Coders

| Feature | UM Agent Coder | Cline | OpenCode |
|---------|---------------|-------|----------|
| Planning | ✅ Explicit | ✅ Implicit | ❌ |
| Cost Tracking | ✅ Detailed | ✅ Basic | ✅ Basic |
| Context Management | ✅ Auto-summarize | ✅ AST-based | ✅ Auto-compact |
| Multi-Provider | ✅ 3+ providers | ❌ Claude only | ✅ Multiple |
| Open Source Support | ✅ Via API | ❌ | ✅ |
| Approval Workflow | ✅ | ✅ | ❌ |

## Best Practices

1. **Model Selection**
   - Use `gpt-5.2` or `claude-opus-4.5` for complex coding tasks
   - Use `gemini-3-flash` for cost-sensitive operations
   - Use `gemini-3-pro` for large context analysis

2. **Context Management**
   - Let auto-summarization handle long conversations
   - Use verbose mode to monitor context usage

3. **Cost Optimization**
   - Set token limits in config
   - Use efficient models for simple tasks
   - Monitor effectiveness scores

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE file for details.