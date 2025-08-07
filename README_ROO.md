# 🦘 UM-Agent-Coder: Roo-Inspired Edition

An advanced AI coding assistant that integrates powerful concepts from [Roo-Code](https://github.com/RooCodeInc/Roo-Code), bringing multi-mode intelligence, smart tools, and customizable workflows to your development process.

## ✨ Key Features

### 🎭 Multiple Agent Modes
Inspired by Roo-Code's adaptive modes, the agent can switch between specialized personas:

- **Code Mode** 💻: General coding and implementation tasks
- **Architect Mode** 🏗️: System design and technical planning  
- **Ask Mode** ❓: Information retrieval and Q&A
- **Debug Mode** 🐛: Problem diagnosis and fixing
- **Review Mode** 🔍: Code review and quality assurance
- **Custom Modes** 🎨: Define your own specialized modes

### 🛠️ Smart Tools
Enhanced tools with validation, safety checks, and intelligent parsing:

- **SmartFileReader**: Intelligent content parsing with metadata extraction
- **SmartFileWriter**: Validation, formatting, and automatic backups
- **SmartCodeSearcher**: Regex, semantic, and AST-based search
- **SmartCommandExecutor**: Safe command execution with environment management
- **SmartProjectAnalyzer**: Deep project insights and pattern detection

### 🎮 Interactive Mode
Run the agent interactively with real-time mode switching and session management:

```bash
python -m um_agent_coder --interactive
```

Features:
- Switch modes on the fly
- View conversation history
- Export sessions
- Clear context
- Approval workflows

### ⚙️ Customizable Instructions
Provide custom instructions that persist across all interactions:

```yaml
modes:
  custom_instructions: |
    Always follow TDD practices.
    Use type hints in Python.
    Prefer functional programming patterns.
```

### 🔌 Multi-Provider Support
Seamlessly switch between different LLM providers:

- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3)
- Google (Gemini Pro)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/um-agent-coder.git
cd um-agent-coder

# Install dependencies
pip install -r requirements.txt

# Set up API keys (choose your provider)
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### Basic Usage

```bash
# Auto-detect mode from prompt
python -m um_agent_coder "Fix the authentication bug in login.py"

# Specify mode explicitly
python -m um_agent_coder --mode architect "Design a microservices architecture for an e-commerce platform"

# Interactive mode
python -m um_agent_coder --interactive

# With custom instructions
python -m um_agent_coder --instructions "Focus on security best practices" "Review the API endpoints"
```

### Using the Roo-Inspired Main Entry Point

```bash
# Use the enhanced Roo-style interface
python -m um_agent_coder.main_roo "Implement a REST API for user management"

# Interactive session with mode switching
python -m um_agent_coder.main_roo --interactive

# View available modes
python -m um_agent_coder.main_roo --help-modes
```

## 📁 Configuration

Create a `config/roo_config.yaml` file (see `roo_config.yaml.example`):

```yaml
llm:
  provider: openai
  openai:
    api_key: ${OPENAI_API_KEY}
    model: gpt-4

agent:
  verbose: true
  auto_mode: true
  require_approval: false

modes:
  default: code
  custom_instructions: |
    Your custom instructions here

tools:
  safe_mode: true
  auto_backup: true
  format_code: true

custom_modes:
  your_mode:
    name: Your Custom Mode
    description: Description
    system_prompt: |
      Your custom prompt
```

## 🎯 Mode Examples

### Code Mode
```bash
python -m um_agent_coder --mode code "Implement a binary search tree with insert and delete operations"
```

### Architect Mode
```bash
python -m um_agent_coder --mode architect "Design a scalable notification system"
```

### Debug Mode
```bash
python -m um_agent_coder --mode debug "The API returns 500 error when processing large payloads"
```

### Ask Mode
```bash
python -m um_agent_coder --mode ask "How does the authentication middleware work?"
```

### Review Mode
```bash
python -m um_agent_coder --mode review "Review the changes in pull request #42"
```

## 🔧 Advanced Features

### Custom Mode Definition

Create specialized modes for your workflow:

```yaml
custom_modes:
  performance:
    name: Performance Optimization Mode
    description: Focus on performance improvements
    system_prompt: |
      You are a performance expert.
      Identify bottlenecks and optimize code.
    temperature: 0.5
    preferred_tools:
      - SmartCodeSearcher
      - SmartCommandExecutor
```

### Tool Configuration

Configure tool behavior:

```yaml
tools:
  safe_mode: true           # Enable command safety checks
  auto_backup: true         # Backup files before modification
  format_code: true         # Auto-format code
  validate_writes: true     # Validate syntax before writing
  
  command_timeout: 30       # Command execution timeout
  allowed_commands:         # Additional safe commands
    - npm test
    - cargo build
```

### Context Management

Intelligent context handling with priorities:

```yaml
context:
  summarization_threshold: 80000
  max_file_size: 50000
  
  default_priorities:
    project_structure: 8
    recent_changes: 9
    error_messages: 10
```

## 📊 Metrics and Cost Tracking

The agent tracks execution metrics and costs:

```bash
# Export metrics after execution
python -m um_agent_coder --export-metrics metrics.json "Your task"
```

Metrics include:
- Token usage per step
- Cost breakdown by mode
- Tool execution statistics
- Success/failure rates

## 🤝 Integration with Existing Code

### Using the RooAgent Programmatically

```python
from um_agent_coder.llm.factory import LLMFactory
from um_agent_coder.agent.roo_agent import RooAgent
from um_agent_coder.agent.modes import AgentMode

# Create LLM
llm = LLMFactory.create("openai", config)

# Create agent
agent = RooAgent(llm, agent_config)

# Run with specific mode
result = agent.run("Your prompt", mode=AgentMode.ARCHITECT)

# Switch modes
agent.switch_mode(AgentMode.DEBUG)

# Get conversation history
history = agent.get_conversation_history()
```

### Custom Tool Development

Create your own smart tools:

```python
from um_agent_coder.tools.base import Tool, ToolResult

class MyCustomTool(Tool):
    def __init__(self):
        super().__init__(
            name="MyCustomTool",
            description="Description of your tool"
        )
    
    def execute(self, **kwargs) -> ToolResult:
        # Your implementation
        return ToolResult(
            success=True,
            data={"result": "data"}
        )
```

## 🏗️ Architecture

The Roo-inspired architecture consists of:

1. **Mode Manager**: Handles mode switching and configuration
2. **Smart Tools**: Enhanced tools with validation and safety
3. **Context Manager**: Intelligent context handling with priorities
4. **Task Planner**: Breaks down complex tasks into steps
5. **Cost Tracker**: Monitors API usage and costs
6. **Interactive Shell**: REPL-style interaction with the agent

## 🔄 Comparison with Roo-Code

| Feature | Roo-Code | UM-Agent-Coder (Roo Edition) |
|---------|----------|------------------------------|
| Multiple Modes | ✅ | ✅ |
| Custom Instructions | ✅ | ✅ |
| Smart Tools | ✅ | ✅ |
| Interactive Mode | ✅ | ✅ |
| Multi-Provider | ✅ | ✅ |
| Cost Tracking | ✅ | ✅ |
| Custom Modes | ✅ | ✅ |
| MCP Support | ✅ | 🚧 (Planned) |
| Web Browser Control | ✅ | 🚧 (Planned) |

## 🚦 Roadmap

- [ ] Model Context Protocol (MCP) integration
- [ ] Web browser control for research
- [ ] Semantic code search with embeddings
- [ ] Auto-test generation
- [ ] GitHub/GitLab integration
- [ ] Plugin system for custom extensions
- [ ] Visual Studio Code extension

## 🤔 FAQ

**Q: How does this differ from the original um-agent-coder?**
A: This version integrates Roo-Code's concepts of multiple modes, smart tools, and interactive workflows while maintaining compatibility with the original codebase.

**Q: Can I use my existing config files?**
A: Yes! The enhanced agent is backward compatible. You can gradually adopt new features.

**Q: Which LLM provider should I use?**
A: Each has strengths:
- OpenAI GPT-4: Best overall performance
- Anthropic Claude: Excellent for complex reasoning
- Google Gemini: Good balance of speed and capability

**Q: How do I create custom modes?**
A: Define them in your config file or programmatically using the ModeManager class.

## 📚 Documentation

- [Mode Guide](docs/modes.md) - Detailed mode documentation
- [Tool Development](docs/tools.md) - Creating custom tools
- [Configuration](docs/config.md) - Configuration options
- [API Reference](docs/api.md) - Programming interface

## 🙏 Acknowledgments

This project integrates concepts and ideas from:
- [Roo-Code](https://github.com/RooCodeInc/Roo-Code) - Multi-mode agent architecture
- The open-source AI community

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.