# Test Provider

Test LLM provider configuration and connectivity.

## Task: $ARGUMENTS

## Available Providers
- `anthropic` - Claude models
- `openai` - GPT models
- `gemini` - Google Gemini

## Commands
```bash
cd /Users/waiyang/Desktop/repo/um-agent-coder
python -c "
from src.um_agent_coder.llm.base import get_provider
provider = get_provider('$ARGUMENTS')
response = provider.complete('Hello, test message')
print(response)
"
```

## Key Files
- `src/um_agent_coder/llm/base.py` - Provider interface
- `src/um_agent_coder/llm/providers/` - Provider implementations

## Configuration
```bash
# Anthropic
export ANTHROPIC_API_KEY=sk-...

# OpenAI
export OPENAI_API_KEY=sk-...

# Gemini
export GOOGLE_API_KEY=...
```

## Example Usage
- `/test-provider anthropic` - Test Claude
- `/test-provider openai` - Test GPT
- `/test-provider all` - Test all providers
