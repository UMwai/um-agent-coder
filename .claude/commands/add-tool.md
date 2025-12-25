# Add Tool

Add a new tool capability to the coding agent.

## Task: $ARGUMENTS

## Agent Configuration

Use the Task tool with `subagent_type="backend-systems-architect"`:

```
Create a new tool for the um-agent-coder framework.

## Tool Structure
1. Inherit from BaseTool (src/um_agent_coder/tools/base.py)
2. Implement execute() method
3. Define tool schema (name, description, parameters)
4. Register in tool registry
```

## Tool Template
```python
from src.um_agent_coder.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Description of what this tool does"

    def execute(self, **kwargs):
        # Implementation
        pass
```

## Key Files
- `src/um_agent_coder/tools/base.py` - Base tool class
- `src/um_agent_coder/tools/` - Existing tools

## Example Usage
- `/add-tool web_scraper "Scrape content from URLs"`
- `/add-tool database_query "Execute SQL queries"`
