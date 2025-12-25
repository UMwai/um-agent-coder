from typing import Dict, Any, List
from um_agent_coder.llm.base import LLM
from um_agent_coder.llm.factory import LLMFactory
from um_agent_coder.tools.cli_tools import ClaudeCodeTool, GeminiCLITool, CodexTool

class SubAgentExecutor:
    """
    Executes the Master Plan using specialized CLI tools/agents.
    """
    
    def __init__(self, config: Dict[str, Any], plan: str):
        self.config = config
        self.plan = plan
        self.tools = {
            "claude_cli": ClaudeCodeTool(),
            "gemini_cli": GeminiCLITool(),
            "codex_cli": CodexTool()
        }
        
        # Initialize the 'Manager' agent who assigns work
        self.manager_agent = self._init_manager()

    def _init_manager(self) -> LLM:
        # Defaults to a strong model to manage execution
        provider = "anthropic" 
        conf = self.config.get("llm", {}).get(provider, {})
        conf["model"] = "claude-3-5-sonnet-20241022" 
        return LLMFactory.create(provider, conf)

    def execute(self) -> str:
        """
        Main execution loop.
        """
        print("\n=== STARTING YOLO EXECUTION PHASE ===")
        
        # In a full implementation, we would loop until completion.
        # For this MVP, we will do a single-pass delegation or a fixed number of steps.
        
        context = f"""
        You are the Execution Manager.
        Your goal is to complete the following Master Plan:
        
        {self.plan}
        
        You have access to the following local CLI tools:
        - claude_cli: Good for general coding and reasoning.
        - gemini_cli: Good for high-context data processing.
        - codex_cli: Good for specific code generation tasks.
        
        Determine the first and most critical step, select the best tool for it, and generate the command prompt for that tool.
        
        Output format:
        TOOL: <tool_name>
        PROMPT: <prompt_for_tool>
        """
        
        # 1. Manager decides what to do
        decision = self.manager_agent.chat(context)
        print(f"\n[Manager Decision]\n{decision}\n")
        
        # 2. Parse decision (Naive parsing for MVP)
        selected_tool_name = None
        tool_prompt = ""
        
        lines = decision.split('\n')
        for line in lines:
            if line.startswith("TOOL:"):
                selected_tool_name = line.replace("TOOL:", "").strip().lower()
            elif line.startswith("PROMPT:"):
                tool_prompt = line.replace("PROMPT:", "").strip()
            # Handle multi-line prompt if needed...
        
        if not selected_tool_name or selected_tool_name not in self.tools:
            return f"Manager failed to select a valid tool. raw output: {decision}"
            
        # 3. Execute the tool
        print(f"--- Spawning SubAgent: {selected_tool_name} ---")
        tool = self.tools[selected_tool_name]
        
        result = tool.execute(prompt=tool_prompt)
        
        if result.success:
            return f"Execution Result from {selected_tool_name}:\n{result.data}"
        else:
            return f"Execution Failed ({selected_tool_name}): {result.error}\nData: {result.data}"
        
