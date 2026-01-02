from typing import Any

from um_agent_coder.llm.base import LLM
from um_agent_coder.llm.factory import LLMFactory


class MultiAgentRouter:
    """
    Routes requests between multiple specialized agents.

    Roles:
    - Orchestrator: Analyzes the request and coordinates the workflow.
    - Planner: Creates detailed execution plans.
    - Executor: Executes the plan (code generation, tool usage).
    - Auditor: Reviews the work.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.agents: dict[str, LLM] = {}
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize the specialized agents based on config."""
        router_config = self.config.get("multi_agent_router", {})

        # Default mapping if not specified
        role_mapping = router_config.get(
            "roles",
            {
                "orchestrator": {"provider": "claude_cli", "model": "claude-3-opus-20240229"},
                "planner": {"provider": "google_adc", "model": "gemini-1.5-pro-latest"},
                "executor": {
                    "provider": "openai",
                    "model": "gpt-4-turbo",
                },  # Fallback to standard OpenAI or custom proxy
                "auditor": {"provider": "claude_cli", "model": "claude-3-opus-20240229"},
            },
        )

        for role, agent_config in role_mapping.items():
            provider_name = agent_config.get("provider")
            # Pass the entire config but focused on this provider
            # The factory expects the root config or a dict for the provider.
            # We'll construct a specific config dict for the factory.

            # We need to merge global llm config with this specific agent config
            # to ensure API keys etc are picked up if they are standard providers.

            # For simplicity, we create the provider directly via factory
            try:
                self.agents[role] = LLMFactory.create(provider_name, agent_config)
            except Exception as e:
                print(f"Warning: Failed to initialize {role} agent ({provider_name}): {e}")
                # Fallback to orchestrator or raise?
                # For now, we'll raise to fail fast during setup
                raise

    def route_request(self, prompt: str) -> str:
        """
        Execute the multi-agent workflow for a given prompt.
        """
        orchestrator = self.agents.get("orchestrator")
        planner = self.agents.get("planner")
        executor = self.agents.get("executor")
        auditor = self.agents.get("auditor")

        if not all([orchestrator, planner, executor]):
            return "Error: specific agents (orchestrator, planner, executor) are not fully initialized."

        print("--- [Orchestrator] Analyzing Request ---")
        # 1. Orchestrator analysis
        analysis_prompt = f"""
        You are the Orchestrator. Analyze the following user request and delegate to the Planner.
        Identify the core objective and any constraints.

        User Request: {prompt}
        """
        analysis = orchestrator.chat(analysis_prompt)
        print(f"Analysis: {analysis[:100]}...")

        print("--- [Planner] Creating Plan ---")
        # 2. Planner creates plan
        plan_prompt = f"""
        You are the Planner. Based on the Orchestrator's analysis, create a step-by-step execution plan.

        Orchestrator Analysis: {analysis}

        User Request: {prompt}
        """
        plan = planner.chat(plan_prompt)
        print(f"Plan: {plan[:100]}...")

        print("--- [Executor] Executing Plan ---")
        # 3. Executor executes (Simulated for now as we don't have full tool loop here yet)
        # In a real scenario, the Executor would have access to tools.
        # Here we ask it to generate the code/solution based on the plan.
        execution_prompt = f"""
        You are the Executor. Execute the following plan to satisfy the user request.
        Provide the final code or answer.

        Plan: {plan}
        """
        result = executor.chat(execution_prompt)

        print("--- [Auditor] Reviewing Result ---")
        # 4. Auditor reviews
        audit_prompt = f"""
        You are the Auditor. Review the Executor's result against the original request and plan.
        If it looks good, approve it. If not, suggest fixes.

        Original Request: {prompt}
        Plan: {plan}
        Result: {result}
        """
        audit = auditor.chat(audit_prompt)

        final_response = f"""
        **Multi-Agent Workflow Complete**

        **Orchestrator Analysis:**
        {analysis}

        **Plan (Gemini):**
        {plan}

        **Execution Result:**
        {result}

        **Audit:**
        {audit}
        """
        return final_response
