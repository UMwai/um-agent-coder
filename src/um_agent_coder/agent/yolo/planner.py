import concurrent.futures
from typing import Any

from um_agent_coder.llm.factory import LLMFactory


class CompetitivePlanner:
    """
    Orchestrates a 'planning competition' between multiple LLM providers
    to generate the best possible execution plan.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.providers = ["openai", "anthropic", "google"]
        # In a real app, we'd check availability in config before adding to list

        self.architect_config = {
            "provider": "anthropic",
            "model": "claude-3-opus-20240229",  # The distinct judge
        }

    def _get_plan_from_provider(self, provider: str, prompt: str) -> str:
        """
        Instantiates a temporary agent for the provider and requests a plan.
        """
        try:
            # We assume the main config has a section for each provider
            provider_config = self.config.get("llm", {}).get(provider, {})
            if not provider_config:
                # Fallback if specific config missing but API keys might be in env
                provider_config = {"api_key": "env"}

            # Use specific high-reasoning models for planning if possible
            if provider == "openai":
                provider_config["model"] = "gpt-4o"
            elif provider == "anthropic":
                provider_config["model"] = "claude-3-5-sonnet-20241022"
            elif provider == "google":
                provider_config["model"] = "gemini-1.5-pro-latest"

            agent = LLMFactory.create(provider, provider_config)

            planning_prompt = f"""
            You are a Senior Software Architect competing to create the best implementation plan.

            User Request: {prompt}

            Please provide a concise but technical step-by-step implementation plan.
            Focus on architecture, file structure, and critical algorithms.
            """

            print(f"[{provider.upper()}] Thinking...")
            response = agent.chat(planning_prompt)
            return f"--- PLAN BY {provider.upper()} ---\n{response}\n"
        except Exception as e:
            return f"[{provider.upper()}] Failed to generate plan: {e}"

    def create_master_plan(self, user_prompt: str) -> str:
        """
        Run the competition and synthesize the results.
        """
        print("\n=== STARTING COMPETITIVE PLANNING PHASE ===")
        plans = []

        # 1. Parallel Generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_provider = {
                executor.submit(self._get_plan_from_provider, p, user_prompt): p
                for p in self.providers
            }

            for future in concurrent.futures.as_completed(future_to_provider):
                plans.append(future.result())

        full_context = "\n\n".join(plans)

        # 2. Synthesis by Architect
        print("\n=== ARCHITECT SYNTHESIZING MASTER PLAN ===")
        try:
            # Use global config for architect authentication
            arch_provider = self.architect_config["provider"]
            arch_config = self.config.get("llm", {}).get(arch_provider, {})
            arch_config["model"] = self.architect_config["model"]

            architect = LLMFactory.create(arch_provider, arch_config)

            synthesis_prompt = f"""
            You are the Chief Architect. You have received implementation proposals from three senior engineers.

            User Request: {user_prompt}

            PROPOSALS:
            {full_context}

            YOUR TASK:
            1. Evaluate the strengths and weaknesses of each plan.
            2. Create a single, unified "MASTER PLAN" that combines the best ideas.
            3. The output should be a clear Markdown list of steps executable by a developer.
            4. Be decisive.
            """

            master_plan = architect.chat(synthesis_prompt)
            return master_plan

        except Exception as e:
            print(f"Architect failed: {e}")
            # Fallback: just return the raw plans
            return f"Architect failed to synthesize. Raw plans:\n{full_context}"
