from um_agent_coder.llm.base import LLM


class Agent:
    """
    The main agent class.
    """

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, prompt: str) -> str:
        """
        Runs the agent with the given prompt.

        Args:
            prompt: The prompt to send to the agent.

        Returns:
            The response from the agent.
        """
        return self.llm.chat(prompt)
