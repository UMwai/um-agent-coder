from um_agent_coder.llm.base import LLM


class OpenAILLM(LLM):
    """
    LLM provider for OpenAI models.
    """

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model

    def chat(self, prompt: str) -> str:
        # TODO: Implement the actual chat logic using the OpenAI API
        return f"Response from OpenAI for prompt: '{prompt}'"
