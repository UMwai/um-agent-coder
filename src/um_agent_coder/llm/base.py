from abc import ABC, abstractmethod


class LLM(ABC):
    """
    Abstract base class for all LLM providers.
    """

    @abstractmethod
    def chat(self, prompt: str) -> str:
        """
        Sends a prompt to the LLM and returns the response.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            The response from the LLM.
        """
        pass
