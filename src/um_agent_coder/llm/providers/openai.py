import json
import os
from typing import Optional, Dict, Any, List
import requests
import time

from um_agent_coder.llm.base import LLM
from um_agent_coder.models import ModelRegistry


class OpenAILLM(LLM):
    """
    LLM provider for OpenAI models.
    """
    
    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "gpt-4o", 
                 temperature: float = 0.7, max_tokens: int = 4096):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_registry = ModelRegistry()
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        # Use a persistent session for connection pooling
        self.session = requests.Session()
    
    def chat(self, prompt: str, messages: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Send a chat request to OpenAI API.
        
        Args:
            prompt: The user prompt
            messages: Optional conversation history
            
        Returns:
            The model's response
        """
        if messages is None:
            messages = []
        
        # Add the new user message
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            # Use session for connection pooling
            response = self.session.post(self.API_URL, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            return f"Error calling OpenAI API: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Error parsing OpenAI response: {str(e)}"
    
    def stream_chat(self, prompt: str, messages: Optional[List[Dict[str, str]]] = None):
        """
        Stream a chat response from OpenAI API.
        """
        if messages is None:
            messages = []
        
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        
        try:
            # Use session for connection pooling
            response = self.session.post(self.API_URL, headers=headers, json=data, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        if line.strip() == 'data: [DONE]':
                            break
                        try:
                            chunk = json.loads(line[6:])
                            if chunk['choices'][0]['delta'].get('content'):
                                yield chunk['choices'][0]['delta']['content']
                        except:
                            continue
                            
        except requests.exceptions.RequestException as e:
            yield f"Error calling OpenAI API: {str(e)}"
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        Note: This is a rough estimate. For accurate counts, use tiktoken library.
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        model_info = self.model_registry.get(self.model)
        if model_info:
            return {
                "name": model_info.name,
                "context_window": model_info.context_window,
                "cost_per_1k_input": model_info.cost_per_1k_input,
                "cost_per_1k_output": model_info.cost_per_1k_output,
                "capabilities": model_info.capabilities
            }
        return {"name": self.model, "info": "Model not in registry"}
