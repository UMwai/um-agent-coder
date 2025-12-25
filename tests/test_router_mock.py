import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Mock missing dependencies BEFORE importing app modules
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["google.auth"] = MagicMock()
sys.modules["google.auth.transport"] = MagicMock()
sys.modules["google.auth.transport.requests"] = MagicMock()

from um_agent_coder.agent.router import MultiAgentRouter
from um_agent_coder.llm.base import LLM

class MockLLM(LLM):
    def __init__(self, name="mock"):
        self.name = name
        
    def chat(self, prompt, messages=None):
        return f"Response from {self.name} for prompt: {prompt[:20]}..."
        
    def stream_chat(self, prompt, messages=None):
        yield self.chat(prompt, messages)
        
    def count_tokens(self, text):
        return len(text)
        
    def get_model_info(self):
        return {"name": self.name}

class TestMultiAgentRouter(unittest.TestCase):
    
    @patch("um_agent_coder.agent.router.LLMFactory")
    def test_router_flow(self, mock_factory):
        # Setup mock factory to return our MockLLM
        def create_mock(provider, config):
            return MockLLM(name=f"{provider}_agent")
            
        mock_factory.create.side_effect = create_mock
        
        # Config
        config = {
            "multi_agent_router": {
                "roles": {
                    "orchestrator": {"provider": "claude_cli"},
                    "planner": {"provider": "google_adc"},
                    "executor": {"provider": "openai"},
                    "auditor": {"provider": "claude_cli"}
                }
            }
        }
        
        # Initialize Router
        router = MultiAgentRouter(config)
        
        # Verify agents initialized
        self.assertEqual(len(router.agents), 4)
        
        # Run route_request
        response = router.route_request("Build a website")
        
        # Check if response contains expected parts
        self.assertIn("Multi-Agent Workflow Complete", response)
        self.assertIn("Response from claude_cli_agent", response) # Orchestrator/Auditor
        self.assertIn("Response from google_adc_agent", response) # Planner
        self.assertIn("Response from openai_agent", response) # Executor
        
        print("\nTest passed! Router output:\n", response)

if __name__ == "__main__":
    unittest.main()
