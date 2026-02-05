import sys
import unittest
from unittest.mock import patch, MagicMock
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from um_agent_coder.main_enhanced import main
from um_agent_coder.utils.colors import ANSI

class TestInteractiveUX(unittest.TestCase):
    @patch('sys.stdin.isatty')
    @patch('builtins.input')
    @patch('sys.argv', ['um-agent'])
    @patch('um_agent_coder.main_enhanced.EnhancedAgent')
    @patch('um_agent_coder.main_enhanced.LLMFactory')
    @patch('um_agent_coder.main_enhanced.Config')
    @patch('os.path.exists')
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    def test_interactive_prompt(self, mock_exists, mock_config_class, mock_factory, mock_agent, mock_input, mock_isatty):
        # Setup
        mock_isatty.return_value = True
        mock_input.return_value = "Test Task"
        mock_exists.return_value = True

        mock_config_instance = MagicMock()
        mock_config_instance.get.return_value = {}
        mock_config_class.return_value = mock_config_instance

        mock_llm = MagicMock()
        mock_factory.create.return_value = mock_llm
        mock_agent_instance = MagicMock()
        mock_agent_instance.run.return_value = {"response": "Task Completed", "metrics": {}}
        mock_agent.return_value = mock_agent_instance

        # Execute
        try:
            main()
        except SystemExit as e:
            self.fail(f"main() raised SystemExit unexpectedly: {e}")

        # Verify
        mock_isatty.assert_called()
        mock_input.assert_called_with("> ")
        mock_agent_instance.run.assert_called_with("Test Task")

    @patch('sys.stdin.isatty')
    @patch('sys.argv', ['um-agent'])
    @patch('os.path.exists')
    @patch('argparse.ArgumentParser.error') # Mock error to prevent stderr noise
    def test_non_interactive_no_prompt(self, mock_error, mock_exists, mock_isatty):
        mock_isatty.return_value = False
        mock_exists.return_value = True # Config exists, so it proceeds to prompt check

        # The parser.error method usually exits, but we mocked it.
        # However, argparse might call it.
        # If we mock parser.error, we can check if it was called.
        # But parser is created inside main.
        # So we need to patch argparse.ArgumentParser.error

        # Actually, main() logic is:
        # if not args.prompt:
        #    parser.error(...)

        # Since we cannot easily access the parser instance created inside main without refactoring,
        # checking for SystemExit is the standard way, assuming parser.error calls exit.
        # But wait, parser.error prints to stderr and exits.

        # Let's just catch SystemExit.
        # But we need to make sure it's the right exit (error), not success.

        # If I mock argparse.ArgumentParser.error to raise a specific exception, I can catch it.
        mock_error.side_effect = SystemExit(2)

        with self.assertRaises(SystemExit) as cm:
            main()

        self.assertEqual(cm.exception.code, 2)
        mock_error.assert_called_with("prompt is required unless using --list-models")

if __name__ == '__main__':
    unittest.main()
