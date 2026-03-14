
import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from um_agent_coder.main_enhanced import list_available_models, main
from um_agent_coder.utils.colors import ANSI

def test_list_models_colors(capsys):
    """Test that performance scores are color-coded correctly."""
    # Patch the ModelRegistry imported in main_enhanced
    with patch("um_agent_coder.main_enhanced.ModelRegistry") as MockRegistry:
        registry_instance = MockRegistry.return_value

        # Mock models with different scores
        model_high = MagicMock()
        model_high.name = "HighPerfModel"
        model_high.provider = "test"
        model_high.performance_score = 95
        model_high.context_window = 1000
        model_high.cost_per_1k_input = 0.01
        model_high.cost_per_1k_output = 0.01
        model_high.description = "High perf"

        model_med = MagicMock()
        model_med.name = "MedPerfModel"
        model_med.provider = "test"
        model_med.performance_score = 85
        model_med.context_window = 1000
        model_med.cost_per_1k_input = 0.01
        model_med.cost_per_1k_output = 0.01
        model_med.description = "Med perf"

        model_low = MagicMock()
        model_low.name = "LowPerfModel"
        model_low.provider = "test"
        model_low.performance_score = 75
        model_low.context_window = 1000
        model_low.cost_per_1k_input = 0.01
        model_low.cost_per_1k_output = 0.01
        model_low.description = "Low perf"

        registry_instance.get_by_category.return_value = [model_high, model_med, model_low]

        list_available_models()

        captured = capsys.readouterr()
        output = captured.out

        # Check for ANSI color codes
        # High score (95) should be Green
        assert f"{ANSI.GREEN}95{ANSI.ENDC}" in output

        # Medium score (85) should be Yellow/Warning
        assert f"{ANSI.WARNING}85{ANSI.ENDC}" in output

        # Low score (75) should be Red/Fail
        assert f"{ANSI.FAIL}75{ANSI.ENDC}" in output


def test_interactive_prompt_tty(monkeypatch):
    """Test interactive prompt when TTY is available and no args provided."""

    # Mock sys.argv to have no arguments (just script name)
    monkeypatch.setattr(sys, 'argv', ['main_enhanced.py'])

    # Mock isatty to return True
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)

    # Mock input to return a prompt
    monkeypatch.setattr('builtins.input', lambda _: "test prompt")

    # Mock os.path.exists to avoid config creation logic
    with patch("os.path.exists", return_value=True):

        # Mock Config to avoid file system access
        with patch("um_agent_coder.main_enhanced.Config") as MockConfig:
            config_instance = MockConfig.return_value
            # Return a config that has provider 'openai' with a valid key
            config_instance.get.side_effect = lambda key, default=None: {
                "provider": "openai",
                "openai": {
                    "api_key": "sk-test-key",
                    "model": "gpt-4o"
                }
            } if key == "llm" else ({} if key == "agent" else default)

            config_instance.config_data = {}

            # Mock LLMFactory to avoid API calls
            with patch("um_agent_coder.main_enhanced.LLMFactory"):

                # Mock EnhancedAgent to verify it gets called with correct prompt
                with patch("um_agent_coder.main_enhanced.EnhancedAgent") as MockAgent:
                    agent_instance = MockAgent.return_value
                    agent_instance.run.return_value = {"response": "Mock response", "metrics": {}}

                    # Run main
                    try:
                        main()
                    except SystemExit:
                        pytest.fail("main() raised SystemExit unexpectedly in interactive mode")

                    # Verify agent was called with the input prompt
                    agent_instance.run.assert_called_once_with("test prompt")

def test_interactive_prompt_eof(monkeypatch):
    """Test interactive prompt handles EOFError (Ctrl+D) gracefully."""

    # Mock sys.argv
    monkeypatch.setattr(sys, 'argv', ['main_enhanced.py'])

    # Mock isatty to return True
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)

    # Mock input to raise EOFError
    def mock_input(_):
        raise EOFError()
    monkeypatch.setattr('builtins.input', mock_input)

    # Mock os.path.exists
    with patch("os.path.exists", return_value=True):
         with pytest.raises(SystemExit) as excinfo:
            main()
         assert excinfo.value.code == 0

def test_missing_prompt_no_tty(monkeypatch, capsys):
    """Test that missing prompt raises error when not TTY."""

    # Mock sys.argv
    monkeypatch.setattr(sys, 'argv', ['main_enhanced.py'])

    # Mock isatty to return False
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: False)

    # Expect SystemExit
    with pytest.raises(SystemExit):
        main()

    captured = capsys.readouterr()
    assert "prompt is required" in captured.err
