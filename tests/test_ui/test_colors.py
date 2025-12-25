from um_agent_coder.utils.colors import ANSI

def test_ansi_colors():
    """Test that ANSI colors are correctly formatted."""
    text = "Hello"
    colored = ANSI.colorize(text, ANSI.RED)
    assert colored == "\033[31mHello\033[0m"

    header = ANSI.header(text)
    assert header == "\033[1m\033[36mHello\033[0m"

    success = ANSI.success(text)
    assert success == "\033[32mHello\033[0m"
