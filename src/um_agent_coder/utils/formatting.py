"""
Formatting utilities for the CLI.
"""

def format_compact_number(n: int) -> str:
    """
    Format a number with suffixes (k, M, B) for compact display.

    Args:
        n: The number to format.

    Returns:
        A string representation with a suffix if appropriate.
    """
    if n >= 1_000_000_000:
        val = n / 1_000_000_000
        return f"{val:.1f}B".replace(".0B", "B")
    if n >= 1_000_000:
        val = n / 1_000_000
        return f"{val:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        val = n / 1_000
        return f"{val:.1f}k".replace(".0k", "k")
    return str(n)
