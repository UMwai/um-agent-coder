"""
Formatting utilities for CLI output.
"""


def format_compact_number(n: float | int) -> str:
    """
    Formats a number with a compact suffix (k, M, B, T).

    Examples:
        1200 -> 1.2k
        1500000 -> 1.5M
    """
    if n < 1000:
        return str(n)

    for suffix in ["k", "M", "B", "T"]:
        n /= 1000
        if n < 1000:
            if n % 1 == 0:
                return f"{int(n)}{suffix}"
            return f"{n:.1f}{suffix}"

    return f"{n:.1f}P"
