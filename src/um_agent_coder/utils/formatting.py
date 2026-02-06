from typing import Union


def format_compact_number(n: Union[int, float]) -> str:
    """
    Format a number into a compact string representation (e.g., 1.2k, 1.5M).

    Args:
        n: The number to format.

    Returns:
        The formatted string.
    """
    if n < 1000:
        return str(int(n)) if isinstance(n, int) or n.is_integer() else f"{n:.1f}".rstrip("0").rstrip(".")

    n = float(n)

    if n < 1_000_000:
        return f"{n/1000:.1f}k".replace(".0k", "k")
    elif n < 1_000_000_000:
        return f"{n/1_000_000:.1f}M".replace(".0M", "M")
    else:
        return f"{n/1_000_000_000:.1f}B".replace(".0B", "B")
