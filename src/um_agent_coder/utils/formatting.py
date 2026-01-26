def format_compact_number(n: int) -> str:
    """
    Format a number with compact suffixes (K, M, B).

    Examples:
        1200 -> 1.2K
        1000000 -> 1M
        250000 -> 250K
    """
    if n < 1000:
        return str(n)

    num = float(n)
    for suffix in ["K", "M", "B", "T"]:
        num /= 1000
        if num < 1000:
            # If it's a whole number, don't show decimal
            if num.is_integer():
                return f"{int(num)}{suffix}"
            return f"{num:.1f}{suffix}"

    return f"{num:.1f}T"
