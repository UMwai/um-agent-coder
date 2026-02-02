from typing import Union


def format_compact_number(number: Union[int, float], decimals: int = 1) -> str:
    """
    Format a number into a compact string with suffixes (k, M, B, T).

    Args:
        number: The number to format.
        decimals: Number of decimal places to show. Defaults to 1.

    Returns:
        Formatted string (e.g., "1.2k", "1M").
    """
    if number == 0:
        return "0"

    suffixes = ["", "k", "M", "B", "T"]
    magnitude = 0
    abs_number = abs(number)

    while abs_number >= 1000 and magnitude < len(suffixes) - 1:
        magnitude += 1
        abs_number /= 1000.0

    # Format the number with specified decimals
    formatted_number = f"{abs_number:.{decimals}f}"

    # Remove trailing zeros and decimal point if unnecessary
    if "." in formatted_number:
        formatted_number = formatted_number.rstrip("0").rstrip(".")

    # Add sign if negative
    sign = "-" if number < 0 else ""

    return f"{sign}{formatted_number}{suffixes[magnitude]}"
