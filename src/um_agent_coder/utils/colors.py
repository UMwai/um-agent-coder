class ANSI:
    """ANSI escape codes for terminal colors."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @staticmethod
    def style(text: str, style: str) -> str:
        """Apply a style to text."""
        return f"{style}{text}{ANSI.ENDC}"
