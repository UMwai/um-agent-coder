"""
ANSI color codes for CLI output.
"""

class ANSI:
    """ANSI escape codes for color output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Apply color to text and reset at the end."""
        return f"{color}{text}{cls.RESET}"

    @classmethod
    def header(cls, text: str) -> str:
        return cls.colorize(text, cls.BOLD + cls.CYAN)

    @classmethod
    def success(cls, text: str) -> str:
        return cls.colorize(text, cls.GREEN)

    @classmethod
    def warning(cls, text: str) -> str:
        return cls.colorize(text, cls.YELLOW)

    @classmethod
    def error(cls, text: str) -> str:
        return cls.colorize(text, cls.RED)

    @classmethod
    def info(cls, text: str) -> str:
        return cls.colorize(text, cls.BLUE)
