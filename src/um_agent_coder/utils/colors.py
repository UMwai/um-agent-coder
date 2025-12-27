"""
ANSI Color codes for CLI output.
"""

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def style(text: str, color: str = ENDC, bold: bool = False) -> str:
        """Apply style to text."""
        style_code = color
        if bold:
            style_code += Colors.BOLD
        return f"{style_code}{text}{Colors.ENDC}"

    @staticmethod
    def header(text: str) -> str:
        return Colors.style(text, Colors.HEADER, bold=True)

    @staticmethod
    def success(text: str) -> str:
        return Colors.style(text, Colors.GREEN)

    @staticmethod
    def warning(text: str) -> str:
        return Colors.style(text, Colors.YELLOW)

    @staticmethod
    def error(text: str) -> str:
        return Colors.style(text, Colors.RED, bold=True)

    @staticmethod
    def info(text: str) -> str:
        return Colors.style(text, Colors.BLUE)
