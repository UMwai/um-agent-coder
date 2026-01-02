"""
Command-line interface for um-agent-coder.

This module provides the main CLI entry point for the package.
"""

import sys


def main():
    """Main CLI entry point."""
    # Check if data mode is requested
    if "--data" in sys.argv or "-D" in sys.argv:
        # Remove the flag and run data agent
        if "--data" in sys.argv:
            sys.argv.remove("--data")
        if "-D" in sys.argv:
            sys.argv.remove("-D")
        from um_agent_coder.main_data import main as data_main

        return data_main()
    else:
        from um_agent_coder.main_enhanced import main as enhanced_main

        return enhanced_main()


if __name__ == "__main__":
    sys.exit(main() or 0)
