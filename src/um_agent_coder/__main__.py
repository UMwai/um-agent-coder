"""Entry point for the UM Agent Coder package."""

import sys

# Check if data mode is requested
if '--data' in sys.argv or '-D' in sys.argv:
    # Remove the flag and run data agent
    if '--data' in sys.argv:
        sys.argv.remove('--data')
    if '-D' in sys.argv:
        sys.argv.remove('-D')
    from um_agent_coder.main_data import main
else:
    from um_agent_coder.main_enhanced import main

if __name__ == "__main__":
    main()
