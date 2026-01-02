"""
Main entry point for Roo-inspired agent with multi-mode support.
Integrates ideas from Roo-Code for a more powerful and flexible coding assistant.
"""

import argparse
import os
import sys

from um_agent_coder.agent.modes import AgentMode
from um_agent_coder.agent.roo_agent import RooAgent
from um_agent_coder.config import Config
from um_agent_coder.llm.factory import LLMFactory


def load_or_create_config(config_path: str) -> Config:
    """Load configuration or create default Roo-style config."""
    if not os.path.exists(config_path):
        print(f"📝 Creating default Roo-style configuration at {config_path}...")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        default_config = {
            "llm": {
                "provider": "openai",
                "openai": {
                    "api_key": os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
                    "model": "gpt-4",
                },
                "anthropic": {
                    "api_key": os.environ.get("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY"),
                    "model": "claude-3-opus-20240229",
                },
                "google": {
                    "api_key": os.environ.get("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY"),
                    "model": "gemini-pro",
                },
            },
            "agent": {
                "verbose": True,
                "auto_mode": True,
                "require_approval": False,
                "auto_summarize": True,
                "interactive": False,
                "max_context_tokens": 100000,
            },
            "modes": {
                "default": "code",
                "custom_instructions": "",
                "auto_approve_patterns": ["read", "search", "analyze"],
            },
            "tools": {
                "safe_mode": True,
                "auto_backup": True,
                "format_code": True,
                "validate_writes": True,
            },
            "custom_modes": {},
        }

        with open(config_path, "w") as f:
            import yaml

            yaml.dump(default_config, f, default_flow_style=False)

    return Config(config_path)


def print_banner():
    """Print the Roo-inspired banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║  🦘 UM-AGENT-CODER                                          ║
║  Roo-Inspired Multi-Mode AI Coding Assistant                 ║
║                                                               ║
║  Modes: Code | Architect | Ask | Debug | Review | Custom     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_mode_help():
    """Print help for available modes."""
    help_text = """
Available Modes:
----------------
• code      : General coding and implementation tasks
• architect : System design and technical planning
• ask       : Information retrieval and Q&A
• debug     : Problem diagnosis and fixing
• review    : Code review and quality assurance
• custom    : User-defined custom mode

Usage Examples:
--------------
# Auto-detect mode from prompt
python -m um_agent_coder "Fix the bug in user authentication"

# Specify mode explicitly
python -m um_agent_coder --mode debug "The login function returns 500 error"

# Interactive mode with approvals
python -m um_agent_coder --interactive "Refactor the database layer"

# Custom instructions
python -m um_agent_coder --instructions "Focus on performance" "Optimize the search algorithm"
    """
    print(help_text)


def interactive_session(agent: RooAgent):
    """Run an interactive session with the agent."""
    print("\n🎮 Interactive Mode - Type 'help' for commands, 'exit' to quit\n")

    commands = {
        "help": "Show available commands",
        "mode <name>": "Switch to a different mode",
        "modes": "List available modes",
        "history": "Show conversation history",
        "clear": "Clear context and start fresh",
        "export <file>": "Export session to file",
        "exit": "Exit interactive mode",
    }

    while True:
        try:
            # Show current mode
            current_mode = agent.mode_manager.current_mode
            prompt = input(f"\n[{current_mode.value}]> ").strip()

            if not prompt:
                continue

            # Handle commands
            if prompt.lower() == "exit":
                print("👋 Goodbye!")
                break

            elif prompt.lower() == "help":
                print("\nAvailable commands:")
                for cmd, desc in commands.items():
                    print(f"  {cmd:20} - {desc}")

            elif prompt.lower() == "modes":
                print("\nAvailable modes:")
                for mode in AgentMode:
                    config = agent.mode_manager.modes.get(mode)
                    if config:
                        print(f"  • {mode.value:10} - {config.description}")

            elif prompt.lower().startswith("mode "):
                new_mode = prompt[5:].strip()
                try:
                    mode_enum = AgentMode(new_mode)
                    result = agent.switch_mode(mode_enum)
                    print(f"✓ {result}")
                except ValueError:
                    print(f"❌ Unknown mode: {new_mode}")

            elif prompt.lower() == "history":
                history = agent.get_conversation_history()
                print(f"\n📜 Conversation History ({len(history)} messages):")
                for msg in history[-10:]:  # Show last 10
                    role = "👤" if msg["role"] == "user" else "🤖"
                    print(f"{role} [{msg['mode']}] {msg['content'][:100]}...")

            elif prompt.lower() == "clear":
                agent.clear_context()
                print("✓ Context cleared")

            elif prompt.lower().startswith("export "):
                filepath = prompt[7:].strip()
                agent.export_session(filepath)
                print(f"✓ Session exported to {filepath}")

            else:
                # Process as normal prompt
                print(f"\n🔍 Processing in {current_mode.value} mode...\n")
                result = agent.run(prompt)

                if result["success"]:
                    print(f"\n{result['response']}")

                    # Show metrics if verbose
                    if agent.verbose:
                        print("\n📊 Metrics:")
                        print(f"  • Mode: {result['mode']}")
                        print(f"  • Steps: {result['execution']['steps_executed']}")
                        print(f"  • Tools: {', '.join(result['execution']['tools_used'])}")
                else:
                    print(f"\n❌ {result['response']}")

        except KeyboardInterrupt:
            print("\n\n💡 Tip: Type 'exit' to quit properly")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def main():
    """Main entry point for Roo-inspired agent."""
    parser = argparse.ArgumentParser(
        description="UM-Agent-Coder: Roo-Inspired Multi-Mode AI Coding Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("prompt", nargs="?", help="The prompt for the agent")

    parser.add_argument(
        "--mode",
        "-m",
        choices=["code", "architect", "ask", "debug", "review", "custom"],
        help="Specify the agent mode to use",
    )

    parser.add_argument(
        "--config", "-c", default="config/roo_config.yaml", help="Path to configuration file"
    )

    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")

    parser.add_argument("--instructions", help="Custom instructions for the agent")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    parser.add_argument("--no-approval", action="store_true", help="Skip approval prompts")

    parser.add_argument(
        "--help-modes", action="store_true", help="Show detailed help for agent modes"
    )

    parser.add_argument("--export-metrics", help="Export metrics to specified file after execution")

    args = parser.parse_args()

    # Show mode help if requested
    if args.help_modes:
        print_mode_help()
        return

    # Check if we have a prompt or interactive mode
    if not args.prompt and not args.interactive:
        print_banner()
        parser.print_help()
        print("\n💡 Use --interactive for interactive mode or provide a prompt")
        return

    # Print banner
    print_banner()

    # Load configuration
    config = load_or_create_config(args.config)

    # Override config with command line arguments
    if args.verbose:
        config.data["agent"]["verbose"] = True
    if args.no_approval:
        config.data["agent"]["require_approval"] = False
    if args.instructions:
        config.data["modes"]["custom_instructions"] = args.instructions

    # Get LLM provider
    llm_provider = config.get("llm.provider")

    try:
        # Create LLM instance using factory
        llm_factory = LLMFactory()
        llm = llm_factory.create(llm_provider, config)

        # Create Roo-inspired agent
        agent_config = config.get("agent", {})
        if isinstance(agent_config, dict):
            agent_config.update(
                {
                    "custom_instructions": config.get("modes.custom_instructions", ""),
                    "custom_modes": config.get("custom_modes", {}),
                }
            )
        else:
            agent_config = {"verbose": True, "custom_instructions": "", "custom_modes": {}}

        agent = RooAgent(llm, agent_config)

        # Run in interactive mode
        if args.interactive:
            interactive_session(agent)
        else:
            # Parse mode if specified
            mode = None
            if args.mode:
                mode = AgentMode(args.mode)

            # Run the agent
            print("\n🚀 Processing your request...\n")
            result = agent.run(args.prompt, mode=mode)

            # Display results
            if result["success"]:
                print(f"{result['response']}\n")

                # Show execution summary
                print("─" * 60)
                print("✅ Task completed successfully")
                print(f"📊 Mode: {result['mode']}")
                print(f"🔧 Tools used: {', '.join(result['execution']['tools_used'])}")
                print(
                    f"📈 Steps: {result['execution']['successful_steps']}/{result['execution']['steps_executed']}"
                )

                # Export metrics if requested
                if args.export_metrics:
                    agent.export_session(args.export_metrics)
                    print(f"📁 Metrics exported to {args.export_metrics}")
            else:
                print(f"❌ Error: {result['response']}")
                if "error" in result:
                    print(f"Details: {result['error']}")

    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
