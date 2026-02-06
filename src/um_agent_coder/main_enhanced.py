import argparse
import os
import sys

from um_agent_coder.agent.agent import Agent
from um_agent_coder.agent.enhanced_agent import EnhancedAgent
from um_agent_coder.agent.router import MultiAgentRouter
from um_agent_coder.agent.yolo.manager import YoloManager
from um_agent_coder.config import Config
from um_agent_coder.llm.factory import LLMFactory
from um_agent_coder.models import ModelCategory, ModelRegistry
from um_agent_coder.utils.colors import ANSI
from um_agent_coder.utils.formatting import format_compact_number


def list_available_models():
    """List all available models organized by category."""
    registry = ModelRegistry()

    print("\n" + ANSI.style("=" * 80, ANSI.BLUE))
    print(ANSI.style("AVAILABLE MODELS", ANSI.BOLD))
    print(ANSI.style("=" * 80, ANSI.BLUE))

    for category in ModelCategory:
        print(f"\n{ANSI.style(category.value.upper().replace('_', ' '), ANSI.CYAN)} MODELS:")
        print("-" * 50)

        models = registry.get_by_category(category)
        for model in sorted(models, key=lambda x: x.performance_score, reverse=True):
            print(f"\n{model.name} ({model.provider})")

            # Color code performance score
            score_str = f"{model.performance_score}/100"
            if model.performance_score >= 90:
                score_colored = ANSI.style(score_str, ANSI.GREEN)
            elif model.performance_score >= 80:
                score_colored = ANSI.style(score_str, ANSI.WARNING)
            else:
                score_colored = ANSI.style(score_str, ANSI.FAIL)

            print(f"  Performance: {score_colored}")
            print(f"  Context: {format_compact_number(model.context_window)} tokens")
            print(
                f"  Cost: ${model.cost_per_1k_input:.4f}/${model.cost_per_1k_output:.4f} per 1K tokens (in/out)"
            )
            print(f"  {model.description}")


def main():
    """
    Main function for the enhanced AI coding agent.
    """
    parser = argparse.ArgumentParser(description="UM Agent Coder - Enhanced AI Coding Assistant")
    parser.add_argument("prompt", nargs="?", help="The prompt for the agent.")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--model",
        help="Override the model specified in config (e.g., claude-3.5-sonnet-20241022, gpt-4o)",
    )
    parser.add_argument(
        "--provider", help="Override the provider specified in config (openai, anthropic, google)"
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        default=True,
        help="Use enhanced agent with planning and cost tracking (default: True)",
    )
    parser.add_argument("--simple", action="store_true", help="Use simple agent without planning")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--export-metrics", help="Export metrics to specified file after execution")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--require-approval", action="store_true", help="Require approval before executing plan"
    )
    parser.add_argument(
        "--router",
        action="store_true",
        help="Use multi-agent router (Claude Orchestrator + Gemini Planner + Executor)",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Use Yolo Mode: Competitive Planning + Autonomous CLI Sub-Agents",
    )
    args = parser.parse_args()

    # Handle list models
    if args.list_models:
        list_available_models()
        return

    # Require prompt if not listing models
    if not args.prompt:
        parser.error("prompt is required unless using --list-models")

    # Create a dummy config file if it doesn't exist
    if not os.path.exists(args.config):
        print("\n" + ANSI.style("=" * 60, ANSI.BLUE))
        print(ANSI.style("WELCOME TO UM AGENT CODER", ANSI.BOLD))
        print(ANSI.style("=" * 60, ANSI.BLUE))

        print(f"\n{ANSI.style('ℹ Config file not found.', ANSI.CYAN)}")
        print(
            f"Creating default configuration at: {ANSI.style(os.path.abspath(args.config), ANSI.BOLD)}"
        )

        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, "w") as f:
            f.write(
                "llm:\n"
                "  provider: openai\n"
                "  openai:\n"
                '    api_key: "YOUR_OPENAI_API_KEY"\n'
                '    model: "gpt-4o"\n'
                "  anthropic:\n"
                '    api_key: "YOUR_ANTHROPIC_API_KEY"\n'
                '    model: "claude-3.5-sonnet-20241022"\n'
                "  google:\n"
                '    api_key: "YOUR_GOOGLE_API_KEY"\n'
                '    model: "gemini-1.5-flash"\n'
                "\n"
                "agent:\n"
                "  max_context_tokens: 100000\n"
                "  auto_summarize: true\n"
                "  require_approval: false\n"
            )

        print(f"\n{ANSI.style('⚠ ACTION REQUIRED', ANSI.WARNING)}")
        print("To get started, you need to provide an API key.")
        print("Option 1: Edit the config file above.")
        print("Option 2: Set an environment variable (e.g., OPENAI_API_KEY).")
        print(f"\n{ANSI.style('Exiting to allow setup.', ANSI.CYAN)}")
        sys.exit(0)

    # Load configuration
    config = Config(args.config)

    # Get LLM configuration
    llm_config = config.get("llm", {})
    provider = args.provider or llm_config.get("provider", "openai")

    # Create provider config
    provider_config = llm_config.get(provider, {})

    # Override model if specified
    if args.model:
        provider_config["model"] = args.model

    # Add API key from environment if not in config or is placeholder
    api_key_env_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    if provider in api_key_env_vars:
        env_var = api_key_env_vars[provider]
        config_key = provider_config.get("api_key", "")
        if not config_key or config_key.startswith("YOUR_"):
            api_key = os.getenv(env_var)
            if api_key:
                provider_config["api_key"] = api_key
            else:
                print(f"{ANSI.style('Error:', ANSI.FAIL)} {env_var} not found in config or environment")
                print(
                    f"Please set the {ANSI.style(env_var, ANSI.BOLD)} environment variable or update config.yaml"
                )
                sys.exit(1)

    # Handle Router Mode
    if args.router:
        print("\n" + ANSI.style("=" * 80, ANSI.BLUE))
        print(ANSI.style("MULTI-AGENT ROUTER MODE", ANSI.BOLD))
        print(ANSI.style("=" * 80, ANSI.BLUE))
        print("Initializing agents (Orchestrator, Planner, Executor, Auditor)...")

        try:
            # Pass the full config so router can extract its section
            router = MultiAgentRouter(config.config_data)
            response = router.route_request(args.prompt)
            print("\n" + ANSI.style("=" * 60, ANSI.GREEN))
            print(ANSI.style("FINAL RESPONSE", ANSI.BOLD))
            print(ANSI.style("=" * 60, ANSI.GREEN))
            print(response)
            return
        except Exception as e:
            print(ANSI.style(f"Error in router mode: {e}", ANSI.FAIL))
            sys.exit(1)

    # Handle Yolo Mode
    if args.yolo:
        try:
            manager = YoloManager(config.config_data)
            response = manager.run(args.prompt)
            print("\n" + ANSI.style("=" * 60, ANSI.GREEN))
            print(ANSI.style("FINAL RESPONSE (YOLO MODE)", ANSI.BOLD))
            print(ANSI.style("=" * 60, ANSI.GREEN))
            print(response)
            return
        except Exception as e:
            print(ANSI.style(f"Error in Yolo Mode: {e}", ANSI.FAIL))
            sys.exit(1)

    # Create LLM instance
    try:
        llm = LLMFactory.create(provider, provider_config)
    except Exception as e:
        print(ANSI.style(f"Error creating LLM: {e}", ANSI.FAIL))
        sys.exit(1)

    # Create and run agent
    if not args.simple:
        # Enhanced agent configuration
        agent_config = {
            "max_context_tokens": config.get("agent", {}).get("max_context_tokens", 100000),
            "verbose": args.verbose,
            "auto_summarize": config.get("agent", {}).get("auto_summarize", True),
            "require_approval": args.require_approval
            or config.get("agent", {}).get("require_approval", False),
        }

        agent = EnhancedAgent(llm, agent_config)
        result = agent.run(args.prompt)

        # Display results
        print("\n" + ANSI.style("=" * 60, ANSI.GREEN))
        print(ANSI.style("RESPONSE", ANSI.BOLD))
        print(ANSI.style("=" * 60, ANSI.GREEN))
        print(result["response"])

        if args.verbose or result.get("metrics", {}).get("total_cost", 0) > 0:
            print("\n" + ANSI.style("=" * 60, ANSI.BLUE))
            print(ANSI.style("METRICS", ANSI.BOLD))
            print(ANSI.style("=" * 60, ANSI.BLUE))
            metrics = result.get("metrics", {})
            success_rate = metrics.get("success_rate", 0)
            total_cost = metrics.get("total_cost", 0)
            effectiveness = metrics.get("effectiveness_score", 0)
            context_usage = result.get("context_usage", {}).get("usage_percentage", 0)
            print(f"Success Rate: {ANSI.style(f'{success_rate:.1f}%', ANSI.CYAN)}")
            print(f"Total Cost: {ANSI.style(f'${total_cost:.4f}', ANSI.CYAN)}")
            print(f"Effectiveness Score: {ANSI.style(f'{effectiveness:.1f}', ANSI.CYAN)}")
            print(f"Context Usage: {ANSI.style(f'{context_usage:.1f}%', ANSI.CYAN)}")

        # Export metrics if requested
        if args.export_metrics:
            agent.export_metrics(args.export_metrics)
            print(f"\nMetrics exported to: {args.export_metrics}")
    else:
        # Simple agent
        agent = Agent(llm)
        response = agent.run(args.prompt)
        print(response)


if __name__ == "__main__":
    main()
