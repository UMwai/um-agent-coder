import argparse
import os
from um_agent_coder.config import Config
from um_agent_coder.llm.providers.openai import OpenAILLM
from um_agent_coder.agent.agent import Agent

def main():
    """
    Main function for the AI coding agent.
    """
    parser = argparse.ArgumentParser(description="AI Coding Agent")
    parser.add_argument("prompt", help="The prompt for the agent.")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    # Create a dummy config file if it doesn't exist
    if not os.path.exists(args.config):
        print(f"Warning: Config file not found at {args.config}.")
        print("Creating a default config file.")
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, "w") as f:
            f.write(
                'llm:\\n'
                '  provider: openai\\n'
                '  openai:\\n'
                '    api_key: "YOUR_OPENAI_API_KEY"\\n'
                '    model: "gpt-3.5-turbo"\\n'
            )

    # Load configuration
    config = Config(args.config)

    # Instantiate the LLM provider
    llm_provider = config.get("llm.provider")
    if llm_provider == "openai":
        api_key = config.get("llm.openai.api_key")
        model = config.get("llm.openai.model")
        if not api_key or api_key == "YOUR_OPENAI_API_KEY":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Error: OpenAI API key not found.")
                print(
                    "Please set the OPENAI_API_KEY environment variable or"
                    " add it to your config file."
                )
                return
        llm = OpenAILLM(api_key=api_key, model=model)
    else:
        print(f"Error: LLM provider '{llm_provider}' is not supported.")
        return

    # Instantiate the agent
    agent = Agent(llm)

    # Run the agent
    response = agent.run(args.prompt)

    # Print the response
    print(response)

if __name__ == "__main__":
    main()
