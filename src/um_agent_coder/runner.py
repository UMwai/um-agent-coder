import argparse
import os
import sys

from um_agent_coder.agent.async_runner import AsyncRunner
from um_agent_coder.agent.persistence import JobStore
from um_agent_coder.config import Config
from um_agent_coder.llm.factory import LLMFactory


def main():
    parser = argparse.ArgumentParser(description="UM Agent Coder - Async Job Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: start
    start_parser = subparsers.add_parser("start", help="Start a new job")
    start_parser.add_argument("prompt", help="The prompt/task description")

    # Command: resume
    resume_parser = subparsers.add_parser("resume", help="Resume a pending/paused job")
    resume_parser.add_argument("job_id", help="The Job ID")
    resume_parser.add_argument("--steps", type=int, default=0, help="Max steps to run (0 for all)")

    # Command: list
    subparsers.add_parser("list", help="List all jobs")

    # Common args
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Load Config & LLM (Same setup as main_enhanced.py)
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return

    config = Config(args.config)
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "openai")
    provider_config = llm_config.get(provider, {})

    # Simple API Key check
    api_key_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    if provider in api_key_vars and not provider_config.get("api_key"):
        provider_config["api_key"] = os.getenv(api_key_vars[provider])

    try:
        llm = LLMFactory.create(provider, provider_config)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)

    # Initialize Job System
    job_store = JobStore()
    runner = AsyncRunner(llm, config.get("agent", {}), job_store)

    if args.command == "start":
        job_id = runner.create_job(args.prompt)
        print("\nJob created successfully.")
        print(f"Run 'python runner.py resume {job_id}' to start execution.")

    elif args.command == "resume":
        runner.resume_job(args.job_id, max_steps=args.steps)

    elif args.command == "list":
        jobs = job_store.list_jobs()
        print("\n" + "=" * 80)
        print(f"{ 'ID':<10} | { 'STATUS':<10} | { 'STEPS':<8} | {'PROMPT'}")
        print("-" * 80)
        for job in jobs:
            total = len(job.plan.get("steps", [])) if job.plan else 0
            prompt_preview = (job.prompt[:50] + "...") if len(job.prompt) > 50 else job.prompt
            print(
                f"{job.job_id:<10} | {job.status:<10} | {job.current_step_index}/{total:<5} | {prompt_preview}"
            )


if __name__ == "__main__":
    main()
