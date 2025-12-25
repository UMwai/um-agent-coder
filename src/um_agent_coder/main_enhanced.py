import argparse
import os
import json
import sys
from typing import Any, Dict, List, Optional
from um_agent_coder.config import Config
from um_agent_coder.llm.providers.openai import OpenAILLM
from um_agent_coder.agent.agent import Agent
from um_agent_coder.agent.enhanced_agent import EnhancedAgent
from um_agent_coder.llm.factory import LLMFactory
from um_agent_coder.models import ModelRegistry, ModelCategory
from um_agent_coder.persistence import TaskCheckpointer, TaskStatus
from um_agent_coder.orchestrator import (
    MultiModelOrchestrator, TaskDecomposer, ParallelExecutor,
    ExecutionMode, DataFetcherRegistry
)
from um_agent_coder.llm.providers.mcp_local import MCPLocalLLM
from um_agent_coder.utils.colors import ANSI


def list_available_models():
    """List all available models organized by category."""
    registry = ModelRegistry()

    print("\n" + ANSI.header("="*80))
    print(ANSI.header("AVAILABLE MODELS"))
    print(ANSI.header("="*80))

    for category in ModelCategory:
        print(f"\n{ANSI.info(category.value.upper().replace('_', ' ') + ' MODELS:')}")
        print("-" * 50)

        models = registry.get_by_category(category)
        for model in sorted(models, key=lambda x: x.performance_score, reverse=True):
            print(f"\n{ANSI.BOLD}{model.name}{ANSI.RESET} ({model.provider})")
            print(f"  Performance: {ANSI.success(str(model.performance_score))}/100")
            print(f"  Context: {model.context_window:,} tokens")
            print(f"  Cost: ${model.cost_per_1k_input:.4f}/${model.cost_per_1k_output:.4f} per 1K tokens (in/out)")
            print(f"  {ANSI.DIM}{model.description}{ANSI.RESET}")


def list_tasks(checkpoint_dir: str = ".task_checkpoints", status_filter: str = None):
    """List all checkpointed tasks."""
    checkpointer = TaskCheckpointer(checkpoint_dir)

    # Parse status filter
    status = None
    if status_filter:
        try:
            status = TaskStatus(status_filter)
        except ValueError:
            print(f"Invalid status filter: {status_filter}")
            print(f"Valid values: {', '.join(s.value for s in TaskStatus)}")
            return

    tasks = checkpointer.list_tasks(status)

    if not tasks:
        print("\nNo tasks found.")
        if status_filter:
            print(f"(filtered by status: {status_filter})")
        return

    print("\n" + ANSI.header("="*80))
    print(ANSI.header("CHECKPOINTED TASKS"))
    print(ANSI.header("="*80))

    # Group by status
    by_status = {}
    for task in tasks:
        s = task["status"]
        if s not in by_status:
            by_status[s] = []
        by_status[s].append(task)

    # Display order
    status_order = ["running", "paused", "failed", "pending", "completed"]

    for status_val in status_order:
        if status_val not in by_status:
            continue

        status_tasks = by_status[status_val]

        # Colorize status header
        header_color = {
            "running": ANSI.CYAN,
            "paused": ANSI.YELLOW,
            "failed": ANSI.RED,
            "pending": ANSI.WHITE,
            "completed": ANSI.GREEN
        }.get(status_val, ANSI.WHITE)

        status_icon = {
            "running": "[RUNNING]",
            "paused": "[PAUSED]",
            "failed": "[FAILED]",
            "pending": "[PENDING]",
            "completed": "[DONE]"
        }.get(status_val, "[?]")

        formatted_header = f"{status_icon} {status_val.upper()} ({len(status_tasks)})"
        print(f"\n{ANSI.colorize(formatted_header, ANSI.BOLD + header_color)}")
        print(ANSI.colorize("-" * 60, header_color))

        for task in status_tasks:
            progress = task.get("progress", "0/0")
            progress_pct = task.get("progress_pct", 0)
            updated = task.get("updated_at", "")[:19] if task.get("updated_at") else "N/A"

            print(f"\n  ID: {ANSI.BOLD}{task['task_id']}{ANSI.RESET}")
            print(f"  Prompt: {task['prompt']}")
            print(f"  Progress: {progress} ({progress_pct:.0f}%)")
            print(f"  Updated: {updated}")

    # Show resumable hint
    resumable = [t for t in tasks if t["status"] in ["running", "paused", "failed"]]
    if resumable:
        print("\n" + "-"*60)
        print(f"To resume a task: python -m um_agent_coder --resume <task_id>")
        print(f"Resumable tasks: {', '.join(t['task_id'] for t in resumable[:5])}")


def run_orchestrated(
    prompt: str,
    gemini_model: str = "gemini-3-pro-preview",
    codex_model: str = "o4-mini",
    claude_model: str = "claude-sonnet",
    verbose: bool = True
):
    """Run a complex task through the multi-model orchestrator."""
    print("\n" + "="*60)
    print("MULTI-MODEL ORCHESTRATION")
    print("="*60)
    print(f"Gemini: {gemini_model}")
    print(f"Codex:  {codex_model}")
    print(f"Claude: {claude_model}")
    print("="*60)

    # Create model instances
    gemini = MCPLocalLLM(backend="gemini", model=gemini_model)
    codex = MCPLocalLLM(backend="codex", model=codex_model, sandbox="workspace-write")
    claude = MCPLocalLLM(backend="claude", model=claude_model)

    # Create orchestrator
    orchestrator = MultiModelOrchestrator(
        gemini=gemini,
        codex=codex,
        claude=claude,
        verbose=verbose
    )

    # Run the task
    result = orchestrator.run(prompt)

    # Display results
    print("\n" + "="*60)
    print("FINAL OUTPUT")
    print("="*60)

    if result["success"]:
        print(result.get("output", "No output"))
        print(f"\nTask ID: {result['task_id']}")
        print(f"Steps completed: {result['steps_completed']}")
    else:
        print(f"ERROR: {result.get('error', 'Unknown error')}")
        print(f"Completed: {result.get('completed_steps', 0)}/{result.get('total_steps', '?')}")
        if result.get("partial_outputs"):
            print("\nPartial outputs available:")
            for k, v in result["partial_outputs"].items():
                print(f"  - {k}: {str(v)[:50]}...")

    return result


def decompose_only(prompt: str, verbose: bool = True):
    """Decompose a task without executing (for planning/review)."""
    # Use gemini for decomposition
    gemini = MCPLocalLLM(backend="gemini", model="gemini-3-pro-preview")
    decomposer = TaskDecomposer(gemini)

    print("\n" + "="*60)
    print("TASK DECOMPOSITION (Preview)")
    print("="*60)

    decomposed = decomposer.decompose(prompt, use_llm=True)
    print(decomposer.visualize(decomposed))

    # Show detailed subtasks
    print("\nDETAILED SUBTASKS:")
    print("-"*40)

    for st in decomposed.subtasks:
        print(f"\n[{st.id}] {st.description}")
        print(f"  Model: {st.model.value}")
        print(f"  Type: {st.type.value}")
        print(f"  Depends on: {st.depends_on or 'None'}")
        print(f"  Prompt: {st.prompt[:80]}...")

    return decomposed


def run_parallel_orchestration(
    prompt: str,
    gemini_model: str = "gemini-3-pro-preview",
    codex_model: str = "o4-mini",
    claude_model: str = "claude-sonnet",
    execution_mode: str = "threads",
    max_workers: int = 4,
    require_approval: bool = False,
    approval_steps: Optional[List[str]] = None,
    verbose: bool = True
):
    """
    Run a complex task with parallel multi-model orchestration.

    This spawns subagents that can run concurrently, respecting dependencies.
    """
    print("\n" + "="*70)
    print("PARALLEL MULTI-MODEL ORCHESTRATION")
    print("="*70)
    print(f"Gemini:     {gemini_model}")
    print(f"Codex:      {codex_model}")
    print(f"Claude:     {claude_model}")
    print(f"Exec Mode:  {execution_mode}")
    print(f"Workers:    {max_workers}")
    print(f"Approval:   {'Required' if require_approval else 'Auto'}")
    print("="*70)

    # Create model instances
    gemini = MCPLocalLLM(backend="gemini", model=gemini_model)
    codex = MCPLocalLLM(backend="codex", model=codex_model, sandbox="workspace-write")
    claude = MCPLocalLLM(backend="claude", model=claude_model)

    # First decompose the task
    print("\n[Step 1] Decomposing task...")
    decomposer = TaskDecomposer(gemini)
    decomposed = decomposer.decompose(prompt, use_llm=True)

    print(decomposer.visualize(decomposed))

    # Set up human approval callback if needed
    approval_callback = None
    if require_approval:
        def approval_callback(task_id, subtask_ids, results):
            print(f"\n{'='*60}")
            print("HUMAN APPROVAL CHECKPOINT")
            print(f"{'='*60}")
            print(f"Task: {task_id}")
            print(f"Next steps requiring approval: {subtask_ids}")
            if results:
                print("\nCompleted steps:")
                for st_id, r in results.items():
                    status = "OK" if r.success else "FAIL"
                    print(f"  [{status}] {st_id}")
            response = input("\nApprove and continue? (y/n): ").strip().lower()
            return response == 'y'

    # Create parallel executor
    exec_mode = ExecutionMode(execution_mode)
    executor = ParallelExecutor(
        gemini_llm=gemini,
        codex_llm=codex,
        claude_llm=claude,
        max_workers=max_workers,
        execution_mode=exec_mode,
        human_approval_callback=approval_callback if require_approval else None,
        verbose=verbose
    )

    # Execute with parallelization
    print(f"\n[Step 2] Executing with {execution_mode} parallelization...")

    result = executor.execute(
        decomposed,
        require_approval_at=approval_steps or []
    )

    # Display results
    print("\n" + "="*70)
    print("FINAL OUTPUT")
    print("="*70)

    if result["success"]:
        print(result.get("output", "No output"))
        print(f"\nTask ID: {result['task_id']}")
        summary = result.get("execution_summary", {})
        print(f"Tasks: {summary.get('completed', 0)}/{summary.get('total_tasks', 0)}")
        print(f"Parallel levels: {summary.get('parallel_levels', 0)}")
    else:
        print(f"ERROR: {result.get('error', 'Unknown error')}")
        print(f"\nPartial results:")
        for k, v in result.get("all_results", {}).items():
            preview = str(v)[:60] + "..." if len(str(v)) > 60 else str(v)
            print(f"  {k}: {preview}")

    return result


def run_with_spec(
    spec_path: str,
    repo_override: Optional[str] = None,
    branch_override: Optional[str] = None,
    webhook_override: Optional[str] = None,
    slack_webhook: Optional[str] = None,
    n8n_webhook: Optional[str] = None,
    gemini_model: str = "gemini-3-pro-preview",
    codex_model: str = "o4-mini",
    claude_model: str = "claude-sonnet",
    execution_mode: str = "threads",
    max_workers: int = 4,
    verbose: bool = True
):
    """
    Run a task using a TaskSpec file.

    The TaskSpec file (YAML or JSON) defines:
    - Task objectives and requirements
    - Target repository
    - Deliverables and acceptance criteria
    - Webhook notifications
    """
    from um_agent_coder.orchestrator import TaskSpec, RepoTarget, UpdateType

    # Load the spec
    if spec_path.endswith('.yaml') or spec_path.endswith('.yml'):
        spec = TaskSpec.from_yaml(spec_path)
    else:
        spec = TaskSpec.from_json(spec_path)

    # Apply overrides
    if repo_override:
        spec.repo = RepoTarget(path=repo_override, branch=branch_override)
    elif branch_override and spec.repo:
        spec.repo.branch = branch_override

    if webhook_override:
        spec.webhook_url = webhook_override
    if slack_webhook:
        spec.slack_webhook = slack_webhook
    if n8n_webhook:
        spec.n8n_webhook = n8n_webhook

    # Reinitialize notifier with overrides
    if any([spec.webhook_url, spec.slack_webhook, spec.n8n_webhook]):
        from um_agent_coder.orchestrator import WebhookNotifier
        spec.notifier = WebhookNotifier(
            webhook_url=spec.webhook_url,
            slack_webhook=spec.slack_webhook,
            n8n_webhook=spec.n8n_webhook,
        )

    print("\n" + "="*70)
    print("TASK SPECIFICATION EXECUTION")
    print("="*70)
    print(f"Name: {spec.name}")
    print(f"Objectives: {len(spec.objectives)}")
    print(f"Deliverables: {len(spec.deliverables)}")

    # Setup repository if specified
    original_cwd = os.getcwd()
    if spec.repo:
        try:
            repo_path = spec.setup_repo()
            print(f"Repository: {repo_path}")
            os.chdir(repo_path)

            repo_info = spec.repo.get_repo_info()
            if repo_info.get('branch'):
                print(f"Branch: {repo_info['branch']}")
            if repo_info.get('commit'):
                print(f"Commit: {repo_info['commit'][:8]}")
        except Exception as e:
            print(f"Error setting up repository: {e}")
            return {"success": False, "error": str(e)}

    print("="*70)

    # Notify start
    spec.notify(UpdateType.STARTED, f"Starting task: {spec.name}")

    try:
        # Convert spec to prompt
        prompt = spec.to_prompt()

        if verbose:
            print("\n[Generated Prompt]")
            print("-" * 40)
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            print("-" * 40)

        # Run the orchestration
        result = run_parallel_orchestration(
            prompt=prompt,
            gemini_model=gemini_model,
            codex_model=codex_model,
            claude_model=claude_model,
            execution_mode=execution_mode,
            max_workers=max_workers,
            require_approval=spec.require_approval,
            approval_steps=spec.approval_steps or None,
            verbose=verbose
        )

        # Notify completion
        if result and result.get("success"):
            spec.notify(
                UpdateType.COMPLETED,
                f"Task completed successfully: {spec.name}",
                data={"task_id": result.get("task_id")}
            )
        else:
            spec.notify(
                UpdateType.ERROR,
                f"Task failed: {result.get('error', 'Unknown error')}",
                data=result
            )

        return result

    except Exception as e:
        spec.notify(UpdateType.ERROR, f"Task exception: {str(e)}")
        raise

    finally:
        # Restore working directory and cleanup
        os.chdir(original_cwd)
        if spec.repo:
            spec.cleanup_repo()


def fetch_data_sources(sources: List[str], params: Dict[str, Any], verbose: bool = True):
    """Fetch data from specified sources."""
    registry = DataFetcherRegistry()

    print("\n" + "="*60)
    print("DATA FETCHER")
    print("="*60)

    results = {}
    for source in sources:
        if verbose:
            print(f"\nFetching from {source}...")

        result = registry.fetch(source, **params.get(source, {}))

        if result.success:
            if verbose:
                print(f"  OK - {len(str(result.data))} bytes")
            results[source] = result.data
        else:
            if verbose:
                print(f"  FAILED - {result.error}")
            results[source] = {"error": result.error}

    return results


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
        help="Override the model specified in config (e.g., claude-3.5-sonnet-20241022, gpt-4o)"
    )
    parser.add_argument(
        "--provider",
        help="Override the provider specified in config (openai, anthropic, google)"
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        default=True,
        help="Use enhanced agent with planning and cost tracking (default: True)"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple agent without planning"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--export-metrics",
        help="Export metrics to specified file after execution"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--require-approval",
        action="store_true",
        help="Require approval before executing plan"
    )
    # Long-running task arguments
    parser.add_argument(
        "--resume",
        metavar="TASK_ID",
        help="Resume a previously paused or failed task by its ID"
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all checkpointed tasks and exit"
    )
    parser.add_argument(
        "--task-status",
        choices=["pending", "running", "paused", "completed", "failed"],
        help="Filter tasks by status (use with --list-tasks)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=".task_checkpoints",
        help="Directory for task checkpoints (default: .task_checkpoints)"
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing for this run"
    )
    # Multi-model orchestration arguments
    parser.add_argument(
        "--orchestrate",
        action="store_true",
        help="Use multi-model orchestration (Gemini + Codex + Claude) for complex tasks"
    )
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Only decompose the task into subtasks (don't execute)"
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-3-pro-preview",
        help="Gemini model for research tasks (default: gemini-3-pro-preview)"
    )
    parser.add_argument(
        "--codex-model",
        default="o4-mini",
        help="Codex model for code generation (default: o4-mini)"
    )
    parser.add_argument(
        "--claude-model",
        default="claude-sonnet",
        help="Claude model for synthesis (default: claude-sonnet)"
    )
    # Parallel execution arguments
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel execution for multi-model orchestration"
    )
    parser.add_argument(
        "--exec-mode",
        choices=["sequential", "threads", "async", "subagent"],
        default="threads",
        help="Execution mode: sequential, threads, async, or subagent (default: threads)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Max parallel workers (default: 4)"
    )
    parser.add_argument(
        "--human-approval",
        action="store_true",
        help="Enable human-in-the-loop approval at each stage"
    )
    parser.add_argument(
        "--approve-at",
        nargs="+",
        help="Specific subtask IDs requiring approval (e.g., --approve-at score_targets generate_report)"
    )
    # Repository and spec arguments
    parser.add_argument(
        "--repo",
        help="Target repository path or git URL (e.g., /path/to/repo or https://github.com/user/repo)"
    )
    parser.add_argument(
        "--branch",
        help="Git branch to use in target repository"
    )
    parser.add_argument(
        "--spec",
        help="Path to task specification file (YAML or JSON)"
    )
    parser.add_argument(
        "--create-spec",
        metavar="PATH",
        help="Create a template task specification file and exit"
    )
    # Webhook/notification arguments
    parser.add_argument(
        "--webhook",
        help="Webhook URL for task updates (generic HTTP POST)"
    )
    parser.add_argument(
        "--slack-webhook",
        help="Slack webhook URL for notifications"
    )
    parser.add_argument(
        "--n8n-webhook",
        help="n8n workflow webhook URL for notifications"
    )
    args = parser.parse_args()
    
    # Handle list models
    if args.list_models:
        list_available_models()
        return

    # Handle list tasks
    if args.list_tasks:
        list_tasks(args.checkpoint_dir, args.task_status)
        return

    # Handle create-spec
    if args.create_spec:
        from um_agent_coder.orchestrator import create_spec_template
        create_spec_template(args.create_spec)
        return

    # Handle spec file - load TaskSpec and run with it
    if args.spec:
        run_with_spec(
            spec_path=args.spec,
            repo_override=args.repo,
            branch_override=args.branch,
            webhook_override=args.webhook,
            slack_webhook=args.slack_webhook,
            n8n_webhook=args.n8n_webhook,
            gemini_model=args.gemini_model,
            codex_model=args.codex_model,
            claude_model=args.claude_model,
            execution_mode=args.exec_mode,
            max_workers=args.workers,
            verbose=args.verbose
        )
        return

    # Require prompt if not listing models/tasks and not resuming
    if not args.prompt and not args.resume and not args.spec:
        parser.error("prompt is required unless using --list-models, --list-tasks, --spec, or --resume")

    # Handle decompose-only mode (no LLM config needed beyond decomposer)
    if args.decompose:
        decompose_only(args.prompt, verbose=args.verbose)
        return

    # Handle orchestrated mode
    if args.orchestrate:
        if args.parallel:
            # Use parallel execution
            run_parallel_orchestration(
                args.prompt,
                gemini_model=args.gemini_model,
                codex_model=args.codex_model,
                claude_model=args.claude_model,
                execution_mode=args.exec_mode,
                max_workers=args.workers,
                require_approval=args.human_approval,
                approval_steps=args.approve_at,
                verbose=args.verbose
            )
        else:
            # Use sequential execution
            run_orchestrated(
                args.prompt,
                gemini_model=args.gemini_model,
                codex_model=args.codex_model,
                claude_model=args.claude_model,
                verbose=args.verbose
            )
        return
    
    # Create a dummy config file if it doesn't exist
    if not os.path.exists(args.config):
        print(f"Warning: Config file not found at {args.config}.")
        print("Creating a default config file.")
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, "w") as f:
            f.write(
                'llm:\n'
                '  provider: openai\n'
                '  openai:\n'
                '    api_key: "YOUR_OPENAI_API_KEY"\n'
                '    model: "gpt-4o"\n'
                '  anthropic:\n'
                '    api_key: "YOUR_ANTHROPIC_API_KEY"\n'
                '    model: "claude-3.5-sonnet-20241022"\n'
                '  google:\n'
                '    api_key: "YOUR_GOOGLE_API_KEY"\n'
                '    model: "gemini-1.5-flash"\n'
                '\n'
                'agent:\n'
                '  max_context_tokens: 100000\n'
                '  auto_summarize: true\n'
                '  require_approval: false\n'
            )
    
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
    
    # Local providers that don't need API keys
    local_providers = {"mcp", "local", "codex", "gemini-local", "claude-local"}

    # Add API key from environment if not in config or is placeholder
    api_key_env_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY"
    }

    if provider in api_key_env_vars and provider not in local_providers:
        env_var = api_key_env_vars[provider]
        config_key = provider_config.get("api_key", "")
        if not config_key or config_key.startswith("YOUR_"):
            api_key = os.getenv(env_var)
            if api_key:
                provider_config["api_key"] = api_key
            else:
                print(f"Error: {env_var} not found in config or environment")
                print(f"Please set the {env_var} environment variable or update config.yaml")
                print(f"\nTip: Use --provider mcp for local MCP tools (no API key needed)")
                sys.exit(1)
    
    # Create LLM instance
    try:
        llm = LLMFactory.create(provider, provider_config)
    except Exception as e:
        print(f"Error creating LLM: {e}")
        sys.exit(1)
    
    # Create and run agent
    if not args.simple:
        # Enhanced agent configuration
        agent_config = {
            "max_context_tokens": config.get("agent", {}).get("max_context_tokens", 100000),
            "verbose": args.verbose,
            "auto_summarize": config.get("agent", {}).get("auto_summarize", True),
            "require_approval": args.require_approval or config.get("agent", {}).get("require_approval", False),
            "checkpoint_dir": args.checkpoint_dir,
            "enable_checkpointing": not args.no_checkpoint
        }

        agent = EnhancedAgent(llm, agent_config)

        # Handle resume vs new task
        if args.resume:
            print(f"\nResuming task: {args.resume}")
            result = agent.resume(args.resume)
        else:
            result = agent.run(args.prompt)

        # Display results
        print("\n" + "="*60)
        print("RESPONSE")
        print("="*60)
        print(result["response"])

        # Show task ID for potential resume
        if result.get("task_id") and not args.no_checkpoint:
            print(f"\nTask ID: {result['task_id']}")
            if not result.get("success"):
                print(f"To resume: python -m um_agent_coder --resume {result['task_id']}")

        if args.verbose or result.get("metrics", {}).get("total_cost", 0) > 0:
            print("\n" + "="*60)
            print("METRICS")
            print("="*60)
            metrics = result.get("metrics", {})
            print(f"Success Rate: {metrics.get('success_rate', 0):.1f}%")
            print(f"Total Cost: ${metrics.get('total_cost', 0):.4f}")
            print(f"Effectiveness Score: {metrics.get('effectiveness_score', 0):.1f}")
            print(f"Context Usage: {result.get('context_usage', {}).get('usage_percentage', 0):.1f}%")

            if result.get("resumed"):
                print("(Resumed from checkpoint)")

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