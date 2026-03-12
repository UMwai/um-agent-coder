"""Act layer: executes planned tasks via the Gemini iterate engine.

Takes PlannedTask objects from the decide step, submits them to the
Gemini intelligence layer (iterate endpoint), and opens GitHub PRs
with the generated code.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from um_agent_coder.daemon.routes.world_agent.models import PlannedTask

logger = logging.getLogger(__name__)

# Track act tasks so we don't double-execute
_running_tasks: dict[str, asyncio.Task] = {}


def _build_iterate_prompt(task: PlannedTask) -> str:
    """Convert a PlannedTask into a prompt for the Gemini iterate engine."""
    parts = [
        f"# Task: {task.title}",
        "",
        f"**Project:** {task.project}",
        f"**Goal:** {task.goal_id}",
        f"**Priority:** {task.priority}/10",
        f"**Effort:** {task.estimated_effort}",
        "",
        "## Description",
        task.description,
        "",
    ]

    if task.success_criteria:
        parts.extend([
            "## Success Criteria",
            task.success_criteria,
            "",
        ])

    if task.context:
        parts.extend(["## Context"])
        for k, v in task.context.items():
            parts.append(f"- **{k}:** {v}")
        parts.append("")

    parts.extend([
        "## Output Requirements",
        "Generate complete, production-ready code that:",
        "1. Implements exactly what is described above",
        "2. Includes necessary imports and dependencies",
        "3. Follows existing project conventions",
        "4. Includes inline comments for non-obvious logic",
        "",
        "Return all files with their full paths and complete content.",
    ])

    return "\n".join(parts)


def _build_system_prompt(task: PlannedTask) -> str:
    """Build a system prompt tailored to the task."""
    return (
        f"You are an expert software engineer working on {task.project}. "
        f"You are implementing a task for the '{task.goal_id}' goal. "
        "Write clean, production-ready code. Follow existing patterns and conventions. "
        "Include all necessary files with complete content — no placeholders or TODOs. "
        "If the task involves multiple files, return all of them."
    )


async def _execute_single_task(
    task: PlannedTask,
    slack_webhook: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a single planned task via the Gemini iterate engine.

    Returns a result dict with iteration_id, status, and output.
    """
    from um_agent_coder.daemon.app import get_db, get_settings
    from um_agent_coder.daemon.routes.gemini.iterate import (
        _build_iterate_response,
        _run_iteration,
    )
    from um_agent_coder.daemon.routes.gemini.models import (
        GeminiModelTier,
        IterateRequest,
    )

    get_settings()
    db = get_db()

    iteration_id = f"wa-{task.id}-{uuid.uuid4().hex[:6]}"
    prompt = _build_iterate_prompt(task)
    system_prompt = _build_system_prompt(task)

    # Map effort to iteration count
    effort_map = {"small": 3, "medium": 5, "large": 8}
    max_iters = effort_map.get(task.estimated_effort, 5)

    req = IterateRequest(
        prompt=prompt,
        system_prompt=system_prompt,
        model=GeminiModelTier.pro_3_1,
        max_iterations=max_iters,
        score_threshold=0.85,
        temperature=0.4,
        max_tokens=65536,
        enable_enhancement=True,
        use_multi_turn=True,
        domain_hint="code",
        webhook_url=slack_webhook,
        webhook_events=["completed", "failed"],
    )

    config = {
        "model": req.model.value,
        "max_iterations": req.max_iterations,
        "score_threshold": req.score_threshold,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "enable_enhancement": req.enable_enhancement,
        "use_multi_turn": req.use_multi_turn,
        "domain_hint": req.domain_hint,
        "world_agent_task_id": task.id,
        "world_agent_goal_id": task.goal_id,
    }

    await db.create_gemini_iteration(
        iteration_id=iteration_id,
        original_prompt=req.prompt,
        system_prompt=req.system_prompt,
        eval_context=req.eval_context,
        config=config,
    )

    logger.info(
        "Act: executing task %s (%s) as iteration %s",
        task.id, task.title, iteration_id,
    )

    try:
        await _run_iteration(iteration_id, req)
        response = await _build_iterate_response(iteration_id)
        return {
            "task_id": task.id,
            "iteration_id": iteration_id,
            "status": "completed",
            "final_score": getattr(response, "best_score", None),
            "steps_taken": getattr(response, "steps_completed", 0),
        }
    except Exception as e:
        logger.error("Act: task %s failed: %s", task.id, e)
        return {
            "task_id": task.id,
            "iteration_id": iteration_id,
            "status": "failed",
            "error": str(e),
        }


async def _push_result_to_github(
    task: PlannedTask,
    iteration_id: str,
    files: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """Push generated files to a branch and open a PR.

    Args:
        task: The original planned task.
        iteration_id: The Gemini iteration ID.
        files: List of {"path": ..., "content": ...} dicts.

    Returns PR info dict or None on failure.
    """
    from um_agent_coder.daemon.app import get_settings
    from um_agent_coder.daemon.routes.world_agent._github_write import GitHubWriteClient

    settings = get_settings()
    if not settings.github_token or not task.project:
        return None

    client = GitHubWriteClient(token=settings.github_token)
    repo = task.project  # "owner/repo"
    branch_name = f"world-agent/{task.id}"
    datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    try:
        # Create branch
        base_sha = await client.get_default_branch_sha(repo)
        await client.create_branch(repo, branch_name, base_sha)

        # Push files
        file_dicts = [
            {
                "path": f["path"],
                "content": f["content"],
                "message": f"feat({task.goal_id}): {task.title} — {f['path']}",
            }
            for f in files
        ]
        await client.push_files(repo, branch_name, file_dicts)

        # Open PR
        pr_body = (
            f"## {task.title}\n\n"
            f"**Goal:** {task.goal_id}\n"
            f"**Priority:** {task.priority}/10\n"
            f"**Iteration:** `{iteration_id}`\n\n"
            f"### Description\n{task.description}\n\n"
            f"### Success Criteria\n{task.success_criteria}\n\n"
            f"---\n"
            f"*Generated by World Agent Act layer*"
        )

        pr = await client.create_pull_request(
            repo=repo,
            title=f"[world-agent] {task.title}",
            body=pr_body,
            head=branch_name,
            base="main",
        )

        logger.info("Act: opened PR #%s on %s", pr.get("number"), repo)
        return {
            "pr_number": pr.get("number"),
            "html_url": pr.get("html_url"),
            "branch": branch_name,
        }

    except Exception as e:
        logger.error("Act: GitHub push failed for task %s: %s", task.id, e)
        return None


async def act(
    planned_tasks: List[PlannedTask],
    max_concurrent: int = 2,
) -> List[Dict[str, Any]]:
    """Execute planned tasks via the Gemini iterate engine.

    Runs tasks concurrently (up to max_concurrent), then attempts to
    push successful results to GitHub as PRs.

    Returns a list of result dicts with execution status.
    """
    if not planned_tasks:
        return []

    from um_agent_coder.daemon.app import get_settings

    settings = get_settings()
    slack_webhook = settings.default_slack_webhook

    # Execute tasks concurrently with semaphore
    sem = asyncio.Semaphore(max_concurrent)
    results: List[Dict[str, Any]] = []

    async def _run_with_sem(task: PlannedTask):
        async with sem:
            return await _execute_single_task(task, slack_webhook=slack_webhook)

    tasks_aws = [_run_with_sem(t) for t in planned_tasks]
    results = await asyncio.gather(*tasks_aws, return_exceptions=True)

    # Process results
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            final_results.append({
                "task_id": planned_tasks[i].id,
                "status": "error",
                "error": str(result),
            })
        else:
            final_results.append(result)

    # For successful iterations, try to extract files and push to GitHub
    for i, result in enumerate(final_results):
        if result.get("status") == "completed":
            iteration_id = result.get("iteration_id", "")
            try:
                from um_agent_coder.daemon.app import get_db
                from um_agent_coder.daemon.routes.gemini._file_extractor import extract_files

                db = get_db()
                row = await db.get_gemini_iteration(iteration_id)
                if row:
                    best_output = row.get("best_output", "")
                    if best_output:
                        files = extract_files(best_output)
                        if files:
                            file_dicts = [
                                {"path": f.path, "content": f.content}
                                for f in files
                                if f.content.strip()
                            ]
                            if file_dicts:
                                pr_info = await _push_result_to_github(
                                    planned_tasks[i], iteration_id, file_dicts,
                                )
                                if pr_info:
                                    result["pr"] = pr_info
            except Exception as e:
                logger.warning(
                    "Act: failed to push results for %s: %s",
                    result.get("task_id"), e,
                )

    logger.info(
        "Act: executed %d tasks — %d completed, %d failed",
        len(final_results),
        sum(1 for r in final_results if r.get("status") == "completed"),
        sum(1 for r in final_results if r.get("status") in ("failed", "error")),
    )

    return final_results
