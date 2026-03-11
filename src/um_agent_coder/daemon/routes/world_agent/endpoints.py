"""World Agent API endpoints."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query

from um_agent_coder.daemon.routes.world_agent import _firestore as store
from um_agent_coder.daemon.routes.world_agent import _goals as goal_store
from um_agent_coder.daemon.routes.world_agent._collectors import GitHubEventsCollector
from um_agent_coder.daemon.routes.world_agent._decide import decide
from um_agent_coder.daemon.routes.world_agent._github_write import GitHubWriteClient
from um_agent_coder.daemon.routes.world_agent._orient import orient
from um_agent_coder.daemon.routes.world_agent.models import (
    CreateBranchRequest,
    CreatePRRequest,
    CycleRequest,
    CycleResponse,
    Goal,
    GoalCreateRequest,
    PostCommentRequest,
    StatusResponse,
    WorldState,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_settings():
    from um_agent_coder.daemon.app import get_settings

    return get_settings()


def _build_github_collector() -> Optional[GitHubEventsCollector]:
    """Build GitHub collector from settings."""
    settings = _get_settings()
    repos_str = settings.world_agent_github_repos
    if not repos_str:
        return None
    repos = [r.strip() for r in repos_str.split(",") if r.strip()]
    if not repos:
        return None
    return GitHubEventsCollector(repos=repos, token=settings.github_token)


def _allowed_repos() -> list[str]:
    """Return the allowlist of repos from settings."""
    settings = _get_settings()
    repos_str = settings.world_agent_github_repos
    if not repos_str:
        return []
    return [r.strip() for r in repos_str.split(",") if r.strip()]


def _validate_repo(owner: str, repo: str) -> str:
    """Validate owner/repo is in the allowlist. Returns full_name or raises 403."""
    full_name = f"{owner}/{repo}"
    if full_name not in _allowed_repos():
        raise HTTPException(status_code=403, detail=f"Repo '{full_name}' not in allowlist")
    return full_name


def _build_write_client() -> GitHubWriteClient:
    """Build GitHub write client. Raises 503 if token is missing."""
    settings = _get_settings()
    if not settings.github_token:
        raise HTTPException(status_code=503, detail="GitHub token not configured")
    return GitHubWriteClient(token=settings.github_token)


# --- Cycle ---


@router.post("/cycle", response_model=CycleResponse)
async def run_cycle(request: CycleRequest):
    """Run a full observe→orient cycle: collect events, filter via LLM, update world state."""
    settings = _get_settings()
    if not settings.world_agent_enabled:
        raise HTTPException(status_code=503, detail="World agent is disabled")

    start = time.time()
    cycle_id = f"cycle-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    try:
        # 1. Observe: collect events
        all_events = []
        gh_collector = _build_github_collector()
        if gh_collector:
            gh_events = await gh_collector.collect()
            all_events.extend(gh_events)

        # Cap events per batch
        max_events = settings.world_agent_max_events_per_batch
        all_events = all_events[:max_events]

        # Persist raw events
        if all_events:
            event_dicts = [e.model_dump(mode="json") for e in all_events]
            # Convert datetime to string for Firestore
            for ed in event_dicts:
                if isinstance(ed.get("timestamp"), datetime):
                    ed["timestamp"] = ed["timestamp"].isoformat()
            await store.save_events(event_dicts)

        # 2. Orient: filter events through LLM
        goals = await goal_store.get_all_goals(status="active")
        threshold = settings.world_agent_relevance_threshold

        summary, signals = await orient(goals, all_events, threshold)

        # 3. Decide: convert signals into planned tasks
        repos = _allowed_repos()
        planned_tasks = await decide(goals, signals, repos)
        tasks_created = len(planned_tasks)

        # Persist planned tasks
        if planned_tasks:
            task_dicts = [t.model_dump(mode="json") for t in planned_tasks]
            await store.save_events(task_dicts)  # reuse events store for now

        # 4. Update world state
        existing_state = await store.get_world_state()
        cycle_count = (existing_state or {}).get("cycle_count", 0) + 1
        total_events = (existing_state or {}).get("total_events_collected", 0) + len(all_events)

        world_state = WorldState(
            summary=summary,
            active_signals=signals,
            cycle_count=cycle_count,
            total_events_collected=total_events,
        )
        await store.save_world_state(world_state.model_dump(mode="json"))

        # Save scheduler state
        await store.save_scheduler_state({
            "last_cycle_id": cycle_id,
            "last_cycle_source": request.source.value,
            "events_collected": len(all_events),
            "signals_generated": len(signals),
        })

        duration_ms = int((time.time() - start) * 1000)
        return CycleResponse(
            cycle_id=cycle_id,
            events_collected=len(all_events),
            signals_generated=len(signals),
            tasks_created=tasks_created,
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        logger.error("Cycle %s failed: %s", cycle_id, e)
        return CycleResponse(
            cycle_id=cycle_id,
            duration_ms=duration_ms,
            error=str(e),
        )


# --- Status ---


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current world state, goals, and cycle count."""
    settings = _get_settings()

    state_doc = await store.get_world_state()
    world_state = None
    cycle_count = 0
    if state_doc:
        try:
            world_state = WorldState(**state_doc)
            cycle_count = world_state.cycle_count
        except Exception:
            pass

    goals = await goal_store.get_all_goals()

    return StatusResponse(
        world_state=world_state,
        goals=goals,
        cycle_count=cycle_count,
        enabled=settings.world_agent_enabled,
    )


# --- Goals ---


@router.post("/goals", response_model=Goal, status_code=201)
async def create_goal(request: GoalCreateRequest):
    """Create a new goal."""
    goal = Goal(**request.model_dump())
    success = await goal_store.create_goal(goal)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create goal")
    return goal


@router.get("/goals", response_model=list[Goal])
async def list_goals(status: Optional[str] = Query(default=None)):
    """List all goals, optionally filtered by status."""
    return await goal_store.get_all_goals(status=status)


@router.get("/goals/{goal_id}", response_model=Goal)
async def get_goal(goal_id: str):
    """Get a single goal by ID."""
    goal = await goal_store.get_goal(goal_id)
    if not goal:
        raise HTTPException(status_code=404, detail=f"Goal '{goal_id}' not found")
    return goal


@router.post("/goals/load-yaml")
async def load_goals_yaml():
    """Load goals from YAML files in the configured goals directory."""
    settings = _get_settings()
    goals = await goal_store.sync_goals_from_yaml(settings.world_agent_goals_path)
    return {"loaded": len(goals), "goal_ids": [g.id for g in goals]}


# --- Events ---


@router.get("/events")
async def list_events(
    since: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    """Query collected events."""
    events = await store.list_events(since=since, source=source, limit=limit)
    return {"events": events, "count": len(events)}


# --- GitHub Write Endpoints ---


@router.get("/repos/{owner}/{repo}/file/{path:path}")
async def get_repo_file(
    owner: str,
    repo: str,
    path: str,
    branch: str = Query(default="main"),
):
    """Read a file from a GitHub repo."""
    full_name = _validate_repo(owner, repo)
    client = _build_write_client()
    try:
        data = await client.get_file(full_name, path, branch)
        return {
            "repo": full_name,
            "path": path,
            "branch": branch,
            "sha": data.get("sha"),
            "size": data.get("size"),
            "content": data.get("decoded_content", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GitHub API error: {e}")


@router.post("/repos/{owner}/{repo}/branch")
async def create_branch(owner: str, repo: str, request: CreateBranchRequest):
    """Create a new branch on a GitHub repo."""
    full_name = _validate_repo(owner, repo)
    client = _build_write_client()
    try:
        base_sha = await client.get_default_branch_sha(full_name, request.base_branch)
        result = await client.create_branch(full_name, request.branch_name, base_sha)
        return {
            "repo": full_name,
            "branch": request.branch_name,
            "sha": base_sha,
            "ref": result.get("ref"),
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GitHub API error: {e}")


@router.post("/repos/{owner}/{repo}/pr")
async def create_pr(owner: str, repo: str, request: CreatePRRequest):
    """Create a PR, optionally pushing files to the head branch first."""
    full_name = _validate_repo(owner, repo)
    client = _build_write_client()
    try:
        # Push files if provided
        if request.files:
            file_dicts = [
                {
                    "path": f.path,
                    "content": f.content,
                    "message": f.message or f"Update {f.path}",
                }
                for f in request.files
            ]
            await client.push_files(full_name, request.head_branch, file_dicts)

        pr = await client.create_pull_request(
            repo=full_name,
            title=request.title,
            body=request.body,
            head=request.head_branch,
            base=request.base_branch,
        )
        return {
            "repo": full_name,
            "pr_number": pr.get("number"),
            "html_url": pr.get("html_url"),
            "state": pr.get("state"),
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GitHub API error: {e}")


@router.post("/repos/{owner}/{repo}/comment")
async def post_comment(owner: str, repo: str, request: PostCommentRequest):
    """Post a comment on a GitHub issue or PR."""
    full_name = _validate_repo(owner, repo)
    client = _build_write_client()
    try:
        result = await client.post_comment(full_name, request.issue_number, request.body)
        return {
            "repo": full_name,
            "issue_number": request.issue_number,
            "comment_id": result.get("id"),
            "html_url": result.get("html_url"),
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GitHub API error: {e}")


@router.get("/repos/{owner}/{repo}/checks/{ref}")
async def get_checks(owner: str, repo: str, ref: str):
    """Get CI check run status for a git ref."""
    full_name = _validate_repo(owner, repo)
    client = _build_write_client()
    try:
        checks = await client.get_check_runs(full_name, ref)
        return {
            "repo": full_name,
            "ref": ref,
            "total": len(checks),
            "check_runs": checks,
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GitHub API error: {e}")
