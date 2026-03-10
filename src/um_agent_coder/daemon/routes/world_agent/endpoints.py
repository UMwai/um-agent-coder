"""World Agent API endpoints."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from um_agent_coder.daemon.routes.world_agent import _firestore as store
from um_agent_coder.daemon.routes.world_agent import _goals as goal_store
from um_agent_coder.daemon.routes.world_agent._collectors import GitHubEventsCollector
from um_agent_coder.daemon.routes.world_agent._orient import orient
from um_agent_coder.daemon.routes.world_agent.models import (
    CycleRequest,
    CycleResponse,
    Goal,
    GoalCreateRequest,
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

        # 3. Update world state
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
