"""World Agent API endpoints."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Path, Query

from um_agent_coder.daemon.routes.world_agent import _firestore as store
from um_agent_coder.daemon.routes.world_agent import _goals as goal_store
from um_agent_coder.daemon.routes.world_agent._act import act
from um_agent_coder.daemon.routes.world_agent._collectors import GitHubEventsCollector
from um_agent_coder.daemon.routes.world_agent._decide import decide
from um_agent_coder.daemon.routes.world_agent._github_write import GitHubWriteClient
from um_agent_coder.daemon.routes.world_agent._journal import generate_journal
from um_agent_coder.daemon.routes.world_agent._local_repo_collector import LocalRepoCollector
from um_agent_coder.daemon.routes.world_agent._market_collectors import (
    CreditStressCollector,
    CryptoFundingCollector,
    MarketMoversCollector,
    NewsCollector,
    SECFilingsCollector,
    VolatilityCollector,
)
from um_agent_coder.daemon.routes.world_agent._orient import orient
from um_agent_coder.daemon.routes.world_agent._reviewer import review_repo
from um_agent_coder.daemon.routes.world_agent._roadmap_gen import (
    generate_roadmap,
    write_roadmap,
)
from um_agent_coder.daemon.routes.world_agent._signal_dispatcher import dispatch_signals
from um_agent_coder.daemon.routes.world_agent._trade_recs import (
    format_trade_recs_discord,
    format_trade_recs_slack,
    generate_premarket_recs,
    generate_trade_recs,
)
from um_agent_coder.daemon.routes.world_agent.models import (
    CreateBranchRequest,
    CreatePRRequest,
    CycleRecord,
    CycleRequest,
    CycleResponse,
    Goal,
    GoalCreateRequest,
    JournalEntry,
    JournalGenerateRequest,
    JournalResponse,
    PendingTasksResponse,
    PostCommentRequest,
    ReviewRequest,
    ReviewResponse,
    StatusResponse,
    TaskCompleteRequest,
    WorldState,
)

logger = logging.getLogger(__name__)

# In-memory pending task queue (tasks produced by review, awaiting harness pickup)
_pending_tasks: list[dict] = []
_completed_tasks: list[dict] = []
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


def _build_local_collector() -> Optional[LocalRepoCollector]:
    """Build local repo collector from settings."""
    settings = _get_settings()
    local_str = settings.world_agent_local_repos
    if not local_str:
        return None
    repos = {}
    for pair in local_str.split(","):
        pair = pair.strip()
        if "=" in pair:
            name, path = pair.split("=", 1)
            repos[name.strip()] = path.strip()
    if not repos:
        return None
    return LocalRepoCollector(repos=repos)


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


# --- Helpers ---


async def _append_cycle_to_journal(
    events_collected: int,
    signals_generated: int,
    tasks_created: int,
) -> None:
    """Increment today's journal counters after each cycle (no LLM call)."""
    from um_agent_coder.daemon.routes.world_agent.models import JournalEntry

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    existing = await store.get_journal_entry(date_str)

    if existing:
        existing["cycles_run"] = existing.get("cycles_run", 0) + 1
        existing["events_collected"] = existing.get("events_collected", 0) + events_collected
        existing["signals_generated"] = existing.get("signals_generated", 0) + signals_generated
        existing["tasks_created"] = existing.get("tasks_created", 0) + tasks_created
        await store.save_journal_entry(existing)
    else:
        entry = JournalEntry(
            date=date_str,
            cycles_run=1,
            events_collected=events_collected,
            signals_generated=signals_generated,
            tasks_created=tasks_created,
        )
        await store.save_journal_entry(entry.model_dump(mode="json"))


async def _collect_market_events() -> list:
    """Run all market collectors in parallel. Failures are logged, not raised."""
    import asyncio

    collectors = [
        MarketMoversCollector(),
        NewsCollector(),
        VolatilityCollector(),
        CryptoFundingCollector(),
        SECFilingsCollector(),
        CreditStressCollector(),
    ]

    async def _safe_collect(c):
        try:
            return await c.collect()
        except Exception as e:
            logger.warning("Market collector %s failed: %s", c.source_id(), e)
            return []

    results = await asyncio.gather(*[_safe_collect(c) for c in collectors])
    events = []
    for batch in results:
        events.extend(batch)
    logger.info("Market collectors produced %d events", len(events))
    return events


async def _dispatch_trade_recs(
    recs: dict,
    slack_webhook: str | None = None,
    discord_bot_token: str | None = None,
) -> None:
    """Post trade recommendation embeds to Discord #trading-signals and Slack."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Discord
        if discord_bot_token:
            embeds = format_trade_recs_discord(recs)
            if embeds:
                from um_agent_coder.daemon.routes.world_agent._signal_dispatcher import (
                    DISCORD_CHANNELS,
                )

                channel = DISCORD_CHANNELS["signals"]
                # Discord max 10 embeds per message — send in batches
                for i in range(0, len(embeds), 10):
                    batch = embeds[i : i + 10]
                    try:
                        resp = await client.post(
                            f"https://discord.com/api/v10/channels/{channel}/messages",
                            json={"embeds": batch},
                            headers={"Authorization": f"Bot {discord_bot_token}"},
                        )
                        if resp.status_code >= 300:
                            logger.warning(
                                "Discord trade recs failed: %d %s",
                                resp.status_code,
                                resp.text[:200],
                            )
                    except Exception as e:
                        logger.warning("Discord trade recs error: %s", e)

        # Slack
        if slack_webhook:
            attachments = format_trade_recs_slack(recs)
            if attachments:
                try:
                    resp = await client.post(slack_webhook, json={"attachments": attachments})
                    if resp.status_code >= 300:
                        logger.warning("Slack trade recs failed: %d", resp.status_code)
                except Exception as e:
                    logger.warning("Slack trade recs error: %s", e)


async def _notify_slack_cycle(
    cycle_id: str,
    events_collected: int,
    signals_generated: int,
    tasks_created: int,
    duration_ms: int,
    summary: str = "",
    signals: list | None = None,
    planned_tasks: list | None = None,
    act_results: list | None = None,
    error: str | None = None,
) -> None:
    """Send a Slack webhook notification after a world agent cycle."""
    settings = _get_settings()
    webhook_url = settings.default_slack_webhook
    if not webhook_url:
        return

    import httpx

    # Build Slack blocks
    if error:
        color = "#dc3545"
        status_emoji = ":x:"
        f"*Cycle Failed*\n`{error[:200]}`"
    elif signals_generated > 0 or tasks_created > 0:
        color = "#28a745"
        status_emoji = ":large_green_circle:"
    else:
        color = "#6c757d"
        status_emoji = ":white_circle:"

    fields = [
        {"title": "Cycle", "value": f"`{cycle_id}`", "short": True},
        {"title": "Duration", "value": f"{duration_ms}ms", "short": True},
    ]

    # Add signal details if any
    if signals:
        signal_lines = []
        for s in signals[:5]:
            urgency = getattr(s, "urgency", None)
            urgency_val = urgency.value if urgency else "?"
            goal = getattr(s, "goal_id", "")
            interp = getattr(s, "interpretation", "")[:80]
            signal_lines.append(f"• [{urgency_val}] {goal}: {interp}")
        if signal_lines:
            fields.append(
                {
                    "title": f"Signals ({len(signals)})",
                    "value": "\n".join(signal_lines),
                    "short": False,
                }
            )

    # Add planned tasks if any
    if planned_tasks:
        task_lines = []
        for t in planned_tasks[:5]:
            title = getattr(t, "title", str(t)[:60])
            project = getattr(t, "project", "")
            task_lines.append(f"• {project}: {title}")
        if task_lines:
            fields.append(
                {
                    "title": f"Planned Tasks ({len(planned_tasks)})",
                    "value": "\n".join(task_lines),
                    "short": False,
                }
            )

    # Add act results if any
    if act_results:
        act_lines = []
        for r in act_results[:5]:
            status = r.get("status", "?")
            task_id = r.get("task_id", "?")
            emoji = ":white_check_mark:" if status == "completed" else ":x:"
            pr = r.get("pr", {})
            pr_link = f" → <{pr['html_url']}|PR #{pr['pr_number']}>" if pr else ""
            score = r.get("final_score")
            score_text = f" (score: {score:.2f})" if score else ""
            act_lines.append(f"{emoji} `{task_id}` {status}{score_text}{pr_link}")
        if act_lines:
            fields.append(
                {
                    "title": f"Executed ({len(act_results)})",
                    "value": "\n".join(act_lines),
                    "short": False,
                }
            )

    if summary:
        fields.append(
            {
                "title": "Summary",
                "value": summary[:500],
                "short": False,
            }
        )

    payload = {
        "attachments": [
            {
                "color": color,
                "fallback": f"{status_emoji} World Agent {cycle_id}: {events_collected} events, {signals_generated} signals, {tasks_created} tasks",
                "pretext": f"{status_emoji} *World Agent Cycle Complete*",
                "fields": fields,
                "footer": "um-agent-daemon | world-agent",
                "ts": int(datetime.now(timezone.utc).timestamp()),
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(webhook_url, json=payload)
            if resp.status_code >= 300:
                logger.warning("Slack webhook returned %d", resp.status_code)
    except Exception as e:
        logger.warning("Slack webhook failed: %s", e)


# --- Cycle ---


@router.post("/cycle", response_model=CycleResponse)
async def run_cycle(request: CycleRequest):
    """Run a full observe→orient cycle: collect events, filter via LLM, update world state."""
    settings = _get_settings()
    if not settings.world_agent_enabled:
        raise HTTPException(status_code=503, detail="World agent is disabled")

    start = time.time()
    cycle_id = (
        f"cycle-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    )

    try:
        # 1. Observe: collect events
        all_events = []
        gh_collector = _build_github_collector()
        if gh_collector:
            gh_events = await gh_collector.collect()
            all_events.extend(gh_events)

        local_collector = _build_local_collector()
        if local_collector:
            local_events = await local_collector.collect()
            all_events.extend(local_events)

        # Market data collectors (run in parallel)
        market_events = await _collect_market_events()
        all_events.extend(market_events)

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

        # 4. Act: execute planned tasks via Gemini iterate engine
        act_results = []
        if planned_tasks:
            try:
                act_results = await act(planned_tasks, max_concurrent=2)
                logger.info(
                    "Cycle %s act: %d/%d tasks executed",
                    cycle_id,
                    len(act_results),
                    len(planned_tasks),
                )
            except Exception as ae:
                logger.error("Act step failed in cycle %s: %s", cycle_id, ae)

        # 5. Update world state
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
        await store.save_scheduler_state(
            {
                "last_cycle_id": cycle_id,
                "last_cycle_source": request.source.value,
                "events_collected": len(all_events),
                "signals_generated": len(signals),
            }
        )

        duration_ms = int((time.time() - start) * 1000)

        # Persist full cycle record (append-only history)
        goal_ids_touched = sorted({s.goal_id for s in signals if s.goal_id})
        cycle_record = CycleRecord(
            cycle_id=cycle_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=request.source.value,
            events_collected=len(all_events),
            signals_generated=len(signals),
            tasks_created=tasks_created,
            duration_ms=duration_ms,
            summary=summary,
            signals=signals,
            planned_tasks=[t.model_dump(mode="json") for t in planned_tasks],
            act_results=act_results,
            event_ids=[e.id for e in all_events],
            goal_ids_touched=goal_ids_touched,
        )
        try:
            await store.save_cycle_record(cycle_record.model_dump(mode="json"))
        except Exception as ce:
            logger.warning("Failed to save cycle record: %s", ce)

        # Append cycle stats to today's journal (lightweight, no LLM)
        try:
            await _append_cycle_to_journal(
                events_collected=len(all_events),
                signals_generated=len(signals),
                tasks_created=tasks_created,
            )
        except Exception as je:
            logger.warning("Failed to update journal from cycle: %s", je)

        # 6. Generate trade recommendations from market data
        #    Use premarket recs during pre-market window (11:00-13:30 UTC = 7:00-9:30 ET)
        trade_recs = {}
        if market_events:
            try:
                now_utc = datetime.now(timezone.utc)
                is_premarket = 11 <= now_utc.hour <= 13 and now_utc.hour * 60 + now_utc.minute < 13 * 60 + 30
                if is_premarket:
                    trade_recs = await generate_premarket_recs(all_events)
                else:
                    trade_recs = await generate_trade_recs(all_events)
                if trade_recs:
                    logger.info(
                        "Cycle %s trade recs: %d recommendations, regime=%s",
                        cycle_id,
                        len(trade_recs.get("recommendations", [])),
                        trade_recs.get("market_regime", "?"),
                    )
                    # Persist to Firestore for accuracy tracking
                    try:
                        await store.save_trade_recs(
                            cycle_id=cycle_id,
                            recs=trade_recs,
                            model=settings.gemini_model_pro,
                            sw_version="0.2.0",
                            market_context_summary=trade_recs.get("market_summary", ""),
                        )
                    except Exception as se:
                        logger.warning("Failed to persist trade recs: %s", se)

                    # Push to Command Center webhook (zero-latency path)
                    if settings.command_center_url:
                        try:
                            from um_agent_coder.daemon.routes.world_agent._push_bridge import (
                                push_recs_to_command_center,
                            )
                            await push_recs_to_command_center(
                                recs=trade_recs,
                                cycle_id=cycle_id,
                                command_center_url=settings.command_center_url,
                            )
                        except Exception as pe:
                            logger.warning("Push to command center failed: %s", pe)
            except Exception as tre:
                logger.warning("Trade rec generation failed: %s", tre)

        # Dispatch rich signals to Slack + Discord
        try:
            await dispatch_signals(
                events=all_events,
                signals=signals,
                planned_tasks=planned_tasks,
                act_results=act_results,
                cycle_id=cycle_id,
                slack_webhook=settings.default_slack_webhook,
                discord_bot_token=settings.discord_bot_token,
            )
        except Exception as de:
            logger.warning("Signal dispatch failed: %s", de)

        # Dispatch trade recommendations to Discord + Slack
        if trade_recs:
            try:
                await _dispatch_trade_recs(
                    trade_recs,
                    slack_webhook=settings.default_slack_webhook,
                    discord_bot_token=settings.discord_bot_token,
                )
            except Exception as tde:
                logger.warning("Trade rec dispatch failed: %s", tde)

        # Cycle summary notification (Slack only — lightweight status line)
        try:
            await _notify_slack_cycle(
                cycle_id=cycle_id,
                events_collected=len(all_events),
                signals_generated=len(signals),
                tasks_created=tasks_created,
                duration_ms=duration_ms,
                summary=summary,
                signals=signals,
                planned_tasks=planned_tasks,
                act_results=act_results,
            )
        except Exception as se:
            logger.warning("Failed to send Slack notification: %s", se)

        # Update Goal KPIs with real performance data (non-blocking)
        if settings.command_center_url:
            try:
                from um_agent_coder.daemon.routes.world_agent._kpi_updater import update_kpis
                kpi_result = await update_kpis(command_center_url=settings.command_center_url)
                if kpi_result:
                    logger.info("KPI update: %s", kpi_result)
            except Exception as ke:
                logger.debug("KPI update failed (non-critical): %s", ke)

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

        # Save failed cycle record too
        try:
            failed_record = CycleRecord(
                cycle_id=cycle_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source=request.source.value,
                duration_ms=duration_ms,
                error=str(e),
            )
            await store.save_cycle_record(failed_record.model_dump(mode="json"))
        except Exception:
            pass

        # Notify failure to Slack
        try:
            await _notify_slack_cycle(
                cycle_id=cycle_id,
                events_collected=0,
                signals_generated=0,
                tasks_created=0,
                duration_ms=duration_ms,
                error=str(e),
            )
        except Exception:
            pass

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


# --- Event-Triggered Fast Cycle ---


@router.post("/cycle/event-triggered")
async def event_triggered_cycle(
    trigger_type: str = Query(..., description="vix_spike, credit_stress, breaking_news"),
    trigger_data: dict = Body(default={}),  # noqa: B008
):
    """Run an immediate fast cycle triggered by a market event.

    Skips GitHub/local collectors and full OODA pipeline. Runs only:
    1. Market data collectors (parallel)
    2. Trade rec generation
    3. Push bridge to command center

    Typical latency: ~5-10s vs ~30s for a full cycle.
    """
    settings = _get_settings()
    if not settings.world_agent_enabled:
        raise HTTPException(status_code=503, detail="World agent is disabled")

    start = time.time()
    cycle_id = (
        f"event-{trigger_type}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        f"-{uuid.uuid4().hex[:4]}"
    )

    logger.info("Event-triggered cycle: type=%s data=%s", trigger_type, trigger_data)

    try:
        # 1. Collect market events only (fast path)
        market_events = await _collect_market_events()

        # 2. Generate trade recs from events + trigger context
        trade_recs = {}
        if market_events:
            trade_recs = await generate_trade_recs(market_events)

            if trade_recs:
                # Persist to Firestore
                try:
                    await store.save_trade_recs(
                        cycle_id=cycle_id,
                        recs=trade_recs,
                        model=settings.gemini_model_pro,
                        sw_version="0.2.0",
                        market_context_summary=f"Event-triggered ({trigger_type}): "
                        + trade_recs.get("market_summary", ""),
                    )
                except Exception:
                    pass

                # 3. Push to command center
                if settings.command_center_url:
                    try:
                        from um_agent_coder.daemon.routes.world_agent._push_bridge import (
                            push_recs_to_command_center,
                        )
                        await push_recs_to_command_center(
                            recs=trade_recs,
                            cycle_id=cycle_id,
                            command_center_url=settings.command_center_url,
                        )
                    except Exception as pe:
                        logger.warning("Event push failed: %s", pe)

                # Dispatch to Discord/Slack
                try:
                    await _dispatch_trade_recs(
                        trade_recs,
                        slack_webhook=settings.default_slack_webhook,
                        discord_bot_token=settings.discord_bot_token,
                    )
                except Exception:
                    pass

        duration_ms = int((time.time() - start) * 1000)
        num_recs = len(trade_recs.get("recommendations", []))
        logger.info(
            "Event-triggered cycle %s completed: %dms, %d recs, regime=%s",
            cycle_id, duration_ms, num_recs, trade_recs.get("market_regime", "?"),
        )

        return {
            "cycle_id": cycle_id,
            "trigger_type": trigger_type,
            "events_collected": len(market_events),
            "recommendations": num_recs,
            "market_regime": trade_recs.get("market_regime", "unknown"),
            "duration_ms": duration_ms,
        }

    except Exception as exc:
        logger.error("Event-triggered cycle failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


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


# --- Cycle History ---


@router.get("/cycles")
async def list_cycles(
    date: Optional[str] = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    limit: int = Query(default=50, ge=1, le=500),
):
    """List cycle records for a date (default: today), newest first."""
    records = await store.list_cycle_records(date_str=date, limit=limit)
    return {"cycles": records, "count": len(records), "date": date}


@router.get("/cycles/stats")
async def get_cycles_stats(
    date: Optional[str] = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    """Get aggregate stats for all cycles on a given date."""
    stats = await store.get_cycle_stats(date_str=date)
    return stats


@router.get("/cycles/{cycle_id}")
async def get_cycle(
    cycle_id: str,
    date: Optional[str] = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    """Get a single cycle record by ID."""
    record = await store.get_cycle_record(cycle_id, date_str=date)
    if not record:
        raise HTTPException(status_code=404, detail=f"Cycle '{cycle_id}' not found")
    return record


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


# --- Journal ---


@router.post("/journal/generate", response_model=JournalResponse)
async def generate_journal_entry(request: JournalGenerateRequest):
    """Generate a daily journal entry from the day's events and cycles.

    Synthesizes events, signals, and world state into a narrative summary
    via LLM, then persists to Firestore. Can be called manually or on a schedule.
    """
    settings = _get_settings()
    if not settings.world_agent_enabled:
        raise HTTPException(status_code=503, detail="World agent is disabled")

    try:
        entry = await generate_journal(date_str=request.date)
        return JournalResponse(entry=entry, generated=True)
    except Exception as e:
        logger.error("Failed to generate journal: %s", e)
        raise HTTPException(status_code=500, detail=f"Journal generation failed: {e}")


@router.get("/journal/{date}", response_model=JournalResponse)
async def get_journal(date: str = Path(..., pattern=r"^\d{4}-\d{2}-\d{2}$")):
    """Get a journal entry for a specific date."""
    entry_data = await store.get_journal_entry(date)
    if not entry_data:
        raise HTTPException(status_code=404, detail=f"No journal entry for {date}")
    return JournalResponse(entry=JournalEntry(**entry_data), generated=False)


@router.get("/journal", response_model=list[JournalEntry])
async def list_journal(
    limit: int = Query(default=30, ge=1, le=365),
):
    """List recent journal entries, newest first."""
    entries = await store.list_journal_entries(limit=limit)
    return [JournalEntry(**e) for e in entries]


# --- Learning ---


@router.post("/learn/reflect")
async def run_reflection(
    days: int = Query(default=7, ge=1, le=30),
):
    """Run a reflection cycle: analyze past journals and cycle history,
    extract lessons about task planning, and store them in the KB.

    Call weekly or on-demand to improve future decision quality.
    """
    settings = _get_settings()
    if not settings.world_agent_enabled:
        raise HTTPException(status_code=503, detail="World agent is disabled")

    from um_agent_coder.daemon.routes.world_agent._learner import reflect

    try:
        result = await reflect(days=days)
        return result
    except Exception as e:
        logger.error("Reflection failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Reflection failed: {e}")


@router.get("/learn/lessons")
async def list_lessons(
    limit: int = Query(default=20, ge=1, le=100),
):
    """List current decision-making lessons from the KB."""
    from um_agent_coder.daemon.routes.kb import _store as kb_store

    lessons, tokens = await kb_store.search_items("decide-lesson", limit=limit)
    return {"lessons": lessons, "count": len(lessons), "search_tokens": tokens}


@router.get("/learn/context")
async def preview_decision_context(
    goal_id: Optional[str] = Query(default=None),
):
    """Preview what learned context would be injected into the next decision cycle."""
    from um_agent_coder.daemon.routes.world_agent._learner import get_decision_context

    goal_ids = [goal_id] if goal_id else None
    context = await get_decision_context(goal_ids=goal_ids)
    return {"context": context, "has_lessons": bool(context)}


# --- Trade Recommendations ---


@router.get("/trade-recs")
async def get_trade_recs(
    date: Optional[str] = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    symbol: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    """List trade recommendations for a date, optionally filtered by symbol."""
    recs = await store.list_trade_recs(date_str=date, symbol=symbol, limit=limit)
    return {"recs": recs, "count": len(recs), "date": date}


@router.post("/trade-recs/{date}/{rec_id}/outcome")
async def update_trade_outcome(
    date: str = Path(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
    rec_id: str = Path(...),
    outcome: str = Query(..., pattern=r"^(win|loss|scratch|expired)$"),
    pnl_pct: Optional[float] = Query(default=None),
    notes: Optional[str] = Query(default=None),
):
    """Record the outcome of a trade recommendation for accuracy tracking.

    outcome: win | loss | scratch | expired
    pnl_pct: realized P&L percentage (e.g. 2.5 for +2.5%, -1.3 for -1.3%)
    """
    success = await store.update_trade_rec_outcome(
        date_str=date,
        rec_id=rec_id,
        outcome=outcome,
        pnl_pct=pnl_pct,
        notes=notes,
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update outcome")

    # Feed outcome into journal for reflection/learning
    try:
        await store.append_outcome_to_journal(date, rec_id, outcome, pnl_pct)
    except Exception:
        pass  # non-critical — the outcome is already persisted

    return {"rec_id": rec_id, "outcome": outcome, "pnl_pct": pnl_pct}


# ---------------------------------------------------------------------------
# Repo Review endpoints
# ---------------------------------------------------------------------------


@router.post("/repos/{owner}/{repo}/review", response_model=ReviewResponse)
async def review_repository(
    owner: str = Path(...),
    repo: str = Path(...),
    request: ReviewRequest = Body(...),  # noqa: B008
):
    """Review a repo against a goal, producing gap analysis and recommended tasks.

    Optionally generates a harness-compatible roadmap file.
    """
    settings = _get_settings()
    if not settings.world_agent_enabled:
        raise HTTPException(status_code=503, detail="World agent is disabled")

    full_name = _validate_repo(owner, repo)

    # Resolve goal
    goal = await goal_store.get_goal(request.goal_id)
    if not goal:
        raise HTTPException(status_code=404, detail=f"Goal '{request.goal_id}' not found")

    # Resolve repo path — from request, settings, or goal projects
    repo_path = request.repo_path
    if not repo_path:
        local_str = settings.world_agent_local_repos or ""
        for pair in local_str.split(","):
            pair = pair.strip()
            if "=" in pair:
                name, rpath = pair.split("=", 1)
                if name.strip() in repo or repo in name.strip():
                    repo_path = rpath.strip()
                    break

    if not repo_path:
        raise HTTPException(
            status_code=400,
            detail=f"Could not resolve local path for {full_name}. "
            "Set repo_path in request or configure UM_DAEMON_WORLD_AGENT_LOCAL_REPOS.",
        )

    result = await review_repo(repo_path, goal, depth=request.depth)

    # Generate roadmap if tasks were recommended
    roadmap_path = None
    if result.get("recommended_tasks"):
        roadmap_content = generate_roadmap(
            review_result=result,
            goal_id=request.goal_id,
            goal_name=goal.name,
            repo_path=repo_path,
        )
        roadmap_file = f".harness/roadmaps/roadmap-{request.goal_id}.md"
        roadmap_path = write_roadmap(roadmap_content, roadmap_file)
        logger.info("Generated roadmap at %s", roadmap_path)

        _pending_tasks.append(
            {
                "id": f"review-{request.goal_id}-{int(time.time())}",
                "goal_id": request.goal_id,
                "repo": full_name,
                "repo_path": repo_path,
                "title": f"Execute roadmap for {goal.name}",
                "roadmap_path": roadmap_path,
                "task_count": len(result["recommended_tasks"]),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return ReviewResponse(
        repo=repo_path,
        goal_id=request.goal_id,
        review_summary=result.get("review_summary", ""),
        kpi_assessment=result.get("kpi_assessment", []),
        gaps=result.get("gaps", []),
        recommended_tasks=result.get("recommended_tasks", []),
        roadmap_path=roadmap_path,
        error=result.get("error"),
    )


# ---------------------------------------------------------------------------
# Task polling endpoints (for local harness bridge)
# ---------------------------------------------------------------------------


@router.get("/tasks/pending", response_model=PendingTasksResponse)
async def list_pending_tasks():
    """List pending tasks awaiting harness execution."""
    return PendingTasksResponse(tasks=_pending_tasks)


@router.post("/tasks/{task_id}/complete")
async def complete_task(
    task_id: str = Path(...),
    request: TaskCompleteRequest = Body(...),  # noqa: B008
):
    """Report task completion from the local harness."""
    global _pending_tasks

    task = None
    remaining = []
    for t in _pending_tasks:
        if t.get("id") == task_id:
            task = t
        else:
            remaining.append(t)
    _pending_tasks = remaining

    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found in pending queue")

    task["completed_at"] = datetime.now(timezone.utc).isoformat()
    task["success"] = request.success
    task["output"] = request.output
    task["pr_url"] = request.pr_url
    _completed_tasks.append(task)

    logger.info(
        "Task %s completed: success=%s pr=%s",
        task_id,
        request.success,
        request.pr_url or "none",
    )
    return {"status": "ok", "task_id": task_id, "success": request.success}
