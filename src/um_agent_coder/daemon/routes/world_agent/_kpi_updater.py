"""KPI auto-updater — fetches real performance data and updates Goal KPIs.

Runs at the end of each World Agent cycle to keep Goal KPIs current
with actual pipeline performance, trade rec accuracy, and signal quality.
"""

from __future__ import annotations

import logging

import httpx

from um_agent_coder.daemon.routes.world_agent import _firestore as store
from um_agent_coder.daemon.routes.world_agent import _goals as goal_store

logger = logging.getLogger(__name__)


async def update_kpis(command_center_url: str = "") -> dict:
    """Fetch performance data from Command Center + Firestore and update Goal KPIs.

    Returns a summary of what was updated.
    """
    updates = {}

    # 1. Fetch pipeline performance from Command Center
    perf_data = {}
    if command_center_url:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{command_center_url}/perf/summary")
                if resp.status_code == 200:
                    perf_data = resp.json()
        except Exception as exc:
            logger.debug("Failed to fetch perf summary: %s", exc)

    # 2. Compute trade rec accuracy from Firestore
    rec_accuracy = await _compute_rec_accuracy()

    # 3. Update Goal KPIs
    goals = await goal_store.get_all_goals(status="active")

    for goal in goals:
        goal_updates = {}

        if goal.id == "hedge-fund-alpha":
            # Update signal generation metrics
            if perf_data:
                day_data = perf_data.get("1d", {})
                week_data = perf_data.get("7d", {})

                # Signal hit rate (7-day window)
                accuracy_7d = week_data.get("overall_accuracy", 0)
                if accuracy_7d > 0:
                    goal_updates["signal_hit_rate"] = f"{accuracy_7d * 100:.1f}%"

                # Active signal sources
                sources = day_data.get("sources", {})
                active_sources = len(
                    [s for s in sources.values() if s.get("total_decisions", 0) > 0]
                )
                if active_sources > 0:
                    goal_updates["signal_sources_active"] = str(active_sources)

                # Daily signals
                daily_decisions = day_data.get("total_decisions", 0)
                if daily_decisions > 0:
                    goal_updates["daily_signals_generated"] = str(daily_decisions)

            # Trade rec accuracy
            if rec_accuracy.get("total", 0) > 0:
                win_rate = rec_accuracy.get("win_rate", 0)
                goal_updates["rec_win_rate"] = f"{win_rate:.1f}%"
                goal_updates["rec_total_evaluated"] = str(rec_accuracy["total"])

        elif goal.id == "hedge-fund-build":
            # System health metrics
            if perf_data:
                week_data = perf_data.get("7d", {})
                accuracy = week_data.get("overall_accuracy", 0)
                if accuracy > 0:
                    goal_updates["backtest_accuracy"] = f"{accuracy * 100:.1f}%"

        # Apply updates to KPIs
        if goal_updates:
            kpi_updates = {}
            for kpi in goal.kpis:
                if kpi.metric in goal_updates:
                    kpi_updates[kpi.metric] = goal_updates[kpi.metric]

            if kpi_updates:
                try:
                    await goal_store.update_goal(
                        goal.id,
                        {
                            "kpis": [
                                {
                                    **kpi.model_dump(),
                                    "current": kpi_updates.get(kpi.metric, kpi.current),
                                }
                                for kpi in goal.kpis
                            ]
                        },
                    )
                    updates[goal.id] = kpi_updates
                    logger.info("Updated KPIs for %s: %s", goal.id, kpi_updates)
                except Exception as exc:
                    logger.warning("Failed to update KPIs for %s: %s", goal.id, exc)

    return updates


async def _compute_rec_accuracy() -> dict:
    """Compute win/loss/scratch rates from Firestore trade rec outcomes."""
    try:
        recs = await store.list_trade_recs(limit=200)

        total = 0
        wins = 0
        losses = 0
        scratches = 0
        total_pnl = 0.0

        for rec in recs:
            outcome = rec.get("outcome")
            if not outcome:
                continue
            total += 1
            if outcome == "win":
                wins += 1
            elif outcome == "loss":
                losses += 1
            elif outcome == "scratch":
                scratches += 1
            pnl = rec.get("outcome_pnl_pct", 0) or 0
            total_pnl += pnl

        win_rate = (wins / total * 100) if total > 0 else 0

        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "scratches": scratches,
            "win_rate": win_rate,
            "total_pnl_pct": total_pnl,
        }
    except Exception as exc:
        logger.debug("Failed to compute rec accuracy: %s", exc)
        return {}
