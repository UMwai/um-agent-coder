"""Signal dispatcher — posts a single consolidated cycle summary to Discord + Slack.

One message per cycle to #trading-signals with:
- Regime + VIX
- Top movers (3 max)
- Trade recommendations (compact table)
- Alerts (credit stress, news) if any
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx

from um_agent_coder.daemon.routes.world_agent.models import Event, Signal

logger = logging.getLogger(__name__)

DISCORD_CHANNELS = {
    "signals": "1481636924023373884",  # #trading-signals
}


async def dispatch_signals(
    events: List[Event],
    signals: List[Signal],
    planned_tasks: list,
    act_results: list,
    cycle_id: str,
    slack_webhook: Optional[str] = None,
    discord_bot_token: Optional[str] = None,
) -> Dict[str, int]:
    """Build and send a single consolidated cycle summary."""
    stats: Dict[str, int] = {"slack": 0, "discord": 0}

    # Classify events
    regime_events = [e for e in events if e.metadata.get("scan_type") == "volatility"]
    mover_events = [
        e for e in events if e.metadata.get("scan_type") in ("price_move", "volume_spike")
    ]
    credit_events = [
        e for e in events
        if e.source == "market.credit_stress" and e.severity.value in ("urgent", "critical")
    ]
    news_events = [
        e for e in events
        if e.source == "market.news" and e.severity.value in ("urgent", "critical")
    ]

    # --- Build consolidated message sections ---
    sections = []

    # 1. Regime line
    for ev in regime_events[:1]:
        regime = ev.metadata.get("regime", "unknown")
        vix = ev.metadata.get("vix", 0)
        change = ev.metadata.get("change_pct", 0)
        emoji = _regime_emoji(regime)
        sections.append(f"{emoji} **Regime: {regime.upper()}** | VIX {vix:.1f} ({change:+.1f}%)")

    # 2. Top movers (3 max)
    if mover_events:
        sorted_movers = sorted(
            mover_events, key=lambda e: abs(e.metadata.get("change_pct", 0)), reverse=True
        )[:3]
        mover_lines = []
        for ev in sorted_movers:
            sym = ev.metadata.get("symbol", "?")
            chg = ev.metadata.get("change_pct", 0)
            price = ev.metadata.get("price", 0)
            vol = ev.metadata.get("vol_ratio", 0)
            arrow = "📈" if chg > 0 else "📉"
            vol_flag = f" ⚡{vol:.0f}x" if vol >= 2.0 else ""
            mover_lines.append(f"{arrow} **{sym}** `{chg:+.1f}%` ${price:,.2f}{vol_flag}")
        sections.append("**Movers:** " + " | ".join(mover_lines))

    # 3. Signals summary (count only)
    if signals:
        immediate = sum(1 for s in signals if s.urgency.value == "immediate")
        today = sum(1 for s in signals if s.urgency.value == "today")
        parts = []
        if immediate:
            parts.append(f"🔴 {immediate} immediate")
        if today:
            parts.append(f"🟡 {today} today")
        if parts:
            sections.append(f"**Signals:** {', '.join(parts)} of {len(signals)} total")

    # 4. Alerts (credit + news, compact)
    alert_parts = []
    if credit_events:
        alert_parts.append(f"🏦 {len(credit_events)} credit stress")
    if news_events:
        top_news = news_events[0].title[:80] if news_events else ""
        alert_parts.append(f"📰 {top_news}")
    if alert_parts:
        sections.append("**Alerts:** " + " | ".join(alert_parts))

    # 5. Act results
    if act_results:
        pr_count = sum(1 for r in act_results if r.get("pr"))
        if pr_count:
            sections.append(f"🚀 {pr_count} PRs opened")

    # 6. Cycle footer
    duration_parts = [f"cycle `{cycle_id[-12:]}`"]
    if planned_tasks:
        duration_parts.append(f"{len(planned_tasks)} tasks")
    sections.append(f"⏱️ {' | '.join(duration_parts)}")

    # Combine into one embed description
    description = "\n".join(sections)

    # Determine color from regime
    regime = "unknown"
    for ev in regime_events[:1]:
        regime = ev.metadata.get("regime", "unknown")
    color = _regime_color_int(regime)

    async with httpx.AsyncClient(timeout=10.0) as client:

        # --- Discord: single message ---
        if discord_bot_token and description:
            try:
                await _post_discord(
                    client,
                    discord_bot_token,
                    DISCORD_CHANNELS["signals"],
                    {
                        "embeds": [
                            {
                                "title": f"📊 World Agent Cycle",
                                "description": description[:4000],
                                "color": color,
                                "footer": {"text": "world-agent"},
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        ]
                    },
                )
                stats["discord"] += 1
            except Exception as e:
                logger.debug("Discord cycle summary failed: %s", e)

        # --- Slack: single message ---
        if slack_webhook and description:
            try:
                await _post_slack(
                    client,
                    slack_webhook,
                    {
                        "attachments": [
                            {
                                "color": _regime_color_hex(regime),
                                "pretext": ":chart_with_upwards_trend: *World Agent Cycle*",
                                "text": description.replace("**", "*"),
                                "footer": "world-agent",
                            }
                        ]
                    },
                )
                stats["slack"] += 1
            except Exception:
                pass

    logger.info(
        "Signal dispatch: %d Slack, %d Discord messages",
        stats["slack"],
        stats["discord"],
    )
    return stats


# --- Helpers ---


async def _post_slack(client: httpx.AsyncClient, webhook_url: str, payload: dict):
    resp = await client.post(webhook_url, json=payload)
    if resp.status_code >= 300:
        logger.warning("Slack post failed: %d", resp.status_code)


async def _post_discord(client: httpx.AsyncClient, bot_token: str, channel_id: str, payload: dict):
    resp = await client.post(
        f"https://discord.com/api/v10/channels/{channel_id}/messages",
        json=payload,
        headers={"Authorization": f"Bot {bot_token}"},
    )
    if resp.status_code >= 300:
        logger.warning("Discord post failed: %d %s", resp.status_code, resp.text[:200])


def _regime_emoji(regime: str) -> str:
    return {"risk-on": "🟢", "neutral": "⚪", "risk-off": "🟡", "crisis": "🔴"}.get(regime, "❓")


def _regime_color_int(regime: str) -> int:
    return {"risk-on": 0x28A745, "neutral": 0x6C757D, "risk-off": 0xFFC107, "crisis": 0xDC3545}.get(
        regime, 0x6C757D
    )


def _regime_color_hex(regime: str) -> str:
    return {"risk-on": "#28a745", "neutral": "#6c757d", "risk-off": "#ffc107", "crisis": "#dc3545"}.get(
        regime, "#6c757d"
    )
