"""Signal dispatcher — posts rich trading intelligence to Slack + Discord.

Dispatches three types of messages:
1. Market regime alerts (VIX, risk-on/off)
2. Individual trade signals/opportunities (movers, funding, earnings)
3. Cycle summary with actionable items

Uses Slack incoming webhook + Discord bot token.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx

from um_agent_coder.daemon.routes.world_agent.models import Event, Signal

logger = logging.getLogger(__name__)

# Discord channel IDs in UM Trades server
DISCORD_CHANNELS = {
    "signals": "1481636924023373884",     # #trading-signals
    "regime": "1448055533935530044",       # #regime
    "daytrading": "1448706889742553272",   # #daytrading
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
    """Dispatch trading intelligence to Slack and Discord.

    Returns dict of {channel: messages_sent} counts.
    """
    stats: Dict[str, int] = {"slack": 0, "discord": 0}

    # Classify events by type
    regime_events = [e for e in events if e.metadata.get("scan_type") == "volatility"]
    mover_events = [e for e in events if e.metadata.get("scan_type") in ("price_move", "volume_spike")]
    funding_events = [e for e in events if e.metadata.get("scan_type") == "funding_rate" and e.metadata.get("apr", 0) and abs(e.metadata.get("apr", 0)) > 10]
    news_events = [e for e in events if e.source == "market.news" and e.severity.value in ("urgent", "critical")]
    [e for e in events if e.source == "market.sec_filings"]

    async with httpx.AsyncClient(timeout=10.0) as client:

        # --- 1. Regime Alert (VIX) → #regime ---
        for ev in regime_events:
            regime = ev.metadata.get("regime", "unknown")
            vix = ev.metadata.get("vix", 0)
            change = ev.metadata.get("change_pct", 0)

            if regime in ("risk-off", "crisis") or abs(change) >= 3:
                emoji = _regime_emoji(regime)
                color = _regime_color(regime)

                if slack_webhook:
                    try:
                        await _post_slack(client, slack_webhook, {
                            "attachments": [{
                                "color": color,
                                "pretext": f"{emoji} *Regime Alert*",
                                "fields": [
                                    {"title": "VIX", "value": f"{vix:.1f} ({change:+.1f}%)", "short": True},
                                    {"title": "Regime", "value": regime.upper(), "short": True},
                                ],
                                "footer": "world-agent | market.volatility",
                            }]
                        })
                        stats["slack"] += 1
                    except Exception as e:
                        logger.debug("Slack regime alert failed: %s", e)

                if discord_bot_token:
                    try:
                        await _post_discord(client, discord_bot_token, DISCORD_CHANNELS["regime"], {
                            "embeds": [{
                                "title": f"{emoji} Regime Alert",
                                "color": _hex_to_int(color),
                                "fields": [
                                    {"name": "VIX", "value": f"{vix:.1f} ({change:+.1f}%)", "inline": True},
                                    {"name": "Regime", "value": regime.upper(), "inline": True},
                                ],
                                "footer": {"text": "world-agent | market.volatility"},
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }]
                        })
                        stats["discord"] += 1
                    except Exception as e:
                        logger.debug("Discord regime alert failed: %s", e)

        # --- 2. Market Movers → single consolidated table ---
        if mover_events:
            # Sort by absolute change descending
            sorted_movers = sorted(
                mover_events,
                key=lambda e: abs(e.metadata.get("change_pct", 0)),
                reverse=True,
            )

            # Build table rows
            lines = []
            for ev in sorted_movers[:20]:
                symbol = ev.metadata.get("symbol", "?")
                change_pct = ev.metadata.get("change_pct", 0)
                price = ev.metadata.get("price", 0)
                vol_ratio = ev.metadata.get("vol_ratio", 0)
                scan_type = ev.metadata.get("scan_type", "")
                arrow = "📈" if change_pct > 0 else "📉"
                vol_flag = f" ⚡{vol_ratio:.0f}x" if scan_type == "volume_spike" or vol_ratio >= 2.0 else ""
                lines.append(f"{arrow} **{symbol}** `{change_pct:+.1f}%` ${price:,.2f}{vol_flag}")

            table_text = "\n".join(lines)

            # Count gainers/losers
            gainers = sum(1 for e in sorted_movers if e.metadata.get("change_pct", 0) > 0)
            losers = len(sorted_movers) - gainers
            net_color = "#28a745" if gainers > losers else "#dc3545" if losers > gainers else "#6c757d"

            if slack_webhook:
                try:
                    # Slack version (no markdown bold)
                    slack_lines = []
                    for ev in sorted_movers[:20]:
                        symbol = ev.metadata.get("symbol", "?")
                        change_pct = ev.metadata.get("change_pct", 0)
                        price = ev.metadata.get("price", 0)
                        vol_ratio = ev.metadata.get("vol_ratio", 0)
                        scan_type = ev.metadata.get("scan_type", "")
                        arrow = ":chart_with_upwards_trend:" if change_pct > 0 else ":chart_with_downwards_trend:"
                        vol_flag = f" :zap:{vol_ratio:.0f}x" if scan_type == "volume_spike" or vol_ratio >= 2.0 else ""
                        slack_lines.append(f"{arrow} *{symbol}* `{change_pct:+.1f}%` ${price:,.2f}{vol_flag}")
                    await _post_slack(client, slack_webhook, {
                        "attachments": [{
                            "color": net_color,
                            "pretext": f":bar_chart: *Market Movers* ({gainers} up / {losers} down)",
                            "text": "\n".join(slack_lines),
                            "footer": "world-agent | market.movers",
                        }]
                    })
                    stats["slack"] += 1
                except Exception:
                    pass

            if discord_bot_token:
                try:
                    await _post_discord(client, discord_bot_token, DISCORD_CHANNELS["signals"], {
                        "embeds": [{
                            "title": f"📊 Market Movers ({gainers} up / {losers} down)",
                            "description": table_text,
                            "color": _hex_to_int(net_color),
                            "footer": {"text": "world-agent | market.movers"},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }]
                    })
                    stats["discord"] += 1
                except Exception:
                    pass

        # --- 3. Crypto Funding Opportunities → single table ---
        actionable_funding = [e for e in funding_events if abs(e.metadata.get("apr", 0)) >= 15]
        if actionable_funding:
            # Sort by absolute APR descending
            actionable_funding.sort(key=lambda e: abs(e.metadata.get("apr", 0)), reverse=True)

            lines = []
            for ev in actionable_funding[:10]:
                symbol = ev.metadata.get("symbol", "?")
                apr = ev.metadata.get("apr", 0)
                rate = ev.metadata.get("funding_rate", 0)
                mark = ev.metadata.get("mark_price", 0)
                emoji = "🟢" if apr > 0 else "🔴"
                trade = "basis" if apr > 0 else "rev basis"
                lines.append(f"{emoji} **{symbol}** `{rate*100:.3f}%` ({apr:.0f}% APR) — ${mark:,.0f} [{trade}]")

            table_text = "\n".join(lines)
            color = "#28a745" if any(e.metadata.get("apr", 0) > 30 for e in actionable_funding) else "#ffc107"

            if slack_webhook:
                try:
                    slack_lines = []
                    for ev in actionable_funding[:10]:
                        symbol = ev.metadata.get("symbol", "?")
                        apr = ev.metadata.get("apr", 0)
                        rate = ev.metadata.get("funding_rate", 0)
                        mark = ev.metadata.get("mark_price", 0)
                        emoji = ":large_green_circle:" if apr > 0 else ":red_circle:"
                        trade = "basis" if apr > 0 else "rev basis"
                        slack_lines.append(f"{emoji} *{symbol}* `{rate*100:.3f}%` ({apr:.0f}% APR) — ${mark:,.0f} [{trade}]")
                    await _post_slack(client, slack_webhook, {
                        "attachments": [{
                            "color": color,
                            "pretext": f":money_with_wings: *Crypto Funding* ({len(actionable_funding)} opportunities)",
                            "text": "\n".join(slack_lines),
                            "footer": "world-agent | market.crypto_funding",
                        }]
                    })
                    stats["slack"] += 1
                except Exception:
                    pass

            if discord_bot_token:
                try:
                    await _post_discord(client, discord_bot_token, DISCORD_CHANNELS["signals"], {
                        "embeds": [{
                            "title": f"💰 Crypto Funding ({len(actionable_funding)} opportunities)",
                            "description": table_text,
                            "color": _hex_to_int(color),
                            "footer": {"text": "world-agent | market.crypto_funding"},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }]
                    })
                    stats["discord"] += 1
                except Exception:
                    pass

        # --- 4. Breaking News → #daytrading ---
        if news_events:
            news_lines = []
            for ev in news_events[:8]:
                title = ev.title.replace("&amp;", "&").replace("&quot;", '"')[:120]
                url = ev.metadata.get("url", "")
                if url:
                    news_lines.append(f"• [{title}]({url})")
                else:
                    news_lines.append(f"• {title}")

            news_text = "\n".join(news_lines)

            if slack_webhook:
                try:
                    await _post_slack(client, slack_webhook, {
                        "attachments": [{
                            "color": "#e83e8c",
                            "pretext": f"📰 *Breaking Market News* ({len(news_events)} stories)",
                            "text": news_text.replace("[", "").replace("]", " ").replace("(", "<").replace(")", ">"),
                            "footer": "world-agent | market.news",
                        }]
                    })
                    stats["slack"] += 1
                except Exception:
                    pass

            if discord_bot_token:
                try:
                    await _post_discord(client, discord_bot_token, DISCORD_CHANNELS["daytrading"], {
                        "embeds": [{
                            "title": f"📰 Breaking Market News ({len(news_events)})",
                            "description": news_text,
                            "color": _hex_to_int("#e83e8c"),
                            "footer": {"text": "world-agent | market.news"},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }]
                    })
                    stats["discord"] += 1
                except Exception:
                    pass

        # --- 5. Oriented Signals Summary → #vol-trades ---
        if signals:
            signal_lines = []
            for s in signals[:10]:
                urgency = s.urgency.value
                urgency_emoji = {"immediate": "🔴", "today": "🟡", "this_week": "🔵"}.get(urgency, "⚪")
                signal_lines.append(
                    f"{urgency_emoji} **{s.goal_id}** (relevance: {s.relevance_score:.0%})\n"
                    f"   {s.interpretation[:150]}\n"
                    f"   → _{s.suggested_action[:120]}_"
                )

            signal_text = "\n\n".join(signal_lines)

            if slack_webhook:
                try:
                    # Slack-formatted version
                    slack_lines = []
                    for s in signals[:10]:
                        urgency = s.urgency.value
                        urgency_emoji = {"immediate": ":red_circle:", "today": ":yellow_circle:", "this_week": ":blue_circle:"}.get(urgency, ":white_circle:")
                        slack_lines.append(
                            f"{urgency_emoji} *{s.goal_id}* (relevance: {s.relevance_score:.0%})\n"
                            f"   {s.interpretation[:150]}\n"
                            f"   → _{s.suggested_action[:120]}_"
                        )
                    await _post_slack(client, slack_webhook, {
                        "attachments": [{
                            "color": "#6f42c1",
                            "pretext": f"🎯 *{len(signals)} Actionable Signals*",
                            "text": "\n\n".join(slack_lines),
                            "footer": f"world-agent | cycle {cycle_id}",
                        }]
                    })
                    stats["slack"] += 1
                except Exception:
                    pass

            if discord_bot_token:
                try:
                    await _post_discord(client, discord_bot_token, DISCORD_CHANNELS["signals"], {
                        "embeds": [{
                            "title": f"🎯 {len(signals)} Actionable Signals",
                            "description": signal_text[:4000],
                            "color": _hex_to_int("#6f42c1"),
                            "footer": {"text": f"world-agent | cycle {cycle_id}"},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }]
                    })
                    stats["discord"] += 1
                except Exception:
                    pass

        # --- 6. Act Results (PRs opened) → #vol-trades ---
        pr_results = [r for r in act_results if r.get("pr")]
        if pr_results:
            pr_lines = []
            for r in pr_results:
                pr = r["pr"]
                pr_lines.append(f"• [PR #{pr['pr_number']}]({pr['html_url']}) — `{r.get('task_id', '?')}`")

            if slack_webhook:
                try:
                    slack_pr = "\n".join(
                        f"• <{r['pr']['html_url']}|PR #{r['pr']['pr_number']}> — `{r.get('task_id', '?')}`"
                        for r in pr_results
                    )
                    await _post_slack(client, slack_webhook, {
                        "attachments": [{
                            "color": "#28a745",
                            "pretext": f"🚀 *{len(pr_results)} PRs Opened by World Agent*",
                            "text": slack_pr,
                            "footer": f"world-agent | cycle {cycle_id}",
                        }]
                    })
                    stats["slack"] += 1
                except Exception:
                    pass

            if discord_bot_token:
                try:
                    await _post_discord(client, discord_bot_token, DISCORD_CHANNELS["signals"], {
                        "embeds": [{
                            "title": f"🚀 {len(pr_results)} PRs Opened",
                            "description": "\n".join(pr_lines),
                            "color": _hex_to_int("#28a745"),
                            "footer": {"text": f"world-agent | cycle {cycle_id}"},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }]
                    })
                    stats["discord"] += 1
                except Exception:
                    pass

    logger.info(
        "Signal dispatch: %d Slack, %d Discord messages",
        stats["slack"], stats["discord"],
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
    return {
        "risk-on": "🟢",
        "neutral": "⚪",
        "risk-off": "🟡",
        "crisis": "🔴",
    }.get(regime, "❓")


def _regime_color(regime: str) -> str:
    return {
        "risk-on": "#28a745",
        "neutral": "#6c757d",
        "risk-off": "#ffc107",
        "crisis": "#dc3545",
    }.get(regime, "#6c757d")


def _hex_to_int(hex_color: str) -> int:
    return int(hex_color.lstrip("#"), 16)
