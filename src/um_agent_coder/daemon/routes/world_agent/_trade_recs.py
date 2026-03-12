"""Trade recommendation engine — analyzes market events via Gemini
and produces structured, actionable trade recommendations.

Takes real market data (prices, VIX, funding rates, news) and outputs
specific trade ideas with entry/stop/target/sizing and reasoning.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from um_agent_coder.daemon.routes.world_agent.models import Event

logger = logging.getLogger(__name__)

TRADE_REC_SYSTEM_PROMPT = """\
You are a quantitative trading analyst for an AI hedge fund.

You receive REAL market data — actual prices, volume, VIX level, funding rates, and news.
Your job is to produce ACTIONABLE trade recommendations.

Rules:
- Use ONLY the data provided. Never invent prices or facts.
- Every recommendation must have: direction, entry, stop loss, target, position size rationale.
- Position sizing follows quarter-Kelly: base 3% of capital, scale with confidence and regime.
- Risk limits: max 10% single position, 25% max drawdown, 3% daily VaR.
- If regime is risk-off or crisis, reduce all sizes by 50-75%.
- Include specific reasoning — what data points support this trade.
- Rate each trade: HIGH / MEDIUM / LOW conviction.
- If no good trades exist, say so. Don't force trades.
- Consider multiple strategies: equity long/short, options (spreads, covered calls, puts),
  crypto basis trades (funding rate arbitrage), mean reversion, momentum.
- For options: mention specific strategy (covered call, put spread, iron condor, etc.)
- Include risk/reward ratio for each trade.

Return a JSON object:
{
  "market_regime": "risk-on | neutral | risk-off | crisis",
  "regime_reasoning": "brief explanation of current regime assessment",
  "recommendations": [
    {
      "symbol": "TICKER",
      "direction": "LONG | SHORT",
      "conviction": "HIGH | MEDIUM | LOW",
      "strategy": "equity_long | equity_short | covered_call | put_spread | iron_condor | basis_trade | momentum | mean_reversion",
      "entry": 150.25,
      "stop_loss": 145.00,
      "target": 162.00,
      "size_pct": 3.5,
      "risk_reward": "1:2.3",
      "timeframe": "intraday | swing (2-5 days) | position (1-4 weeks)",
      "reasoning": "Detailed reasoning with specific data points",
      "risks": "Key risks to this trade",
      "options_detail": "If options strategy: specific strikes, expiry, premium expected (null if equity)"
    }
  ],
  "watchlist": [
    {
      "symbol": "TICKER",
      "note": "Why watching — what trigger would make this actionable"
    }
  ],
  "market_summary": "2-3 sentence overall market assessment"
}

If there are no good trades, return empty recommendations with a clear market_summary explaining why.
"""


def _build_market_context(events: List[Event]) -> str:
    """Build a structured market context prompt from collected events."""
    sections = []

    # Group events by source
    by_source: Dict[str, List[Event]] = {}
    for e in events:
        by_source.setdefault(e.source, []).append(e)

    # VIX / Volatility
    vol_events = by_source.get("market.volatility", [])
    if vol_events:
        sections.append("## VOLATILITY")
        for e in vol_events:
            vix = e.metadata.get("vix", 0)
            change = e.metadata.get("change_pct", 0)
            regime = e.metadata.get("regime", "unknown")
            sections.append(f"- VIX: {vix:.1f} ({change:+.1f}%) — regime: {regime}")

    # Price Movers
    mover_events = by_source.get("market.movers", [])
    if mover_events:
        sections.append("\n## EQUITY PRICES (today's data)")
        # Separate actual movers from all quotes
        for e in mover_events:
            symbol = e.metadata.get("symbol", "?")
            price = e.metadata.get("price", 0)
            change = e.metadata.get("change_pct", 0)
            volume = e.metadata.get("volume", 0)
            vol_ratio = e.metadata.get("vol_ratio", 0)
            scan_type = e.metadata.get("scan_type", "")
            if scan_type == "price_move":
                sections.append(
                    f"- {symbol}: ${price:.2f} ({change:+.1f}%) — SIGNIFICANT MOVE — vol {vol_ratio:.1f}x avg"
                )
            elif scan_type == "volume_spike":
                sections.append(
                    f"- {symbol}: ${price:.2f} ({change:+.1f}%) — VOLUME SPIKE {vol_ratio:.1f}x avg ({volume:,})"
                )
            else:
                sections.append(f"- {symbol}: ${price:.2f} ({change:+.1f}%)")

    # All quotes (even non-movers, from metadata)
    # Include raw quote data if available
    quote_events = [
        e for e in events if e.source == "market.movers" and e.metadata.get("scan_type") == "quote"
    ]
    if quote_events:
        for e in quote_events:
            symbol = e.metadata.get("symbol", "?")
            price = e.metadata.get("price", 0)
            change = e.metadata.get("change_pct", 0)
            sections.append(f"- {symbol}: ${price:.2f} ({change:+.1f}%)")

    # Crypto Funding
    funding_events = by_source.get("market.crypto_funding", [])
    if funding_events:
        sections.append("\n## CRYPTO FUNDING RATES")
        for e in funding_events:
            symbol = e.metadata.get("symbol", "?")
            rate = e.metadata.get("funding_rate", 0)
            apr = e.metadata.get("apr", 0)
            mark = e.metadata.get("mark_price", 0)
            sections.append(
                f"- {symbol}: funding {rate*100:.4f}% ({apr:.1f}% APR) | mark ${mark:,.2f}"
            )

    # News
    news_events = by_source.get("market.news", [])
    if news_events:
        sections.append("\n## NEWS (today)")
        for e in news_events[:12]:
            title = e.title.replace("&amp;", "&").replace("&quot;", '"')
            severity = e.severity.value
            sections.append(f"- [{severity}] {title}")

    # SEC Filings
    sec_events = by_source.get("market.sec_filings", [])
    if sec_events:
        sections.append("\n## SEC FILINGS")
        for e in sec_events[:5]:
            sections.append(f"- {e.title}")

    # GitHub (dev context)
    gh_events = by_source.get("dev.github_events", [])
    if gh_events:
        sections.append("\n## DEV ACTIVITY")
        sections.append(f"- {len(gh_events)} GitHub events across monitored repos")

    return "\n".join(sections)


def _parse_trade_recs(text: str) -> Dict[str, Any]:
    """Parse LLM response into structured trade recommendations."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    logger.warning("Failed to parse trade recommendations")
    return {}


async def generate_trade_recs(events: List[Event]) -> Dict[str, Any]:
    """Generate trade recommendations from market events via Gemini.

    Returns structured dict with regime, recommendations, watchlist, summary.
    """
    from um_agent_coder.daemon.app import get_gemini_client, get_settings

    settings = get_settings()
    client = get_gemini_client()
    if not client:
        logger.error("Gemini client not available for trade recs")
        return {}

    market_context = _build_market_context(events)
    if not market_context.strip():
        return {}

    now = datetime.now(timezone.utc)
    user_prompt = (
        f"## CURRENT TIME\n{now.strftime('%Y-%m-%d %H:%M UTC')} "
        f"({'market hours' if 14 <= now.hour <= 21 else 'after hours'})\n\n"
        f"{market_context}\n\n"
        f"Analyze this market data and produce trade recommendations. "
        f"Use the ACTUAL prices shown above. Do not invent any data."
    )

    try:
        response = await client.generate(
            prompt=user_prompt,
            system_prompt=TRADE_REC_SYSTEM_PROMPT,
            model=settings.gemini_model_pro,
            temperature=0.3,
            max_tokens=8192,
        )

        text = response["text"] if isinstance(response, dict) else str(response)
        recs = _parse_trade_recs(text)
        if recs:
            logger.info(
                "Trade recs: %d recommendations, regime=%s",
                len(recs.get("recommendations", [])),
                recs.get("market_regime", "?"),
            )
        return recs

    except Exception as e:
        logger.error("Trade rec generation failed: %s", e)
        return {}


def format_trade_recs_discord(recs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Format trade recs as Discord embeds.

    Layout:
      Embed 1 — Regime + summary + trades table (all plays at a glance)
      Embed 2..N — One per trade with full detail (reasoning, risks, options)
      Last — Watchlist
    """
    embeds = []

    if not recs:
        return embeds

    regime = recs.get("market_regime", "unknown")
    regime_emoji = {"risk-on": "🟢", "neutral": "⚪", "risk-off": "🟡", "crisis": "🔴"}.get(
        regime, "❓"
    )
    regime_color = {
        "risk-on": 0x28A745,
        "neutral": 0x6C757D,
        "risk-off": 0xFFC107,
        "crisis": 0xDC3545,
    }.get(regime, 0x6C757D)

    recommendations = recs.get("recommendations", [])[:6]

    # --- Embed 1: Overview + summary table ---
    summary = recs.get("market_summary", "")
    regime_reasoning = recs.get("regime_reasoning", "")

    desc_parts = [f"**Regime:** {regime_emoji} {regime.upper()}"]
    if regime_reasoning:
        desc_parts.append(f"_{regime_reasoning}_")
    if summary:
        desc_parts.append(f"\n{summary}")

    # Build the trades table
    if recommendations:
        # Rich table — no code block, uses emoji for color
        table_lines = [""]
        for rec in recommendations:
            symbol = rec.get("symbol", "?")
            direction = rec.get("direction", "?")
            entry = rec.get("entry", 0)
            stop = rec.get("stop_loss", 0)
            target = rec.get("target", 0)
            rr = rec.get("risk_reward", "?")
            conviction = rec.get("conviction", "?")
            dir_emoji = "🟢" if direction == "LONG" else "🔴"
            conv_dot = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "⚪"}.get(conviction, "⚪")
            conv_letter = conviction[0] if conviction else "?"

            def _p(v: float) -> str:
                if not v:
                    return "—"
                return f"${v:,.0f}" if v >= 100 else f"${v:,.2f}"

            table_lines.append(
                f"{dir_emoji} **{symbol}** "
                f"`{_p(entry)}→{_p(target)}` "
                f"stop `{_p(stop)}` "
                f"R:R `{rr}` "
                f"{conv_dot}{conv_letter}"
            )
        desc_parts.append("\n".join(table_lines))

    embeds.append(
        {
            "title": f"📊 Trade Recommendations ({len(recommendations)} plays)",
            "description": "\n".join(desc_parts),
            "color": regime_color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    # --- Embeds 2..N: Detail card per trade ---
    for i, rec in enumerate(recommendations):
        symbol = rec.get("symbol", "?")
        direction = rec.get("direction", "?")
        conviction = rec.get("conviction", "?")
        strategy = rec.get("strategy", "?").replace("_", " ").title()
        entry = rec.get("entry", 0)
        stop = rec.get("stop_loss", 0)
        target = rec.get("target", 0)
        size = rec.get("size_pct", 0)
        rr = rec.get("risk_reward", "?")
        timeframe = rec.get("timeframe", "?")
        reasoning = rec.get("reasoning", "")
        risks = rec.get("risks", "")
        options_detail = rec.get("options_detail")

        dir_emoji = "🟢" if direction == "LONG" else "🔴"
        conv_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💡"}.get(conviction, "❓")
        conv_color = {"HIGH": 0x28A745, "MEDIUM": 0xFD7E14, "LOW": 0x6C757D}.get(
            conviction, 0x6C757D
        )

        # Compact key stats line
        stats_line = (
            f"{dir_emoji} **{direction}** · {conv_emoji} {conviction} · "
            f"Entry **${entry:,.2f}** → Target **${target:,.2f}** · "
            f"Stop ${stop:,.2f} · R:R {rr} · {size:.1f}% · {timeframe}"
        )
        if strategy not in ("?", "Equity Long", "Equity Short"):
            stats_line = f"**{strategy}** — " + stats_line

        detail_parts = [stats_line]

        if options_detail:
            detail_parts.append(f"\n📋 **Options:** {options_detail}")

        detail_parts.append(f"\n📝 **Reasoning**\n{reasoning[:1000]}")

        if risks:
            detail_parts.append(f"\n⚠️ **Risks**\n{risks[:500]}")

        embeds.append(
            {
                "title": f"{i+1}. {symbol}",
                "description": "\n".join(detail_parts),
                "color": conv_color,
            }
        )

    # --- Watchlist ---
    watchlist = recs.get("watchlist", [])
    if watchlist:
        watch_lines = [f"**{w.get('symbol', '?')}** — {w.get('note', '')}" for w in watchlist[:8]]
        embeds.append(
            {
                "title": "👀 Watchlist",
                "description": "\n".join(watch_lines),
                "color": 0x17A2B8,
            }
        )

    return embeds


def format_trade_recs_slack(recs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Format trade recs as Slack attachments.

    Layout: overview + summary table, then detail per trade, then watchlist.
    """
    attachments = []

    if not recs:
        return attachments

    regime = recs.get("market_regime", "unknown")
    regime_emoji = {
        "risk-on": ":large_green_circle:",
        "neutral": ":white_circle:",
        "risk-off": ":warning:",
        "crisis": ":red_circle:",
    }.get(regime, ":question:")
    regime_color = {
        "risk-on": "#28a745",
        "neutral": "#6c757d",
        "risk-off": "#ffc107",
        "crisis": "#dc3545",
    }.get(regime, "#6c757d")

    recommendations = recs.get("recommendations", [])[:6]

    # --- Attachment 1: Overview + table ---
    summary = recs.get("market_summary", "")
    regime_reasoning = recs.get("regime_reasoning", "")

    overview_parts = [f"{regime_emoji} *Regime: {regime.upper()}*"]
    if regime_reasoning:
        overview_parts.append(f"_{regime_reasoning}_")
    if summary:
        overview_parts.append(f"\n{summary}")

    if recommendations:
        table_lines = []
        for rec in recommendations:
            symbol = rec.get("symbol", "?")
            direction = rec.get("direction", "?")
            entry = rec.get("entry", 0)
            stop = rec.get("stop_loss", 0)
            target = rec.get("target", 0)
            rr = rec.get("risk_reward", "?")
            conviction = rec.get("conviction", "?")
            dir_emoji = ":arrow_up:" if direction == "LONG" else ":arrow_down:"
            conv_dot = {
                "HIGH": ":large_green_circle:",
                "MEDIUM": ":yellow_circle:",
                "LOW": ":white_circle:",
            }.get(conviction, ":white_circle:")
            conv_letter = conviction[0] if conviction else "?"

            def _p(v: float) -> str:
                if not v:
                    return "—"
                return f"${v:,.0f}" if v >= 100 else f"${v:,.2f}"

            table_lines.append(
                f"{dir_emoji} *{symbol}* "
                f"`{_p(entry)}→{_p(target)}` "
                f"stop `{_p(stop)}` "
                f"R:R `{rr}` "
                f"{conv_dot}{conv_letter}"
            )
        overview_parts.append("\n".join(table_lines))

    attachments.append(
        {
            "color": regime_color,
            "pretext": f":chart_with_upwards_trend: *Trade Recommendations ({len(recommendations)} plays)*",
            "text": "\n".join(overview_parts),
            "footer": "world-agent | trade-recs",
        }
    )

    # --- Detail per trade ---
    for i, rec in enumerate(recommendations):
        symbol = rec.get("symbol", "?")
        direction = rec.get("direction", "?")
        conviction = rec.get("conviction", "?")
        strategy = rec.get("strategy", "?").replace("_", " ").title()
        entry = rec.get("entry", 0)
        stop = rec.get("stop_loss", 0)
        target = rec.get("target", 0)
        size = rec.get("size_pct", 0)
        rr = rec.get("risk_reward", "?")
        timeframe = rec.get("timeframe", "?")
        reasoning = rec.get("reasoning", "")
        risks = rec.get("risks", "")
        options_detail = rec.get("options_detail")

        dir_emoji = ":arrow_up:" if direction == "LONG" else ":arrow_down:"
        conv_emoji = {"HIGH": ":fire:", "MEDIUM": ":zap:", "LOW": ":bulb:"}.get(conviction, "")
        conv_color = {"HIGH": "#28a745", "MEDIUM": "#fd7e14", "LOW": "#6c757d"}.get(
            conviction, "#6c757d"
        )

        stats_line = (
            f"{dir_emoji} *{direction}* · {conv_emoji} {conviction} · "
            f"Entry *${entry:,.2f}* → Target *${target:,.2f}* · "
            f"Stop ${stop:,.2f} · R:R {rr} · {size:.1f}% · {timeframe}"
        )

        detail_parts = [stats_line]
        if options_detail:
            detail_parts.append(f"\n:clipboard: *Options:* {options_detail}")
        detail_parts.append(f"\n:memo: *Reasoning*\n{reasoning[:500]}")
        if risks:
            detail_parts.append(f"\n:warning: *Risks*\n{risks[:300]}")

        attachments.append(
            {
                "color": conv_color,
                "title": f"{i+1}. {symbol} — {strategy}",
                "text": "\n".join(detail_parts),
            }
        )

    # --- Watchlist ---
    watchlist = recs.get("watchlist", [])
    if watchlist:
        watch_text = "\n".join(
            f"*{w.get('symbol', '?')}* — {w.get('note', '')}" for w in watchlist[:8]
        )
        attachments.append(
            {
                "color": "#17a2b8",
                "title": ":eyes: Watchlist",
                "text": watch_text,
            }
        )

    return attachments
