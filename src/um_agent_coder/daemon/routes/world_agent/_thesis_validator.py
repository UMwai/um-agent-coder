"""Thesis validator — reviews held positions against current market events
via Gemini Pro to determine if the original trade thesis is still valid.

Cross-references SEC filings, news, price moves, and regime changes
against each position's thesis to produce HOLD/EXIT/REDUCE/ADD verdicts.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from um_agent_coder.daemon.routes.world_agent.models import Event

logger = logging.getLogger(__name__)

THESIS_VALIDATION_SYSTEM_PROMPT = """\
You are a quantitative portfolio analyst for an AI hedge fund.

You receive:
1. A list of CURRENTLY HELD POSITIONS with their original trade thesis, entry price, P&L, and entry date.
2. CURRENT MARKET EVENTS — real prices, news, SEC filings, VIX, and volume data.

Your job is to validate or invalidate each position's thesis against the latest evidence.

Rules:
- Use ONLY the data provided. Never invent prices or facts.
- For each position, assess whether the original thesis still holds given current events.
- Consider: price action since entry, relevant news, SEC filings, sector moves, VIX regime changes.
- A thesis can be invalidated by: contradicting news, broken technical levels, fundamental changes,
  regime shifts, or exceeded time horizon.
- Be specific — cite which market events support or undermine each thesis.
- If a position is profitable but the thesis is weakening, recommend REDUCE not EXIT.
- If new events strengthen the thesis, recommend ADD (with size guidance).

Return a JSON object:
{
  "validation_timestamp": "ISO timestamp",
  "market_regime": "risk-on | neutral | risk-off | crisis",
  "validations": [
    {
      "symbol": "AAPL",
      "verdict": "HOLD | EXIT | REDUCE | ADD",
      "confidence": 0.85,
      "thesis_status": "intact | weakening | invalidated | strengthened",
      "reasoning": "Detailed reasoning with specific data points from the events",
      "invalidation_triggers": [
        "Price closes below $240 (current support)",
        "iPhone demand revision in next earnings call"
      ],
      "suggested_action": "Specific action if not HOLD, e.g. 'Reduce position by 50% at market open'",
      "time_urgency": "immediate | today | this_week | monitor"
    }
  ],
  "portfolio_notes": "Any cross-position observations (correlations, concentration risk, etc.)"
}
"""


def _build_positions_context(positions: List[Dict[str, Any]]) -> str:
    """Build a structured prompt section from held positions."""
    if not positions:
        return "No positions currently held."

    lines = ["## CURRENT POSITIONS"]
    for p in positions:
        symbol = p.get("symbol", "?")
        direction = p.get("direction", "?")
        quantity = p.get("quantity", 0)
        cost_basis = p.get("cost_basis", 0)
        current_price = p.get("current_price", 0)
        unrealized_pnl = p.get("unrealized_pnl", 0)
        pnl_pct = p.get("pnl_pct", 0)
        entry_date = p.get("entry_date", "?")
        thesis = p.get("thesis", "No thesis recorded")

        lines.append(
            f"- **{symbol}** {direction} x{quantity} | "
            f"Entry: ${cost_basis:,.2f} | Current: ${current_price:,.2f} | "
            f"P&L: ${unrealized_pnl:+,.2f} ({pnl_pct:+.1%}) | "
            f"Since: {entry_date}"
        )
        lines.append(f"  Thesis: {thesis}")

    return "\n".join(lines)


def _build_events_context(events: List[Event]) -> str:
    """Build a market events context for thesis validation (reuses trade recs pattern)."""
    from um_agent_coder.daemon.routes.world_agent._trade_recs import _build_market_context

    return _build_market_context(events)


def _parse_validation_result(text: str) -> Dict[str, Any]:
    """Parse LLM response into structured validation results."""
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
    logger.warning("Failed to parse thesis validation response")
    return {}


async def validate_theses(
    positions: List[Dict[str, Any]],
    events: List[Event],
) -> Dict[str, Any]:
    """Validate trade theses for held positions against current market events.

    Uses Gemini Pro to cross-reference positions with real market data
    and produce HOLD/EXIT/REDUCE/ADD verdicts.

    Returns structured dict with validations per position.
    """
    from um_agent_coder.daemon.app import get_gemini_client, get_settings

    settings = get_settings()
    client = get_gemini_client()
    if not client:
        logger.error("Gemini client not available for thesis validation")
        return {}

    if not positions:
        return {"validations": [], "portfolio_notes": "No positions to validate."}

    positions_context = _build_positions_context(positions)
    events_context = _build_events_context(events)

    if not events_context.strip():
        events_context = "No recent market events available."

    now = datetime.now(timezone.utc)
    user_prompt = (
        f"## CURRENT TIME\n{now.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"{positions_context}\n\n"
        f"{events_context}\n\n"
        f"Validate each position's thesis against the current market data. "
        f"Use the ACTUAL prices and events shown above. Do not invent any data."
    )

    try:
        response = await client.generate(
            prompt=user_prompt,
            system_prompt=THESIS_VALIDATION_SYSTEM_PROMPT,
            model=settings.gemini_model_pro,
            temperature=0.2,
            max_tokens=8192,
        )

        text = response["text"] if isinstance(response, dict) else str(response)
        result = _parse_validation_result(text)
        if result:
            validations = result.get("validations", [])
            logger.info(
                "Thesis validation: %d positions reviewed, verdicts: %s",
                len(validations),
                ", ".join(f"{v.get('symbol','?')}={v.get('verdict','?')}" for v in validations),
            )
        return result

    except Exception as e:
        logger.error("Thesis validation failed: %s", e)
        return {}
