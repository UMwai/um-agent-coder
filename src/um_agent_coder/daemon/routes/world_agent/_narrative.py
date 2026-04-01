"""Narrative Intelligence — tracks valuation stories the market is pricing.

Each OODA cycle, this module:
1. Loads existing per-ticker narrative states from the Knowledge Base
2. Compares incoming market events against those narratives
3. Classifies drift: reinforced, challenged, shifting, unchanged
4. Persists updated states and generates signals for drifting narratives

One flash-model LLM call per cycle, batching all watched tickers.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# --- KB constants ---
NARRATIVE_KB_TYPE = "narrative-state"
NARRATIVE_KB_SOURCE = "world-agent-narrator"

# --- LLM Prompts ---

NARRATIVE_SYSTEM_PROMPT = """\
You are a narrative intelligence analyst for a quantitative hedge fund.

Your task: Given market events and existing per-ticker narrative states,
assess whether each ticker's valuation story has changed.

The "narrative" is NOT a DCF model — it is the STORY that justifies the
market's willingness to pay a certain multiple. Examples:
- "NVDA trades at 35x forward because the market believes AI infrastructure
  spending will sustain 50%+ revenue growth through 2027"
- "TSLA's 60x PE is justified by the market pricing robotaxi optionality,
  not current auto margins"
- "META re-rated from 12x to 25x as the market accepted the AI capex story
  would drive engagement + ad revenue, not just burn cash"

For each ticker, classify the narrative drift:
- "reinforced" — new data supports the current story (earnings beat, guidance raise, peer validation)
- "challenged" — new data weakens the story but doesn't break it yet (miss on one metric, competitor gains)
- "shifting" — the dominant narrative is changing to a new story (pivot, new catalyst, reframing)
- "unchanged" — no relevant new information

Return JSON:
{
  "tickers": [
    {
      "ticker": "AAPL",
      "dominant_narrative": "1-2 sentence story the market prices",
      "themes": ["services_growth", "buyback_machine"],
      "confidence": 0.75,
      "drivers": ["services revenue acceleration", "installed base growth"],
      "risks": ["china_revenue_decline", "antitrust_regulation"],
      "freshness_days": 2,
      "multiple_direction": "stable",
      "last_drift": "reinforced",
      "drift_detail": "Q1 services beat reinforces premium multiple"
    }
  ],
  "narrative_signals": [
    {
      "ticker": "TSLA",
      "drift": "shifting",
      "old_narrative": "margin expansion via manufacturing efficiency",
      "new_narrative": "robotaxi pivot + energy storage",
      "urgency": "today",
      "implication": "Multiple expansion if market buys new story; compression if execution doubt"
    }
  ]
}

Rules:
- Only include tickers in narrative_signals if drift is "challenged" or "shifting"
- Be concise. Each field should be brief.
- Themes should be snake_case tags, max 5 per ticker
- If no events relate to a ticker, set drift to "unchanged" and keep existing state
- confidence: 0.9+ = everyone agrees, 0.5 = split, 0.3 = contrarian story gaining
- multiple_direction: "expanding" if market paying more for the story, "compressing" if less, "stable" if flat
"""


def extract_tickers_from_events(events: list) -> list[str]:
    """Extract unique ticker symbols from event metadata."""
    tickers: set[str] = set()
    for event in events:
        meta = getattr(event, "metadata", {}) or {}
        sym = meta.get("symbol", "")
        if sym and len(sym) <= 5 and sym.isalpha():
            tickers.add(sym.upper())
    return sorted(tickers)


def _events_for_ticker(events: list, ticker: str) -> list[str]:
    """Get event titles relevant to a ticker (simple keyword match)."""
    relevant = []
    t_lower = ticker.lower()
    for event in events:
        meta = getattr(event, "metadata", {}) or {}
        title = getattr(event, "title", "") or ""
        body = getattr(event, "body", "") or ""
        sym = (meta.get("symbol", "") or "").upper()
        if sym == ticker or t_lower in title.lower() or t_lower in body[:200].lower():
            relevant.append(title)
    return relevant[:10]


def build_narrative_prompt(
    events: list,
    existing_narratives: dict[str, Any],
    tickers: list[str],
) -> str:
    """Build the user prompt that batches all tickers into one LLM call."""
    sections = []

    # Market events summary
    event_lines = []
    for event in events[:50]:
        title = getattr(event, "title", "")
        severity = getattr(event, "severity", "info")
        source = getattr(event, "source", "")
        if title:
            event_lines.append(f"- [{severity}] ({source}) {title}")
    if event_lines:
        sections.append("## MARKET EVENTS\n" + "\n".join(event_lines))

    # Per-ticker context
    ticker_sections = []
    for ticker in tickers:
        lines = [f"### {ticker}"]
        existing = existing_narratives.get(ticker)
        if existing:
            lines.append(f"Current narrative: {existing.get('dominant_narrative', 'unknown')}")
            lines.append(f"Themes: {', '.join(existing.get('themes', []))}")
            lines.append(f"Confidence: {existing.get('confidence', 0.5)}")
            lines.append(f"Multiple: {existing.get('multiple_direction', 'stable')}")
            lines.append(f"Last drift: {existing.get('last_drift', 'unchanged')}")
            freshness = existing.get("freshness_days", 0)
            if freshness > 0:
                lines.append(f"Days since last catalyst: {freshness}")
        else:
            lines.append("No existing narrative state — generate initial assessment.")

        relevant_events = _events_for_ticker(events, ticker)
        if relevant_events:
            lines.append("Recent events:")
            for evt in relevant_events:
                lines.append(f"  - {evt}")
        else:
            lines.append("No directly relevant events this cycle.")

        ticker_sections.append("\n".join(lines))

    sections.append("## TICKERS TO ANALYZE\n\n" + "\n\n".join(ticker_sections))
    sections.append(
        "Analyze each ticker's narrative. Return JSON with updated states "
        "and narrative_signals for any drifting tickers."
    )
    return "\n\n".join(sections)


def parse_narrative_response(text: str) -> dict:
    """Parse JSON response from LLM, handling markdown fences."""
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
    logger.warning("Failed to parse narrative response")
    return {"tickers": [], "narrative_signals": []}


async def load_narrative_states(tickers: list[str]) -> dict[str, dict]:
    """Load latest narrative state per ticker from Knowledge Base."""
    from um_agent_coder.daemon.routes.kb import _store as kb_store

    states: dict[str, dict] = {}
    for ticker in tickers:
        items = await kb_store.list_items(
            item_type=NARRATIVE_KB_TYPE,
            status="active",
            tag=f"ticker-{ticker.lower()}",
            limit=1,
        )
        if items:
            content = items[0].get("content", "{}")
            try:
                state = json.loads(content) if isinstance(content, str) else content
                states[ticker] = state
            except (json.JSONDecodeError, TypeError):
                pass
    return states


async def persist_narrative_states(
    states: list[dict], cycle_id: str
) -> int:
    """Save narrative states to KB. Upserts: archives old, creates new."""
    from um_agent_coder.daemon.routes.kb import _store as kb_store

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    persisted = 0

    for state in states:
        ticker = state.get("ticker", "")
        if not ticker:
            continue

        tag = f"ticker-{ticker.lower()}"

        # Archive existing active state for this ticker
        existing = await kb_store.list_items(
            item_type=NARRATIVE_KB_TYPE, status="active", tag=tag, limit=1
        )
        for item in existing:
            await kb_store.archive_item(item["id"])

        # Create new state
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["cycle_id"] = cycle_id

        result = await kb_store.create_item({
            "type": NARRATIVE_KB_TYPE,
            "title": f"{ticker} narrative state {today}",
            "content": json.dumps(state),
            "tags": [tag, "narrative", f"date-{today}"],
            "source": NARRATIVE_KB_SOURCE,
            "source_ref": cycle_id,
        })
        if result:
            persisted += 1

    return persisted


async def analyze_narratives(
    events: list,
    tickers: list[str],
    cycle_id: str,
) -> Optional[Dict[str, Any]]:
    """Main entry: load existing states → LLM analysis → persist → return result.

    Returns dict with keys: tickers, narrative_signals, analysis_model, analysis_duration_ms.
    Returns None if analysis fails or is skipped.
    """
    from um_agent_coder.daemon.app import get_llm_router, get_settings

    settings = get_settings()

    if not tickers:
        return None

    # Cap tickers
    tickers = tickers[: settings.world_agent_narrative_max_tickers]

    # Load existing narrative states
    existing = await load_narrative_states(tickers)
    logger.info(
        "Narrative analysis: %d tickers (%d with existing state)",
        len(tickers),
        len(existing),
    )

    # Build prompt
    user_prompt = build_narrative_prompt(events, existing, tickers)

    # Pick model
    model = settings.world_agent_narrative_model or settings.gemini_model_flash

    # Call LLM
    router = get_llm_router()
    start = time.monotonic()
    try:
        response = await router.generate(
            prompt=user_prompt,
            system_prompt=NARRATIVE_SYSTEM_PROMPT,
            model=model,
            temperature=0.3,
            max_tokens=4096,
        )
    except Exception as e:
        logger.warning("Narrative LLM call failed: %s", e)
        return None

    duration_ms = int((time.monotonic() - start) * 1000)
    text = response.get("text", "") if isinstance(response, dict) else str(response)

    # Parse
    parsed = parse_narrative_response(text)
    ticker_states = parsed.get("tickers", [])
    narrative_signals = parsed.get("narrative_signals", [])

    # Persist updated states
    if ticker_states:
        count = await persist_narrative_states(ticker_states, cycle_id)
        logger.info(
            "Narrative analysis complete: %d states persisted, %d drift signals, %dms",
            count,
            len(narrative_signals),
            duration_ms,
        )

    return {
        "tickers": ticker_states,
        "narrative_signals": narrative_signals,
        "analysis_model": model,
        "analysis_duration_ms": duration_ms,
    }


def build_narrative_context_for_trade_recs(
    narrative_result: Optional[Dict[str, Any]],
) -> str:
    """Format narrative analysis as a prompt section for trade rec LLM call."""
    if not narrative_result:
        return ""

    ticker_states = narrative_result.get("tickers", [])
    signals = narrative_result.get("narrative_signals", [])

    if not ticker_states and not signals:
        return ""

    lines = ["## NARRATIVE CONTEXT"]
    lines.append(
        "Valuation stories the market is currently pricing for watched tickers. "
        "Use to modulate conviction — narrative-reinforced + technical = higher conviction; "
        "technical + narrative-deteriorating = reduce size or flag risk."
    )

    for state in ticker_states:
        ticker = state.get("ticker", "?")
        narrative = state.get("dominant_narrative", "unknown")
        drift = state.get("last_drift", "unchanged")
        confidence = state.get("confidence", 0.5)
        multiple = state.get("multiple_direction", "stable")
        drift_icon = {
            "reinforced": "+",
            "challenged": "!",
            "shifting": ">>",
            "unchanged": "=",
        }.get(drift, "=")
        lines.append(
            f"- {ticker} [{drift_icon}{drift.upper()}] (conf={confidence:.2f}, "
            f"multiple={multiple}): {narrative}"
        )

    if signals:
        lines.append("\n### NARRATIVE DRIFT ALERTS")
        for sig in signals:
            ticker = sig.get("ticker", "?")
            old = sig.get("old_narrative", "")
            new = sig.get("new_narrative", "")
            impl = sig.get("implication", "")
            lines.append(f"- {ticker}: \"{old}\" → \"{new}\". {impl}")

    return "\n".join(lines)
