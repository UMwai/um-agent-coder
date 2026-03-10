"""Orientation layer: LLM-powered event filtering and signal generation.

Sends batched events + goals to Gemini Flash for relevance filtering.
Produces Signal objects from raw events.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from um_agent_coder.daemon.routes.world_agent.models import (
    Event,
    Goal,
    Signal,
    SignalUrgency,
)

logger = logging.getLogger(__name__)

ORIENTATION_SYSTEM_PROMPT = """\
You are an event-filtering intelligence layer for an autonomous agent.

Your task: Given a set of high-level GOALS and a batch of raw EVENTS,
identify which events are relevant to which goals and produce actionable signals.

Return a JSON object with exactly these fields:
{
  "summary": "Brief narrative of what's happening (2-3 sentences)",
  "signals": [
    {
      "event_id": "the event ID",
      "goal_id": "the related goal ID",
      "relevance_score": 0.0 to 1.0,
      "interpretation": "what this event means for the goal",
      "suggested_action": "what the agent should do about it",
      "urgency": "immediate|today|this_week|backlog"
    }
  ]
}

Rules:
- Only include events with relevance_score >= THRESHOLD
- Be selective: most events are noise. Only flag truly relevant ones.
- urgency should reflect how time-sensitive the response should be
- suggested_action should be concrete and actionable
- Return an empty signals array if nothing is relevant
"""


def build_orientation_prompt(
    goals: List[Goal],
    events: List[Event],
    threshold: float = 0.3,
) -> str:
    """Build the user prompt with goals and events for the orientation LLM."""
    goals_text = ""
    for g in goals:
        constraints_str = ", ".join(g.constraints) if g.constraints else "none"
        goals_text += (
            f"- **{g.id}** (priority={g.priority}): {g.name}\n"
            f"  Description: {g.description.strip()}\n"
            f"  Constraints: {constraints_str}\n"
            f"  Event sources: {', '.join(g.event_sources)}\n\n"
        )

    events_text = ""
    for e in events:
        events_text += (
            f"- [{e.severity.value}] {e.id} ({e.source}, {e.timestamp.isoformat()})\n"
            f"  {e.title}\n"
            f"  {e.body[:500]}\n\n"
        )

    return (
        f"RELEVANCE_THRESHOLD: {threshold}\n\n"
        f"## GOALS\n\n{goals_text}\n"
        f"## EVENTS ({len(events)} total)\n\n{events_text}\n"
        f"Analyze and return the JSON response."
    )


def _parse_orientation_response(text: str) -> Dict[str, Any]:
    """Parse the LLM orientation response, handling markdown fences."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    logger.warning("Failed to parse orientation response")
    return {"summary": "", "signals": []}


async def orient(
    goals: List[Goal],
    events: List[Event],
    threshold: float = 0.3,
) -> tuple[str, List[Signal]]:
    """Run orientation: filter events through LLM against goals.

    Returns (summary, signals).
    """
    if not events:
        return "No new events to process.", []

    if not goals:
        return "No active goals defined.", []

    from um_agent_coder.daemon.app import get_gemini_client, get_settings

    settings = get_settings()
    model = settings.world_agent_orientation_model or settings.gemini_model_flash

    user_prompt = build_orientation_prompt(goals, events, threshold)

    try:
        client = get_gemini_client()
        if not client:
            logger.error("Gemini client not available for orientation")
            return "Orientation failed: Gemini client unavailable.", []

        response = await client.generate(
            prompt=user_prompt,
            system_prompt=ORIENTATION_SYSTEM_PROMPT,
            model=model,
            temperature=0.3,
            max_tokens=4096,
        )

        result = _parse_orientation_response(response)
        summary = result.get("summary", "")
        raw_signals = result.get("signals", [])

        signals: List[Signal] = []
        for s in raw_signals:
            try:
                urgency_str = s.get("urgency", "backlog")
                try:
                    urgency = SignalUrgency(urgency_str)
                except ValueError:
                    urgency = SignalUrgency.backlog

                signal = Signal(
                    event_id=s.get("event_id", ""),
                    goal_id=s.get("goal_id", ""),
                    relevance_score=float(s.get("relevance_score", 0.0)),
                    interpretation=s.get("interpretation", ""),
                    suggested_action=s.get("suggested_action", ""),
                    urgency=urgency,
                )
                if signal.relevance_score >= threshold:
                    signals.append(signal)
            except Exception as e:
                logger.warning("Failed to parse signal: %s", e)

        logger.info(
            "Orientation complete: %d events → %d signals (threshold=%.2f)",
            len(events),
            len(signals),
            threshold,
        )
        return summary, signals

    except Exception as e:
        logger.error("Orientation failed: %s", e)
        return f"Orientation failed: {e}", []
