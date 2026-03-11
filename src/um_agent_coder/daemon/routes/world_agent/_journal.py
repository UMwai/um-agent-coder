"""Daily journal synthesis for the World Agent.

Collects the day's cycle history, events, and world state,
then synthesizes a narrative journal entry via LLM.

Primary data source: world_agent_cycles/{date}/runs/* (append-only cycle records)
Secondary: world_agent_events/{date}, world_agent_state/current
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from um_agent_coder.daemon.routes.world_agent import _firestore as store
from um_agent_coder.daemon.routes.world_agent import _goals as goal_store
from um_agent_coder.daemon.routes.world_agent.models import JournalEntry

logger = logging.getLogger(__name__)

JOURNAL_SYSTEM_PROMPT = """\
You are the World Agent's journal writer. Given a day's OODA cycle history, \
events, signals, decisions, and goals, write a concise daily journal entry \
summarizing the work performed.

Write in first person ("I observed...", "I created...", "I decided...").
Be factual and specific — mention repos, PRs, issues by name/number.
Keep the summary under 300 words.

Also extract:
- highlights: key accomplishments (list of short strings)
- key_decisions: notable decisions made (list of short strings)
- errors: any errors encountered (list of short strings)
- goals_progressed: goal IDs that saw meaningful progress (list of strings)

Return valid JSON with keys: summary, highlights, key_decisions, errors, goals_progressed
"""


async def _gather_day_data(date_str: str) -> Dict[str, Any]:
    """Gather all data for a given date to feed into journal synthesis."""
    # Primary source: cycle history (append-only, has everything)
    cycles = await store.list_cycle_records(date_str=date_str, limit=500)
    cycle_stats = await store.get_cycle_stats(date_str=date_str)

    # Secondary: raw events and current world state
    events = await store.list_events(limit=200)
    goals = await goal_store.get_all_goals(status="active")

    return {
        "date": date_str,
        "cycles": cycles,
        "cycle_stats": cycle_stats,
        "events": events,
        "goals": [{"id": g.id, "name": g.name, "status": g.status.value} for g in goals],
    }


def _build_cycles_section(cycles: List[Dict[str, Any]]) -> str:
    """Build a text summary of cycle history for the LLM prompt."""
    if not cycles:
        return "(no cycles recorded)"

    lines = []
    for c in cycles:
        status = "FAILED" if c.get("error") else "OK"
        line = (
            f"- [{status}] {c.get('cycle_id', '?')} "
            f"({c.get('source', '?')}): "
            f"{c.get('events_collected', 0)} events, "
            f"{c.get('signals_generated', 0)} signals, "
            f"{c.get('tasks_created', 0)} tasks, "
            f"{c.get('duration_ms', 0)}ms"
        )
        if c.get("error"):
            line += f" — error: {c['error'][:100]}"
        if c.get("summary"):
            line += f"\n  Summary: {c['summary'][:200]}"
        lines.append(line)

    return "\n".join(lines)


def _build_signals_section(cycles: List[Dict[str, Any]]) -> str:
    """Collect all signals across cycles for the prompt."""
    all_signals = []
    for c in cycles:
        for s in c.get("signals", []):
            interp = s.get("interpretation", "")
            action = s.get("suggested_action", "")
            goal = s.get("goal_id", "")
            urgency = s.get("urgency", "")
            if interp or action:
                all_signals.append(f"- [{urgency}] goal={goal}: {interp}. Action: {action}")

    if not all_signals:
        return "(no signals)"
    # Deduplicate similar signals, keep up to 30
    seen = set()
    unique = []
    for s in all_signals:
        key = s[:80]
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return "\n".join(unique[:30])


def _build_tasks_section(cycles: List[Dict[str, Any]]) -> str:
    """Collect all planned tasks across cycles."""
    all_tasks = []
    for c in cycles:
        for t in c.get("planned_tasks", []):
            title = t.get("title", "?")
            goal = t.get("goal_id", "")
            cli = t.get("cli", "")
            all_tasks.append(f"- {title} (goal={goal}, cli={cli})")

    if not all_tasks:
        return "(no tasks planned)"
    return "\n".join(all_tasks[:30])


async def _synthesize_via_llm(day_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call LLM to synthesize a journal narrative from the day's data."""
    import json

    try:
        from um_agent_coder.daemon.app import get_gemini_client, get_settings

        client = get_gemini_client()
        if not client:
            logger.warning("Gemini client unavailable, using fallback")
            return _fallback_synthesis(day_data)

        settings = get_settings()
        model = settings.gemini_model_flash

        cycles = day_data.get("cycles", [])
        stats = day_data.get("cycle_stats", {})

        goals_text = "\n".join(
            f"- {g['id']}: {g['name']} ({g['status']})" for g in day_data.get("goals", [])
        )

        # Collect errors from failed cycles
        errors = [c["error"] for c in cycles if c.get("error")]
        errors_text = "\n".join(f"- {e[:200]}" for e in errors) if errors else "(none)"

        prompt = f"""Date: {day_data['date']}

## Day Statistics
- Total cycles: {stats.get('total_cycles', 0)} ({stats.get('successful_cycles', 0)} OK, {stats.get('failed_cycles', 0)} failed)
- Events collected: {stats.get('total_events', 0)}
- Signals generated: {stats.get('total_signals', 0)}
- Tasks planned: {stats.get('total_tasks', 0)}
- Total processing time: {stats.get('total_duration_ms', 0)}ms
- Goals touched: {', '.join(stats.get('goal_ids_touched', [])) or 'none'}

## OODA Cycle History ({len(cycles)} cycles)
{_build_cycles_section(cycles)}

## Signals Generated
{_build_signals_section(cycles)}

## Tasks Planned
{_build_tasks_section(cycles)}

## Errors
{errors_text}

## Active Goals
{goals_text or '(none)'}

Write the daily journal entry as JSON."""

        response = await client.generate(
            prompt=prompt,
            system_prompt=JOURNAL_SYSTEM_PROMPT,
            model=model,
            temperature=0.3,
            max_tokens=2048,
        )

        # Parse JSON from response
        text = response if isinstance(response, str) else str(response)
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]

        return json.loads(text.strip())

    except Exception as e:
        logger.warning("LLM synthesis failed, using fallback: %s", e)
        return _fallback_synthesis(day_data)


def _fallback_synthesis(day_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a basic journal entry without LLM when synthesis fails."""
    stats = day_data.get("cycle_stats", {})
    cycles = day_data.get("cycles", [])

    errors = [c["error"][:100] for c in cycles if c.get("error")]
    goal_ids = stats.get("goal_ids_touched", [])

    summary = (
        f"Ran {stats.get('total_cycles', 0)} OODA cycles "
        f"({stats.get('successful_cycles', 0)} successful, "
        f"{stats.get('failed_cycles', 0)} failed). "
        f"Collected {stats.get('total_events', 0)} events, "
        f"generated {stats.get('total_signals', 0)} signals, "
        f"planned {stats.get('total_tasks', 0)} tasks."
    )

    if goal_ids:
        summary += f" Goals touched: {', '.join(goal_ids)}."

    return {
        "summary": summary.strip(),
        "highlights": [],
        "key_decisions": [],
        "errors": errors[:10],
        "goals_progressed": goal_ids,
    }


async def generate_journal(date_str: Optional[str] = None) -> JournalEntry:
    """Generate (or regenerate) a journal entry for the given date.

    Pulls from cycle history as primary data source, then synthesizes
    via LLM. If no date is provided, uses today (UTC).
    """
    if not date_str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Gather data from cycle history + events
    day_data = await _gather_day_data(date_str)
    stats = day_data.get("cycle_stats", {})

    # Synthesize narrative
    synthesis = await _synthesize_via_llm(day_data)

    entry = JournalEntry(
        date=date_str,
        summary=synthesis.get("summary", ""),
        cycles_run=stats.get("total_cycles", 0),
        events_collected=stats.get("total_events", 0),
        signals_generated=stats.get("total_signals", 0),
        tasks_created=stats.get("total_tasks", 0),
        goals_progressed=synthesis.get("goals_progressed", []),
        key_decisions=synthesis.get("key_decisions", []),
        errors=synthesis.get("errors", []),
        highlights=synthesis.get("highlights", []),
    )

    # Persist to Firestore
    await store.save_journal_entry(entry.model_dump(mode="json"))

    return entry
