"""Learning system for the World Agent's decision layer.

Analyzes past cycle history and journal entries to extract lessons
about task planning quality. Stores lessons in the KB module and
injects them into the decide prompt as context.

Flow:
1. reflect() — periodic (weekly/on-demand): reads journals + cycle history,
   synthesizes lessons via LLM, writes to KB with tag "decide-lesson"
2. get_decision_context() — called every cycle by _decide.py:
   retrieves relevant lessons from KB to inject into the decide prompt
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from um_agent_coder.daemon.routes.world_agent import _firestore as store

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM_PROMPT = """\
You are the World Agent's learning system. You analyze past OODA cycle \
history and daily journals to identify patterns that improve task planning.

Focus on:
1. Which planned tasks were effective vs wasteful (too vague, wrong CLI, wrong priority)
2. Recurring signals that always/never need action
3. Goal-specific patterns (what works for each goal)
4. Effort estimation accuracy (small tasks that took hours, large tasks that were quick)
5. Which repos/projects need what kind of attention

Output a JSON object:
{
  "lessons": [
    {
      "title": "short lesson title",
      "content": "detailed lesson — what to do differently",
      "tags": ["decide-lesson", "goal-id-if-applicable", "other-relevant-tags"],
      "priority": "high|medium|low",
      "applies_to": "goal ID or 'all'"
    }
  ],
  "obsolete_lessons": ["titles of previous lessons that are no longer accurate"]
}

Be specific and actionable. Bad: "plan better tasks". Good: "For goal hedge-fund-build, \
prefer codex CLI for implementation tasks and gemini for research — codex tasks complete \
2x faster on code changes."
"""


async def _gather_reflection_data(days: int = 7) -> Dict[str, Any]:
    """Gather journal entries and cycle history for the reflection window."""
    today = datetime.now(timezone.utc)
    journals = []
    all_cycles = []

    for i in range(days):
        date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")

        # Journal entry for this day
        journal = await store.get_journal_entry(date_str)
        if journal:
            journals.append(journal)

        # Cycle records for this day
        cycles = await store.list_cycle_records(date_str=date_str, limit=200)
        all_cycles.extend(cycles)

    # Aggregate task data across all cycles
    all_tasks = []
    all_errors = []
    signals_by_goal: Dict[str, int] = {}
    tasks_by_goal: Dict[str, int] = {}
    tasks_by_cli: Dict[str, int] = {}

    for c in all_cycles:
        for t in c.get("planned_tasks", []):
            all_tasks.append(t)
            goal = t.get("goal_id", "unknown")
            cli = t.get("cli", "unknown")
            tasks_by_goal[goal] = tasks_by_goal.get(goal, 0) + 1
            tasks_by_cli[cli] = tasks_by_cli.get(cli, 0) + 1

        for s in c.get("signals", []):
            goal = s.get("goal_id", "unknown")
            signals_by_goal[goal] = signals_by_goal.get(goal, 0) + 1

        if c.get("error"):
            all_errors.append(c["error"][:200])

    return {
        "days": days,
        "journals": journals,
        "total_cycles": len(all_cycles),
        "total_tasks_planned": len(all_tasks),
        "tasks_sample": all_tasks[:30],
        "errors": all_errors[:20],
        "signals_by_goal": signals_by_goal,
        "tasks_by_goal": tasks_by_goal,
        "tasks_by_cli": tasks_by_cli,
    }


def _build_reflection_prompt(data: Dict[str, Any], existing_lessons: List[Dict[str, Any]]) -> str:
    """Build the LLM prompt for reflection."""
    # Journal summaries
    journal_text = ""
    for j in data.get("journals", []):
        date = j.get("date", "?")
        summary = j.get("summary", "(no summary)")[:300]
        errors = j.get("errors", [])
        highlights = j.get("highlights", [])
        decisions = j.get("key_decisions", [])
        journal_text += f"\n### {date}\n{summary}\n"
        if highlights:
            journal_text += f"Highlights: {', '.join(highlights[:5])}\n"
        if decisions:
            journal_text += f"Decisions: {', '.join(decisions[:5])}\n"
        if errors:
            journal_text += f"Errors: {', '.join(errors[:5])}\n"

    # Task samples
    task_text = ""
    for t in data.get("tasks_sample", []):
        task_text += (
            f"- [{t.get('cli', '?')}] {t.get('title', '?')} "
            f"(goal={t.get('goal_id', '?')}, effort={t.get('estimated_effort', '?')}, "
            f"priority={t.get('priority', '?')})\n"
        )

    # Distribution stats
    goal_dist = "\n".join(
        f"  {g}: {c} signals, {data['tasks_by_goal'].get(g, 0)} tasks"
        for g, c in sorted(data.get("signals_by_goal", {}).items(), key=lambda x: -x[1])
    )
    cli_dist = "\n".join(
        f"  {c}: {n} tasks" for c, n in data.get("tasks_by_cli", {}).items()
    )

    # Existing lessons
    existing_text = ""
    if existing_lessons:
        for l in existing_lessons:
            existing_text += f"- [{l.get('priority', '?')}] {l.get('title', '?')}: {l.get('content', '')[:150]}\n"
    else:
        existing_text = "(none yet)"

    return f"""## Reflection Period: last {data['days']} days
Total cycles: {data['total_cycles']}
Total tasks planned: {data['total_tasks_planned']}

## Daily Journals
{journal_text or '(no journals)'}

## Task Samples ({len(data.get('tasks_sample', []))})
{task_text or '(none)'}

## Signal Distribution by Goal
{goal_dist or '(none)'}

## Task Distribution by CLI
{cli_dist or '(none)'}

## Errors ({len(data.get('errors', []))})
{chr(10).join(f'- {e}' for e in data.get('errors', [])[:10]) or '(none)'}

## Existing Lessons (update or mark obsolete if needed)
{existing_text}

Analyze these patterns and produce lessons to improve future task planning."""


async def reflect(days: int = 7) -> Dict[str, Any]:
    """Run a reflection cycle: analyze past data and store lessons in KB.

    Returns summary of lessons created/updated/archived.
    """
    from um_agent_coder.daemon.routes.kb import _store as kb_store

    # Gather historical data
    data = await _gather_reflection_data(days=days)

    if data["total_cycles"] == 0:
        return {"status": "skipped", "reason": "no cycles in reflection window"}

    # Get existing lessons from KB
    existing_lessons, _ = await kb_store.search_items("decide-lesson", limit=20)

    # Synthesize via LLM
    try:
        from um_agent_coder.daemon.app import get_gemini_client, get_settings

        client = get_gemini_client()
        if not client:
            return {"status": "error", "reason": "Gemini client unavailable"}

        settings = get_settings()
        model = settings.gemini_model_pro

        prompt = _build_reflection_prompt(data, existing_lessons)

        response = await client.generate(
            prompt=prompt,
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            model=model,
            temperature=0.3,
            max_tokens=4096,
        )

        # Parse response
        text = response if isinstance(response, str) else str(response)
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]

        result = json.loads(text.strip())

    except Exception as e:
        logger.error("Reflection LLM call failed: %s", e)
        return {"status": "error", "reason": str(e)}

    # Store new lessons in KB
    lessons_created = 0
    for lesson in result.get("lessons", []):
        tags = lesson.get("tags", ["decide-lesson"])
        if "decide-lesson" not in tags:
            tags.insert(0, "decide-lesson")

        item = await kb_store.create_item({
            "type": "lesson",
            "title": lesson.get("title", "Untitled lesson"),
            "content": lesson.get("content", ""),
            "tags": tags,
            "priority": lesson.get("priority", "medium"),
            "source": "world-agent-learner",
            "source_ref": f"reflection-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        })
        if item:
            lessons_created += 1

    # Archive obsolete lessons
    lessons_archived = 0
    obsolete_titles = set(result.get("obsolete_lessons", []))
    if obsolete_titles:
        for existing in existing_lessons:
            if existing.get("title") in obsolete_titles:
                archived = await kb_store.archive_item(existing["id"])
                if archived:
                    lessons_archived += 1

    return {
        "status": "completed",
        "days_analyzed": days,
        "cycles_analyzed": data["total_cycles"],
        "lessons_created": lessons_created,
        "lessons_archived": lessons_archived,
    }


async def get_decision_context(
    goal_ids: Optional[List[str]] = None,
    limit: int = 10,
) -> str:
    """Retrieve relevant lessons from KB to inject into the decide prompt.

    Called by _decide.py before each decision cycle.
    """
    from um_agent_coder.daemon.routes.kb import _store as kb_store

    # Search for decide-lessons
    lessons, _ = await kb_store.search_items("decide-lesson", limit=limit)

    if not lessons:
        return ""

    # Filter by goal if specified, but always include "all" lessons
    if goal_ids:
        goal_set = set(goal_ids)
        relevant = []
        for l in lessons:
            applies_to = ""
            # Check tags for goal IDs
            for tag in l.get("tags", []):
                if tag in goal_set or tag == "all":
                    relevant.append(l)
                    break
            # Also check content for goal references
            content = l.get("content", "").lower()
            title = l.get("title", "").lower()
            if any(g.lower() in content or g.lower() in title for g in goal_ids):
                if l not in relevant:
                    relevant.append(l)
        # If we have goal-specific lessons, prefer them; otherwise use all
        if relevant:
            lessons = relevant

    # Format as text for prompt injection
    lines = ["## LEARNED LESSONS (from past experience)"]
    for l in lessons[:limit]:
        priority = l.get("priority", "medium")
        lines.append(f"- [{priority}] **{l.get('title', '?')}**: {l.get('content', '')}")

    lines.append(
        "\nApply these lessons when planning tasks. "
        "If a lesson contradicts the current signals, note it in the task context."
    )

    return "\n".join(lines)
