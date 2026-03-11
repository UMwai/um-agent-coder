"""Decision layer: converts oriented signals into planned tasks.

Calls Gemini Pro to prioritise signals against goals and produce
a concrete list of PlannedTask objects the agent can execute.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List

from um_agent_coder.daemon.routes.world_agent.models import (
    Goal,
    PlannedTask,
    Signal,
)

logger = logging.getLogger(__name__)

DECIDE_SYSTEM_PROMPT = """\
You are the decision-making layer of an autonomous agent.

Given GOALS and SIGNALS (events that passed the relevance filter),
produce a prioritised list of concrete tasks the agent should execute.

Return a JSON object:
{
  "tasks": [
    {
      "goal_id": "parent goal ID",
      "project": "owner/repo",
      "title": "short imperative title",
      "description": "detailed description of what to do",
      "priority": 1-10 (1=highest),
      "estimated_effort": "small|medium|large",
      "cli": "codex|gemini|claude",
      "timeout": "30min|1h|2h",
      "success_criteria": "how to verify completion",
      "context": {"key": "value pairs with relevant data"}
    }
  ]
}

Rules:
- Only create tasks that are actionable RIGHT NOW
- Each task should map to a single GitHub operation (branch+PR, comment, code review)
- Include enough context for the executor to act without further research
- If no action is needed, return {"tasks": []}
- Maximum 5 tasks per decision cycle
"""


def _build_decide_prompt(
    goals: List[Goal],
    signals: List[Signal],
    repos: List[str],
) -> str:
    """Build the user prompt for the decision LLM."""
    goals_text = ""
    for g in goals:
        goals_text += f"- **{g.id}** (p={g.priority}): {g.name} — {g.description}\n"

    signals_text = ""
    for s in signals:
        signals_text += (
            f"- event={s.event_id} goal={s.goal_id} "
            f"relevance={s.relevance_score:.2f} urgency={s.urgency.value}\n"
            f"  {s.interpretation}\n"
            f"  Suggested: {s.suggested_action}\n\n"
        )

    repos_text = ", ".join(repos)

    return (
        f"## GOALS\n{goals_text}\n\n"
        f"## SIGNALS ({len(signals)})\n{signals_text}\n"
        f"## AVAILABLE REPOS\n{repos_text}\n\n"
        f"Produce the task list JSON."
    )


def _parse_decide_response(text: str) -> List[Dict[str, Any]]:
    """Parse the LLM decision response."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("Failed to parse decide response")
                return []
        else:
            return []
    return data.get("tasks", [])


async def decide(
    goals: List[Goal],
    signals: List[Signal],
    repos: List[str],
) -> List[PlannedTask]:
    """Run decision: convert signals into planned tasks via LLM.

    Returns a list of PlannedTask objects.
    """
    if not signals:
        return []

    from um_agent_coder.daemon.app import get_gemini_client, get_settings

    settings = get_settings()
    model = settings.gemini_model_pro

    user_prompt = _build_decide_prompt(goals, signals, repos)

    try:
        client = get_gemini_client()
        if not client:
            logger.error("Gemini client not available for decision layer")
            return []

        response = await client.generate(
            prompt=user_prompt,
            system_prompt=DECIDE_SYSTEM_PROMPT,
            model=model,
            temperature=0.3,
            max_tokens=4096,
        )

        raw_tasks = _parse_decide_response(response)
        tasks: List[PlannedTask] = []
        for t in raw_tasks[:5]:  # cap at 5
            try:
                task = PlannedTask(
                    id=f"task-{uuid.uuid4().hex[:8]}",
                    goal_id=t.get("goal_id", ""),
                    project=t.get("project", ""),
                    title=t.get("title", "Untitled task"),
                    description=t.get("description", ""),
                    priority=int(t.get("priority", 5)),
                    estimated_effort=t.get("estimated_effort", "medium"),
                    cli=t.get("cli", "codex"),
                    timeout=t.get("timeout", "1h"),
                    success_criteria=t.get("success_criteria", ""),
                    context=t.get("context", {}),
                )
                tasks.append(task)
            except Exception as e:
                logger.warning("Failed to parse planned task: %s", e)

        logger.info("Decision layer produced %d tasks from %d signals", len(tasks), len(signals))
        return tasks

    except Exception as e:
        logger.error("Decision layer failed: %s", e)
        return []
