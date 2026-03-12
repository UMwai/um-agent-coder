"""Generate harness-compatible roadmap.md and goal YAML from review results.

Pure template generation — no LLM calls. Produces markdown that
``RoadmapParser`` can consume directly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_EFFORT_TIMEOUT = {"small": "15min", "medium": "30min", "large": "60min"}
_EFFORT_ITERATIONS = {"small": 15, "medium": 30, "large": 50}

_CATEGORY_PHASE: dict[str, tuple[int, str]] = {
    "feature": (1, "Feature Implementation"),
    "test": (2, "Testing & Validation"),
    "infra": (3, "Infrastructure & DevOps"),
}


def _goal_prefix(goal_id: str) -> str:
    """Derive a short prefix from a goal id (e.g. 'hedge-fund-build' -> 'hfb')."""
    parts = re.split(r"[-_]", goal_id)
    prefix = "".join(p[0] for p in parts if p)
    return prefix[:4] or "t"


def _categorise(task: dict[str, Any]) -> str:
    """Assign a task to a phase category based on heuristics."""
    title = (task.get("title") or "").lower()
    desc = (task.get("description") or "").lower()
    combined = f"{title} {desc}"
    if any(kw in combined for kw in ("deploy", "infra", "docker", "k8s", "pipeline", "monitor")):
        return "infra"
    if any(kw in combined for kw in ("test", "coverage", "lint", "ci", "validate")):
        return "test"
    return "feature"


def generate_roadmap(
    review_result: dict[str, Any],
    goal_id: str,
    goal_name: str,
    repo_path: str,
) -> str:
    """Return a harness-compatible roadmap markdown string.

    Parameters
    ----------
    review_result:
        Output of ``_reviewer.py`` containing ``recommended_tasks``,
        ``gaps``, ``kpi_assessment``, and ``review_summary``.
    goal_id:
        Unique goal identifier (e.g. ``"hedge-fund-build"``).
    goal_name:
        Human-readable goal name.
    repo_path:
        Absolute path to the target repository.
    """
    tasks: list[dict[str, Any]] = review_result.get("recommended_tasks", [])
    kpis: list[dict[str, Any]] = review_result.get("kpi_assessment", [])
    summary: str = review_result.get("review_summary", "")

    prefix = _goal_prefix(goal_id)

    # --- group tasks by category ------------------------------------------
    buckets: dict[str, list[dict[str, Any]]] = {}
    for t in tasks:
        cat = _categorise(t)
        buckets.setdefault(cat, []).append(t)

    # --- build success criteria from KPIs ---------------------------------
    criteria_lines: list[str] = []
    for kpi in kpis:
        metric = kpi.get("metric", "metric")
        target = kpi.get("target", "TBD")
        criteria_lines.append(f"- [ ] {metric} reaches {target}")
    if not criteria_lines:
        criteria_lines.append("- [ ] All tasks pass their success criteria")

    # --- assign IDs and render phases -------------------------------------
    phase_blocks: list[str] = []
    task_counter = 0
    prev_phase_last_id: str | None = None

    for cat in ("feature", "test", "infra"):
        cat_tasks = buckets.get(cat)
        if not cat_tasks:
            continue
        phase_num, phase_label = _CATEGORY_PHASE[cat]

        lines: list[str] = [f"### Phase {phase_num}: {phase_label}\n"]
        first_id_in_phase: str | None = None

        for t in sorted(cat_tasks, key=lambda x: x.get("priority", 99)):
            task_counter += 1
            tid = f"{prefix}-{task_counter:03d}"
            tid_upper = tid.replace("-", "_").upper()
            if first_id_in_phase is None:
                first_id_in_phase = tid

            effort = (t.get("estimated_effort") or "medium").lower()
            timeout = _EFFORT_TIMEOUT.get(effort, "30min")
            max_iter = _EFFORT_ITERATIONS.get(effort, 30)
            cli = t.get("cli") or "codex"
            success = t.get("success_criteria") or "Implementation complete and tests pass"
            depends = prev_phase_last_id or "none"

            lines.append(f"- [ ] **{tid}**: {t.get('title', 'Unnamed task')}")
            lines.append(f"  - timeout: {timeout}")
            lines.append(f"  - depends: {depends}")
            lines.append(f"  - success: {success}")
            lines.append(f"  - cwd: {repo_path}")
            lines.append(f"  - cli: {cli}")
            lines.append("  - ralph: true")
            lines.append(f"  - max_iterations: {max_iter}")
            lines.append(f"  - completion_promise: {tid_upper}_COMPLETE")
            lines.append("")

        prev_phase_last_id = f"{prefix}-{task_counter:03d}"
        phase_blocks.append("\n".join(lines))

    # --- assemble full document -------------------------------------------
    objective = summary or f"Complete all tasks for goal: {goal_name}"
    criteria = "\n".join(criteria_lines)
    phases = "\n".join(phase_blocks)
    doc = (
        f"# Roadmap: {goal_name}\n\n"
        f"## Objective\n{objective}\n\n"
        f"## Constraints\n"
        f"- Max time per task: 60 min\n"
        f"- Max retries per task: 3\n"
        f"- Working directory: {repo_path}\n\n"
        f"## Success Criteria\n{criteria}\n\n"
        f"## Tasks\n\n{phases}\n"
        f"## Growth Mode\n"
        f"1. Generate improvement tasks when all phases complete\n"
    )
    return doc


def write_roadmap(content: str, output_path: str) -> str:
    """Write roadmap markdown to *output_path*, creating parents as needed.

    Returns the resolved absolute path.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return str(p.resolve())


def generate_goal_yaml(
    goal_id: str,
    goal_name: str,
    description: str,
    repo_path: str,
    kpis: list[dict[str, Any]] | None = None,
    constraints: list[str] | None = None,
) -> str:
    """Return a YAML string defining a world-agent goal."""
    kpis = kpis or []
    constraints = constraints or []

    lines = [
        "goals:",
        f'  - id: "{goal_id}"',
        f'    name: "{goal_name}"',
        f'    description: "{description}"',
        f'    repo: "{repo_path}"',
        "    priority: 1",
        "    status: active",
    ]
    if kpis:
        lines.append("    kpis:")
        for k in kpis:
            lines.append(f'      - metric: "{k.get("metric", "metric")}"')
            lines.append(f'        target: "{k.get("target", "TBD")}"')
            lines.append(f'        current_estimate: "{k.get("current_estimate", "unknown")}"')
    if constraints:
        lines.append("    constraints:")
        for c in constraints:
            lines.append(f'      - "{c}"')
    return "\n".join(lines) + "\n"
