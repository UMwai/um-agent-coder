"""Strategic retry engine for the Ralph autonomous loop.

Maps goal validation failures to targeted retry prompts, adapting
the Gemini Intelligence Layer's strategy principles for CLI-based
task execution.

Key principles ported from Gemini layer:
1. Dimension-specific fixes (functional vs kpi vs constraint)
2. Severity-prioritized feedback (breaking first)
3. Previous output context for continuity
4. Oscillation detection recommendations
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from .goal_validator import CriterionResult, GoalCriterion, GoalValidationResult

logger = logging.getLogger(__name__)

# Severity to priority mapping (lower = more important)
_SEVERITY_PRIORITY: Dict[str, int] = {
    "breaking": 0,
    "important": 1,
    "nice_to_have": 2,
}

# Max chars of previous output to include in retry prompt
_MAX_PREVIOUS_OUTPUT = 3000


@dataclass
class RalphStrategy:
    """A targeted fix strategy for a failing goal dimension."""

    name: str
    dimension: str  # functional, kpi, constraint
    prompt_section: str
    priority: int  # lower = more important (breaking=0, important=1, nice_to_have=2)


# ---------------------------------------------------------------------------
# Dimension-specific strategy builders
# ---------------------------------------------------------------------------


def _functional_strategy(failures: List[CriterionResult]) -> RalphStrategy:
    """Build strategy for failing functional criteria."""
    lines = ["The following core features are missing or broken:\n"]
    for f in failures:
        severity_tag = f.severity.upper()
        detail = f": {f.detail}" if f.detail else ""
        lines.append(f"- [{severity_tag}] {f.description}{detail}")

    lines.append(
        "\nThese core features are missing or broken. "
        "Implement them fully before addressing other issues."
    )

    worst_priority = min(_SEVERITY_PRIORITY.get(f.severity, 2) for f in failures)

    return RalphStrategy(
        name="functional_fix",
        dimension="functional",
        prompt_section="\n".join(lines),
        priority=worst_priority,
    )


def _kpi_strategy(failures: List[CriterionResult]) -> RalphStrategy:
    """Build strategy for failing KPI criteria."""
    lines = ["The following measurable targets are not met:\n"]
    for f in failures:
        severity_tag = f.severity.upper()
        detail = f": {f.detail}" if f.detail else ""
        lines.append(f"- [{severity_tag}] {f.description}{detail}")

    lines.append(
        "\nThese measurable targets are not met. "
        "Add benchmarks, tests, or metrics to verify each target is hit."
    )

    worst_priority = min(_SEVERITY_PRIORITY.get(f.severity, 2) for f in failures)

    return RalphStrategy(
        name="kpi_fix",
        dimension="kpi",
        prompt_section="\n".join(lines),
        priority=worst_priority,
    )


def _constraint_strategy(failures: List[CriterionResult]) -> RalphStrategy:
    """Build strategy for failing constraint criteria."""
    lines = ["The following hard constraints are violated:\n"]
    for f in failures:
        severity_tag = f.severity.upper()
        detail = f": {f.detail}" if f.detail else ""
        lines.append(f"- [{severity_tag}] {f.description}{detail}")

    lines.append(
        "\nThese hard constraints are violated. "
        "Refactor to comply with each constraint before proceeding."
    )

    worst_priority = min(_SEVERITY_PRIORITY.get(f.severity, 2) for f in failures)

    return RalphStrategy(
        name="constraint_fix",
        dimension="constraint",
        prompt_section="\n".join(lines),
        priority=worst_priority,
    )


# Mapping from dimension to strategy builder
_DIMENSION_BUILDERS = {
    "functional": _functional_strategy,
    "kpi": _kpi_strategy,
    "constraint": _constraint_strategy,
}


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


def _build_dimension_map(
    criteria: Optional[List[GoalCriterion]],
) -> Dict[str, str]:
    """Build criterion_id -> dimension lookup from GoalCriterion list."""
    if not criteria:
        return {}
    return {c.id: c.dimension for c in criteria}


def select_strategies(
    result: GoalValidationResult,
    max_strategies: int = 3,
    criteria: Optional[List[GoalCriterion]] = None,
) -> List[RalphStrategy]:
    """Select targeted fix strategies based on failing goal criteria.

    Groups failing criteria by dimension (functional, kpi, constraint),
    builds a strategy per dimension, and returns the most critical ones
    sorted by priority.

    When ``criteria`` is provided, criterion IDs are mapped to their
    original dimension. When not provided, all failures are grouped
    as ``functional`` (the most conservative default).

    Args:
        result: Goal validation result with per-criterion pass/fail.
        max_strategies: Maximum number of strategies to return.
        criteria: Optional list of GoalCriterion for dimension lookup.

    Returns:
        List of RalphStrategy sorted by priority (most critical first).
    """
    failures = result.failing_criteria
    if not failures:
        return []

    # Map criterion_id -> dimension for grouping
    dim_map = _build_dimension_map(criteria)

    # Group failures by dimension
    by_dimension: Dict[str, List[CriterionResult]] = defaultdict(list)
    for f in failures:
        dimension = dim_map.get(f.criterion_id, "functional")
        by_dimension[dimension].append(f)

    # Build one strategy per dimension that has failures
    strategies: List[RalphStrategy] = []
    for dimension, dim_failures in by_dimension.items():
        builder = _DIMENSION_BUILDERS.get(dimension, _functional_strategy)
        strategy = builder(dim_failures)
        strategies.append(strategy)

    # Sort by priority (most critical first) and cap
    strategies.sort(key=lambda s: s.priority)
    if len(strategies) > max_strategies:
        strategies = strategies[:max_strategies]
        logger.debug(
            "Capped strategies to %d (dropped lower-priority dimensions)",
            max_strategies,
        )

    logger.info(
        "Selected %d strategies for score=%.2f (%d failing criteria)",
        len(strategies),
        result.score,
        len(failures),
    )

    return strategies


# ---------------------------------------------------------------------------
# Trend-aware guidance
# ---------------------------------------------------------------------------

_TREND_GUIDANCE: Dict[str, str] = {
    "improving": ("You are making progress. Focus on the remaining failures listed above."),
    "stuck": (
        "You have been producing the same score for multiple iterations. "
        "Try a fundamentally different approach -- consider restructuring, "
        "using different libraries, or breaking the problem down differently."
    ),
    "regressing": (
        "Your score has dropped compared to your best attempt. "
        "Revert to your best approach and make only targeted, minimal fixes. "
        "Do not rewrite large sections that were previously working."
    ),
    "unknown": "",
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_strategic_prompt(
    task_description: str,
    success_criteria: str,
    context: str,
    result: GoalValidationResult,
    strategies: List[RalphStrategy],
    previous_output: Optional[str] = None,
    completion_promise: str = "COMPLETE",
    trend_info: Optional[Dict] = None,
) -> str:
    """Build a complete retry prompt incorporating goal validation feedback.

    Constructs a prompt that gives the executing CLI targeted guidance on
    what failed, grouped by severity, with trend-aware coaching and
    previous output for continuity.

    Args:
        task_description: Original task description.
        success_criteria: Human-readable success criteria string.
        context: Additional context (repo state, file listing, etc.).
        result: GoalValidationResult from the most recent iteration.
        strategies: Selected RalphStrategy objects from select_strategies().
        previous_output: Raw output from the previous CLI iteration.
        completion_promise: The promise tag the CLI should emit on success.
        trend_info: Dict with keys "trend", "best_score", "current_score".

    Returns:
        Complete prompt string ready to send to the executing CLI.
    """
    parts: List[str] = []

    # -- Header with score summary --
    parts.append("## Retry: Goal Validation Did Not Pass")
    parts.append("")
    passing_count = len(result.criteria_results) - len(result.failing_criteria)
    total_count = len(result.criteria_results)
    parts.append(f"Score: {result.score:.2f} | " f"Passing: {passing_count}/{total_count} criteria")
    if trend_info and trend_info.get("best_score") is not None:
        parts.append(
            f"Best score so far: {trend_info['best_score']:.2f} | "
            f"Current: {trend_info['current_score']:.2f}"
        )
    parts.append("")

    # -- Failing criteria grouped by severity --
    breaking = [r for r in result.failing_criteria if r.severity == "breaking"]
    important = [r for r in result.failing_criteria if r.severity == "important"]
    nice = [r for r in result.failing_criteria if r.severity == "nice_to_have"]

    if breaking:
        parts.append("### MUST FIX (blocking)")
        for r in breaking:
            detail = f" -- {r.detail}" if r.detail else ""
            parts.append(f"- **{r.description}**{detail}")
        parts.append("")

    if important:
        parts.append("### SHOULD FIX (important)")
        for r in important:
            detail = f" -- {r.detail}" if r.detail else ""
            parts.append(f"- {r.description}{detail}")
        parts.append("")

    if nice:
        parts.append("### NICE TO HAVE")
        for r in nice:
            detail = f" -- {r.detail}" if r.detail else ""
            parts.append(f"- {r.description}{detail}")
        parts.append("")

    # -- Strategy-specific fix instructions --
    if strategies:
        parts.append("### Fix Instructions")
        parts.append("")
        for strategy in strategies:
            parts.append(strategy.prompt_section)
            parts.append("")

    # -- Trend guidance --
    if trend_info:
        trend = trend_info.get("trend", "unknown")
        guidance = _TREND_GUIDANCE.get(trend, "")
        if guidance:
            parts.append("### Progress Trend")
            parts.append("")
            parts.append(f"Trend: **{trend}**")
            parts.append("")
            parts.append(guidance)
            parts.append("")

    # -- Issues list (capped to avoid prompt bloat) --
    if result.issues:
        max_issues = 10
        capped = result.issues[:max_issues]
        parts.append("### Specific Issues")
        parts.append("")
        for issue in capped:
            parts.append(f"- {issue}")
        if len(result.issues) > max_issues:
            parts.append(f"- ... and {len(result.issues) - max_issues} more issues")
        parts.append("")

    # -- Previous output excerpt for continuity --
    if previous_output:
        excerpt = previous_output[-_MAX_PREVIOUS_OUTPUT:]
        if len(previous_output) > _MAX_PREVIOUS_OUTPUT:
            truncated = len(previous_output) - _MAX_PREVIOUS_OUTPUT
            excerpt = f"[... truncated {truncated} chars ...]\n" + excerpt
        parts.append("### Previous Output (for continuity)")
        parts.append("")
        parts.append("```")
        parts.append(excerpt)
        parts.append("```")
        parts.append("")

    # -- Original task and success criteria --
    parts.append("---")
    parts.append("")
    parts.append("## Original Task")
    parts.append("")
    parts.append(task_description)
    parts.append("")

    if success_criteria:
        parts.append("## Success Criteria")
        parts.append("")
        parts.append(success_criteria)
        parts.append("")

    if context:
        parts.append("## Context")
        parts.append("")
        parts.append(context)
        parts.append("")

    # -- Completion promise reminder --
    parts.append("## Completion")
    parts.append("")
    parts.append(
        f"When ALL criteria are satisfied, output "
        f"<promise>{completion_promise}</promise> to signal completion."
    )
    parts.append("Do NOT output the promise until every MUST FIX item is resolved.")

    return "\n".join(parts)
