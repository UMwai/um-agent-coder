"""Fix strategy engine — maps eval dimension failures to targeted prompt modifications.

Given an EvalResult, selects strategies for failing dimensions and builds a
strategic retry prompt that addresses specific weaknesses rather than generic
"do better" instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ._evaluator import (
    AccuracyCheck,
    ActionabilityCheck,
    ClarityCheck,
    CompletenessCheck,
    EvalResult,
    FulfillmentCheck,
)


@dataclass
class FixStrategy:
    """A targeted fix strategy for a failing eval dimension."""

    name: str
    dimension: str
    prompt_addendum: str = ""
    system_addendum: str = ""
    temperature_delta: float = 0.0


# --- Strategy definitions ---


def _accuracy_with_checklist(
    eval_context: str,
    checks: List[AccuracyCheck],
) -> FixStrategy:
    """Build accuracy fix with specific failing checks from the checklist."""
    # Group failures by severity, cap each category to avoid prompt bloat
    breaking = [c for c in checks if c.status == "fail" and c.severity == "breaking"][:5]
    foreign = [c for c in checks if c.status == "fail" and c.severity == "foreign"][:5]
    style = [c for c in checks if c.status == "fail" and c.severity == "style"][:3]

    failures_text = ""
    if breaking:
        failures_text += "\nCRITICAL (must fix):\n"
        for c in breaking:
            failures_text += f"- {c.check}: {c.detail}\n"
    if foreign:
        failures_text += "\nDEPENDENCY (must fix):\n"
        for c in foreign:
            failures_text += f"- {c.check}: {c.detail}\n"
    if style:
        failures_text += "\nSTYLE (should fix):\n"
        for c in style:
            failures_text += f"- {c.check}: {c.detail}\n"

    return FixStrategy(
        name="accuracy_checklist_fix",
        dimension="accuracy",
        prompt_addendum=(
            "\n\n[ACCURACY ISSUES TO FIX]\n"
            f"{failures_text}\n"
            "=== REFERENCE MATERIAL ===\n"
            f"{eval_context}\n"
            "=== END REFERENCE MATERIAL ===\n\n"
            "Verify each function call, import, and pattern against the reference material."
        ),
        system_addendum=(
            "Fix the accuracy issues listed in the prompt. Verify every API call "
            "against the provided reference material."
        ),
        temperature_delta=-0.1,
    )


def _accuracy_with_context(eval_context: str) -> FixStrategy:
    return FixStrategy(
        name="accuracy_with_context",
        dimension="accuracy",
        prompt_addendum=(
            "\n\n[ACCURACY FIX — REFERENCE MATERIAL PROVIDED]\n"
            "The following reference material contains the correct API signatures, "
            "schemas, and interfaces. You MUST verify every function call, import, "
            "and type reference against this material. Do NOT invent APIs that are "
            "not listed here.\n\n"
            "=== REFERENCE MATERIAL ===\n"
            f"{eval_context}\n"
            "=== END REFERENCE MATERIAL ===\n\n"
            "For each API call or import in your response, mentally check it against "
            "the reference material above. If it doesn't match, fix it."
        ),
        temperature_delta=-0.2,
    )


def _accuracy_fix() -> FixStrategy:
    return FixStrategy(
        name="accuracy_fix",
        dimension="accuracy",
        prompt_addendum=(
            "\n\n[ACCURACY FIX]\n"
            "Your previous response contained factual errors or incorrect API usage. "
            "Before writing any code or making claims:\n"
            "1. Verify all function signatures and import paths are correct\n"
            "2. Double-check type annotations and return types\n"
            "3. Ensure all referenced libraries and methods actually exist\n"
            "4. If unsure about an API, use a comment noting the uncertainty"
        ),
        temperature_delta=-0.2,
    )


def _completeness_fix() -> FixStrategy:
    return FixStrategy(
        name="completeness_fix",
        dimension="completeness",
        prompt_addendum=(
            "\n\n[COMPLETENESS FIX]\n"
            "Your previous response was incomplete. You MUST:\n"
            "1. Include ALL files mentioned in the task — list them as a manifest first\n"
            "2. Implement every function, class, and method fully — no stubs\n"
            "3. Do NOT stop mid-file or mid-function\n"
            "4. If the response is long, that's OK — completeness is more important than brevity"
        ),
        system_addendum=(
            "You MUST complete the entire response. Never truncate, never say "
            "'I'll continue in the next message', never leave placeholders. "
            "Output the full implementation no matter how long it is."
        ),
    )


def _clarity_fix() -> FixStrategy:
    return FixStrategy(
        name="clarity_fix",
        dimension="clarity",
        prompt_addendum=(
            "\n\n[CLARITY FIX]\n"
            "Your previous response lacked clear structure. Please:\n"
            "1. Use clear section headers (## or ###) to organize content\n"
            "2. Add brief comments explaining non-obvious logic\n"
            "3. Group related code together with separating headers\n"
            "4. Start with a brief overview of your approach before diving into code"
        ),
    )


def _actionability_fix() -> FixStrategy:
    return FixStrategy(
        name="actionability_fix",
        dimension="actionability",
        prompt_addendum=(
            "\n\n[ACTIONABILITY FIX]\n"
            "Your previous response was not directly usable. You MUST:\n"
            "1. NO TODOs — implement everything fully\n"
            "2. NO stubs — every function must have a real implementation\n"
            "3. NO '...' or 'pass' placeholders — write the actual code\n"
            "4. Include all imports, dependencies, and configuration needed\n"
            "5. The output should be copy-paste ready and immediately runnable"
        ),
    )


def _fulfillment_with_checklist(
    checks: List[FulfillmentCheck],
) -> FixStrategy:
    """Build fulfillment fix with specific unfulfilled requirements."""
    breaking = [c for c in checks if c.status == "fail" and c.severity == "breaking"]
    foreign = [c for c in checks if c.status == "fail" and c.severity == "foreign"]
    style = [c for c in checks if c.status == "fail" and c.severity == "style"]

    failures_text = ""
    if breaking:
        failures_text += (
            "\n\n🚨 MISSING REQUIREMENTS (must add — these were explicitly requested):\n"
        )
        for c in breaking:
            failures_text += f"- {c.check}: {c.detail}\n"
    if foreign:
        failures_text += (
            "\n\n⚠️ PARTIAL REQUIREMENTS (must complete — implementation is incomplete):\n"
        )
        for c in foreign:
            failures_text += f"- {c.check}: {c.detail}\n"
    if style:
        failures_text += "\n\n📋 MINOR GAPS (should address):\n"
        for c in style:
            failures_text += f"- {c.check}: {c.detail}\n"

    # Build explicit implementation instructions for breaking failures
    impl_instructions = ""
    if breaking:
        impl_instructions = (
            "\n\nFor each MISSING REQUIREMENT above, you MUST:\n"
            "1. Write the COMPLETE implementation (not a stub or placeholder)\n"
            "2. Include all necessary functions, classes, and imports\n"
            "3. Show realistic logic — not just an empty function body\n"
            "Do NOT skip any of the missing requirements. Each one listed "
            "as MISSING was explicitly requested in the original task."
        )

    return FixStrategy(
        name="fulfillment_checklist_fix",
        dimension="fulfillment",
        prompt_addendum=(
            "\n\n[UNFULFILLED REQUIREMENTS]\n"
            f"{failures_text}"
            f"{impl_instructions}\n\n"
            "Your response must include ALL previously working code PLUS "
            "the missing features listed above. Do not remove existing "
            "functionality while adding new features."
        ),
        system_addendum=(
            "The response is missing explicitly requested features. "
            "Add every missing feature with full implementation."
            if breaking
            else ""
        ),
    )


def _completeness_with_checklist(
    checks: List[CompletenessCheck],
) -> FixStrategy:
    """Build completeness fix with specific missing items from the checklist."""
    breaking = [c for c in checks if c.status == "fail" and c.severity == "breaking"]
    foreign = [c for c in checks if c.status == "fail" and c.severity == "foreign"]
    style = [c for c in checks if c.status == "fail" and c.severity == "style"]

    failures_text = ""
    if breaking:
        failures_text += "\n\n🚨 MISSING COMPONENTS (must add — entirely absent):\n"
        for c in breaking:
            failures_text += f"- {c.check}: {c.detail}\n"
    if foreign:
        failures_text += "\n\n⚠️ INCOMPLETE COMPONENTS (must complete — partially present):\n"
        for c in foreign:
            failures_text += f"- {c.check}: {c.detail}\n"
    if style:
        failures_text += "\n\n📋 MINOR OMISSIONS (should address):\n"
        for c in style:
            failures_text += f"- {c.check}: {c.detail}\n"

    return FixStrategy(
        name="completeness_checklist_fix",
        dimension="completeness",
        prompt_addendum=(
            "\n\n[COMPLETENESS GAPS]\n"
            f"{failures_text}\n"
            "Ensure ALL requested components are present and complete."
        ),
        system_addendum="",
    )


def _clarity_with_checklist(
    checks: List[ClarityCheck],
) -> FixStrategy:
    """Build clarity fix with specific issues from the checklist."""
    failing = [c for c in checks if c.status == "fail"]

    failures_text = ""
    for c in failing:
        failures_text += f"- [{c.severity.upper()}] {c.check}: {c.detail}\n"

    return FixStrategy(
        name="clarity_checklist_fix",
        dimension="clarity",
        prompt_addendum=(
            "\n\n[CLARITY FIX — CHECKLIST-DRIVEN]\n"
            "An automated audit found the following clarity issues:\n"
            f"{failures_text}\n"
            "Improve the organization and readability of your response."
        ),
    )


def _actionability_with_checklist(
    checks: List[ActionabilityCheck],
) -> FixStrategy:
    """Build actionability fix with specific issues from the checklist."""
    breaking = [c for c in checks if c.status == "fail" and c.severity == "breaking"]
    foreign = [c for c in checks if c.status == "fail" and c.severity == "foreign"]
    style = [c for c in checks if c.status == "fail" and c.severity == "style"]

    failures_text = ""
    if breaking:
        failures_text += "\n\n🚨 STUBS/PLACEHOLDERS (must implement — code won't run):\n"
        for c in breaking:
            failures_text += f"- {c.check}: {c.detail}\n"
    if foreign:
        failures_text += "\n\n⚠️ MISSING DEPENDENCIES (must add — code incomplete):\n"
        for c in foreign:
            failures_text += f"- {c.check}: {c.detail}\n"
    if style:
        failures_text += "\n\n📋 USABILITY GAPS (should address):\n"
        for c in style:
            failures_text += f"- {c.check}: {c.detail}\n"

    return FixStrategy(
        name="actionability_checklist_fix",
        dimension="actionability",
        prompt_addendum=(
            "\n\n[ACTIONABILITY ISSUES]\n"
            f"{failures_text}\n"
            "Replace all stubs, TODOs, and placeholders with real implementations."
        ),
        system_addendum="",
    )


def _fulfillment_fix() -> FixStrategy:
    return FixStrategy(
        name="fulfillment_fix",
        dimension="fulfillment",
        prompt_addendum=(
            "\n\n[FULFILLMENT FIX]\n"
            "Your previous response did not fully address the task requirements. "
            "Re-read the original task carefully and ensure:\n"
            "1. Every requested file/component is present\n"
            "2. Every requested feature is fully implemented\n"
            "3. All constraints are respected (no stubs, no TODOs, etc.)\n"
            "4. Quantitative requirements are met (file counts, endpoint counts)"
        ),
    )


def select_strategies(
    eval_result: EvalResult,
    eval_context: Optional[str] = None,
    threshold: float = 0.7,
    max_strategies: int = 2,
) -> List[FixStrategy]:
    """Select targeted fix strategies based on failing eval dimensions.

    Skips dimensions that failed to parse (score=0.5 default) since we have
    no useful check data to base a fix strategy on — adding generic fix text
    just creates noise in the retry prompt.

    After collecting all failing strategies, caps at ``max_strategies`` by
    keeping only the worst-scoring dimensions.  This prevents prompt overload
    that causes the model to spread thin and regress.

    Args:
        eval_result: The evaluation result with per-dimension scores.
        eval_context: Reference material for accuracy checking.
        threshold: Scores below this trigger a fix strategy.
        max_strategies: Maximum strategies to return (worst dims first).

    Returns:
        List of FixStrategy objects to apply.
    """
    strategies: List[FixStrategy] = []
    skip = set(eval_result.parse_failed_dimensions)

    # Also skip cascade-skipped dims: score=0.0 with no checks means the
    # dimension was never evaluated (accuracy gate blocked full eval).
    # Adding generic strategies for these just creates prompt noise.
    def _cascade_skipped(score: float, checks: list) -> bool:
        return score == 0.0 and len(checks) == 0

    if eval_result.accuracy < threshold and "accuracy" not in skip:
        if eval_result.accuracy_checks and eval_context:
            strategies.append(_accuracy_with_checklist(eval_context, eval_result.accuracy_checks))
        elif eval_context:
            strategies.append(_accuracy_with_context(eval_context))
        else:
            strategies.append(_accuracy_fix())

    if (
        eval_result.completeness < threshold
        and "completeness" not in skip
        and not _cascade_skipped(eval_result.completeness, eval_result.completeness_checks)
    ):
        if eval_result.completeness_checks:
            strategies.append(_completeness_with_checklist(eval_result.completeness_checks))
        else:
            strategies.append(_completeness_fix())

    if (
        eval_result.clarity < threshold
        and "clarity" not in skip
        and not _cascade_skipped(eval_result.clarity, eval_result.clarity_checks)
    ):
        if eval_result.clarity_checks:
            strategies.append(_clarity_with_checklist(eval_result.clarity_checks))
        else:
            strategies.append(_clarity_fix())

    if (
        eval_result.actionability < threshold
        and "actionability" not in skip
        and not _cascade_skipped(eval_result.actionability, eval_result.actionability_checks)
    ):
        if eval_result.actionability_checks:
            strategies.append(_actionability_with_checklist(eval_result.actionability_checks))
        else:
            strategies.append(_actionability_fix())

    if (
        eval_result.fulfillment < threshold
        and "fulfillment" not in skip
        and not _cascade_skipped(eval_result.fulfillment, eval_result.fulfillment_checks)
    ):
        if eval_result.fulfillment_checks:
            strategies.append(_fulfillment_with_checklist(eval_result.fulfillment_checks))
        else:
            strategies.append(_fulfillment_fix())

    # Cap to worst N dimensions to keep retry prompts focused
    if len(strategies) > max_strategies:
        dim_scores = {
            "accuracy": eval_result.accuracy,
            "completeness": eval_result.completeness,
            "clarity": eval_result.clarity,
            "actionability": eval_result.actionability,
            "fulfillment": eval_result.fulfillment,
        }
        strategies.sort(key=lambda s: dim_scores.get(s.dimension, 1.0))
        strategies = strategies[:max_strategies]

    return strategies


def build_strategic_retry_prompt(
    original_prompt: str,
    previous_response: str,
    eval_result: EvalResult,
    strategies: List[FixStrategy],
    eval_context: Optional[str] = None,
) -> Tuple[str, str, float]:
    """Build a strategic retry prompt from selected fix strategies.

    Args:
        original_prompt: The original user prompt.
        previous_response: The previous model response (for context).
        eval_result: The evaluation result.
        strategies: Selected fix strategies.
        eval_context: Reference material (already embedded in accuracy strategy if used).

    Returns:
        Tuple of (prompt, system_addendum, temperature_delta).
    """
    # Build score summary — only show dimensions that were actually evaluated
    score_parts = [f"accuracy={eval_result.accuracy:.2f}"]
    if eval_result.completeness > 0:
        score_parts.append(f"completeness={eval_result.completeness:.2f}")
    if eval_result.clarity > 0:
        score_parts.append(f"clarity={eval_result.clarity:.2f}")
    if eval_result.actionability > 0:
        score_parts.append(f"actionability={eval_result.actionability:.2f}")
    if eval_result.fulfillment > 0:
        score_parts.append(f"fulfillment={eval_result.fulfillment:.2f}")
    score_summary = (
        f"Previous attempt scores: {', '.join(score_parts)} " f"(overall={eval_result.score:.2f})"
    )

    # Collect issues (cap at 15 to avoid prompt bloat)
    issues_text = ""
    if eval_result.issues:
        capped = eval_result.issues[:15]
        issues_text = "\n\nSpecific issues found:\n" + "\n".join(f"- {issue}" for issue in capped)
        if len(eval_result.issues) > 15:
            issues_text += f"\n... and {len(eval_result.issues) - 15} more issues"

    # Collect strategy addenda
    prompt_parts = [original_prompt]
    system_parts = []
    total_temp_delta = 0.0

    prompt_parts.append(f"\n\n[RETRY CONTEXT]\n{score_summary}{issues_text}")

    # Always include eval_context as reference material if available
    # (accuracy strategy also embeds it, but it must be present even when accuracy passes)
    if eval_context:
        # Check if any accuracy strategy already embeds it
        has_accuracy_strategy = any(s.dimension == "accuracy" for s in strategies)
        if not has_accuracy_strategy:
            prompt_parts.append(
                "\n\n=== REFERENCE MATERIAL ===\n"
                f"{eval_context}\n"
                "=== END REFERENCE MATERIAL ===\n"
                "Verify your code against this reference material."
            )

    for strategy in strategies:
        if strategy.prompt_addendum:
            prompt_parts.append(strategy.prompt_addendum)
        if strategy.system_addendum:
            system_parts.append(strategy.system_addendum)
        total_temp_delta += strategy.temperature_delta

    # Include a truncated excerpt of the previous response for context
    max_prev = 4000
    if len(previous_response) > max_prev:
        prev_excerpt = (
            previous_response[: max_prev // 2]
            + f"\n[... {len(previous_response) - max_prev} chars omitted ...]\n"
            + previous_response[-(max_prev // 2) :]
        )
    else:
        prev_excerpt = previous_response

    prompt_parts.append(
        f"\n\n[PREVIOUS RESPONSE FOR REFERENCE]\n{prev_excerpt}\n"
        "[END PREVIOUS RESPONSE]\n\n"
        "Now provide an improved, complete response addressing all issues above."
    )

    prompt = "\n".join(prompt_parts)
    system_addendum = " ".join(system_parts)

    return prompt, system_addendum, total_temp_delta
