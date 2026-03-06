"""Fix strategy engine — maps eval dimension failures to targeted prompt modifications.

Given an EvalResult, selects strategies for failing dimensions and builds a
strategic retry prompt that addresses specific weaknesses rather than generic
"do better" instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ._evaluator import AccuracyCheck, EvalResult


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
    # Group failures by severity
    breaking = [c for c in checks if c.status == "fail" and c.severity == "breaking"]
    foreign = [c for c in checks if c.status == "fail" and c.severity == "foreign"]
    style = [c for c in checks if c.status == "fail" and c.severity == "style"]

    failures_text = ""
    if breaking:
        failures_text += "\n\n🚨 CRITICAL FAILURES (must fix — these cause runtime errors):\n"
        for c in breaking:
            failures_text += f"- {c.check}: {c.detail}\n"
    if foreign:
        failures_text += "\n\n⚠️ FOREIGN DEPENDENCY ISSUES (must fix — use project infrastructure):\n"
        for c in foreign:
            failures_text += f"- {c.check}: {c.detail}\n"
    if style:
        failures_text += "\n\n📋 PATTERN DEVIATIONS (should fix — match project conventions):\n"
        for c in style:
            failures_text += f"- {c.check}: {c.detail}\n"

    return FixStrategy(
        name="accuracy_checklist_fix",
        dimension="accuracy",
        prompt_addendum=(
            "\n\n[ACCURACY FIX — CHECKLIST-DRIVEN]\n"
            "An automated code audit found the following specific issues in your "
            "previous response. You MUST fix ALL of them.\n"
            f"{failures_text}\n"
            "=== REFERENCE MATERIAL (ground truth) ===\n"
            f"{eval_context}\n"
            "=== END REFERENCE MATERIAL ===\n\n"
            "Fix EVERY issue listed above. Verify each function call, import, and "
            "pattern against the reference material. Do NOT repeat the same mistakes."
        ),
        system_addendum=(
            "You are fixing specific code accuracy issues identified by an automated audit. "
            "Each issue has been categorized by severity. Fix ALL critical and foreign "
            "dependency issues. Fix pattern deviations where possible. "
            "Verify every API call against the provided reference material."
        ),
        temperature_delta=-0.2,
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


def select_strategies(
    eval_result: EvalResult,
    eval_context: Optional[str] = None,
    threshold: float = 0.7,
) -> List[FixStrategy]:
    """Select targeted fix strategies based on failing eval dimensions.

    Args:
        eval_result: The evaluation result with per-dimension scores.
        eval_context: Reference material for accuracy checking.
        threshold: Scores below this trigger a fix strategy.

    Returns:
        List of FixStrategy objects to apply.
    """
    strategies: List[FixStrategy] = []

    if eval_result.accuracy < threshold:
        if eval_result.accuracy_checks and eval_context:
            # Use checklist-driven fix with specific failures
            strategies.append(_accuracy_with_checklist(eval_context, eval_result.accuracy_checks))
        elif eval_context:
            strategies.append(_accuracy_with_context(eval_context))
        else:
            strategies.append(_accuracy_fix())

    if eval_result.completeness < threshold:
        strategies.append(_completeness_fix())

    if eval_result.clarity < threshold:
        strategies.append(_clarity_fix())

    if eval_result.actionability < threshold:
        strategies.append(_actionability_fix())

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
    # Build score summary
    score_summary = (
        f"Previous attempt scores: "
        f"accuracy={eval_result.accuracy:.2f}, "
        f"completeness={eval_result.completeness:.2f}, "
        f"clarity={eval_result.clarity:.2f}, "
        f"actionability={eval_result.actionability:.2f} "
        f"(overall={eval_result.score:.2f})"
    )

    # Collect issues
    issues_text = ""
    if eval_result.issues:
        issues_text = "\n\nSpecific issues found:\n" + "\n".join(
            f"- {issue}" for issue in eval_result.issues
        )

    # Collect strategy addenda
    prompt_parts = [original_prompt]
    system_parts = []
    total_temp_delta = 0.0

    prompt_parts.append(f"\n\n[RETRY CONTEXT]\n{score_summary}{issues_text}")

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
            previous_response[:max_prev // 2]
            + f"\n[... {len(previous_response) - max_prev} chars omitted ...]\n"
            + previous_response[-(max_prev // 2):]
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
