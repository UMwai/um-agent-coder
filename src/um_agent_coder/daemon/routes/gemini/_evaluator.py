"""OutputEvaluator — rates response quality and suggests improvements.

Supports two evaluation modes:
1. General eval: Scores accuracy, completeness, clarity, actionability (1–10).
2. Accuracy-first cascade: Runs a specialized accuracy pass with pass/fail
   checklist against eval_context. If accuracy < threshold, skips full eval
   and returns immediately for targeted retry.

Accuracy checks use tiered penalties:
- BREAKING (0 pts): Wrong signatures, missing params, runtime errors
- FOREIGN (0.5 pts): External dependencies not in the project
- STYLE (0.75 pts): Pattern deviations, missing __init__.py, file structure
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_EVAL_MODEL = "gemini-3-flash-preview"
ACCURACY_EVAL_MODEL = "gemini-3.1-pro-preview"

# --- Severity weights for tiered penalties ---
SEVERITY_WEIGHTS: Dict[str, float] = {
    "breaking": 0.0,    # Wrong signatures, runtime errors → 0 points
    "foreign": 0.5,     # External deps not in project → half credit
    "style": 0.75,      # Pattern deviations → 3/4 credit
}

EVAL_SYSTEM_PROMPT = """You are a strict JSON-only response evaluator. You will receive a task description, a response to evaluate, and optionally reference material.

Your ONLY job is to output a JSON object rating the response. Do NOT discuss, summarize, or continue the response.

Output EXACTLY this JSON format and NOTHING else — no markdown, no explanation, no preamble:
{"accuracy": N, "completeness": N, "clarity": N, "actionability": N, "issues": ["issue1", "issue2"]}

Scoring (1-10 scale):
- accuracy: Are facts, API calls, and code patterns correct?
- completeness: Does it fully address every part of the task?
- clarity: Is it well-organized and easy to understand?
- actionability: Can someone directly use this output?
- issues: List specific problems found (empty list if none)"""


ACCURACY_SYSTEM_PROMPT = """You are a strict code accuracy auditor. You will receive:
1. A task description
2. Reference material containing the CORRECT API signatures, database patterns, and project conventions
3. A code response to audit

Your job is to produce a JSON checklist of accuracy checks. Each check is pass/fail with a severity level.

CHECK CATEGORIES:

**API Signature Verification** — For EVERY function call in the response:
- Does the function name match the reference exactly?
- Are parameters correct (names, order, types)?
- Is it called as instance method vs static/class method correctly?
- Are required parameters present? Are there invented parameters?

**Structural Correctness** — For the overall code structure:
- Are file paths correct per the task requirements?
- Are __init__.py files present for Python packages?
- Are imports referencing modules that exist in the project (not foreign dependencies)?
- Are enums used where the reference specifies enums (not string literals)?
- Does the code use the project's own infrastructure (not external services)?

**Runtime Viability** — Will this code actually run?
- Are async/await used correctly (not awaiting sync functions)?
- Are all imports resolvable within the project?
- Are there undefined variables or type mismatches?
- Will the code crash on first execution?

SEVERITY LEVELS:
- "breaking": Would cause runtime error, wrong behavior, or uses completely wrong API. Score: 0 points.
- "foreign": Uses external dependency not in the project, or invents APIs. Score: 0.5 points.
- "style": Pattern deviation that works but doesn't match project conventions. Score: 0.75 points.

Output EXACTLY this JSON format and NOTHING else:
{"checks": [{"check": "description", "status": "pass"|"fail", "severity": "breaking"|"foreign"|"style", "detail": "explanation"}]}

Be thorough. Check EVERY function call against the reference material. Check EVERY import path. Check EVERY file path. Missing checks are worse than false positives."""


@dataclass
class AccuracyCheck:
    """A single pass/fail accuracy check."""
    check: str
    status: str  # "pass" or "fail"
    severity: str  # "breaking", "foreign", "style"
    detail: str = ""


@dataclass
class EvalResult:
    score: float = 0.0
    accuracy: float = 0.0
    completeness: float = 0.0
    clarity: float = 0.0
    actionability: float = 0.0
    issues: List[str] = field(default_factory=list)
    retry_count: int = 0
    accuracy_checks: List[AccuracyCheck] = field(default_factory=list)
    parse_failed: bool = False


def _parse_eval_response(text: str) -> Optional[dict]:
    """Extract JSON scores from evaluator response."""
    # Try direct JSON parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object (including nested arrays/objects)
    match = re.search(r"\{[^{}]*(?:\[.*?\])?[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort: find all JSON-like blocks and try each
    for match in re.finditer(r"\{[^}]+\}", text):
        try:
            candidate = match.group(0)
            parsed = json.loads(candidate)
            if "accuracy" in parsed or "completeness" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    logger.warning("Could not parse eval JSON from: %s", text[:300])
    return None


def _parse_accuracy_checks(text: str) -> Optional[dict]:
    """Extract JSON checklist from accuracy evaluator response."""
    # Try direct parse
    try:
        data = json.loads(text.strip())
        if "checks" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Try markdown code blocks
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "checks" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Try finding the outermost JSON object containing "checks"
    # Use a greedy match from first { to last }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if "checks" in data:
                return data
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse accuracy checks JSON from: %s", text[:300])
    return None


def _score_accuracy_checks(checks: List[AccuracyCheck]) -> float:
    """Calculate accuracy score from pass/fail checklist with tiered penalties.

    Each check contributes equally to the total. Passing checks get 1.0 points.
    Failing checks get points based on severity:
    - breaking: 0.0 points
    - foreign: 0.5 points
    - style: 0.75 points
    """
    if not checks:
        return 0.7  # No checks = default

    total_points = 0.0
    for check in checks:
        if check.status == "pass":
            total_points += 1.0
        else:
            total_points += SEVERITY_WEIGHTS.get(check.severity, 0.0)

    return total_points / len(checks)


async def evaluate_accuracy(
    client,
    prompt: str,
    response: str,
    eval_context: str,
    *,
    model: Optional[str] = None,
) -> EvalResult:
    """Run accuracy-first evaluation with pass/fail checklist.

    Uses a specialized system prompt that checks API signatures, structural
    correctness, and runtime viability against eval_context.

    Args:
        client: GeminiCodeAssistClient instance.
        prompt: The original user prompt.
        response: The model's response to evaluate.
        eval_context: REQUIRED — reference material (API signatures, schemas).
        model: Model for accuracy eval. Defaults to ACCURACY_EVAL_MODEL (Pro 3.1).

    Returns:
        EvalResult with accuracy score from checklist and accuracy_checks populated.
        Other dimensions (completeness, clarity, actionability) are zeroed — caller
        should run evaluate_response() separately if accuracy passes.
    """
    eval_model = model or ACCURACY_EVAL_MODEL

    # Truncate for eval
    MAX_PROMPT_CHARS = 8_000
    MAX_RESPONSE_CHARS = 60_000  # Larger limit for accuracy — need to see all code
    truncated_prompt = prompt[:MAX_PROMPT_CHARS] + ("..." if len(prompt) > MAX_PROMPT_CHARS else "")
    if len(response) > MAX_RESPONSE_CHARS:
        half = MAX_RESPONSE_CHARS // 2
        truncated_response = (
            response[:half]
            + f"\n\n[... {len(response) - MAX_RESPONSE_CHARS} chars omitted ...]\n\n"
            + response[-half:]
        )
    else:
        truncated_response = response

    eval_prompt = (
        "=== TASK DESCRIPTION ===\n"
        f"{truncated_prompt}\n"
        "=== END TASK DESCRIPTION ===\n\n"
        "=== REFERENCE MATERIAL (ground truth — check ALL code against this) ===\n"
        f"{eval_context}\n"
        "=== END REFERENCE MATERIAL ===\n\n"
        "=== CODE RESPONSE TO AUDIT ===\n"
        f"{truncated_response}\n"
        "=== END CODE RESPONSE ===\n\n"
        "Now audit this code response against the reference material. "
        "Check EVERY function call, import, file path, and pattern. "
        "Output your checklist as JSON:\n"
        '{"checks": [{"check": "...", "status": "pass"|"fail", '
        '"severity": "breaking"|"foreign"|"style", "detail": "..."}]}'
    )

    try:
        result = await client.generate(
            prompt=eval_prompt,
            model=eval_model,
            system_prompt=ACCURACY_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=16384,  # Checklist can be long
            timeout=180.0,
        )

        eval_text = result.get("text", "")
        finish = result.get("usage", {}).get("finish_reason", "unknown")
        logger.info(
            "Accuracy eval raw (%d chars, finish=%s): %s",
            len(eval_text), finish, eval_text[:500],
        )

        # Handle truncated JSON
        cleaned = eval_text.strip()
        if cleaned.startswith("{") and not cleaned.endswith("}"):
            logger.warning("Accuracy eval JSON truncated, attempting repair")
            # Find last complete check object
            last_brace = cleaned.rfind("}")
            if last_brace > 0:
                cleaned = cleaned[:last_brace + 1] + "]}"
            eval_text = cleaned

        parsed = _parse_accuracy_checks(eval_text)
        if not parsed or "checks" not in parsed:
            logger.warning("Failed to parse accuracy checks, returning default")
            return EvalResult(score=0.5, accuracy=0.5, parse_failed=True)

        # Convert to AccuracyCheck objects
        checks: List[AccuracyCheck] = []
        for raw in parsed["checks"]:
            checks.append(AccuracyCheck(
                check=raw.get("check", "unknown"),
                status=raw.get("status", "fail"),
                severity=raw.get("severity", "breaking"),
                detail=raw.get("detail", ""),
            ))

        accuracy_score = _score_accuracy_checks(checks)

        # Build issues from failing checks
        issues = [
            f"[{c.severity.upper()}] {c.check}: {c.detail}"
            for c in checks if c.status == "fail"
        ]

        logger.info(
            "Accuracy eval: %d checks, %d passed, %d failed, score=%.3f",
            len(checks),
            sum(1 for c in checks if c.status == "pass"),
            sum(1 for c in checks if c.status == "fail"),
            accuracy_score,
        )

        return EvalResult(
            score=accuracy_score,  # Overall = accuracy in cascade mode
            accuracy=accuracy_score,
            issues=issues,
            accuracy_checks=checks,
        )

    except Exception as e:
        logger.warning("Accuracy eval failed: %s", e)
        return EvalResult(score=0.5, accuracy=0.5, parse_failed=True)


async def evaluate_response(
    client,
    prompt: str,
    response: str,
    *,
    model: Optional[str] = None,
    eval_context: Optional[str] = None,
) -> EvalResult:
    """Evaluate a response quality (general dimensions).

    Args:
        client: GeminiCodeAssistClient instance.
        prompt: The original user prompt.
        response: The model's response to evaluate.
        model: Model to use for evaluation. None → DEFAULT_EVAL_MODEL (Flash).
        eval_context: Reference material (API signatures, schemas) to check against.

    Returns:
        EvalResult with scores and issues.
    """
    eval_model = model or DEFAULT_EVAL_MODEL

    # Truncate very long prompts/responses to keep eval focused
    MAX_PROMPT_CHARS = 8_000
    MAX_RESPONSE_CHARS = 40_000
    truncated_prompt = prompt[:MAX_PROMPT_CHARS] + ("..." if len(prompt) > MAX_PROMPT_CHARS else "")
    if len(response) > MAX_RESPONSE_CHARS:
        # Keep start + end so evaluator can check completeness
        half = MAX_RESPONSE_CHARS // 2
        truncated_response = (
            response[:half]
            + f"\n\n[... {len(response) - MAX_RESPONSE_CHARS} chars omitted ...]\n\n"
            + response[-half:]
        )
    else:
        truncated_response = response

    parts = [
        "=== TASK DESCRIPTION ===\n",
        truncated_prompt,
        "\n=== END TASK DESCRIPTION ===\n\n",
    ]
    if eval_context:
        parts.append("=== REFERENCE MATERIAL ===\n")
        parts.append(eval_context)
        parts.append("\n=== END REFERENCE MATERIAL ===\n\n")
    parts.extend([
        "=== RESPONSE TO EVALUATE ===\n",
        truncated_response,
        "\n=== END RESPONSE ===\n\n",
        "Now output your evaluation as a single JSON object. "
        "Do NOT output anything except the JSON:\n"
        '{"accuracy": N, "completeness": N, "clarity": N, "actionability": N, "issues": ["..."]}',
    ])
    eval_prompt = "".join(parts)
    system_prompt = EVAL_SYSTEM_PROMPT

    try:
        result = await client.generate(
            prompt=eval_prompt,
            model=eval_model,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=8192,
            timeout=120.0,
        )

        eval_text = result.get("text", "")
        finish = result.get("usage", {}).get("finish_reason", "unknown")
        logger.info(
            "Eval raw response (%d chars, finish=%s): %s",
            len(eval_text), finish, eval_text[:500],
        )

        # If the response looks like truncated JSON, try to repair it
        cleaned = eval_text.strip()
        if cleaned.startswith("{") and not cleaned.endswith("}"):
            logger.warning("Eval JSON appears truncated, attempting repair")
            last_quote = cleaned.rfind('"')
            if last_quote > 0:
                if '"issues"' in cleaned:
                    cleaned = cleaned[:last_quote + 1] + "]}"
                else:
                    cleaned = cleaned[:last_quote + 1] + "}"
            eval_text = cleaned

        scores = _parse_eval_response(eval_text)
        if not scores:
            logger.warning("Failed to parse eval response, returning default scores")
            return EvalResult(score=0.7, parse_failed=True)

        accuracy = float(scores.get("accuracy", 5)) / 10
        completeness = float(scores.get("completeness", 5)) / 10
        clarity = float(scores.get("clarity", 5)) / 10
        actionability = float(scores.get("actionability", 5)) / 10
        overall = (accuracy + completeness + clarity + actionability) / 4

        return EvalResult(
            score=overall,
            accuracy=accuracy,
            completeness=completeness,
            clarity=clarity,
            actionability=actionability,
            issues=scores.get("issues", []),
        )
    except Exception as e:
        logger.warning("Eval failed, returning default score: %s", e)
        return EvalResult(score=0.7, parse_failed=True)


def build_retry_prompt(
    original_prompt: str,
    previous_response: str,
    eval_result: EvalResult,
) -> str:
    """Build an improved prompt incorporating eval feedback."""
    issues_text = "\n".join(f"- {issue}" for issue in eval_result.issues)

    return (
        f"{original_prompt}\n\n"
        f"[IMPORTANT: A previous attempt had these issues:\n{issues_text}\n"
        f"Please address these issues specifically. "
        f"Scores were: accuracy={eval_result.accuracy:.1f}, "
        f"completeness={eval_result.completeness:.1f}, "
        f"clarity={eval_result.clarity:.1f}. "
        f"Improve on all dimensions.]"
    )
