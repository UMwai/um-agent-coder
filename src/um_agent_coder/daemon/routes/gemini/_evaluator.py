"""OutputEvaluator — rates response quality and suggests improvements.

Supports three evaluation modes:
1. General eval: Scores accuracy, completeness, clarity, actionability (1–10).
2. Accuracy-first cascade: Runs a specialized accuracy pass with pass/fail
   checklist against eval_context. If accuracy < threshold, skips full eval
   and returns immediately for targeted retry.
3. Fulfillment eval: Checks whether the response actually addresses every
   requirement stated in the original prompt. Pass/fail checklist.

Accuracy/fulfillment checks use tiered penalties:
- BREAKING (0 pts): Wrong signatures, missing params, runtime errors
- FOREIGN (0.25 pts): External dependencies not in the project
- STYLE (0.5 pts): Pattern deviations, missing __init__.py, file structure
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_RATE_LIMIT_MAX_RETRIES = 3
_RATE_LIMIT_BASE_DELAY = 15  # seconds


async def _generate_with_retry(client, **kwargs) -> dict:
    """Call client.generate() with exponential backoff on rate limits."""
    from um_agent_coder.daemon.gemini_client import RateLimitError

    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return await client.generate(**kwargs)
        except RateLimitError:
            if attempt >= _RATE_LIMIT_MAX_RETRIES:
                raise
            delay = _RATE_LIMIT_BASE_DELAY * (2**attempt)
            logger.warning(
                "Rate limited (attempt %d/%d), waiting %ds before retry",
                attempt + 1,
                _RATE_LIMIT_MAX_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)
    # unreachable, but keeps type checkers happy
    raise RuntimeError("Exhausted rate limit retries")


def _get_eval_model() -> str:
    """Get default eval model from config."""
    try:
        from um_agent_coder.daemon.app import get_settings

        return get_settings().gemini_eval_model
    except Exception:
        return "gemini-3-flash-preview"


def _get_accuracy_eval_model() -> str:
    """Get accuracy/fulfillment eval model from config."""
    try:
        from um_agent_coder.daemon.app import get_settings

        return get_settings().gemini_accuracy_eval_model
    except Exception:
        return "gemini-3.1-pro-preview"


# Keep as fallback constants for scripts that import directly
DEFAULT_EVAL_MODEL = "gemini-3-flash-preview"
ACCURACY_EVAL_MODEL = "gemini-3.1-pro-preview"

# --- Severity weights for tiered penalties ---
SEVERITY_WEIGHTS: Dict[str, float] = {
    "breaking": 0.0,  # Wrong signatures, runtime errors → 0 points
    "foreign": 0.25,  # External deps not in project → quarter credit
    "style": 0.5,  # Pattern deviations → half credit
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

Output one check per line in this pipe-delimited format (NO JSON, no markdown, no commentary):
PASS|breaking|Check description|Detail text
FAIL|foreign|Check description|Detail text

Example:
PASS|breaking|create_task() takes task_id as first param|Matches reference signature
FAIL|foreign|Uses requests library|Project uses httpx, not requests

Be thorough. Check EVERY function call against the reference material. Check EVERY import path. Check EVERY file path. Missing checks are worse than false positives."""


FULFILLMENT_SYSTEM_PROMPT = """You are a strict requirements fulfillment auditor. You will receive:
1. A task description (the original prompt with specific requirements)
2. A code response to audit

Your job is to extract EVERY specific requirement from the task description and check whether the response actually fulfills it. This is NOT about code correctness (that's a separate check) — this is about whether the response DOES what was ASKED.

CHECK CATEGORIES:

**Deliverable Presence** — For each file, feature, or component explicitly requested:
- Is the deliverable present in the response?
- Is it complete or just a stub/placeholder?
- Does it match what was described (e.g., "Full CRUD" means create + read + update + delete)?

**Feature Completeness** — For each functional requirement:
- Is the feature actually implemented, not just mentioned?
- Does it cover the full scope described (all sub-requirements)?
- Are edge cases addressed if the prompt specifies them?

**Negative Requirements** — For constraints and prohibitions:
- "No TODOs" — are there any TODO comments?
- "No stubs" — are all functions fully implemented?
- "No placeholders" — are there pass statements, "...", or "not yet implemented" messages?
- "Must be async" — is everything actually async?

SEVERITY LEVELS:
- "breaking": A core required deliverable is completely missing or a critical requirement is violated. Score: 0 points.
- "foreign": A requirement is only partially fulfilled or the implementation doesn't match what was asked. Score: 0.5 points.
- "style": A minor or implied requirement is not fully met. Score: 0.75 points.

Output one check per line in this pipe-delimited format (NO JSON, no markdown, no commentary):
PASS|breaking|Check description|Detail text
FAIL|foreign|Check description|Detail text

Example:
PASS|breaking|All 11 requested files are present|Counted 11 files in response
FAIL|breaking|No TODO comments|Found 3 TODO comments in auth.py

Be thorough. Extract EVERY requirement — explicit and clearly implied. Check quantitative claims (e.g., "11 files" — count them). Check qualitative claims (e.g., "complete implementations" — verify no stubs)."""


COMPLETENESS_SYSTEM_PROMPT = """You are a strict completeness auditor. You will receive:
1. A task description
2. A code response to audit

Your job is to check whether the response covers the FULL scope of the task. This is NOT about correctness — it is about whether everything requested is present.

CHECK CATEGORIES:

**File/Component Coverage** — For each requested deliverable:
- Is the file/component present in the response?
- Is it complete (not truncated mid-function or mid-class)?
- Are all sections/methods of the component included?

**Feature Coverage** — For each feature area:
- Are ALL sub-features implemented?
- Are edge cases handled when requested?
- Are error handling paths present?
- Are configuration/setup steps included?

**Scope Completeness** — Overall scope coverage:
- Does the response address EVERY part of a multi-part request?
- Are dependencies between components addressed?
- Are integration points covered?
- Is any section cut short with "etc.", "...", or "similar for other X"?

SEVERITY LEVELS:
- "breaking": An entire requested file, component, or feature is missing. Score: 0 points.
- "foreign": A component is present but significantly incomplete (major sections missing). Score: 0.5 points.
- "style": Minor omission that doesn't block functionality. Score: 0.75 points.

Output one check per line in this pipe-delimited format (NO JSON, no markdown, no commentary):
PASS|breaking|Check description|Detail text
FAIL|foreign|Check description|Detail text

Example:
PASS|breaking|All 5 requested files present|Found models.py, routes.py, tests.py, config.py, utils.py
FAIL|breaking|Error handling paths present|Missing try/except in database module

Be thorough. Count files, count functions, count features. If the task asks for 5 files and 3 are present, that's a failing check."""


CLARITY_SYSTEM_PROMPT = """You are a strict clarity and organization auditor. You will receive:
1. A task description
2. A code response to audit

Your job is to check whether the response is well-organized, readable, and easy to follow. This is NOT about correctness — it is about communication quality.

CHECK CATEGORIES:

**Structure and Organization** — For the overall response:
- Are code blocks clearly delimited with file paths?
- Are sections logically ordered (models before usage, base before derived)?
- Is there a clear overview/summary before diving into code?
- Are related pieces grouped together?

**Code Readability** — For code quality:
- Are variable/function names descriptive and consistent?
- Are complex logic blocks commented or explained?
- Is indentation and formatting consistent?
- Are magic numbers or strings explained?

**Documentation Quality** — For explanatory content:
- Are function/class docstrings present for public APIs?
- Are non-obvious design decisions explained?
- Are usage examples provided when helpful?
- Is terminology consistent throughout?

SEVERITY LEVELS:
- "breaking": Response is fundamentally disorganized — code mixed with prose, no file boundaries, unreadable structure. Score: 0 points.
- "foreign": Significant clarity issues — missing explanations for complex logic, inconsistent naming, poor organization. Score: 0.5 points.
- "style": Minor clarity issues — could use better comments, slightly inconsistent formatting. Score: 0.75 points.

Output one check per line in this pipe-delimited format (NO JSON, no markdown, no commentary):
PASS|breaking|Check description|Detail text
FAIL|foreign|Check description|Detail text

Example:
PASS|style|Code blocks clearly delimited with file paths|Each file has header comment
FAIL|foreign|Complex logic blocks commented|Database query builder has no explanation

Focus on whether someone could understand and USE this response easily."""


ACTIONABILITY_SYSTEM_PROMPT = """You are a strict actionability auditor. You will receive a task and a code response.

Check whether the response is DIRECTLY USABLE — can someone copy-paste it and run without extra work?

CHECK THESE (combine related issues into single checks):
1. Stubs/TODOs: Any `pass`, `...`, `raise NotImplementedError`, TODO comments, or "add your code here"?
2. Imports: Are all import statements present?
3. Config: Are config values real (not "YOUR_KEY_HERE")?
4. File paths: Are files clearly labeled for placement?
5. Dependencies: Are requirements/setup steps mentioned?

SEVERITY: "breaking" (stubs/TODOs, 0 pts), "foreign" (missing deps, 0.5 pts), "style" (minor, 0.75 pts).

RULES:
- Maximum 15 checks. Group related issues.
- Keep detail to 1 SHORT sentence.

Output one check per line in this pipe-delimited format (NO JSON, no markdown, no commentary):
PASS|breaking|Check description|Detail text
FAIL|foreign|Check description|Detail text

Example:
PASS|breaking|No stubs or TODOs|All functions fully implemented
FAIL|breaking|All imports present|Missing import for datetime module"""


@dataclass
class AccuracyCheck:
    """A single pass/fail accuracy check."""

    check: str
    status: str  # "pass" or "fail"
    severity: str  # "breaking", "foreign", "style"
    detail: str = ""


# Type aliases — all checklist dimensions reuse AccuracyCheck structure
FulfillmentCheck = AccuracyCheck
CompletenessCheck = AccuracyCheck
ClarityCheck = AccuracyCheck
ActionabilityCheck = AccuracyCheck


@dataclass
class PreGenCheck:
    """A single pre-generated check for accuracy/fulfillment/completeness."""

    dimension: str  # "accuracy", "fulfillment", "completeness"
    check: str
    severity: str = "breaking"
    detail: str = ""  # Filled during scoring
    source: str = "pre_gen"  # "pre_gen" or "evaluator"


@dataclass
class PreGenChecklist:
    """Fixed checklist generated BEFORE code generation."""

    checks: List[PreGenCheck] = field(default_factory=list)
    generation_tokens: int = 0

    @property
    def accuracy_checks(self) -> List[PreGenCheck]:
        return [c for c in self.checks if c.dimension == "accuracy"]

    @property
    def fulfillment_checks(self) -> List[PreGenCheck]:
        return [c for c in self.checks if c.dimension == "fulfillment"]

    @property
    def completeness_checks(self) -> List[PreGenCheck]:
        return [c for c in self.checks if c.dimension == "completeness"]

    def format_for_prompt(self) -> str:
        """Format checklist for injection into generation prompt."""
        lines = ["[EVALUATION CHECKLIST]"]
        for c in self.checks:
            lines.append(f"- [{c.severity.upper()}] [{c.dimension}] {c.check}")
        return "\n".join(lines)


CHECKLIST_GENERATION_SYSTEM_PROMPT = """You are a strict evaluation checklist generator. You will receive a task description and optionally reference material (eval_context).

Your job is to produce a FIXED checklist of checks that will be used to evaluate ANY response to this task. The checks must be:
1. Derived ONLY from the task description and reference material — NOT from any response
2. Specific and objectively verifiable (pass/fail, not subjective)
3. Categorized by dimension: accuracy, fulfillment, or completeness

DIMENSIONS:

**accuracy** — Checks verifiable against the reference material:
- Expected API calls, function signatures, parameter names
- Required imports and module paths
- Expected types, enums, database patterns
- Runtime correctness patterns (async/await, error handling)
Only generate accuracy checks if eval_context is provided.

**fulfillment** — Checks derived from the task requirements:
- Each explicit requirement in the prompt
- Each deliverable (file, component, feature) requested
- Negative requirements ("no TODOs", "no stubs", "must be async")
- Quantitative requirements ("11 files", "full CRUD")

**completeness** — Checks for expected scope:
- Each file or component explicitly requested
- Sub-features within larger features
- Integration points and dependencies
- Configuration and setup requirements

SEVERITY LEVELS:
- "breaking": Core requirement, wrong/missing = unusable. Score: 0 points.
- "foreign": Important but partial fulfillment still has value. Score: 0.25 points.
- "style": Minor convention or preference. Score: 0.5 points.

RULES:
- Maximum 40 checks total (5-15 per dimension)
- Each check must be independently verifiable
- Do NOT check for code quality, clarity, or style (those are separate dimensions)
- Keep check descriptions concise (1 sentence)

Output one check per line in this pipe-delimited format (NO JSON, no markdown, no commentary):
DIMENSION|SEVERITY|Check description

Example:
accuracy|breaking|create_task() signature matches reference (task_id, name, config)
fulfillment|breaking|All 11 requested files are present
completeness|foreign|Error handling for API failures included"""


@dataclass
class EvalResult:
    score: float = 0.0
    accuracy: float = 0.0
    completeness: float = 0.0
    clarity: float = 0.0
    actionability: float = 0.0
    fulfillment: float = 0.0
    issues: List[str] = field(default_factory=list)
    retry_count: int = 0
    accuracy_checks: List[AccuracyCheck] = field(default_factory=list)
    fulfillment_checks: List[FulfillmentCheck] = field(default_factory=list)
    completeness_checks: List[CompletenessCheck] = field(default_factory=list)
    clarity_checks: List[ClarityCheck] = field(default_factory=list)
    actionability_checks: List[ActionabilityCheck] = field(default_factory=list)
    parse_failed: bool = False
    parse_failed_dimensions: List[str] = field(default_factory=list)


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
    """Extract JSON checklist from evaluator response.

    Handles multiple failure modes:
    1. Direct JSON (clean LLM output)
    2. JSON inside markdown code blocks
    3. JSON with commentary before/after
    4. Truncated JSON (response cut off mid-array)
    5. Individual check objects (no wrapper)
    """
    cleaned = text.strip()

    # 1. Direct parse
    try:
        data = json.loads(cleaned)
        if "checks" in data:
            return data
    except json.JSONDecodeError:
        pass

    # 2. Markdown code blocks (```json ... ``` or ``` ... ```)
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "checks" in data:
                return data
        except json.JSONDecodeError:
            pass

    # 3. Find outermost { ... } containing "checks"
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            data = json.loads(candidate)
            if "checks" in data:
                return data
        except json.JSONDecodeError:
            # 4. Truncated JSON — find last complete check object and close the array
            last_complete = candidate.rfind("},")
            if last_complete == -1:
                last_complete = candidate.rfind("}")
            if last_complete > 0:
                repaired = candidate[: last_complete + 1] + "]}"
                try:
                    data = json.loads(repaired)
                    if "checks" in data:
                        logger.info(
                            "Repaired truncated JSON: recovered %d checks",
                            len(data.get("checks", [])),
                        )
                        return data
                except json.JSONDecodeError:
                    pass

    # 5. Individual check objects scattered in text (no wrapper)
    check_objects = []
    for m in re.finditer(r'\{\s*"check"\s*:\s*"[^"]*"[^}]*\}', cleaned, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if "check" in obj and "status" in obj:
                check_objects.append(obj)
        except json.JSONDecodeError:
            continue
    if check_objects:
        logger.info("Recovered %d individual check objects", len(check_objects))
        return {"checks": check_objects}

    logger.warning("Could not parse accuracy checks JSON from: %s", cleaned[:500])
    return None


def _parse_checklist_lines(text: str, *, format: str = "standard") -> Optional[dict]:
    """Parse pipe-delimited checklist lines with JSON fallback.

    Formats:
      standard:  STATUS|SEVERITY|Check description|Detail text
      pre_gen:   DIMENSION|SEVERITY|Check description
      scoring:   STATUS|SEVERITY|SOURCE|Check description|Detail text

    Falls back to _parse_accuracy_checks() (JSON) if no pipe lines found.
    """
    lines = text.strip().splitlines()
    checks = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("="):
            continue
        # Strip leading "- " or "* " bullet markers
        if line.startswith(("- ", "* ")):
            line = line[2:]

        parts = [p.strip() for p in line.split("|")]

        if format == "standard" and len(parts) >= 3:
            status = parts[0].lower()
            if status not in ("pass", "fail"):
                continue
            severity = parts[1].lower()
            if severity not in ("breaking", "foreign", "style"):
                severity = "breaking"
            check_desc = parts[2]
            detail = parts[3] if len(parts) >= 4 else ""
            checks.append(
                {
                    "check": check_desc,
                    "status": status,
                    "severity": severity,
                    "detail": detail,
                }
            )

        elif format == "pre_gen" and len(parts) >= 3:
            dim = parts[0].lower()
            if dim not in ("accuracy", "fulfillment", "completeness"):
                continue
            severity = parts[1].lower()
            if severity not in ("breaking", "foreign", "style"):
                severity = "breaking"
            check_desc = parts[2]
            checks.append(
                {
                    "dimension": dim,
                    "check": check_desc,
                    "severity": severity,
                }
            )

        elif format == "scoring" and len(parts) >= 4:
            status = parts[0].lower()
            if status not in ("pass", "fail"):
                continue
            severity = parts[1].lower()
            if severity not in ("breaking", "foreign", "style"):
                severity = "breaking"
            source = parts[2].lower()
            if source not in ("pre_gen", "evaluator"):
                source = "pre_gen"
            check_desc = parts[3]
            detail = parts[4] if len(parts) >= 5 else ""
            checks.append(
                {
                    "check": check_desc,
                    "status": status,
                    "severity": severity,
                    "detail": detail,
                    "source": source,
                }
            )

    if checks:
        return {"checks": checks}

    # Fallback to JSON parser
    return _parse_accuracy_checks(text)


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
    eval_model = model or _get_accuracy_eval_model()

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
        "One check per line: STATUS|SEVERITY|Check description|Detail"
    )

    try:
        result = await _generate_with_retry(
            client,
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
            len(eval_text),
            finish,
            eval_text[:500],
        )

        parsed = _parse_checklist_lines(eval_text, format="standard")
        if not parsed or "checks" not in parsed:
            # Retry with Flash model
            logger.warning("Failed to parse accuracy checks, falling back to Flash")
            flash_model = _get_eval_model()  # Flash
            FLASH_MAX = 15_000
            flash_response = response[:FLASH_MAX] if len(response) > FLASH_MAX else response
            flash_ctx = eval_context[:20_000] if len(eval_context) > 20_000 else eval_context
            flash_prompt = (
                f"TASK:\n{truncated_prompt[:4000]}\n\n"
                f"REFERENCE:\n{flash_ctx}\n\n"
                f"CODE TO AUDIT:\n{flash_response}\n\n"
                "Audit for accuracy. One check per line: PASS|SEVERITY|Check description|Detail"
            )
            flash_system = (
                "You are an accuracy auditor. Check code against the reference material. "
                "Output ONLY pipe-delimited lines. No JSON, no markdown, no commentary. "
                "Format: STATUS|SEVERITY|Check description|Detail text"
            )
            try:
                flash_result = await _generate_with_retry(
                    client,
                    prompt=flash_prompt,
                    model=flash_model,
                    system_prompt=flash_system,
                    temperature=0.0,
                    max_tokens=8192,
                    timeout=90.0,
                )
                eval_text = flash_result.get("text", "")
                logger.info(
                    "Accuracy Flash fallback (%d chars): %s", len(eval_text), eval_text[:300]
                )
                parsed = _parse_checklist_lines(eval_text, format="standard")
            except Exception as flash_err:
                logger.warning("Accuracy Flash fallback failed: %s", flash_err)

        if not parsed or "checks" not in parsed:
            logger.warning("Failed to parse accuracy checks after all retries")
            return EvalResult(score=0.5, accuracy=0.5, parse_failed=True)

        # Convert to AccuracyCheck objects
        checks: List[AccuracyCheck] = []
        for raw in parsed["checks"]:
            checks.append(
                AccuracyCheck(
                    check=raw.get("check", "unknown"),
                    status=raw.get("status", "fail"),
                    severity=raw.get("severity", "breaking"),
                    detail=raw.get("detail", ""),
                )
            )

        accuracy_score = _score_accuracy_checks(checks)

        # Build issues from failing checks
        issues = [
            f"[{c.severity.upper()}] {c.check}: {c.detail}" for c in checks if c.status == "fail"
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


async def evaluate_fulfillment(
    client,
    prompt: str,
    response: str,
    *,
    model: Optional[str] = None,
) -> EvalResult:
    """Run fulfillment evaluation with pass/fail checklist.

    Uses the shared _run_checklist_eval which includes retry logic on parse failure.
    """
    return await _run_checklist_eval(
        client,
        prompt,
        response,
        system_prompt=FULFILLMENT_SYSTEM_PROMPT,
        dimension="fulfillment",
        model=model,
    )


async def _run_checklist_eval(
    client,
    prompt: str,
    response: str,
    system_prompt: str,
    dimension: str,
    *,
    model: Optional[str] = None,
) -> EvalResult:
    """Generic checklist evaluation for any dimension.

    Shares the same pattern: build eval prompt → call Pro 3.1 → parse →
    score via _score_accuracy_checks.
    """
    eval_model = model or _get_accuracy_eval_model()

    MAX_PROMPT_CHARS = 12_000
    MAX_RESPONSE_CHARS = 60_000
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
        "=== RESPONSE TO AUDIT ===\n"
        f"{truncated_response}\n"
        "=== END RESPONSE ===\n\n"
        f"Now audit this response for {dimension}. "
        "One check per line: STATUS|SEVERITY|Check description|Detail"
    )

    try:
        result = await _generate_with_retry(
            client,
            prompt=eval_prompt,
            model=eval_model,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=16384,
            timeout=180.0,
        )

        eval_text = result.get("text", "")
        finish = result.get("usage", {}).get("finish_reason", "unknown")
        logger.info(
            "%s eval raw (%d chars, finish=%s): %s",
            dimension.capitalize(),
            len(eval_text),
            finish,
            eval_text[:500],
        )

        parsed = _parse_checklist_lines(eval_text, format="standard")
        if not parsed or "checks" not in parsed:
            # Single retry with shorter response excerpt and strict prompt
            logger.warning(
                "Failed to parse %s checks (finish=%s), retrying with stricter prompt",
                dimension,
                finish,
            )
            # Truncate response more aggressively for retry
            RETRY_MAX = 30_000
            retry_response = response[:RETRY_MAX] if len(response) > RETRY_MAX else response
            retry_prompt = (
                f"TASK: {truncated_prompt[:6000]}\n\n"
                f"RESPONSE TO AUDIT:\n{retry_response}\n\n"
                f"Audit for {dimension}. One check per line: STATUS|SEVERITY|Check description|Detail"
            )
            try:
                retry_result = await _generate_with_retry(
                    client,
                    prompt=retry_prompt,
                    model=eval_model,
                    system_prompt=system_prompt
                    + "\nYou MUST output ONLY pipe-delimited lines. No JSON, no markdown, no commentary.",
                    temperature=0.0,
                    max_tokens=16384,
                    timeout=120.0,
                )
                eval_text = retry_result.get("text", "")
                logger.info(
                    "%s eval retry (%d chars): %s",
                    dimension.capitalize(),
                    len(eval_text),
                    eval_text[:300],
                )
                parsed = _parse_checklist_lines(eval_text, format="standard")
            except Exception as retry_err:
                logger.warning("%s eval retry failed: %s", dimension.capitalize(), retry_err)

        if not parsed or "checks" not in parsed:
            # 2nd retry: Flash model fallback
            logger.warning(
                "Failed to parse %s checks after retry, falling back to Flash",
                dimension,
            )
            flash_model = _get_eval_model()  # Flash
            FLASH_MAX = 15_000
            flash_response = response[:FLASH_MAX] if len(response) > FLASH_MAX else response
            flash_prompt = (
                f"TASK:\n{truncated_prompt[:4000]}\n\n"
                f"RESPONSE:\n{flash_response}\n\n"
                f"Audit for {dimension}. One check per line: STATUS|SEVERITY|Check description|Detail"
            )
            flash_system = (
                f"You are a {dimension} auditor. Output ONLY pipe-delimited lines. "
                "No JSON, no markdown, no commentary. "
                "Format: STATUS|SEVERITY|Check description|Detail text"
            )
            try:
                flash_result = await _generate_with_retry(
                    client,
                    prompt=flash_prompt,
                    model=flash_model,
                    system_prompt=flash_system,
                    temperature=0.0,
                    max_tokens=8192,
                    timeout=90.0,
                )
                eval_text = flash_result.get("text", "")
                logger.info(
                    "%s eval Flash fallback (%d chars): %s",
                    dimension.capitalize(),
                    len(eval_text),
                    eval_text[:300],
                )
                parsed = _parse_checklist_lines(eval_text, format="standard")
            except Exception as flash_err:
                logger.warning("%s Flash fallback failed: %s", dimension.capitalize(), flash_err)

        if not parsed or "checks" not in parsed:
            logger.warning(
                "Failed to parse %s checks after all retries, returning default", dimension
            )
            return EvalResult(score=0.5, parse_failed=True, **{dimension: 0.5})

        checks: List[AccuracyCheck] = []
        for raw in parsed["checks"]:
            checks.append(
                AccuracyCheck(
                    check=raw.get("check", "unknown"),
                    status=raw.get("status", "fail"),
                    severity=raw.get("severity", "breaking"),
                    detail=raw.get("detail", ""),
                )
            )

        score = _score_accuracy_checks(checks)

        issues = [
            f"[{dimension.upper()}:{c.severity.upper()}] {c.check}: {c.detail}"
            for c in checks
            if c.status == "fail"
        ]

        logger.info(
            "%s eval: %d checks, %d passed, %d failed, score=%.3f",
            dimension.capitalize(),
            len(checks),
            sum(1 for c in checks if c.status == "pass"),
            sum(1 for c in checks if c.status == "fail"),
            score,
        )

        return EvalResult(
            score=score,
            issues=issues,
            **{dimension: score, f"{dimension}_checks": checks},
        )

    except Exception as e:
        logger.warning("%s eval failed: %s", dimension.capitalize(), e)
        return EvalResult(score=0.5, parse_failed=True, **{dimension: 0.5})


async def evaluate_completeness(
    client,
    prompt: str,
    response: str,
    *,
    model: Optional[str] = None,
) -> EvalResult:
    """Run completeness evaluation with pass/fail checklist."""
    return await _run_checklist_eval(
        client,
        prompt,
        response,
        system_prompt=COMPLETENESS_SYSTEM_PROMPT,
        dimension="completeness",
        model=model,
    )


async def evaluate_clarity(
    client,
    prompt: str,
    response: str,
    *,
    model: Optional[str] = None,
) -> EvalResult:
    """Run clarity evaluation with pass/fail checklist."""
    return await _run_checklist_eval(
        client,
        prompt,
        response,
        system_prompt=CLARITY_SYSTEM_PROMPT,
        dimension="clarity",
        model=model,
    )


async def evaluate_actionability(
    client,
    prompt: str,
    response: str,
    *,
    model: Optional[str] = None,
) -> EvalResult:
    """Run actionability evaluation with pass/fail checklist."""
    return await _run_checklist_eval(
        client,
        prompt,
        response,
        system_prompt=ACTIONABILITY_SYSTEM_PROMPT,
        dimension="actionability",
        model=model,
    )


async def generate_pre_gen_checklist(
    client,
    prompt: str,
    eval_context: Optional[str] = None,
    *,
    model: Optional[str] = None,
    max_checks: int = 40,
) -> PreGenChecklist:
    """Generate a fixed checklist BEFORE code generation.

    One LLM call that produces checks for accuracy/fulfillment/completeness.
    These checks stay constant across all iteration steps.
    """
    eval_model = model or _get_accuracy_eval_model()

    MAX_PROMPT_CHARS = 12_000
    truncated_prompt = prompt[:MAX_PROMPT_CHARS] + ("..." if len(prompt) > MAX_PROMPT_CHARS else "")

    parts = [
        "=== TASK DESCRIPTION ===\n",
        truncated_prompt,
        "\n=== END TASK DESCRIPTION ===\n\n",
    ]
    if eval_context:
        MAX_CONTEXT_CHARS = 60_000
        truncated_context = eval_context[:MAX_CONTEXT_CHARS] + (
            "..." if len(eval_context) > MAX_CONTEXT_CHARS else ""
        )
        parts.append("=== REFERENCE MATERIAL (eval_context) ===\n")
        parts.append(truncated_context)
        parts.append("\n=== END REFERENCE MATERIAL ===\n\n")

    parts.append(
        "Generate a checklist of checks to evaluate ANY response to this task. "
        "One check per line: DIMENSION|SEVERITY|Check description"
    )
    gen_prompt = "".join(parts)

    try:
        result = await _generate_with_retry(
            client,
            prompt=gen_prompt,
            model=eval_model,
            system_prompt=CHECKLIST_GENERATION_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=8192,
            timeout=120.0,
        )

        gen_text = result.get("text", "")
        gen_tokens = result.get("usage", {}).get("total_tokens", 0)

        logger.info(
            "Pre-gen checklist raw (%d chars): %s",
            len(gen_text),
            gen_text[:500],
        )

        parsed = _parse_checklist_lines(gen_text, format="pre_gen")
        if not parsed or "checks" not in parsed:
            # Retry with stricter prompt
            logger.warning("Failed to parse pre-gen checklist, retrying")
            retry_result = await _generate_with_retry(
                client,
                prompt=gen_prompt,
                model=eval_model,
                system_prompt=CHECKLIST_GENERATION_SYSTEM_PROMPT
                + "\nYou MUST output ONLY pipe-delimited lines. No JSON, no markdown, no commentary.",
                temperature=0.0,
                max_tokens=8192,
                timeout=90.0,
            )
            gen_text = retry_result.get("text", "")
            gen_tokens += retry_result.get("usage", {}).get("total_tokens", 0)
            parsed = _parse_checklist_lines(gen_text, format="pre_gen")

        if not parsed or "checks" not in parsed:
            logger.warning("Pre-gen checklist generation failed after retry")
            return PreGenChecklist(generation_tokens=gen_tokens)

        valid_dims = {"accuracy", "fulfillment", "completeness"}
        checks: List[PreGenCheck] = []
        for raw in parsed["checks"][:max_checks]:
            dim = raw.get("dimension", "")
            if dim not in valid_dims:
                continue
            # Skip accuracy checks when no eval_context provided
            if dim == "accuracy" and not eval_context:
                continue
            checks.append(
                PreGenCheck(
                    dimension=dim,
                    check=raw.get("check", "unknown"),
                    severity=raw.get("severity", "breaking"),
                )
            )

        logger.info(
            "Pre-gen checklist: %d checks (accuracy=%d, fulfillment=%d, completeness=%d)",
            len(checks),
            sum(1 for c in checks if c.dimension == "accuracy"),
            sum(1 for c in checks if c.dimension == "fulfillment"),
            sum(1 for c in checks if c.dimension == "completeness"),
        )

        return PreGenChecklist(checks=checks, generation_tokens=gen_tokens)

    except Exception as e:
        logger.warning("Pre-gen checklist generation failed: %s", e)
        return PreGenChecklist()


PREGEN_SCORING_SYSTEM_PROMPT = """You are a strict code auditor. You will receive:
1. A task description
2. A fixed checklist of checks to score
3. A code response to audit

Your job is to score EACH check in the checklist as pass or fail, AND you may add up to 5 NEW checks for important issues the checklist missed.

For EACH check in the provided checklist, output it with status "pass" or "fail" and a brief detail.
For any NEW checks you add, include source="evaluator" to distinguish them.

SEVERITY LEVELS (for new checks only — pre-gen checks keep their original severity):
- "breaking": Would cause runtime error, wrong behavior, or missing core requirement. Score: 0 points.
- "foreign": Uses external dependency not in project, or partially fulfills requirement. Score: 0.25 points.
- "style": Pattern deviation or minor omission. Score: 0.5 points.

Output one check per line in this pipe-delimited format (NO JSON, no markdown, no commentary):
STATUS|SEVERITY|SOURCE|Check description|Detail text

Example:
PASS|breaking|pre_gen|create_task() signature matches reference|All params correct
FAIL|foreign|evaluator|Uses external caching library|Project has built-in cache"""


async def score_pre_gen_checklist(
    client,
    prompt: str,
    response: str,
    pre_gen_checks: List[PreGenCheck],
    dimension: str,
    *,
    eval_context: Optional[str] = None,
    model: Optional[str] = None,
) -> EvalResult:
    """Score a response against a fixed pre-generated checklist for one dimension.

    The evaluator scores each pre-gen check and may add up to 5 new checks.
    """
    eval_model = model or _get_accuracy_eval_model()

    MAX_PROMPT_CHARS = 12_000
    MAX_RESPONSE_CHARS = 60_000
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

    # Format the pre-gen checks for the evaluator
    checks_text = "\n".join(f"- [{c.severity.upper()}] {c.check}" for c in pre_gen_checks)

    parts = [
        "=== TASK DESCRIPTION ===\n",
        truncated_prompt,
        "\n=== END TASK DESCRIPTION ===\n\n",
    ]
    if eval_context and dimension == "accuracy":
        MAX_CONTEXT_CHARS = 60_000
        truncated_context = eval_context[:MAX_CONTEXT_CHARS] + (
            "..." if len(eval_context) > MAX_CONTEXT_CHARS else ""
        )
        parts.append("=== REFERENCE MATERIAL ===\n")
        parts.append(truncated_context)
        parts.append("\n=== END REFERENCE MATERIAL ===\n\n")
    parts.extend(
        [
            f"=== FIXED CHECKLIST FOR {dimension.upper()} ===\n",
            checks_text,
            "\n=== END CHECKLIST ===\n\n",
            "=== RESPONSE TO AUDIT ===\n",
            truncated_response,
            "\n=== END RESPONSE ===\n\n",
            f"Score EACH check in the {dimension} checklist as pass/fail. "
            "You may add up to 5 NEW checks for important issues the checklist missed "
            "(tag them source=evaluator). One check per line: STATUS|SEVERITY|SOURCE|Check description|Detail",
        ]
    )
    eval_prompt = "".join(parts)

    try:
        result = await _generate_with_retry(
            client,
            prompt=eval_prompt,
            model=eval_model,
            system_prompt=PREGEN_SCORING_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=16384,
            timeout=180.0,
        )

        eval_text = result.get("text", "")
        finish = result.get("usage", {}).get("finish_reason", "unknown")
        logger.info(
            "Pre-gen %s scoring raw (%d chars, finish=%s): %s",
            dimension,
            len(eval_text),
            finish,
            eval_text[:500],
        )

        parsed = _parse_checklist_lines(eval_text, format="scoring")
        if not parsed or "checks" not in parsed:
            # Retry
            logger.warning("Failed to parse pre-gen %s scoring, retrying", dimension)
            RETRY_MAX = 30_000
            retry_response = response[:RETRY_MAX] if len(response) > RETRY_MAX else response
            retry_prompt = (
                f"TASK: {truncated_prompt[:6000]}\n\n"
                f"CHECKLIST:\n{checks_text}\n\n"
                f"RESPONSE:\n{retry_response}\n\n"
                f"Score each check as pass/fail. One check per line: STATUS|SEVERITY|SOURCE|Check description|Detail"
            )
            try:
                retry_result = await _generate_with_retry(
                    client,
                    prompt=retry_prompt,
                    model=eval_model,
                    system_prompt=PREGEN_SCORING_SYSTEM_PROMPT
                    + "\nYou MUST output ONLY pipe-delimited lines. No JSON, no markdown, no commentary.",
                    temperature=0.0,
                    max_tokens=16384,
                    timeout=120.0,
                )
                eval_text = retry_result.get("text", "")
                parsed = _parse_checklist_lines(eval_text, format="scoring")
            except Exception as retry_err:
                logger.warning("Pre-gen %s retry failed: %s", dimension, retry_err)

        if not parsed or "checks" not in parsed:
            # 2nd retry: Flash model fallback
            logger.warning(
                "Pre-gen %s scoring failed after retry, falling back to Flash",
                dimension,
            )
            flash_model = _get_eval_model()  # Flash
            FLASH_MAX = 15_000
            flash_response = response[:FLASH_MAX] if len(response) > FLASH_MAX else response
            flash_prompt = (
                f"TASK:\n{truncated_prompt[:4000]}\n\n"
                f"CHECKLIST:\n{checks_text}\n\n"
                f"RESPONSE:\n{flash_response}\n\n"
                f"Score each check as pass/fail. One check per line: STATUS|SEVERITY|SOURCE|Check description|Detail"
            )
            flash_system = (
                f"You are a {dimension} auditor. Score the given checklist against the response. "
                "Output ONLY pipe-delimited lines. No JSON, no markdown, no commentary. "
                "Format: STATUS|SEVERITY|SOURCE|Check description|Detail text"
            )
            try:
                flash_result = await _generate_with_retry(
                    client,
                    prompt=flash_prompt,
                    model=flash_model,
                    system_prompt=flash_system,
                    temperature=0.0,
                    max_tokens=8192,
                    timeout=90.0,
                )
                eval_text = flash_result.get("text", "")
                logger.info(
                    "Pre-gen %s Flash fallback (%d chars): %s",
                    dimension,
                    len(eval_text),
                    eval_text[:300],
                )
                parsed = _parse_checklist_lines(eval_text, format="scoring")
            except Exception as flash_err:
                logger.warning("Pre-gen %s Flash fallback failed: %s", dimension, flash_err)

        if not parsed or "checks" not in parsed:
            logger.warning("Pre-gen %s scoring failed after all retries", dimension)
            return EvalResult(score=0.5, parse_failed=True, **{dimension: 0.5})

        # Cap evaluator-added checks at 5
        evaluator_count = 0
        checks: List[AccuracyCheck] = []
        for raw in parsed["checks"]:
            src = raw.get("source", "pre_gen")
            if src == "evaluator":
                evaluator_count += 1
                if evaluator_count > 5:
                    continue
            checks.append(
                AccuracyCheck(
                    check=raw.get("check", "unknown"),
                    status=raw.get("status", "fail"),
                    severity=raw.get("severity", "breaking"),
                    detail=raw.get("detail", ""),
                )
            )

        score = _score_accuracy_checks(checks)

        issues = [
            f"[{dimension.upper()}:{c.severity.upper()}] {c.check}: {c.detail}"
            for c in checks
            if c.status == "fail"
        ]

        logger.info(
            "Pre-gen %s scoring: %d checks (%d pre-gen + %d evaluator), "
            "%d passed, %d failed, score=%.3f",
            dimension,
            len(checks),
            sum(1 for r in parsed["checks"] if r.get("source", "pre_gen") == "pre_gen"),
            min(evaluator_count, 5),
            sum(1 for c in checks if c.status == "pass"),
            sum(1 for c in checks if c.status == "fail"),
            score,
        )

        return EvalResult(
            score=score,
            issues=issues,
            **{dimension: score, f"{dimension}_checks": checks},
        )

    except Exception as e:
        logger.warning("Pre-gen %s scoring failed: %s", dimension, e)
        return EvalResult(score=0.5, parse_failed=True, **{dimension: 0.5})


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
    eval_model = model or _get_eval_model()

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
    parts.extend(
        [
            "=== RESPONSE TO EVALUATE ===\n",
            truncated_response,
            "\n=== END RESPONSE ===\n\n",
            "Now output your evaluation as a single JSON object. "
            "Do NOT output anything except the JSON:\n"
            '{"accuracy": N, "completeness": N, "clarity": N, "actionability": N, "issues": ["..."]}',
        ]
    )
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
            len(eval_text),
            finish,
            eval_text[:500],
        )

        # If the response looks like truncated JSON, try to repair it
        cleaned = eval_text.strip()
        if cleaned.startswith("{") and not cleaned.endswith("}"):
            logger.warning("Eval JSON appears truncated, attempting repair")
            last_quote = cleaned.rfind('"')
            if last_quote > 0:
                if '"issues"' in cleaned:
                    cleaned = cleaned[: last_quote + 1] + "]}"
                else:
                    cleaned = cleaned[: last_quote + 1] + "}"
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
