"""Goal validation endpoints for autonomous loop integration.

POST /api/gemini/goal-validate/checklist — Decompose a goal into verifiable criteria
POST /api/gemini/goal-validate/score    — Score task output against goal criteria
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from um_agent_coder.daemon.auth import verify_api_key

from ._evaluator import _generate_with_retry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/goal-validate")


# --- Request / Response Models ---


class GoalCriterionInfo(BaseModel):
    id: str = ""
    description: str
    dimension: str = "functional"  # functional, kpi, constraint
    severity: str = "breaking"  # breaking, important, nice_to_have


class CriterionResultInfo(BaseModel):
    criterion_id: str
    description: str
    status: str  # "pass" or "fail"
    severity: str
    detail: str = ""


class GoalChecklistRequest(BaseModel):
    goal_description: str = Field(..., min_length=1)
    kpis: List[str] = Field(default_factory=list)
    success_criteria: str = ""
    constraints: List[str] = Field(default_factory=list)


class GoalChecklistResponse(BaseModel):
    id: str
    criteria: List[GoalCriterionInfo]
    duration_ms: int


class GoalScoreRequest(BaseModel):
    output: str = Field(..., min_length=1)
    criteria: List[GoalCriterionInfo]
    goal_description: str = ""
    iteration: int = 0


class GoalScoreResponse(BaseModel):
    id: str
    passed: bool
    score: float
    criteria_results: List[CriterionResultInfo]
    issues: List[str]
    duration_ms: int


# --- System Prompts ---


CHECKLIST_SYSTEM_PROMPT = """\
You are a goal decomposition engine. Given a high-level goal with optional KPIs, \
success criteria, and constraints, produce a checklist of verifiable criteria that \
can be checked against task output.

DIMENSIONS:
- "functional": Core functionality that must work (features, endpoints, logic)
- "kpi": Measurable outcomes (performance targets, coverage thresholds)
- "constraint": Hard constraints that must not be violated (no external deps, must be async, etc.)

SEVERITY LEVELS:
- "breaking": Must pass or the goal is not met. Score: 0 points.
- "important": Should pass for quality. Score: 0.5 points.
- "nice_to_have": Ideal but not required. Score: 0.75 points.

RULES:
- Generate 5-15 criteria total
- Each criterion must be independently verifiable from task output
- Be specific: "JWT auth endpoint returns 401 on invalid token" not "auth works"
- Derive criteria ONLY from the provided goal, KPIs, and constraints
- Do NOT invent requirements not implied by the goal

Output one criterion per line in pipe-delimited format (NO JSON, no markdown):
DIMENSION|SEVERITY|Criterion description

Example:
functional|breaking|Login endpoint accepts email+password and returns JWT token
kpi|important|API response time under 200ms for auth endpoints
constraint|breaking|No external authentication services used (self-contained JWT)
"""

SCORING_SYSTEM_PROMPT = """\
You are a goal validation auditor. Given task output and a checklist of criteria, \
determine whether each criterion is met.

For each criterion, output a line in this pipe-delimited format:
STATUS|CRITERION_ID|Detail explanation

STATUS is "PASS" or "FAIL".
CRITERION_ID matches the id field of the criterion.
Detail is a brief explanation of why it passed or failed.

Be strict but fair:
- PASS only if the output clearly satisfies the criterion
- FAIL if the criterion is not addressed, partially addressed, or incorrectly implemented
- If the output doesn't contain enough info to judge, FAIL with explanation

Output one line per criterion. No JSON, no markdown, no preamble.

Example:
PASS|c-001|Login endpoint implementation present with email+password validation
FAIL|c-002|No performance benchmarks or response time measurements in output
PASS|c-003|JWT generation uses built-in jsonwebtoken, no external auth service
"""


# --- Helpers ---


def _get_client():
    from um_agent_coder.daemon.app import get_gemini_client

    return get_gemini_client()


def _get_eval_model() -> str:
    try:
        from um_agent_coder.daemon.app import get_settings

        return get_settings().gemini_model_flash
    except Exception:
        return "gemini-3-flash-preview"


def _parse_checklist_lines(text: str) -> List[GoalCriterionInfo]:
    """Parse pipe-delimited checklist lines into GoalCriterionInfo objects."""
    criteria = []
    counter = 0
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        dimension = parts[0].strip().lower()
        severity = parts[1].strip().lower()
        description = parts[2].strip()

        if dimension not in ("functional", "kpi", "constraint"):
            continue
        if severity not in ("breaking", "important", "nice_to_have"):
            severity = "important"

        counter += 1
        criteria.append(
            GoalCriterionInfo(
                id=f"c-{counter:03d}",
                description=description,
                dimension=dimension,
                severity=severity,
            )
        )
    return criteria


def _parse_score_lines(
    text: str,
    criteria: List[GoalCriterionInfo],
) -> List[CriterionResultInfo]:
    """Parse pipe-delimited score lines into CriterionResultInfo objects."""
    results = []
    criteria_by_id = {c.id: c for c in criteria}

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 2)
        if len(parts) < 2:
            continue

        status = parts[0].strip().upper()
        criterion_id = parts[1].strip()
        detail = parts[2].strip() if len(parts) > 2 else ""

        if status not in ("PASS", "FAIL"):
            continue

        criterion = criteria_by_id.get(criterion_id)
        if not criterion:
            continue

        results.append(
            CriterionResultInfo(
                criterion_id=criterion_id,
                description=criterion.description,
                status=status.lower(),
                severity=criterion.severity,
                detail=detail,
            )
        )

    return results


SEVERITY_WEIGHTS = {
    "breaking": 0.0,
    "important": 0.5,
    "nice_to_have": 0.75,
}


def _compute_score(results: List[CriterionResultInfo]) -> float:
    """Compute weighted score from criterion results."""
    if not results:
        return 0.0
    total = 0.0
    for r in results:
        if r.status == "pass":
            total += 1.0
        else:
            total += SEVERITY_WEIGHTS.get(r.severity, 0.0)
    return total / len(results)


# --- Endpoints ---


@router.post("/checklist", response_model=GoalChecklistResponse)
async def generate_goal_checklist(
    req: GoalChecklistRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Decompose a goal into verifiable criteria."""
    client = _get_client()
    query_id = f"gc-{uuid.uuid4().hex[:8]}"
    start = time.monotonic()

    # Build user prompt from goal components
    prompt_parts = [f"## GOAL\n{req.goal_description}"]
    if req.success_criteria:
        prompt_parts.append(f"\n## SUCCESS CRITERIA\n{req.success_criteria}")
    if req.kpis:
        prompt_parts.append("\n## KPIs\n" + "\n".join(f"- {k}" for k in req.kpis))
    if req.constraints:
        prompt_parts.append("\n## CONSTRAINTS\n" + "\n".join(f"- {c}" for c in req.constraints))
    prompt_parts.append("\nGenerate the evaluation checklist.")

    user_prompt = "\n".join(prompt_parts)

    result = await _generate_with_retry(
        client,
        prompt=user_prompt,
        system_prompt=CHECKLIST_SYSTEM_PROMPT,
        model=_get_eval_model(),
        temperature=0.2,
        max_tokens=4096,
    )

    raw_text = result.get("text", "")
    criteria = _parse_checklist_lines(raw_text)

    duration_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "Goal checklist %s: %d criteria in %dms",
        query_id,
        len(criteria),
        duration_ms,
    )

    return GoalChecklistResponse(
        id=query_id,
        criteria=criteria,
        duration_ms=duration_ms,
    )


@router.post("/score", response_model=GoalScoreResponse)
async def score_goal_output(
    req: GoalScoreRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Score task output against goal criteria."""
    client = _get_client()
    query_id = f"gs-{uuid.uuid4().hex[:8]}"
    start = time.monotonic()

    # Build criteria text for the prompt
    criteria_text = ""
    for c in req.criteria:
        criteria_text += f"- {c.id} [{c.dimension}|{c.severity}]: {c.description}\n"

    # Truncate output to avoid exceeding context
    max_output = 60_000
    output_text = req.output
    if len(output_text) > max_output:
        half = max_output // 2
        output_text = (
            output_text[:half]
            + f"\n[... {len(req.output) - max_output} chars omitted ...]\n"
            + output_text[-half:]
        )

    user_prompt = (
        f"## GOAL\n{req.goal_description}\n\n"
        f"## CRITERIA TO CHECK\n{criteria_text}\n"
        f"## TASK OUTPUT\n{output_text}\n\n"
        f"Score each criterion as PASS or FAIL."
    )

    result = await _generate_with_retry(
        client,
        prompt=user_prompt,
        system_prompt=SCORING_SYSTEM_PROMPT,
        model=_get_eval_model(),
        temperature=0.1,
        max_tokens=8192,
    )

    raw_text = result.get("text", "")
    criteria_results = _parse_score_lines(raw_text, req.criteria)

    # Fill in any criteria that weren't scored (assume fail)
    scored_ids = {r.criterion_id for r in criteria_results}
    for c in req.criteria:
        if c.id not in scored_ids:
            criteria_results.append(
                CriterionResultInfo(
                    criterion_id=c.id,
                    description=c.description,
                    status="fail",
                    severity=c.severity,
                    detail="Not evaluated by scorer",
                )
            )

    score = _compute_score(criteria_results)
    issues = [
        f"[{r.severity}] {r.description}: {r.detail}"
        for r in criteria_results
        if r.status == "fail"
    ]
    # Passed = no breaking failures and score >= 0.8
    has_breaking_failure = any(
        r.status == "fail" and r.severity == "breaking" for r in criteria_results
    )
    passed = not has_breaking_failure and score >= 0.8

    duration_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "Goal score %s: score=%.2f passed=%s (%d/%d criteria) in %dms",
        query_id,
        score,
        passed,
        sum(1 for r in criteria_results if r.status == "pass"),
        len(criteria_results),
        duration_ms,
    )

    return GoalScoreResponse(
        id=query_id,
        passed=passed,
        score=score,
        criteria_results=criteria_results,
        issues=issues,
        duration_ms=duration_ms,
    )
