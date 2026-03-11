"""Goal-level validation for the Ralph autonomous loop.

Calls the deployed Gemini Intelligence Layer to:
1. Decompose a goal into verifiable criteria (once at task start)
2. Score task output against those criteria (on promise detection or per-iteration)
3. Track progress across iterations and detect stuck/regression

Plugs into RalphExecutor alongside TestRunner as a validation gate.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 60.0


@dataclass
class GoalCriterion:
    """A single verifiable criterion derived from the goal."""

    id: str
    description: str
    dimension: str  # functional, kpi, constraint
    severity: str  # breaking, important, nice_to_have


@dataclass
class CriterionResult:
    """Result of evaluating a single criterion."""

    criterion_id: str
    description: str
    status: str  # pass, fail
    severity: str
    detail: str = ""


@dataclass
class GoalValidationResult:
    """Result from validating output against goal criteria."""

    passed: bool
    score: float  # 0.0-1.0
    criteria_results: List[CriterionResult] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    @property
    def failing_criteria(self) -> List[CriterionResult]:
        return [r for r in self.criteria_results if r.status == "fail"]

    @property
    def breaking_failures(self) -> List[CriterionResult]:
        return [
            r
            for r in self.criteria_results
            if r.status == "fail" and r.severity == "breaking"
        ]


class GoalValidator:
    """Validates Ralph loop output against goal-level criteria.

    Usage:
        validator = GoalValidator(daemon_url="https://...", api_key="...")
        criteria = validator.initialize(goal_description="Build JWT auth", ...)
        # ... after each iteration or promise detection:
        result = validator.validate(output)
        if not result.passed:
            prompt = validator.build_failure_prompt(result)
    """

    def __init__(
        self,
        daemon_url: str,
        api_key: str = "",
        threshold: float = 0.8,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self.daemon_url = daemon_url.rstrip("/")
        self.api_key = api_key
        self.threshold = threshold
        self.timeout = timeout

        self._criteria: List[GoalCriterion] = []
        self._goal_description: str = ""
        self._progress_history: List[Dict] = []

        self._client = httpx.Client(timeout=timeout)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def initialize(
        self,
        goal_description: str,
        kpis: Optional[List[str]] = None,
        success_criteria: str = "",
        constraints: Optional[List[str]] = None,
    ) -> List[GoalCriterion]:
        """Decompose a goal into verifiable criteria via the daemon.

        Call once at task start. Stores criteria internally.
        """
        self._goal_description = goal_description

        payload = {
            "goal_description": goal_description,
            "kpis": kpis or [],
            "success_criteria": success_criteria,
            "constraints": constraints or [],
        }

        try:
            resp = self._client.post(
                f"{self.daemon_url}/api/gemini/goal-validate/checklist",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

            self._criteria = [
                GoalCriterion(
                    id=c["id"],
                    description=c["description"],
                    dimension=c.get("dimension", "functional"),
                    severity=c.get("severity", "breaking"),
                )
                for c in data.get("criteria", [])
            ]

            logger.info(
                "Goal decomposed into %d criteria in %dms",
                len(self._criteria),
                data.get("duration_ms", 0),
            )
            return self._criteria

        except Exception as e:
            logger.error("Goal checklist generation failed: %s", e)
            return []

    def validate(self, output: str, iteration: int = 0) -> GoalValidationResult:
        """Validate task output against stored goal criteria.

        Returns GoalValidationResult with pass/fail per criterion.
        """
        if not self._criteria:
            logger.warning("No criteria loaded, skipping goal validation")
            return GoalValidationResult(passed=True, score=1.0)

        payload = {
            "output": output,
            "criteria": [
                {
                    "id": c.id,
                    "description": c.description,
                    "dimension": c.dimension,
                    "severity": c.severity,
                }
                for c in self._criteria
            ],
            "goal_description": self._goal_description,
            "iteration": iteration,
        }

        try:
            resp = self._client.post(
                f"{self.daemon_url}/api/gemini/goal-validate/score",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

            result = GoalValidationResult(
                passed=data.get("passed", False),
                score=data.get("score", 0.0),
                criteria_results=[
                    CriterionResult(
                        criterion_id=r["criterion_id"],
                        description=r["description"],
                        status=r["status"],
                        severity=r["severity"],
                        detail=r.get("detail", ""),
                    )
                    for r in data.get("criteria_results", [])
                ],
                issues=data.get("issues", []),
            )

            # Track progress
            self._progress_history.append(
                {
                    "iteration": iteration,
                    "score": result.score,
                    "passed": result.passed,
                    "failing_count": len(result.failing_criteria),
                }
            )

            logger.info(
                "Goal validation: score=%.2f passed=%s (%d/%d criteria, iteration=%d)",
                result.score,
                result.passed,
                len(result.criteria_results) - len(result.failing_criteria),
                len(result.criteria_results),
                iteration,
            )

            return result

        except Exception as e:
            logger.error("Goal validation failed: %s", e)
            # On error, don't block — return passing to avoid false rejections
            return GoalValidationResult(passed=True, score=0.0)

    def build_failure_prompt(
        self,
        result: GoalValidationResult,
        task_description: str = "",
        context: str = "",
    ) -> str:
        """Build a prompt for the next iteration that addresses failing criteria."""
        parts = [
            "## Goal Validation Failed",
            "",
            "Your output was evaluated against the goal criteria and **did not pass**.",
            f"Score: {result.score:.2f} (threshold: {self.threshold})",
            "",
        ]

        # Group failures by severity
        breaking = [r for r in result.failing_criteria if r.severity == "breaking"]
        important = [r for r in result.failing_criteria if r.severity == "important"]
        nice = [r for r in result.failing_criteria if r.severity == "nice_to_have"]

        if breaking:
            parts.append("### MUST FIX (blocking)")
            for r in breaking:
                parts.append(f"- **{r.description}**: {r.detail}")
            parts.append("")

        if important:
            parts.append("### SHOULD FIX (important)")
            for r in important:
                parts.append(f"- {r.description}: {r.detail}")
            parts.append("")

        if nice:
            parts.append("### NICE TO HAVE")
            for r in nice:
                parts.append(f"- {r.description}: {r.detail}")
            parts.append("")

        # Add progress trend if available
        trend = self.detect_trend()
        if trend["trend"] != "unknown":
            parts.extend(
                [
                    "### Progress Trend",
                    f"- Trend: **{trend['trend']}**",
                    f"- Best score: {trend['best_score']:.2f}",
                    f"- Current: {trend['current_score']:.2f}",
                    "",
                ]
            )

        # Add original task context
        if task_description:
            parts.extend(["---", "", "## Original Task", "", task_description, ""])

        if context:
            parts.extend(["## Context", "", context, ""])

        # Completion reminder
        parts.extend(
            [
                "## Next Steps",
                "",
                "1. Address all MUST FIX items first",
                "2. Then address SHOULD FIX items",
                "3. Only output the completion promise when the goal is fully met",
            ]
        )

        return "\n".join(parts)

    def detect_trend(self, window: int = 3) -> Dict:
        """Analyze score trend across recent iterations.

        Returns dict with trend info:
        - trend: "improving" | "stuck" | "regressing" | "unknown"
        - best_score: float
        - current_score: float
        """
        if len(self._progress_history) < 2:
            return {
                "trend": "unknown",
                "best_score": self._progress_history[-1]["score"]
                if self._progress_history
                else 0.0,
                "current_score": self._progress_history[-1]["score"]
                if self._progress_history
                else 0.0,
            }

        recent = self._progress_history[-window:]
        scores = [h["score"] for h in recent]
        best = max(h["score"] for h in self._progress_history)
        current = scores[-1]

        # Stuck: all scores within 0.02 of each other
        score_range = max(scores) - min(scores)
        if score_range < 0.02 and len(scores) >= window:
            trend = "stuck"
        # Regressing: current score dropped > 0.1 from best
        elif best - current > 0.1:
            trend = "regressing"
        # Improving: latest > previous
        elif len(scores) >= 2 and scores[-1] > scores[-2]:
            trend = "improving"
        else:
            trend = "stuck"

        return {
            "trend": trend,
            "best_score": best,
            "current_score": current,
        }

    @property
    def criteria(self) -> List[GoalCriterion]:
        return list(self._criteria)

    @property
    def progress_history(self) -> List[Dict]:
        return list(self._progress_history)

    def close(self):
        self._client.close()
