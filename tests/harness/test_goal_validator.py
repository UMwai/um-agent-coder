"""Tests for GoalValidator and daemon goal_validate endpoint."""

from um_agent_coder.daemon.routes.gemini.goal_validate import (
    CriterionResultInfo,
    GoalCriterionInfo,
    _compute_score,
    _parse_checklist_lines,
    _parse_score_lines,
)
from um_agent_coder.harness.ralph.goal_validator import (
    CriterionResult,
    GoalValidationResult,
    GoalValidator,
)

# --- Daemon-side parsing tests ---


class TestChecklistParsing:
    def test_parse_valid_lines(self):
        text = (
            "functional|breaking|Login endpoint accepts email and returns JWT\n"
            "kpi|important|Response time under 200ms\n"
            "constraint|breaking|No external auth services\n"
        )
        criteria = _parse_checklist_lines(text)
        assert len(criteria) == 3
        assert criteria[0].id == "c-001"
        assert criteria[0].dimension == "functional"
        assert criteria[0].severity == "breaking"
        assert criteria[1].severity == "important"
        assert criteria[2].dimension == "constraint"

    def test_skip_invalid_lines(self):
        text = (
            "functional|breaking|Valid criterion\n"
            "# comment\n"
            "\n"
            "bad line no pipes\n"
            "kpi|important|Another valid one\n"
        )
        criteria = _parse_checklist_lines(text)
        assert len(criteria) == 2

    def test_fix_unknown_severity(self):
        text = "functional|unknown_severity|Some check\n"
        criteria = _parse_checklist_lines(text)
        assert len(criteria) == 1
        assert criteria[0].severity == "important"  # default fallback

    def test_skip_unknown_dimension(self):
        text = "bogus|breaking|Some check\n"
        criteria = _parse_checklist_lines(text)
        assert len(criteria) == 0


class TestScoreParsing:
    def setup_method(self):
        self.criteria = [
            GoalCriterionInfo(
                id="c-001",
                description="Login works",
                dimension="functional",
                severity="breaking",
            ),
            GoalCriterionInfo(
                id="c-002",
                description="Fast response",
                dimension="kpi",
                severity="important",
            ),
            GoalCriterionInfo(
                id="c-003",
                description="No external deps",
                dimension="constraint",
                severity="breaking",
            ),
        ]

    def test_parse_valid_scores(self):
        text = (
            "PASS|c-001|Found login endpoint\n"
            "FAIL|c-002|No benchmarks\n"
            "PASS|c-003|Self-contained\n"
        )
        results = _parse_score_lines(text, self.criteria)
        assert len(results) == 3
        assert results[0].status == "pass"
        assert results[1].status == "fail"

    def test_skip_unknown_criterion_id(self):
        text = "PASS|c-999|Unknown criterion\n"
        results = _parse_score_lines(text, self.criteria)
        assert len(results) == 0


class TestScoreComputation:
    def test_all_pass(self):
        results = [
            CriterionResultInfo(
                criterion_id="c-001",
                description="x",
                status="pass",
                severity="breaking",
            ),
            CriterionResultInfo(
                criterion_id="c-002",
                description="x",
                status="pass",
                severity="important",
            ),
        ]
        assert _compute_score(results) == 1.0

    def test_all_fail_breaking(self):
        results = [
            CriterionResultInfo(
                criterion_id="c-001",
                description="x",
                status="fail",
                severity="breaking",
            ),
        ]
        assert _compute_score(results) == 0.0

    def test_mixed(self):
        results = [
            CriterionResultInfo(
                criterion_id="c-001",
                description="x",
                status="pass",
                severity="breaking",
            ),
            CriterionResultInfo(
                criterion_id="c-002",
                description="x",
                status="fail",
                severity="important",
            ),
            CriterionResultInfo(
                criterion_id="c-003",
                description="x",
                status="pass",
                severity="breaking",
            ),
        ]
        # (1.0 + 0.5 + 1.0) / 3 = 0.833...
        score = _compute_score(results)
        assert abs(score - 0.833) < 0.01

    def test_empty(self):
        assert _compute_score([]) == 0.0


# --- Client-side GoalValidator tests ---


class TestGoalValidationResult:
    def test_failing_criteria(self):
        result = GoalValidationResult(
            passed=False,
            score=0.5,
            criteria_results=[
                CriterionResult(
                    criterion_id="c-001",
                    description="x",
                    status="pass",
                    severity="breaking",
                ),
                CriterionResult(
                    criterion_id="c-002",
                    description="y",
                    status="fail",
                    severity="breaking",
                ),
                CriterionResult(
                    criterion_id="c-003",
                    description="z",
                    status="fail",
                    severity="important",
                ),
            ],
        )
        assert len(result.failing_criteria) == 2
        assert len(result.breaking_failures) == 1


class TestGoalValidatorTrend:
    def test_trend_improving(self):
        validator = GoalValidator(daemon_url="http://localhost:8080")
        validator._progress_history = [
            {"iteration": 1, "score": 0.3, "passed": False, "failing_count": 5},
            {"iteration": 2, "score": 0.5, "passed": False, "failing_count": 3},
            {"iteration": 3, "score": 0.7, "passed": False, "failing_count": 1},
        ]
        trend = validator.detect_trend()
        assert trend["trend"] == "improving"
        assert trend["best_score"] == 0.7

    def test_trend_stuck(self):
        validator = GoalValidator(daemon_url="http://localhost:8080")
        validator._progress_history = [
            {"iteration": 1, "score": 0.6, "passed": False, "failing_count": 2},
            {"iteration": 2, "score": 0.61, "passed": False, "failing_count": 2},
            {"iteration": 3, "score": 0.6, "passed": False, "failing_count": 2},
        ]
        trend = validator.detect_trend()
        assert trend["trend"] == "stuck"

    def test_trend_regressing(self):
        validator = GoalValidator(daemon_url="http://localhost:8080")
        validator._progress_history = [
            {"iteration": 1, "score": 0.8, "passed": False, "failing_count": 1},
            {"iteration": 2, "score": 0.9, "passed": False, "failing_count": 0},
            {"iteration": 3, "score": 0.7, "passed": False, "failing_count": 2},
        ]
        trend = validator.detect_trend()
        assert trend["trend"] == "regressing"

    def test_trend_unknown_single(self):
        validator = GoalValidator(daemon_url="http://localhost:8080")
        validator._progress_history = [
            {"iteration": 1, "score": 0.5, "passed": False, "failing_count": 3},
        ]
        trend = validator.detect_trend()
        assert trend["trend"] == "unknown"


class TestGoalValidatorFailurePrompt:
    def test_build_failure_prompt(self):
        validator = GoalValidator(daemon_url="http://localhost:8080")
        result = GoalValidationResult(
            passed=False,
            score=0.5,
            criteria_results=[
                CriterionResult(
                    criterion_id="c-001",
                    description="Login endpoint",
                    status="fail",
                    severity="breaking",
                    detail="No login endpoint found",
                ),
                CriterionResult(
                    criterion_id="c-002",
                    description="Tests pass",
                    status="pass",
                    severity="important",
                ),
            ],
        )
        prompt = validator.build_failure_prompt(result, task_description="Build auth")
        assert "Goal Validation Failed" in prompt
        assert "MUST FIX" in prompt
        assert "Login endpoint" in prompt
        assert "Build auth" in prompt
