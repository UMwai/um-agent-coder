"""Tests for Ralph intelligent loop improvements.

Tests:
1. Iteration tracker - oscillation detection, score trajectory, scoring schedule
2. Strategies - strategy selection and prompt building
3. Persistence - eval_score column migration and round-trip
4. Roadmap parser - new intelligent loop properties
"""

import os
import sys
import tempfile
import unittest

sys.path.append(os.path.join(os.getcwd(), "src"))

from um_agent_coder.harness.ralph.iteration_tracker import (
    IterationTracker,
)
from um_agent_coder.harness.ralph.persistence import RalphPersistence
from um_agent_coder.harness.ralph.strategies import (
    RalphStrategy,
    build_strategic_prompt,
    select_strategies,
)
from um_agent_coder.harness.roadmap_parser import RoadmapParser

# ---------------------------------------------------------------------------
# IterationTracker tests
# ---------------------------------------------------------------------------


class TestOscillationDetection(unittest.TestCase):
    """Test the oscillation detection feature."""

    def _tracker_with_scores(self, scores: list[float]) -> IterationTracker:
        tracker = IterationTracker(task_id="t-001", max_iterations=100)
        for i, score in enumerate(scores, 1):
            tracker.start_iteration()
            tracker.end_iteration(output=f"iter-{i}", promise_found=False)
            tracker.iteration_history[-1].eval_score = score
        return tracker

    def test_no_oscillation_insufficient_data(self):
        tracker = self._tracker_with_scores([0.5, 0.6])
        result = tracker.detect_oscillation(window=4, spread=0.03)
        self.assertFalse(result["oscillating"])
        self.assertEqual(result["suggestion"], "continue")

    def test_oscillation_detected_low_mean(self):
        tracker = self._tracker_with_scores([0.40, 0.41, 0.40, 0.42])
        result = tracker.detect_oscillation(window=4, spread=0.03)
        self.assertTrue(result["oscillating"])
        self.assertEqual(result["suggestion"], "escalate_model")

    def test_oscillation_detected_mid_mean(self):
        tracker = self._tracker_with_scores([0.70, 0.71, 0.72, 0.70])
        result = tracker.detect_oscillation(window=4, spread=0.03)
        self.assertTrue(result["oscillating"])
        self.assertEqual(result["suggestion"], "mutate_prompt")

    def test_no_oscillation_large_spread(self):
        tracker = self._tracker_with_scores([0.40, 0.50, 0.60, 0.70])
        result = tracker.detect_oscillation(window=4, spread=0.03)
        self.assertFalse(result["oscillating"])

    def test_oscillation_high_mean_continues(self):
        tracker = self._tracker_with_scores([0.90, 0.91, 0.90, 0.91])
        result = tracker.detect_oscillation(window=4, spread=0.03)
        self.assertTrue(result["oscillating"])
        self.assertEqual(result["suggestion"], "continue")


class TestScoreTrajectory(unittest.TestCase):
    """Test the score trajectory feature."""

    def _tracker_with_scores(self, scores: list[float]) -> IterationTracker:
        tracker = IterationTracker(task_id="t-002", max_iterations=100)
        for i, score in enumerate(scores, 1):
            tracker.start_iteration()
            tracker.end_iteration(output=f"iter-{i}", promise_found=False)
            tracker.iteration_history[-1].eval_score = score
        return tracker

    def test_insufficient_data(self):
        tracker = self._tracker_with_scores([0.5])
        result = tracker.get_score_trajectory()
        self.assertEqual(result["trend"], "insufficient_data")
        self.assertEqual(result["best_score"], 0.5)

    def test_improving_trend(self):
        tracker = self._tracker_with_scores([0.3, 0.5, 0.6, 0.7, 0.8])
        result = tracker.get_score_trajectory()
        self.assertEqual(result["trend"], "improving")
        self.assertEqual(result["best_score"], 0.8)
        self.assertGreater(result["improvement_rate"], 0)

    def test_declining_trend(self):
        tracker = self._tracker_with_scores([0.8, 0.7, 0.6, 0.5, 0.4])
        result = tracker.get_score_trajectory()
        self.assertEqual(result["trend"], "declining")
        self.assertEqual(result["best_score"], 0.8)

    def test_flat_trend(self):
        tracker = self._tracker_with_scores([0.5, 0.6, 0.5, 0.6, 0.5])
        result = tracker.get_score_trajectory()
        self.assertEqual(result["trend"], "flat")


class TestScoringSchedule(unittest.TestCase):
    """Test should_score_this_iteration."""

    def test_always_scores_first_iteration(self):
        tracker = IterationTracker(task_id="t-003", max_iterations=100)
        tracker.start_iteration()
        self.assertTrue(tracker.should_score_this_iteration(scoring_interval=3))

    def test_scores_on_interval(self):
        tracker = IterationTracker(task_id="t-003", max_iterations=100)
        for _i in range(6):
            tracker.start_iteration()
            tracker.end_iteration(output="x", promise_found=False)

        # At iteration 6, should score (6 % 3 == 0)
        self.assertTrue(tracker.should_score_this_iteration(scoring_interval=3))

    def test_skips_between_intervals(self):
        tracker = IterationTracker(task_id="t-003", max_iterations=100)
        for _i in range(4):
            tracker.start_iteration()
            tracker.end_iteration(output="x", promise_found=False)

        # At iteration 4, should not score (4 % 3 != 0)
        self.assertFalse(tracker.should_score_this_iteration(scoring_interval=3))

    def test_scores_after_promise(self):
        tracker = IterationTracker(task_id="t-003", max_iterations=100)
        tracker.start_iteration()
        tracker.end_iteration(output="x", promise_found=True)
        # Reset completed state for testing
        tracker._completed = False

        tracker.start_iteration()
        # Previous iteration had promise_found=True, so should score
        self.assertTrue(tracker.should_score_this_iteration(scoring_interval=10))


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------


class TestStrategySelection(unittest.TestCase):
    """Test strategy selection from goal validation results."""

    def _make_result(self, failing_criteria):
        """Build a minimal GoalValidationResult-like mock."""
        from unittest.mock import MagicMock

        result = MagicMock()
        result.failing_criteria = failing_criteria
        result.criteria_results = failing_criteria  # simplified
        result.score = 0.5
        result.issues = []
        return result

    def _make_criterion(self, cid, description, severity="breaking", detail=""):
        from unittest.mock import MagicMock

        c = MagicMock()
        c.criterion_id = cid
        c.description = description
        c.severity = severity
        c.detail = detail
        return c

    def _make_goal_criterion(self, cid, dimension):
        from unittest.mock import MagicMock

        gc = MagicMock()
        gc.id = cid
        gc.dimension = dimension
        return gc

    def test_no_failures_returns_empty(self):
        result = self._make_result([])
        strategies = select_strategies(result)
        self.assertEqual(len(strategies), 0)

    def test_groups_by_dimension(self):
        f1 = self._make_criterion("c1", "Missing login", "breaking")
        f2 = self._make_criterion("c2", "High latency", "important")

        criteria = [
            self._make_goal_criterion("c1", "functional"),
            self._make_goal_criterion("c2", "kpi"),
        ]

        result = self._make_result([f1, f2])
        strategies = select_strategies(result, criteria=criteria)

        self.assertEqual(len(strategies), 2)
        dims = {s.dimension for s in strategies}
        self.assertEqual(dims, {"functional", "kpi"})

    def test_caps_at_max_strategies(self):
        failures = [self._make_criterion(f"c{i}", f"Failure {i}", "important") for i in range(10)]
        criteria = [self._make_goal_criterion(f"c{i}", f"dim{i}") for i in range(10)]

        result = self._make_result(failures)
        strategies = select_strategies(result, max_strategies=2, criteria=criteria)
        self.assertLessEqual(len(strategies), 2)


class TestStrategicPromptBuilder(unittest.TestCase):
    """Test the strategic prompt builder."""

    def test_builds_prompt_with_all_sections(self):
        from unittest.mock import MagicMock

        result = MagicMock()
        result.score = 0.65
        result.criteria_results = [MagicMock() for _ in range(5)]
        result.failing_criteria = [
            MagicMock(severity="breaking", description="Missing auth", detail="no JWT"),
            MagicMock(severity="important", description="Slow API", detail=""),
        ]
        result.issues = ["Issue 1", "Issue 2"]

        strategy = RalphStrategy(
            name="functional_fix",
            dimension="functional",
            prompt_section="Fix the auth module.",
            priority=0,
        )

        prompt = build_strategic_prompt(
            task_description="Build auth system",
            success_criteria="JWT auth works",
            context="Node.js project",
            result=result,
            strategies=[strategy],
            previous_output="const app = express();",
            completion_promise="AUTH_DONE",
            trend_info={"trend": "improving", "best_score": 0.7, "current_score": 0.65},
        )

        self.assertIn("Score: 0.65", prompt)
        self.assertIn("MUST FIX", prompt)
        self.assertIn("Missing auth", prompt)
        self.assertIn("Fix Instructions", prompt)
        self.assertIn("AUTH_DONE", prompt)
        self.assertIn("improving", prompt)
        self.assertIn("Build auth system", prompt)


# ---------------------------------------------------------------------------
# Persistence eval_score tests
# ---------------------------------------------------------------------------


class TestPersistenceEvalScore(unittest.TestCase):
    """Test that eval_score is persisted and loaded correctly."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "ralph_state.db")
        self.persistence = RalphPersistence(db_path=self.db_path)

    def test_eval_score_round_trip(self):
        tracker = IterationTracker(task_id="t-eval", max_iterations=10)
        tracker.start_iteration()
        tracker.end_iteration(output="hello", promise_found=False)
        tracker.iteration_history[-1].eval_score = 0.75

        tracker.start_iteration()
        tracker.end_iteration(output="world", promise_found=False)
        tracker.iteration_history[-1].eval_score = 0.82

        self.persistence.save_tracker(tracker)

        loaded = self.persistence.load_tracker("t-eval")
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded.iteration_history), 2)
        self.assertAlmostEqual(loaded.iteration_history[0].eval_score, 0.75)
        self.assertAlmostEqual(loaded.iteration_history[1].eval_score, 0.82)

    def test_eval_score_none_when_not_set(self):
        tracker = IterationTracker(task_id="t-none", max_iterations=10)
        tracker.start_iteration()
        tracker.end_iteration(output="no score", promise_found=False)

        self.persistence.save_tracker(tracker)

        loaded = self.persistence.load_tracker("t-none")
        self.assertIsNone(loaded.iteration_history[0].eval_score)

    def test_get_iteration_history_includes_eval_score(self):
        tracker = IterationTracker(task_id="t-hist", max_iterations=10)
        tracker.start_iteration()
        tracker.end_iteration(output="scored", promise_found=False)
        tracker.iteration_history[-1].eval_score = 0.91

        self.persistence.save_tracker(tracker)

        history = self.persistence.get_iteration_history("t-hist")
        self.assertEqual(len(history), 1)
        self.assertAlmostEqual(history[0].eval_score, 0.91)


# ---------------------------------------------------------------------------
# Roadmap parser intelligent loop properties
# ---------------------------------------------------------------------------


class TestRoadmapParserIntelligentLoop(unittest.TestCase):
    """Test that roadmap parser handles intelligent loop properties."""

    def test_parses_intelligent_loop_properties(self):
        roadmap_content = """# Roadmap: Test

## Objective
Test intelligent loop

## Constraints
- Max time per task: 15 min

## Tasks

### Phase 1: Build

- [ ] **task-001**: Build feature X. Output <promise>DONE</promise> when complete.
  - ralph: true
  - max_iterations: 50
  - completion_promise: DONE
  - scoring_interval: 5
  - inject_checklist: false
  - oscillation_detection: true
  - timeout: 30min
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(roadmap_content)
            f.flush()

            parser = RoadmapParser(f.name)
            roadmap = parser.parse()

        os.unlink(f.name)

        self.assertEqual(len(roadmap.phases), 1)
        task = roadmap.phases[0].tasks[0]
        self.assertIsNotNone(task.ralph_config)
        self.assertEqual(task.ralph_config.scoring_interval, 5)
        self.assertFalse(task.ralph_config.inject_checklist)
        self.assertTrue(task.ralph_config.enable_oscillation_detection)

    def test_defaults_when_not_specified(self):
        roadmap_content = """# Roadmap: Test

## Objective
Test defaults

## Tasks

### Phase 1: Build

- [ ] **task-002**: Simple ralph task
  - ralph: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(roadmap_content)
            f.flush()

            parser = RoadmapParser(f.name)
            roadmap = parser.parse()

        os.unlink(f.name)

        task = roadmap.phases[0].tasks[0]
        self.assertIsNotNone(task.ralph_config)
        self.assertEqual(task.ralph_config.scoring_interval, 3)
        self.assertTrue(task.ralph_config.inject_checklist)
        self.assertTrue(task.ralph_config.enable_oscillation_detection)


if __name__ == "__main__":
    unittest.main()
