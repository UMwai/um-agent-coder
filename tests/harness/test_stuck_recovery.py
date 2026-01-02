"""Tests for stuck recovery system."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from um_agent_coder.harness.autonomous.recovery.stuck_detector import (
    ProgressRecord,
    StuckDetector,
    StuckState,
)
from um_agent_coder.harness.autonomous.recovery.prompt_mutator import (
    MutationResult,
    MutationType,
    PromptMutator,
)
from um_agent_coder.harness.autonomous.recovery.model_escalator import (
    ESCALATION_ORDER,
    EscalationResult,
    ModelEscalator,
)
from um_agent_coder.harness.autonomous.recovery.branch_explorer import (
    BranchApproach,
    BranchExplorer,
    ExplorationBranch,
)
from um_agent_coder.harness.autonomous.recovery.recovery_manager import (
    RecoveryManager,
    RecoveryResult,
    RecoveryStrategy,
)


class TestStuckDetector:
    """Tests for StuckDetector."""

    def test_initial_state(self):
        """Test initial state is progressing."""
        detector = StuckDetector()
        assert detector.current_state == StuckState.PROGRESSING
        assert detector.consecutive_no_progress == 0
        assert not detector.is_stuck()

    def test_progress_resets_counter(self):
        """Test that progress resets the consecutive counter."""
        detector = StuckDetector()

        # No progress for 2 iterations
        detector.record_iteration(1, 0.1, False)
        detector.record_iteration(2, 0.1, False)
        assert detector.consecutive_no_progress == 2

        # Progress resets counter
        detector.record_iteration(3, 0.5, True)
        assert detector.consecutive_no_progress == 0
        assert detector.current_state == StuckState.PROGRESSING

    def test_warning_state(self):
        """Test warning state at warning threshold."""
        detector = StuckDetector(warning_threshold=2, stuck_threshold=4)

        detector.record_iteration(1, 0.1, False)
        assert detector.current_state == StuckState.PROGRESSING

        detector.record_iteration(2, 0.1, False)
        assert detector.current_state == StuckState.WARNING
        assert detector.is_warning()
        assert not detector.is_stuck()

    def test_stuck_state(self):
        """Test stuck state at stuck threshold."""
        detector = StuckDetector(stuck_threshold=3)

        detector.record_iteration(1, 0.1, False)
        detector.record_iteration(2, 0.1, False)
        assert not detector.is_stuck()

        detector.record_iteration(3, 0.1, False)
        assert detector.is_stuck()
        assert detector.needs_recovery()

    def test_recovery_budget(self):
        """Test recovery budget management."""
        detector = StuckDetector(recovery_budget=5)

        assert detector.recovery_budget_remaining() == 5

        assert detector.use_recovery_iteration()
        assert detector.recovery_budget_remaining() == 4

        # Use remaining budget
        for _ in range(4):
            assert detector.use_recovery_iteration()

        # Budget exhausted
        assert not detector.use_recovery_iteration()
        assert detector.recovery_budget_remaining() == 0

    def test_recovery_lifecycle(self):
        """Test recovery start and end."""
        detector = StuckDetector(stuck_threshold=2)

        detector.record_iteration(1, 0.1, False)
        detector.record_iteration(2, 0.1, False)
        assert detector.is_stuck()

        detector.start_recovery()
        assert detector.current_state == StuckState.RECOVERING

        detector.end_recovery(True)
        assert detector.current_state == StuckState.PROGRESSING
        assert detector.consecutive_no_progress == 0

    def test_recovery_failure(self):
        """Test recovery failure keeps stuck state."""
        detector = StuckDetector(stuck_threshold=2)

        detector.record_iteration(1, 0.1, False)
        detector.record_iteration(2, 0.1, False)

        detector.start_recovery()
        detector.end_recovery(False)
        assert detector.current_state == StuckState.STUCK

    def test_serialization(self):
        """Test serialization and deserialization."""
        detector = StuckDetector(stuck_threshold=5, recovery_budget=10)
        detector.record_iteration(1, 0.3, False)
        detector.record_iteration(2, 0.5, True)
        detector.use_recovery_iteration()

        data = detector.to_dict()
        restored = StuckDetector.from_dict(data)

        assert restored.stuck_threshold == 5
        assert restored.recovery_budget == 10
        assert restored.recovery_iterations_used == 1
        assert len(restored.history) == 2

    def test_get_summary(self):
        """Test summary generation."""
        detector = StuckDetector()
        detector.record_iteration(1, 0.3, True)
        detector.record_iteration(2, 0.1, False)

        summary = detector.get_summary()
        assert "current_state" in summary
        assert "consecutive_no_progress" in summary
        assert "recent_scores" in summary
        assert len(summary["recent_scores"]) == 2

    def test_reset(self):
        """Test reset clears all state."""
        detector = StuckDetector()
        detector.record_iteration(1, 0.1, False)
        detector.record_iteration(2, 0.1, False)
        detector.record_iteration(3, 0.1, False)
        detector.use_recovery_iteration()

        detector.reset()

        assert detector.consecutive_no_progress == 0
        assert detector.total_no_progress == 0
        assert len(detector.history) == 0
        assert detector.recovery_iterations_used == 0


class TestPromptMutator:
    """Tests for PromptMutator."""

    def test_rephrase_mutation(self):
        """Test rephrase mutation."""
        mutator = PromptMutator()
        result = mutator.mutate("Implement a login system", MutationType.REPHRASE)

        assert result.success
        assert result.mutation_type == MutationType.REPHRASE
        assert result.original_prompt == "Implement a login system"
        assert "Reword" in result.mutated_prompt or "rephras" in result.mutated_prompt.lower()

    def test_decompose_mutation(self):
        """Test decompose mutation."""
        mutator = PromptMutator()
        result = mutator.mutate("Build a web API", MutationType.DECOMPOSE)

        assert result.success
        assert "Step" in result.mutated_prompt or "step" in result.mutated_prompt.lower()

    def test_constrain_mutation(self):
        """Test constrain mutation."""
        mutator = PromptMutator()
        result = mutator.mutate("Add authentication", MutationType.CONSTRAIN)

        assert result.success
        assert "Constrain" in result.mutated_prompt or "Focus" in result.mutated_prompt

    def test_mutation_sequence(self):
        """Test getting mutation sequence."""
        mutator = PromptMutator()
        sequence = mutator.get_mutation_sequence()

        assert len(sequence) == 5
        assert MutationType.REPHRASE in sequence
        assert MutationType.DECOMPOSE in sequence

    def test_mutate_with_fallback(self):
        """Test fallback through mutations."""
        mutator = PromptMutator()

        # First call returns first mutation
        result1 = mutator.mutate_with_fallback("Test task", [])
        assert result1 is not None
        assert result1.mutation_type == MutationType.REPHRASE

        # Second call skips already tried
        result2 = mutator.mutate_with_fallback("Test task", [MutationType.REPHRASE])
        assert result2 is not None
        assert result2.mutation_type == MutationType.DECOMPOSE

    def test_mutate_with_fallback_exhausted(self):
        """Test fallback when all mutations exhausted."""
        mutator = PromptMutator()
        all_types = list(MutationType)

        result = mutator.mutate_with_fallback("Test", all_types)
        assert result is None

    def test_mutation_with_context(self):
        """Test mutation includes context."""
        mutator = PromptMutator()
        result = mutator.mutate(
            "Implement feature",
            MutationType.REPHRASE,
            context="Previous attempt failed because of X",
        )
        assert result.success


class TestModelEscalator:
    """Tests for ModelEscalator."""

    def test_default_escalation_order(self):
        """Test default escalation order exists."""
        assert len(ESCALATION_ORDER) > 0

        # Verify capability scores are increasing
        scores = [score for _, _, score in ESCALATION_ORDER]
        assert scores == sorted(scores)

    def test_can_escalate_from_start(self):
        """Test can escalate from cheapest model."""
        escalator = ModelEscalator()
        assert escalator.can_escalate("gemini", "gemini-3-flash")

    def test_cannot_escalate_from_max(self):
        """Test cannot escalate from most capable model."""
        escalator = ModelEscalator()
        assert not escalator.can_escalate("claude", "claude-opus-4.5")

    def test_escalate_step(self):
        """Test single escalation step."""
        escalator = ModelEscalator()
        result = escalator.escalate("gemini", "gemini-3-flash")

        assert result is not None
        assert result.previous_cli == "gemini"
        assert result.previous_model == "gemini-3-flash"
        assert result.new_cli == "gemini"
        assert result.new_model == "gemini-3-pro"
        assert result.capability_increase > 0

    def test_escalate_from_max_returns_none(self):
        """Test escalation from max returns None."""
        escalator = ModelEscalator()
        result = escalator.escalate("claude", "claude-opus-4.5")
        assert result is None

    def test_escalate_unknown_model(self):
        """Test escalation from unknown model starts at beginning."""
        escalator = ModelEscalator()
        result = escalator.escalate("unknown", "unknown-model")

        assert result is not None
        assert result.new_cli == "gemini"
        assert result.new_model == "gemini-3-flash"

    def test_enabled_clis_filter(self):
        """Test filtering by enabled CLIs."""
        escalator = ModelEscalator(enabled_clis=["codex", "claude"])

        # Gemini should be filtered out
        result = escalator.escalate("codex", "gpt-5.2")
        assert result is not None
        assert result.new_cli != "gemini"

    def test_get_smartest(self):
        """Test getting smartest model."""
        escalator = ModelEscalator()
        cli, model = escalator.get_smartest()
        assert cli == "claude"
        assert model == "claude-opus-4.5"

    def test_get_cheapest(self):
        """Test getting cheapest model."""
        escalator = ModelEscalator()
        cli, model = escalator.get_cheapest()
        assert cli == "gemini"
        assert model == "gemini-3-flash"

    def test_get_capability_score(self):
        """Test getting capability score."""
        escalator = ModelEscalator()
        score = escalator.get_capability_score("claude", "claude-opus-4.5")
        assert score == 1.0

        score = escalator.get_capability_score("unknown", "model")
        assert score == 0.0

    def test_is_at_max_capability(self):
        """Test max capability check."""
        escalator = ModelEscalator()
        assert escalator.is_at_max_capability("claude", "claude-opus-4.5")
        assert not escalator.is_at_max_capability("gemini", "gemini-3-flash")


class TestBranchExplorer:
    """Tests for BranchExplorer."""

    def test_generate_branches(self):
        """Test branch generation."""
        explorer = BranchExplorer(enabled_clis=["codex", "gemini"])
        branches = explorer.generate_branches("Implement feature X", branch_count=3)

        assert len(branches) == 3
        for branch in branches:
            assert branch.branch_id
            assert branch.approach in BranchApproach
            assert branch.cli in ["codex", "gemini"]
            assert branch.prompt_variant

    def test_generate_branches_with_context(self):
        """Test branch generation includes context."""
        explorer = BranchExplorer()
        branches = explorer.generate_branches(
            "Implement feature",
            context="Previous attempts failed because of X",
        )

        for branch in branches:
            assert "Previous" in branch.prompt_variant or "context" in branch.prompt_variant.lower()

    def test_exploration_branch_serialization(self):
        """Test branch serialization."""
        branch = ExplorationBranch(
            branch_id="test-123",
            approach=BranchApproach.BOTTOM_UP,
            prompt_variant="Test prompt",
            cli="codex",
            model="gpt-5.2",
            max_iterations=5,
        )
        branch.executed = True
        branch.progress_score = 0.7
        branch.success = True
        branch.started_at = datetime.now()
        branch.completed_at = datetime.now()

        data = branch.to_dict()
        restored = ExplorationBranch.from_dict(data)

        assert restored.branch_id == "test-123"
        assert restored.approach == BranchApproach.BOTTOM_UP
        assert restored.executed
        assert restored.progress_score == 0.7

    def test_execute_branch_no_executor(self):
        """Test branch execution without executor."""
        explorer = BranchExplorer()
        branch = ExplorationBranch(
            branch_id="test",
            approach=BranchApproach.TOP_DOWN,
            prompt_variant="Test",
            cli="codex",
            model="gpt-5.2",
        )

        result = explorer.execute_branch(branch)
        assert result.executed
        assert not result.success
        assert result.error == "No executor available"

    def test_select_best_branch(self):
        """Test selecting best branch."""
        explorer = BranchExplorer()

        branches = [
            ExplorationBranch(
                branch_id="1",
                approach=BranchApproach.BOTTOM_UP,
                prompt_variant="P1",
                cli="codex",
                model="gpt-5.2",
            ),
            ExplorationBranch(
                branch_id="2",
                approach=BranchApproach.TOP_DOWN,
                prompt_variant="P2",
                cli="gemini",
                model="gemini-3-pro",
            ),
        ]

        # Set execution results
        branches[0].executed = True
        branches[0].progress_score = 0.2
        branches[1].executed = True
        branches[1].progress_score = 0.5

        best = explorer.select_best_branch(branches, min_progress_threshold=0.3)
        assert best is not None
        assert best.branch_id == "2"

    def test_select_best_branch_none_pass_threshold(self):
        """Test no branch passes threshold."""
        explorer = BranchExplorer()

        branches = [
            ExplorationBranch(
                branch_id="1",
                approach=BranchApproach.BOTTOM_UP,
                prompt_variant="P1",
                cli="codex",
                model="gpt-5.2",
            ),
        ]
        branches[0].executed = True
        branches[0].progress_score = 0.1

        best = explorer.select_best_branch(branches, min_progress_threshold=0.5)
        assert best is None


class TestRecoveryManager:
    """Tests for RecoveryManager."""

    def test_initialization(self):
        """Test recovery manager initialization."""
        manager = RecoveryManager()
        assert manager.stuck_detector is not None
        assert manager.prompt_mutator is not None
        assert manager.model_escalator is not None
        assert manager.branch_explorer is not None

    def test_needs_recovery_false_initially(self):
        """Test needs_recovery is False initially."""
        manager = RecoveryManager()
        assert not manager.needs_recovery()

    def test_needs_recovery_true_when_stuck(self):
        """Test needs_recovery is True when stuck."""
        manager = RecoveryManager()

        # Simulate stuck state
        for i in range(3):
            manager.stuck_detector.record_iteration(i, 0.1, False)

        assert manager.needs_recovery()

    def test_recover_with_prompt_mutation(self):
        """Test recovery via prompt mutation."""
        manager = RecoveryManager(max_mutation_attempts=1)

        # Mock execute_fn that returns success
        def mock_execute(prompt, max_iterations=5, **kwargs):
            return {"made_progress": True, "iterations": 1}

        # Make stuck
        for i in range(3):
            manager.stuck_detector.record_iteration(i, 0.1, False)

        result = manager.recover(
            goal="Test goal",
            current_cli="codex",
            current_model="gpt-5.2",
            execute_fn=mock_execute,
        )

        assert result.success
        assert result.strategy_used == RecoveryStrategy.PROMPT_MUTATION
        assert result.new_prompt is not None

    def test_recover_all_strategies_fail(self):
        """Test recovery when all strategies fail."""
        manager = RecoveryManager(max_mutation_attempts=1)

        # Make stuck
        for i in range(3):
            manager.stuck_detector.record_iteration(i, 0.1, False)

        result = manager.recover(
            goal="Test goal",
            current_cli="claude",
            current_model="claude-opus-4.5",  # Already at max
        )

        # Should end with human escalation
        assert result.strategy_used == RecoveryStrategy.HUMAN_ESCALATION
        assert result.needs_human

    def test_recovery_result_serialization(self):
        """Test recovery result serialization."""
        result = RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.PROMPT_MUTATION,
            new_prompt="Mutated prompt",
            total_iterations_used=5,
        )

        data = result.to_dict()
        assert data["success"]
        assert data["strategy_used"] == "prompt_mutation"
        assert data["new_prompt"] == "Mutated prompt"

    def test_get_recovery_summary(self):
        """Test recovery summary generation."""
        manager = RecoveryManager()
        manager.stuck_detector.record_iteration(1, 0.1, False)

        summary = manager.get_recovery_summary()
        assert "stuck_state" in summary
        assert "recovery_budget_remaining" in summary


class TestRecoveryIntegration:
    """Integration tests for recovery system."""

    def test_full_recovery_flow_mutation_success(self):
        """Test full recovery flow ending in mutation success."""
        manager = RecoveryManager(max_mutation_attempts=3)

        call_count = [0]

        def mock_execute(prompt, max_iterations=5, **kwargs):
            call_count[0] += 1
            # Succeed on second call
            return {
                "made_progress": call_count[0] >= 2,
                "iterations": 1,
            }

        # Make stuck
        for i in range(3):
            manager.stuck_detector.record_iteration(i, 0.1, False)

        result = manager.recover(
            goal="Implement feature",
            current_cli="codex",
            current_model="gpt-5.2",
            execute_fn=mock_execute,
        )

        assert result.success
        assert result.strategy_used == RecoveryStrategy.PROMPT_MUTATION
        assert len(result.attempts) >= 1

    def test_recovery_respects_budget(self):
        """Test recovery respects iteration budget."""
        detector = StuckDetector(recovery_budget=2)
        manager = RecoveryManager(
            stuck_detector=detector,
            max_mutation_attempts=10,
        )

        # Make stuck
        for i in range(3):
            detector.record_iteration(i, 0.1, False)

        # Execute with budget of 2
        result = manager.recover(
            goal="Test",
            current_cli="codex",
            current_model="gpt-5.2",
        )

        # Should have used limited iterations
        assert manager.stuck_detector.recovery_budget_remaining() <= 2
