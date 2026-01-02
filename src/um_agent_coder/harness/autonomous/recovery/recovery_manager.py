"""Recovery manager orchestrating all recovery strategies.

Coordinates stuck detection with prompt mutation, model escalation,
and branch exploration to recover from stuck states.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from .branch_explorer import BranchExplorer, ExplorationBranch
from .model_escalator import ModelEscalator
from .prompt_mutator import MutationType, PromptMutator
from .stuck_detector import StuckDetector


class RecoveryStrategy(Enum):
    """Recovery strategies in order of application."""

    PROMPT_MUTATION = "prompt_mutation"
    MODEL_ESCALATION = "model_escalation"
    BRANCH_EXPLORATION = "branch_exploration"
    HUMAN_ESCALATION = "human_escalation"


@dataclass
class RecoveryAttempt:
    """Record of a single recovery attempt."""

    strategy: RecoveryStrategy
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    iterations_used: int = 0


@dataclass
class RecoveryResult:
    """Result of recovery process."""

    success: bool
    strategy_used: Optional[RecoveryStrategy] = None
    new_prompt: Optional[str] = None
    new_cli: Optional[str] = None
    new_model: Optional[str] = None
    branch_selected: Optional[ExplorationBranch] = None
    needs_human: bool = False
    attempts: list[RecoveryAttempt] = field(default_factory=list)
    total_iterations_used: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "strategy_used": self.strategy_used.value if self.strategy_used else None,
            "new_prompt": self.new_prompt,
            "new_cli": self.new_cli,
            "new_model": self.new_model,
            "branch_selected": self.branch_selected.to_dict() if self.branch_selected else None,
            "needs_human": self.needs_human,
            "total_iterations_used": self.total_iterations_used,
            "error": self.error,
            "attempts": [
                {
                    "strategy": a.strategy.value,
                    "started_at": a.started_at.isoformat(),
                    "completed_at": a.completed_at.isoformat() if a.completed_at else None,
                    "success": a.success,
                    "details": a.details,
                    "iterations_used": a.iterations_used,
                }
                for a in self.attempts
            ],
        }


class RecoveryManager:
    """Orchestrate recovery from stuck states.

    Applies recovery strategies in order:
    1. Prompt mutations (3 attempts)
    2. Model escalation
    3. Branch exploration (parallel)
    4. Human escalation (last resort)
    """

    def __init__(
        self,
        stuck_detector: Optional[StuckDetector] = None,
        prompt_mutator: Optional[PromptMutator] = None,
        model_escalator: Optional[ModelEscalator] = None,
        branch_explorer: Optional[BranchExplorer] = None,
        max_mutation_attempts: int = 3,
        executor_factory=None,
    ):
        """Initialize recovery manager.

        Args:
            stuck_detector: StuckDetector instance.
            prompt_mutator: PromptMutator instance.
            model_escalator: ModelEscalator instance.
            branch_explorer: BranchExplorer instance.
            max_mutation_attempts: Max prompt mutations before escalating.
            executor_factory: Factory for creating executors.
        """
        self.stuck_detector = stuck_detector or StuckDetector()
        self.prompt_mutator = prompt_mutator or PromptMutator()
        self.model_escalator = model_escalator or ModelEscalator()
        self.branch_explorer = branch_explorer or BranchExplorer()
        self.max_mutation_attempts = max_mutation_attempts
        self.executor_factory = executor_factory

    def needs_recovery(self) -> bool:
        """Check if recovery is needed."""
        return self.stuck_detector.needs_recovery()

    def recover(
        self,
        goal: str,
        current_cli: str,
        current_model: str,
        context: str = "",
        execute_fn=None,
    ) -> RecoveryResult:
        """Attempt to recover from stuck state.

        Args:
            goal: The original task goal.
            current_cli: Current CLI backend.
            current_model: Current model.
            context: Context from previous iterations.
            execute_fn: Optional function to execute recovery attempts.

        Returns:
            RecoveryResult with recovery outcome.
        """
        result = RecoveryResult(success=False)
        self.stuck_detector.start_recovery()

        try:
            # Stage 1: Prompt Mutations
            mutation_result = self._try_prompt_mutations(goal, context, execute_fn)
            result.attempts.extend(mutation_result.get("attempts", []))
            result.total_iterations_used += mutation_result.get("iterations", 0)

            if mutation_result.get("success"):
                result.success = True
                result.strategy_used = RecoveryStrategy.PROMPT_MUTATION
                result.new_prompt = mutation_result.get("new_prompt")
                self.stuck_detector.end_recovery(True)
                return result

            # Stage 2: Model Escalation
            escalation_result = self._try_model_escalation(
                goal, current_cli, current_model, context, execute_fn
            )
            result.attempts.extend(escalation_result.get("attempts", []))
            result.total_iterations_used += escalation_result.get("iterations", 0)

            if escalation_result.get("success"):
                result.success = True
                result.strategy_used = RecoveryStrategy.MODEL_ESCALATION
                result.new_cli = escalation_result.get("new_cli")
                result.new_model = escalation_result.get("new_model")
                self.stuck_detector.end_recovery(True)
                return result

            # Stage 3: Branch Exploration
            exploration_result = self._try_branch_exploration(goal, context)
            result.attempts.extend(exploration_result.get("attempts", []))
            result.total_iterations_used += exploration_result.get("iterations", 0)

            if exploration_result.get("success"):
                result.success = True
                result.strategy_used = RecoveryStrategy.BRANCH_EXPLORATION
                result.branch_selected = exploration_result.get("branch")
                result.new_prompt = exploration_result.get("new_prompt")
                result.new_cli = exploration_result.get("new_cli")
                result.new_model = exploration_result.get("new_model")
                self.stuck_detector.end_recovery(True)
                return result

            # Stage 4: Human Escalation (last resort)
            result.strategy_used = RecoveryStrategy.HUMAN_ESCALATION
            result.needs_human = True
            self.stuck_detector.end_recovery(False)
            return result

        except Exception as e:
            result.error = str(e)
            self.stuck_detector.end_recovery(False)
            return result

    def _try_prompt_mutations(
        self,
        goal: str,
        context: str,
        execute_fn=None,
    ) -> dict[str, Any]:
        """Try prompt mutations to escape stuck state.

        Args:
            goal: Original goal.
            context: Previous context.
            execute_fn: Function to execute with mutated prompt.

        Returns:
            Dict with success status and details.
        """
        attempts = []
        total_iterations = 0
        tried_mutations: list[MutationType] = []

        for _ in range(self.max_mutation_attempts):
            if not self.stuck_detector.use_recovery_iteration():
                break

            attempt = RecoveryAttempt(
                strategy=RecoveryStrategy.PROMPT_MUTATION,
                started_at=datetime.now(),
            )

            mutation_result = self.prompt_mutator.mutate_with_fallback(
                goal, tried_mutations, context
            )

            if mutation_result is None:
                # All mutations exhausted
                attempt.completed_at = datetime.now()
                attempt.success = False
                attempt.details = {"reason": "all_mutations_exhausted"}
                attempts.append(attempt)
                break

            tried_mutations.append(mutation_result.mutation_type)
            attempt.details = {
                "mutation_type": mutation_result.mutation_type.value,
                "mutated_prompt": mutation_result.mutated_prompt[:200],
            }

            # Execute with mutated prompt if function provided
            if execute_fn is not None:
                try:
                    exec_result = execute_fn(mutation_result.mutated_prompt, max_iterations=5)
                    attempt.iterations_used = exec_result.get("iterations", 1)
                    total_iterations += attempt.iterations_used

                    if exec_result.get("made_progress", False):
                        attempt.success = True
                        attempt.completed_at = datetime.now()
                        attempts.append(attempt)
                        return {
                            "success": True,
                            "new_prompt": mutation_result.mutated_prompt,
                            "attempts": attempts,
                            "iterations": total_iterations,
                        }
                except Exception as e:
                    attempt.details["error"] = str(e)

            attempt.completed_at = datetime.now()
            attempt.success = False
            attempts.append(attempt)
            total_iterations += 1

        return {
            "success": False,
            "attempts": attempts,
            "iterations": total_iterations,
        }

    def _try_model_escalation(
        self,
        goal: str,
        current_cli: str,
        current_model: str,
        context: str,
        execute_fn=None,
    ) -> dict[str, Any]:
        """Try escalating to a more capable model.

        Args:
            goal: Original goal.
            current_cli: Current CLI.
            current_model: Current model.
            context: Previous context.
            execute_fn: Function to execute with new model.

        Returns:
            Dict with success status and details.
        """
        attempts = []
        total_iterations = 0

        if not self.stuck_detector.use_recovery_iteration():
            return {"success": False, "attempts": [], "iterations": 0}

        attempt = RecoveryAttempt(
            strategy=RecoveryStrategy.MODEL_ESCALATION,
            started_at=datetime.now(),
        )

        escalation = self.model_escalator.escalate(current_cli, current_model)
        if escalation is None:
            attempt.completed_at = datetime.now()
            attempt.success = False
            attempt.details = {"reason": "already_at_max_capability"}
            attempts.append(attempt)
            return {"success": False, "attempts": attempts, "iterations": 0}

        attempt.details = {
            "from_cli": escalation.previous_cli,
            "from_model": escalation.previous_model,
            "to_cli": escalation.new_cli,
            "to_model": escalation.new_model,
            "capability_increase": escalation.capability_increase,
        }

        # Execute with new model if function provided
        if execute_fn is not None:
            try:
                exec_result = execute_fn(
                    goal,
                    max_iterations=10,
                    cli=escalation.new_cli,
                    model=escalation.new_model,
                )
                attempt.iterations_used = exec_result.get("iterations", 1)
                total_iterations += attempt.iterations_used

                if exec_result.get("made_progress", False):
                    attempt.success = True
                    attempt.completed_at = datetime.now()
                    attempts.append(attempt)
                    return {
                        "success": True,
                        "new_cli": escalation.new_cli,
                        "new_model": escalation.new_model,
                        "attempts": attempts,
                        "iterations": total_iterations,
                    }
            except Exception as e:
                attempt.details["error"] = str(e)

        attempt.completed_at = datetime.now()
        attempt.success = False
        attempts.append(attempt)
        total_iterations += 1

        return {
            "success": False,
            "attempts": attempts,
            "iterations": total_iterations,
        }

    def _try_branch_exploration(
        self,
        goal: str,
        context: str,
    ) -> dict[str, Any]:
        """Try parallel branch exploration.

        Args:
            goal: Original goal.
            context: Previous context.

        Returns:
            Dict with success status and details.
        """
        attempts = []

        attempt = RecoveryAttempt(
            strategy=RecoveryStrategy.BRANCH_EXPLORATION,
            started_at=datetime.now(),
        )

        try:
            # Generate and execute branches
            branches = self.branch_explorer.generate_branches(goal, context)
            attempt.details = {"branch_count": len(branches)}

            # Track iterations used
            iterations_per_branch = self.branch_explorer.default_iterations_per_branch
            for _ in range(len(branches) * iterations_per_branch):
                if not self.stuck_detector.use_recovery_iteration():
                    break

            executed = self.branch_explorer.execute_branches_parallel(branches)
            attempt.iterations_used = sum(b.iterations_used for b in executed)

            # Select best branch
            best = self.branch_explorer.select_best_branch(executed)
            if best is not None:
                attempt.success = True
                attempt.completed_at = datetime.now()
                attempt.details["selected_branch"] = best.branch_id
                attempt.details["selected_approach"] = best.approach.value
                attempts.append(attempt)
                return {
                    "success": True,
                    "branch": best,
                    "new_prompt": best.prompt_variant,
                    "new_cli": best.cli,
                    "new_model": best.model,
                    "attempts": attempts,
                    "iterations": attempt.iterations_used,
                }

            attempt.details["reason"] = "no_branch_met_threshold"

        except Exception as e:
            attempt.details["error"] = str(e)

        attempt.completed_at = datetime.now()
        attempt.success = False
        attempts.append(attempt)

        return {
            "success": False,
            "attempts": attempts,
            "iterations": attempt.iterations_used if hasattr(attempt, "iterations_used") else 0,
        }

    def get_recovery_summary(self) -> dict[str, Any]:
        """Get summary of recovery state."""
        return {
            "stuck_state": self.stuck_detector.get_summary(),
            "recovery_budget_remaining": self.stuck_detector.recovery_budget_remaining(),
            "model_at_max": self.model_escalator.is_at_max_capability(
                (
                    self.model_escalator.get_cheapest()[0]
                    if self.model_escalator.get_cheapest()
                    else ""
                ),
                (
                    self.model_escalator.get_cheapest()[1]
                    if self.model_escalator.get_cheapest()
                    else ""
                ),
            ),
        }
