"""Branch exploration for stuck recovery.

When stuck, fork into multiple parallel approaches to explore
different solution strategies simultaneously.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class BranchApproach(Enum):
    """Types of exploration approaches."""

    BOTTOM_UP = "bottom_up"  # Start with small pieces, build up
    TOP_DOWN = "top_down"  # Start with structure, fill in details
    RESEARCH_FIRST = "research_first"  # Gather info before implementing
    CONSTRAINT_DRIVEN = "constraint_driven"  # Work within strict bounds
    EXAMPLE_DRIVEN = "example_driven"  # Start with examples, generalize


@dataclass
class ExplorationBranch:
    """A single exploration branch."""

    branch_id: str
    approach: BranchApproach
    prompt_variant: str
    cli: str
    model: str
    max_iterations: int = 10

    # Results (filled after execution)
    executed: bool = False
    progress_score: float = 0.0
    iterations_used: int = 0
    final_output: str = ""
    success: bool = False
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "branch_id": self.branch_id,
            "approach": self.approach.value,
            "prompt_variant": self.prompt_variant,
            "cli": self.cli,
            "model": self.model,
            "max_iterations": self.max_iterations,
            "executed": self.executed,
            "progress_score": self.progress_score,
            "iterations_used": self.iterations_used,
            "final_output": self.final_output[:500],  # Truncate for storage
            "success": self.success,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExplorationBranch":
        """Deserialize from dictionary."""
        branch = cls(
            branch_id=data["branch_id"],
            approach=BranchApproach(data["approach"]),
            prompt_variant=data["prompt_variant"],
            cli=data["cli"],
            model=data["model"],
            max_iterations=data.get("max_iterations", 10),
        )
        branch.executed = data.get("executed", False)
        branch.progress_score = data.get("progress_score", 0.0)
        branch.iterations_used = data.get("iterations_used", 0)
        branch.final_output = data.get("final_output", "")
        branch.success = data.get("success", False)
        branch.error = data.get("error")
        if data.get("started_at"):
            branch.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            branch.completed_at = datetime.fromisoformat(data["completed_at"])
        return branch


# Approach-specific prompt templates
APPROACH_TEMPLATES = {
    BranchApproach.BOTTOM_UP: """
Approach: Build from small pieces up to the complete solution.

Task: {goal}

Strategy:
1. Identify the smallest functional unit needed
2. Implement and test that unit
3. Add the next smallest piece
4. Continue building up until complete

Start with the smallest piece first:
""",
    BranchApproach.TOP_DOWN: """
Approach: Start with the high-level structure, then fill in details.

Task: {goal}

Strategy:
1. Define the overall structure/architecture
2. Create placeholder functions/components
3. Fill in each placeholder one by one
4. Connect all pieces together

Start by defining the structure:
""",
    BranchApproach.RESEARCH_FIRST: """
Approach: Gather information and understanding before implementing.

Task: {goal}

Strategy:
1. Read and understand relevant existing code
2. Research similar implementations or patterns
3. Document the approach before coding
4. Implement based on research findings

Start by exploring and understanding:
""",
    BranchApproach.CONSTRAINT_DRIVEN: """
Approach: Work within strict constraints to force focused solutions.

Task: {goal}

Constraints:
- Must complete in minimal iterations
- No unnecessary changes to existing code
- Focus only on the core requirement
- Test immediately after each change

Work within these constraints:
""",
    BranchApproach.EXAMPLE_DRIVEN: """
Approach: Start with concrete examples, then generalize.

Task: {goal}

Strategy:
1. Create a specific working example first
2. Test the example thoroughly
3. Generalize from the working example
4. Ensure generalization maintains example behavior

Start with a concrete example:
""",
}


class BranchExplorer:
    """Explore multiple solution branches in parallel.

    When stuck, generate and execute multiple exploration branches
    with different approaches to find a way forward.
    """

    def __init__(
        self,
        enabled_clis: Optional[list[str]] = None,
        default_branch_count: int = 3,
        default_iterations_per_branch: int = 10,
        executor_factory=None,
    ):
        """Initialize branch explorer.

        Args:
            enabled_clis: List of enabled CLI backends.
            default_branch_count: Default number of branches to create.
            default_iterations_per_branch: Default max iterations per branch.
            executor_factory: Optional factory for creating executors.
        """
        self.enabled_clis = enabled_clis or ["codex", "gemini"]
        self.default_branch_count = default_branch_count
        self.default_iterations_per_branch = default_iterations_per_branch
        self.executor_factory = executor_factory

    def generate_branches(
        self,
        goal: str,
        context: str = "",
        branch_count: Optional[int] = None,
    ) -> list[ExplorationBranch]:
        """Generate exploration branches for a goal.

        Args:
            goal: The task goal to achieve.
            context: Optional context from previous attempts.
            branch_count: Number of branches to generate.

        Returns:
            List of ExplorationBranch objects.
        """
        count = branch_count or self.default_branch_count
        branches = []

        # Select approaches to try (cycle through if more branches than approaches)
        approaches = list(BranchApproach)

        for i in range(count):
            approach = approaches[i % len(approaches)]
            cli = self.enabled_clis[i % len(self.enabled_clis)]

            # Generate prompt variant using approach template
            template = APPROACH_TEMPLATES.get(approach, "{goal}")
            prompt_variant = template.format(goal=goal)

            if context:
                prompt_variant = f"Previous context:\n{context}\n\n{prompt_variant}"

            # Determine model based on CLI
            model = self._get_default_model(cli)

            branch = ExplorationBranch(
                branch_id=str(uuid.uuid4())[:8],
                approach=approach,
                prompt_variant=prompt_variant,
                cli=cli,
                model=model,
                max_iterations=self.default_iterations_per_branch,
            )
            branches.append(branch)

        return branches

    def _get_default_model(self, cli: str) -> str:
        """Get default model for a CLI."""
        defaults = {
            "gemini": "gemini-3-pro",
            "codex": "gpt-5.2",
            "claude": "claude-sonnet-4",
        }
        return defaults.get(cli, "default")

    def execute_branch(
        self,
        branch: ExplorationBranch,
        executor=None,
    ) -> ExplorationBranch:
        """Execute a single exploration branch.

        Args:
            branch: The branch to execute.
            executor: Optional executor to use.

        Returns:
            Updated branch with results.
        """
        branch.started_at = datetime.now()
        branch.executed = True

        try:
            if executor is None and self.executor_factory is not None:
                executor = self.executor_factory(branch.cli, branch.model)

            if executor is None:
                # No executor available, simulate execution
                branch.progress_score = 0.0
                branch.iterations_used = 0
                branch.success = False
                branch.error = "No executor available"
            else:
                # Execute with the provided executor
                result = executor.execute(
                    prompt=branch.prompt_variant,
                    max_iterations=branch.max_iterations,
                )
                branch.progress_score = result.get("progress_score", 0.0)
                branch.iterations_used = result.get("iterations", 0)
                branch.final_output = result.get("output", "")
                branch.success = result.get("success", False)

        except Exception as e:
            branch.success = False
            branch.error = str(e)

        branch.completed_at = datetime.now()
        return branch

    def execute_branches_parallel(
        self,
        branches: list[ExplorationBranch],
        max_workers: int = 3,
    ) -> list[ExplorationBranch]:
        """Execute multiple branches in parallel.

        Args:
            branches: List of branches to execute.
            max_workers: Maximum parallel workers.

        Returns:
            List of executed branches with results.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.execute_branch, branch): branch for branch in branches}

            results = []
            for future in as_completed(futures):
                branch = future.result()
                results.append(branch)

        return results

    def select_best_branch(
        self,
        branches: list[ExplorationBranch],
        min_progress_threshold: float = 0.3,
    ) -> Optional[ExplorationBranch]:
        """Select the best performing branch.

        Args:
            branches: List of executed branches.
            min_progress_threshold: Minimum progress score to consider.

        Returns:
            Best branch, or None if none meet threshold.
        """
        executed = [b for b in branches if b.executed]
        if not executed:
            return None

        # Filter by threshold
        passing = [b for b in executed if b.progress_score >= min_progress_threshold]
        if not passing:
            return None

        # Return highest progress score
        return max(passing, key=lambda b: b.progress_score)

    def explore_and_select(
        self,
        goal: str,
        context: str = "",
        branch_count: Optional[int] = None,
        min_progress_threshold: float = 0.3,
    ) -> Optional[ExplorationBranch]:
        """Generate, execute, and select best branch.

        Args:
            goal: The task goal.
            context: Optional context from previous attempts.
            branch_count: Number of branches to try.
            min_progress_threshold: Minimum progress to accept.

        Returns:
            Best branch, or None if exploration failed.
        """
        branches = self.generate_branches(goal, context, branch_count)
        executed = self.execute_branches_parallel(branches)
        return self.select_best_branch(executed, min_progress_threshold)
