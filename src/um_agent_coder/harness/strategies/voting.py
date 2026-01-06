"""
Voting coordination strategy.

Multiple harnesses complete, best result is selected by criteria.
"""

import logging
from datetime import datetime
from typing import Callable, List, Optional, TYPE_CHECKING

from .base import BaseStrategy, StrategyConfig

if TYPE_CHECKING:
    from ..handle import HarnessHandle
    from ..result import AggregatedResult, HarnessResult

logger = logging.getLogger(__name__)


class VotingStrategy(BaseStrategy):
    """
    Voting coordination strategy.

    Multiple harnesses run and complete. The best result is selected
    based on configurable criteria (first, best_progress, best_tests).

    Example:
        strategy = VotingStrategy(StrategyConfig(
            min_votes=2,
            selection_criteria="best_progress",
        ))
        result = strategy.execute(handles, manager.wait_for, manager.wait_for_any)
    """

    @property
    def name(self) -> str:
        return "voting"

    def execute(
        self,
        handles: List["HarnessHandle"],
        wait_for: Callable,
        wait_for_any: Callable,
        on_complete: Optional[Callable[["HarnessHandle"], None]] = None,
    ) -> "AggregatedResult":
        """Execute voting coordination.

        Wait for minimum votes, then select winner by criteria.

        Args:
            handles: List of HarnessHandles to coordinate
            wait_for: Function to wait for harnesses
            wait_for_any: Function to wait for any harness
            on_complete: Optional callback when a harness completes

        Returns:
            AggregatedResult with winner (best result)
        """
        from ..result import AggregatedResult, HarnessResult

        started_at = datetime.now()
        min_votes = self.config.min_votes
        selection_criteria = self.config.selection_criteria
        poll_interval = self.config.poll_interval_seconds

        successful_results: List[HarnessResult] = []
        all_results: List[HarnessResult] = []
        pending = list(handles)

        logger.info(
            f"Starting voting execution with {len(handles)} candidates, "
            f"min_votes={min_votes}, criteria={selection_criteria}"
        )

        # Wait for minimum successful completions
        import time
        while len(successful_results) < min_votes and pending:
            for handle in list(pending):
                if handle.is_complete():
                    result = handle.get_result()
                    all_results.append(result)
                    pending.remove(handle)

                    if on_complete:
                        on_complete(handle)

                    if result.success:
                        successful_results.append(result)
                        logger.info(
                            f"Successful vote from {handle.harness_id} "
                            f"({len(successful_results)}/{min_votes})"
                        )
                    else:
                        logger.warning(
                            f"Failed vote from {handle.harness_id}"
                        )

            # Check if we have enough votes
            if len(successful_results) >= min_votes:
                break

            # Check if we can still reach min_votes
            if len(successful_results) + len(pending) < min_votes:
                logger.warning("Cannot reach minimum votes, stopping")
                break

            if pending:
                time.sleep(poll_interval)

        completed_at = datetime.now()

        # Add results from remaining pending harnesses
        for handle in pending:
            result = handle.get_result()
            all_results.append(result)
            if result.success:
                successful_results.append(result)

        # No successful votes
        if not successful_results:
            logger.error("No successful results to vote on")
            return AggregatedResult(
                strategy=self.name,
                success=False,
                results=all_results,
                winner=None,
                started_at=started_at,
                completed_at=completed_at,
            )

        # Select winner based on criteria
        winner = self._select_winner(successful_results, selection_criteria)

        logger.info(
            f"Voting complete. Winner: {winner.harness_id} "
            f"(criteria: {selection_criteria})"
        )

        # Reorder results to put winner first
        ordered_results = [winner] + [r for r in all_results if r != winner]

        return AggregatedResult(
            strategy=self.name,
            success=True,
            results=ordered_results,
            winner=winner,
            started_at=started_at,
            completed_at=completed_at,
        )

    def _select_winner(
        self,
        results: List["HarnessResult"],
        criteria: str,
    ) -> "HarnessResult":
        """Select winner based on criteria.

        Args:
            results: List of successful results
            criteria: Selection criteria

        Returns:
            Winning result
        """
        if not results:
            raise ValueError("No results to select from")

        if len(results) == 1:
            return results[0]

        if criteria == "first":
            # First result (already in order)
            return results[0]

        elif criteria == "best_progress":
            # Highest progress score
            return max(results, key=lambda r: r.progress)

        elif criteria == "best_tests":
            # Most tests passed
            return max(results, key=lambda r: r.metrics.tests_passed)

        elif criteria == "lowest_failures":
            # Fewest tasks failed
            return min(results, key=lambda r: r.tasks_failed)

        elif criteria == "fastest":
            # Shortest duration
            return min(
                results,
                key=lambda r: (
                    r.completed_at - r.started_at
                    if r.completed_at and r.started_at
                    else float("inf")
                ),
            )

        else:
            logger.warning(f"Unknown criteria {criteria}, using 'first'")
            return results[0]
