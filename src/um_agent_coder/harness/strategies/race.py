"""
Race coordination strategy.

First harness to complete wins, others are terminated.
"""

import logging
from datetime import datetime
from typing import Callable, List, Optional, TYPE_CHECKING

from .base import BaseStrategy, StrategyConfig

if TYPE_CHECKING:
    from ..handle import HarnessHandle
    from ..result import AggregatedResult, HarnessResult

logger = logging.getLogger(__name__)


class RaceStrategy(BaseStrategy):
    """
    Race coordination strategy.

    All harnesses run simultaneously. The first one to complete
    successfully is declared the winner. Other harnesses are
    terminated (optionally).

    Example:
        strategy = RaceStrategy(StrategyConfig(
            min_progress_to_win=0.8,
            terminate_losers=True,
        ))
        result = strategy.execute(handles, manager.wait_for, manager.wait_for_any)
    """

    @property
    def name(self) -> str:
        return "race"

    def execute(
        self,
        handles: List["HarnessHandle"],
        wait_for: Callable,
        wait_for_any: Callable,
        on_complete: Optional[Callable[["HarnessHandle"], None]] = None,
    ) -> "AggregatedResult":
        """Execute race coordination.

        All harnesses run. First to complete wins.

        Args:
            handles: List of HarnessHandles to coordinate
            wait_for: Function to wait for harnesses
            wait_for_any: Function to wait for any harness
            on_complete: Optional callback when a harness completes

        Returns:
            AggregatedResult with winner and loser results
        """
        from ..result import AggregatedResult, HarnessResult

        started_at = datetime.now()
        min_progress = self.config.min_progress_to_win
        terminate_losers = self.config.terminate_losers
        results: List[HarnessResult] = []

        logger.info(f"Starting race execution with {len(handles)} competitors")

        # Wait for first to complete
        try:
            winner_handle, winner_result = wait_for_any(
                handles,
                timeout=self.config.timeout,
            )
        except TimeoutError:
            logger.error("Race timed out with no winner")
            # Get current results from all
            for handle in handles:
                results.append(handle.get_result())
            return AggregatedResult(
                strategy=self.name,
                success=False,
                results=results,
                winner=None,
                started_at=started_at,
                completed_at=datetime.now(),
            )

        logger.info(f"Race winner: {winner_handle.harness_id}")

        if on_complete:
            on_complete(winner_handle)

        # Check if winner meets criteria
        if winner_result.progress < min_progress and winner_result.success:
            logger.warning(
                f"Winner {winner_handle.harness_id} has progress "
                f"{winner_result.progress:.1%} < {min_progress:.1%}"
            )

        results.append(winner_result)

        # Handle losers
        for handle in handles:
            if handle.harness_id == winner_handle.harness_id:
                continue

            if terminate_losers:
                logger.info(f"Terminating loser: {handle.harness_id}")
                handle.request_stop()

            # Get final result
            loser_result = handle.get_result()
            results.append(loser_result)

            if on_complete:
                on_complete(handle)

        completed_at = datetime.now()

        # Race is successful if winner succeeded
        success = winner_result.success

        return AggregatedResult(
            strategy=self.name,
            success=success,
            results=results,
            winner=winner_result,
            started_at=started_at,
            completed_at=completed_at,
        )
