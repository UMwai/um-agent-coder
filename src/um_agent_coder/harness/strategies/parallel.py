"""
Parallel coordination strategy.

All harnesses run simultaneously, results are aggregated when all complete.
"""

import logging
from datetime import datetime
from typing import Callable, List, Optional, TYPE_CHECKING

from .base import BaseStrategy, StrategyConfig

if TYPE_CHECKING:
    from ..handle import HarnessHandle
    from ..result import AggregatedResult, HarnessResult

logger = logging.getLogger(__name__)


class ParallelStrategy(BaseStrategy):
    """
    Parallel coordination strategy.

    All harnesses run simultaneously. Results are aggregated when all
    complete. Optionally supports fail-fast mode where execution stops
    on first failure.

    Example:
        strategy = ParallelStrategy(StrategyConfig(fail_fast=True))
        result = strategy.execute(handles, manager.wait_for, manager.wait_for_any)
    """

    @property
    def name(self) -> str:
        return "parallel"

    def execute(
        self,
        handles: List["HarnessHandle"],
        wait_for: Callable,
        wait_for_any: Callable,
        on_complete: Optional[Callable[["HarnessHandle"], None]] = None,
    ) -> "AggregatedResult":
        """Execute parallel coordination.

        All harnesses run at the same time. Waits for all to complete
        (or until fail-fast triggers).

        Args:
            handles: List of HarnessHandles to coordinate
            wait_for: Function to wait for harnesses
            wait_for_any: Function to wait for any harness
            on_complete: Optional callback when a harness completes

        Returns:
            AggregatedResult with all results
        """
        from ..result import AggregatedResult

        started_at = datetime.now()
        fail_fast = self.config.fail_fast
        failed = False

        logger.info(f"Starting parallel execution of {len(handles)} harnesses")

        def callback(handle: "HarnessHandle") -> None:
            nonlocal failed
            result = handle.get_result()

            if on_complete:
                on_complete(handle)

            if fail_fast and not result.success:
                logger.warning(
                    f"Fail-fast: {handle.harness_id} failed, stopping others"
                )
                failed = True
                # Signal stop to other handles
                for h in handles:
                    if h.harness_id != handle.harness_id:
                        h.request_stop()

        # Wait for all harnesses with callback
        results = wait_for(
            handles,
            timeout=self.config.timeout,
            callback=callback,
        )

        completed_at = datetime.now()

        # Determine overall success
        all_success = all(r.success for r in results)
        success = all_success if not fail_fast else not failed

        return AggregatedResult(
            strategy=self.name,
            success=success,
            results=results,
            winner=None,
            started_at=started_at,
            completed_at=completed_at,
        )
