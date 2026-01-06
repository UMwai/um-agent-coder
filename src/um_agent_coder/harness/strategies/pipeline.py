"""
Pipeline coordination strategy.

Harnesses execute sequentially, with output from one stage feeding
into the next as context.
"""

import json
import logging
from datetime import datetime
from typing import Callable, List, Optional, TYPE_CHECKING

from .base import BaseStrategy, StrategyConfig

if TYPE_CHECKING:
    from ..handle import HarnessHandle
    from ..result import AggregatedResult, HarnessResult

logger = logging.getLogger(__name__)


class PipelineStrategy(BaseStrategy):
    """
    Pipeline coordination strategy.

    Harnesses execute sequentially in order. The output from each
    stage is passed as context to the next stage. If a stage fails
    and stop_on_failure is True, the pipeline stops.

    Example:
        strategy = PipelineStrategy(StrategyConfig(
            stop_on_failure=True,
            pass_context=True,
        ))
        result = strategy.execute(handles, manager.wait_for, manager.wait_for_any)
    """

    @property
    def name(self) -> str:
        return "pipeline"

    def execute(
        self,
        handles: List["HarnessHandle"],
        wait_for: Callable,
        wait_for_any: Callable,
        on_complete: Optional[Callable[["HarnessHandle"], None]] = None,
    ) -> "AggregatedResult":
        """Execute pipeline coordination.

        Harnesses execute one at a time in order. Each stage's output
        is passed to the next stage as context.

        Args:
            handles: List of HarnessHandles to coordinate (in order)
            wait_for: Function to wait for harnesses
            wait_for_any: Function to wait for any harness
            on_complete: Optional callback when a harness completes

        Returns:
            AggregatedResult with all results
        """
        from ..result import AggregatedResult, HarnessResult

        started_at = datetime.now()
        results: List[HarnessResult] = []
        context: dict = self.config.extra.get("initial_context", {})
        stop_on_failure = self.config.stop_on_failure
        pass_context = self.config.pass_context
        pipeline_failed = False

        logger.info(f"Starting pipeline execution of {len(handles)} stages")

        for i, handle in enumerate(handles):
            stage_num = i + 1
            logger.info(f"Pipeline stage {stage_num}/{len(handles)}: {handle.harness_id}")

            # Send context to this stage if enabled
            if pass_context and context:
                context_msg = (
                    f"Context from previous pipeline stages:\n"
                    f"{json.dumps(context, indent=2)}"
                )
                handle.send_instruction(context_msg)

            # Wait for this stage to complete
            stage_results = wait_for([handle], timeout=self.config.timeout)
            result = stage_results[0] if stage_results else handle.get_result()
            results.append(result)

            if on_complete:
                on_complete(handle)

            # Check for failure
            if not result.success:
                pipeline_failed = True
                logger.warning(
                    f"Pipeline stage {stage_num} ({handle.harness_id}) failed"
                )

                if stop_on_failure:
                    logger.info("Stopping pipeline due to failure")
                    break

            # Update context for next stage
            if pass_context:
                context["previous_stage"] = {
                    "harness_id": handle.harness_id,
                    "stage_number": stage_num,
                    "output": result.final_output,
                    "metrics": {
                        "tasks_completed": result.tasks_completed,
                        "tasks_failed": result.tasks_failed,
                    },
                }

                # Accumulate stage outputs
                if "stage_outputs" not in context:
                    context["stage_outputs"] = {}
                context["stage_outputs"][handle.harness_id] = {
                    "output": result.final_output,
                    "success": result.success,
                }

        completed_at = datetime.now()

        # Determine overall success
        all_complete = len(results) == len(handles)
        all_success = all(r.success for r in results)
        success = all_complete and all_success

        return AggregatedResult(
            strategy=self.name,
            success=success,
            results=results,
            winner=results[-1] if results else None,  # Last stage is "winner"
            aggregated_output=json.dumps(context.get("stage_outputs", {}), indent=2),
            started_at=started_at,
            completed_at=completed_at,
        )
