"""
Coordination strategies for meta-harness execution.

Provides different strategies for coordinating multiple sub-harnesses:
- PARALLEL: All run simultaneously, aggregate all results
- PIPELINE: Sequential, output feeds next input
- RACE: First to complete wins, others terminated
- VOTING: Multiple complete, pick best by criteria
"""

from .base import BaseStrategy, StrategyConfig
from .parallel import ParallelStrategy
from .pipeline import PipelineStrategy
from .race import RaceStrategy
from .voting import VotingStrategy

__all__ = [
    "BaseStrategy",
    "ParallelStrategy",
    "PipelineStrategy",
    "RaceStrategy",
    "StrategyConfig",
    "VotingStrategy",
]


def get_strategy(name: str, config: StrategyConfig = None) -> BaseStrategy:
    """Get a strategy by name.

    Args:
        name: Strategy name (parallel, pipeline, race, voting)
        config: Optional strategy configuration

    Returns:
        Strategy instance

    Raises:
        ValueError: If strategy name is unknown
    """
    strategies = {
        "parallel": ParallelStrategy,
        "pipeline": PipelineStrategy,
        "race": RaceStrategy,
        "voting": VotingStrategy,
    }

    if name.lower() not in strategies:
        raise ValueError(
            f"Unknown strategy: {name}. "
            f"Available: {', '.join(strategies.keys())}"
        )

    return strategies[name.lower()](config or StrategyConfig())
