"""
Base strategy class for coordination strategies.

Defines the interface that all coordination strategies must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..handle import HarnessHandle
    from ..result import AggregatedResult, HarnessResult


@dataclass
class StrategyConfig:
    """Configuration for coordination strategies."""

    # General settings
    timeout: Optional[timedelta] = None
    poll_interval_seconds: float = 5.0

    # Parallel strategy
    fail_fast: bool = False
    max_concurrent: int = 10

    # Race strategy
    min_progress_to_win: float = 0.8
    terminate_losers: bool = True

    # Voting strategy
    min_votes: int = 2
    selection_criteria: str = "first"  # first, best_progress, best_tests

    # Pipeline strategy
    stop_on_failure: bool = True
    pass_context: bool = True

    # Extra config
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    Abstract base class for coordination strategies.

    All coordination strategies must inherit from this class and
    implement the execute() method.

    Example:
        class MyStrategy(BaseStrategy):
            def execute(self, handles, manager) -> AggregatedResult:
                # Custom coordination logic
                ...
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        """Initialize the strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config or StrategyConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the strategy name."""
        pass

    @abstractmethod
    def execute(
        self,
        handles: List["HarnessHandle"],
        wait_for: Callable,
        wait_for_any: Callable,
        on_complete: Optional[Callable[["HarnessHandle"], None]] = None,
    ) -> "AggregatedResult":
        """Execute the coordination strategy.

        Args:
            handles: List of HarnessHandles to coordinate
            wait_for: Function to wait for harnesses (from manager)
            wait_for_any: Function to wait for any harness (from manager)
            on_complete: Optional callback when a harness completes

        Returns:
            AggregatedResult with all results
        """
        pass

    def validate_handles(self, handles: List["HarnessHandle"]) -> bool:
        """Validate that handles are suitable for this strategy.

        Args:
            handles: List of handles to validate

        Returns:
            True if valid
        """
        if not handles:
            return False
        return True

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Config key (supports dot notation)
            default: Default value if not found

        Returns:
            Config value
        """
        # Check extra config first
        if key in self.config.extra:
            return self.config.extra[key]

        # Check main config
        if hasattr(self.config, key):
            return getattr(self.config, key)

        return default
