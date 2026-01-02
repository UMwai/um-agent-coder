"""Multi-CLI router for autonomous loop.

Routes tasks to appropriate CLI backends based on task characteristics,
model strengths, and token efficiency. Supports explicit CLI lists
or auto-routing.

Reference: specs/autonomous-loop-spec.md Section 7
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional


class TaskType(Enum):
    """Task type categories for routing."""

    RESEARCH = "research"
    IMPLEMENTATION = "implementation"
    COMPLEX_REASONING = "complex_reasoning"
    SIMPLE = "simple"
    STUCK_RECOVERY = "stuck_recovery"


@dataclass
class TaskAnalysis:
    """Analysis of task characteristics for routing."""

    requires_large_context: bool = False  # > 100k tokens
    is_implementation: bool = False  # Writing code
    is_research: bool = False  # Exploring/researching
    is_complex_reasoning: bool = False  # Multi-step logic
    is_stuck_recovery: bool = False  # Recovering from stuck
    estimated_difficulty: float = 0.5  # 0.0 (trivial) to 1.0 (hardest)
    keywords_detected: list[str] = None

    def __post_init__(self):
        if self.keywords_detected is None:
            self.keywords_detected = []

    def get_primary_type(self) -> TaskType:
        """Get the primary task type."""
        if self.is_stuck_recovery:
            return TaskType.STUCK_RECOVERY
        if self.is_complex_reasoning and self.estimated_difficulty > 0.7:
            return TaskType.COMPLEX_REASONING
        if self.is_implementation:
            return TaskType.IMPLEMENTATION
        if self.is_research:
            return TaskType.RESEARCH
        return TaskType.SIMPLE


# Keywords for task type detection
IMPLEMENTATION_KEYWORDS = [
    "implement",
    "write",
    "create",
    "build",
    "add",
    "fix",
    "update",
    "modify",
    "change",
    "develop",
    "code",
    "function",
    "class",
]

RESEARCH_KEYWORDS = [
    "research",
    "explore",
    "find",
    "investigate",
    "analyze",
    "understand",
    "search",
    "look",
    "examine",
    "study",
    "review",
    "check",
]

COMPLEX_KEYWORDS = [
    "design",
    "architect",
    "optimize",
    "refactor",
    "debug",
    "complex",
    "difficult",
    "challenging",
    "tricky",
    "sophisticated",
]


class TaskAnalyzer:
    """Analyze tasks to determine routing characteristics."""

    def __init__(
        self,
        large_context_threshold: int = 100000,
    ):
        """Initialize task analyzer.

        Args:
            large_context_threshold: Token count above which is "large context".
        """
        self.large_context_threshold = large_context_threshold

    def analyze(
        self,
        goal: str,
        context_size: int = 0,
        consecutive_no_progress: int = 0,
        total_iterations: int = 0,
    ) -> TaskAnalysis:
        """Analyze a task goal for routing.

        Args:
            goal: The task goal description.
            context_size: Estimated context size in tokens.
            consecutive_no_progress: Consecutive iterations without progress.
            total_iterations: Total iterations so far.

        Returns:
            TaskAnalysis with detected characteristics.
        """
        goal_lower = goal.lower()

        # Detect task types by keywords
        impl_keywords = [kw for kw in IMPLEMENTATION_KEYWORDS if kw in goal_lower]
        research_keywords = [kw for kw in RESEARCH_KEYWORDS if kw in goal_lower]
        complex_keywords = [kw for kw in COMPLEX_KEYWORDS if kw in goal_lower]

        # Estimate difficulty
        difficulty = self._estimate_difficulty(goal, total_iterations, consecutive_no_progress)

        return TaskAnalysis(
            requires_large_context=context_size > self.large_context_threshold,
            is_implementation=len(impl_keywords) > 0,
            is_research=len(research_keywords) > 0,
            is_complex_reasoning=len(complex_keywords) > 0 or difficulty > 0.7,
            is_stuck_recovery=consecutive_no_progress >= 3,
            estimated_difficulty=difficulty,
            keywords_detected=impl_keywords + research_keywords + complex_keywords,
        )

    def _estimate_difficulty(
        self,
        goal: str,
        total_iterations: int,
        consecutive_no_progress: int,
    ) -> float:
        """Estimate task difficulty from 0.0 to 1.0."""
        factors = []

        # Length of goal (longer = more complex)
        if len(goal) > 500:
            factors.append(0.2)
        elif len(goal) > 200:
            factors.append(0.1)

        # Many iterations = harder
        if total_iterations > 20:
            factors.append(0.3)
        elif total_iterations > 10:
            factors.append(0.15)

        # Stuck = harder
        if consecutive_no_progress > 0:
            factors.append(0.2)

        # Complex keywords
        goal_lower = goal.lower()
        if any(kw in goal_lower for kw in COMPLEX_KEYWORDS):
            factors.append(0.3)

        return min(1.0, sum(factors))


class OpusGuard:
    """Guard to preserve scarce Opus tokens.

    Limits the number of Opus (claude-opus-4.5) iterations per day
    to prevent overuse of the most expensive model.
    """

    def __init__(self, daily_limit: int = 50):
        """Initialize Opus guard.

        Args:
            daily_limit: Maximum Opus iterations per day.
        """
        self.daily_limit = daily_limit
        self.used_today = 0
        self.last_reset = date.today()

    def can_use_opus(self) -> bool:
        """Check if Opus can be used."""
        self._maybe_reset()
        return self.used_today < self.daily_limit

    def record_opus_use(self) -> None:
        """Record an Opus usage."""
        self._maybe_reset()
        self.used_today += 1

    def get_remaining(self) -> int:
        """Get remaining Opus iterations today."""
        self._maybe_reset()
        return max(0, self.daily_limit - self.used_today)

    def _maybe_reset(self) -> None:
        """Reset counter if new day."""
        if date.today() != self.last_reset:
            self.used_today = 0
            self.last_reset = date.today()


# Default routing table: (cli, model) for each task type
DEFAULT_ROUTING: dict[TaskType, tuple[str, str]] = {
    TaskType.RESEARCH: ("gemini", "gemini-3-pro"),  # 1M context
    TaskType.IMPLEMENTATION: ("codex", "gpt-5.2"),  # Strong implementation
    TaskType.COMPLEX_REASONING: ("claude", "claude-opus-4.5"),  # Most capable
    TaskType.SIMPLE: ("gemini", "gemini-3-flash"),  # Cheapest
    TaskType.STUCK_RECOVERY: ("claude", "claude-opus-4.5"),  # Smartest for recovery
}


class AutoRouter:
    """Automatically route tasks to optimal CLI/model.

    Uses task analysis to select the best available model
    based on task characteristics and token efficiency.
    """

    def __init__(
        self,
        enabled_clis: Optional[list[str]] = None,
        opus_guard: Optional[OpusGuard] = None,
        prefer_cheap: bool = True,
    ):
        """Initialize auto-router.

        Args:
            enabled_clis: List of enabled CLI backends. None = all enabled.
            opus_guard: OpusGuard instance for Opus rate limiting.
            prefer_cheap: Whether to prefer cheaper models when possible.
        """
        self.enabled_clis: set[str] = (
            set(enabled_clis) if enabled_clis else {"codex", "gemini", "claude"}
        )
        self.opus_guard = opus_guard or OpusGuard()
        self.prefer_cheap = prefer_cheap
        self.analyzer = TaskAnalyzer()

    def route(
        self,
        goal: str,
        context_size: int = 0,
        consecutive_no_progress: int = 0,
        total_iterations: int = 0,
    ) -> tuple[str, str]:
        """Route a task to the best CLI/model.

        Args:
            goal: The task goal.
            context_size: Estimated context size.
            consecutive_no_progress: Consecutive no-progress iterations.
            total_iterations: Total iterations so far.

        Returns:
            Tuple of (cli, model).
        """
        analysis = self.analyzer.analyze(
            goal, context_size, consecutive_no_progress, total_iterations
        )

        return self.route_by_analysis(analysis)

    def route_by_analysis(self, analysis: TaskAnalysis) -> tuple[str, str]:
        """Route based on pre-computed analysis.

        Args:
            analysis: TaskAnalysis from analyzer.

        Returns:
            Tuple of (cli, model).
        """
        task_type = analysis.get_primary_type()

        # Priority 1: Stuck recovery → smartest available
        if analysis.is_stuck_recovery:
            return self._get_smartest()

        # Priority 2: Large context → Gemini
        if analysis.requires_large_context and "gemini" in self.enabled_clis:
            return ("gemini", "gemini-3-pro")

        # Priority 3: Implementation → Codex
        if analysis.is_implementation and "codex" in self.enabled_clis:
            return ("codex", "gpt-5.2")

        # Priority 4: Research → Gemini
        if analysis.is_research and "gemini" in self.enabled_clis:
            return ("gemini", "gemini-3-pro")

        # Priority 5: Complex reasoning → Claude (if budget allows)
        if analysis.is_complex_reasoning and analysis.estimated_difficulty > 0.7:
            if "claude" in self.enabled_clis:
                if self.opus_guard.can_use_opus():
                    self.opus_guard.record_opus_use()
                    return ("claude", "claude-opus-4.5")
                # Fallback to Sonnet if Opus exhausted
                return ("claude", "claude-sonnet-4")

        # Default: prefer cheap if enabled
        if self.prefer_cheap:
            return self._get_cheapest()

        # Use default routing
        default = DEFAULT_ROUTING.get(task_type)
        if default and default[0] in self.enabled_clis:
            return default

        return self._get_cheapest()

    def _get_smartest(self) -> tuple[str, str]:
        """Get the smartest available model."""
        if "claude" in self.enabled_clis:
            if self.opus_guard.can_use_opus():
                self.opus_guard.record_opus_use()
                return ("claude", "claude-opus-4.5")
            return ("claude", "claude-sonnet-4")
        if "codex" in self.enabled_clis:
            return ("codex", "gpt-5.2")
        if "gemini" in self.enabled_clis:
            return ("gemini", "gemini-3-pro")
        # Fallback
        return ("codex", "gpt-5.2")

    def _get_cheapest(self) -> tuple[str, str]:
        """Get the cheapest available model."""
        if "gemini" in self.enabled_clis:
            return ("gemini", "gemini-3-flash")
        if "codex" in self.enabled_clis:
            return ("codex", "gpt-5.2")
        if "claude" in self.enabled_clis:
            return ("claude", "claude-sonnet-4")
        # Fallback
        return ("codex", "gpt-5.2")


def parse_cli_list(cli_string: str) -> list[str]:
    """Parse comma-separated CLI list.

    Args:
        cli_string: String like "codex,gemini" or "auto".

    Returns:
        List of CLI names, or ["auto"] for auto-routing.
    """
    if not cli_string:
        return ["auto"]

    cli_string = cli_string.strip().lower()

    if cli_string == "auto":
        return ["auto"]

    return [c.strip() for c in cli_string.split(",") if c.strip()]


class CLIRouter:
    """Main CLI router supporting explicit lists and auto-routing.

    Parses CLI specification (e.g., "codex,gemini" or "auto") and
    routes tasks appropriately.
    """

    def __init__(
        self,
        cli_spec: str = "auto",
        opus_daily_limit: int = 50,
        prefer_cheap: bool = True,
    ):
        """Initialize CLI router.

        Args:
            cli_spec: CLI specification string ("auto" or comma-separated list).
            opus_daily_limit: Daily Opus iteration limit.
            prefer_cheap: Whether to prefer cheaper models.
        """
        self.cli_list = parse_cli_list(cli_spec)
        self.is_auto = self.cli_list == ["auto"]

        # For auto mode
        self.opus_guard = OpusGuard(daily_limit=opus_daily_limit)
        self.auto_router = (
            AutoRouter(
                enabled_clis=None if self.is_auto else self.cli_list,
                opus_guard=self.opus_guard,
                prefer_cheap=prefer_cheap,
            )
            if self.is_auto
            else None
        )

        # For explicit mode, prepare enabled set
        self.enabled_clis = set(self.cli_list) if not self.is_auto else set()

        # Current index for round-robin in explicit mode
        self._current_index = 0

    def route(
        self,
        goal: str = "",
        context_size: int = 0,
        consecutive_no_progress: int = 0,
        total_iterations: int = 0,
    ) -> tuple[str, str]:
        """Route to appropriate CLI/model.

        Args:
            goal: The task goal (used for auto-routing).
            context_size: Context size estimate (used for auto-routing).
            consecutive_no_progress: No-progress count (used for auto-routing).
            total_iterations: Total iterations (used for auto-routing).

        Returns:
            Tuple of (cli, model).
        """
        if self.is_auto:
            return self.auto_router.route(
                goal, context_size, consecutive_no_progress, total_iterations
            )

        # Explicit mode: use round-robin or task-based selection
        cli = self.cli_list[self._current_index % len(self.cli_list)]
        self._current_index += 1

        return self._get_default_model_for_cli(cli)

    def _get_default_model_for_cli(self, cli: str) -> tuple[str, str]:
        """Get default model for a CLI."""
        defaults = {
            "gemini": ("gemini", "gemini-3-pro"),
            "codex": ("codex", "gpt-5.2"),
            "claude": ("claude", "claude-sonnet-4"),
        }
        return defaults.get(cli, ("codex", "gpt-5.2"))

    def get_enabled_clis(self) -> list[str]:
        """Get list of enabled CLIs."""
        if self.is_auto:
            return ["codex", "gemini", "claude"]
        return self.cli_list

    def get_opus_remaining(self) -> int:
        """Get remaining Opus iterations today."""
        return self.opus_guard.get_remaining()
