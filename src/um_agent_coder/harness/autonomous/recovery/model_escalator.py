"""Model escalation for stuck recovery.

When stuck, escalate to more capable models in a defined order,
from cheapest/fastest to most expensive/capable.
"""

from dataclasses import dataclass
from typing import Optional

# Escalation order from cheapest to most capable
# Each entry: (cli, model, capability_score)
ESCALATION_ORDER: list[tuple[str, str, float]] = [
    ("gemini", "gemini-3-flash", 0.3),  # Cheapest, fastest
    ("gemini", "gemini-3-pro", 0.5),  # Better reasoning, 1M context
    ("codex", "gpt-5.2", 0.7),  # Strong implementation
    ("claude", "claude-sonnet-4", 0.85),  # Good balance
    ("claude", "claude-opus-4.5", 1.0),  # Most capable (use sparingly)
]


@dataclass
class EscalationResult:
    """Result of model escalation."""

    previous_cli: str
    previous_model: str
    new_cli: str
    new_model: str
    capability_increase: float
    is_max_capability: bool


class ModelEscalator:
    """Escalate to more capable models when stuck.

    Moves through a defined order of models from cheapest/fastest
    to most expensive/capable.
    """

    def __init__(
        self,
        enabled_clis: Optional[list[str]] = None,
        escalation_order: Optional[list[tuple[str, str, float]]] = None,
    ):
        """Initialize model escalator.

        Args:
            enabled_clis: List of enabled CLI backends. If None, all are enabled.
            escalation_order: Custom escalation order. Defaults to ESCALATION_ORDER.
        """
        self.enabled_clis = set(enabled_clis) if enabled_clis else None
        self.escalation_order = escalation_order or ESCALATION_ORDER

    def _filter_order(self) -> list[tuple[str, str, float]]:
        """Filter escalation order to only enabled CLIs."""
        if self.enabled_clis is None:
            return self.escalation_order

        return [
            (cli, model, score)
            for cli, model, score in self.escalation_order
            if cli in self.enabled_clis
        ]

    def get_current_index(self, cli: str, model: str) -> int:
        """Get index of current model in escalation order.

        Args:
            cli: Current CLI backend.
            model: Current model.

        Returns:
            Index in escalation order, or -1 if not found.
        """
        filtered = self._filter_order()
        for i, (c, m, _) in enumerate(filtered):
            if c == cli and m == model:
                return i
        return -1

    def get_capability_score(self, cli: str, model: str) -> float:
        """Get capability score for a model.

        Args:
            cli: CLI backend.
            model: Model name.

        Returns:
            Capability score (0.0-1.0), or 0.0 if not found.
        """
        for c, m, score in self.escalation_order:
            if c == cli and m == model:
                return score
        return 0.0

    def can_escalate(self, cli: str, model: str) -> bool:
        """Check if escalation is possible from current model.

        Args:
            cli: Current CLI backend.
            model: Current model.

        Returns:
            True if a more capable model is available.
        """
        filtered = self._filter_order()
        if not filtered:
            return False

        current_idx = self.get_current_index(cli, model)
        if current_idx < 0:
            # Not in order, can escalate to first model
            return True

        return current_idx < len(filtered) - 1

    def escalate(self, cli: str, model: str) -> Optional[EscalationResult]:
        """Escalate to next more capable model.

        Args:
            cli: Current CLI backend.
            model: Current model.

        Returns:
            EscalationResult with new model, or None if already at max.
        """
        filtered = self._filter_order()
        if not filtered:
            return None

        current_idx = self.get_current_index(cli, model)

        if current_idx < 0:
            # Not in order, start at beginning
            new_cli, new_model, new_score = filtered[0]
            return EscalationResult(
                previous_cli=cli,
                previous_model=model,
                new_cli=new_cli,
                new_model=new_model,
                capability_increase=new_score,
                is_max_capability=(len(filtered) == 1),
            )

        if current_idx >= len(filtered) - 1:
            # Already at max
            return None

        # Move to next
        prev_cli, prev_model, prev_score = filtered[current_idx]
        new_cli, new_model, new_score = filtered[current_idx + 1]

        return EscalationResult(
            previous_cli=prev_cli,
            previous_model=prev_model,
            new_cli=new_cli,
            new_model=new_model,
            capability_increase=new_score - prev_score,
            is_max_capability=(current_idx + 1 == len(filtered) - 1),
        )

    def get_smartest(self) -> Optional[tuple[str, str]]:
        """Get the smartest available model.

        Returns:
            Tuple of (cli, model) for smartest model, or None if none available.
        """
        filtered = self._filter_order()
        if not filtered:
            return None
        cli, model, _ = filtered[-1]
        return (cli, model)

    def get_cheapest(self) -> Optional[tuple[str, str]]:
        """Get the cheapest available model.

        Returns:
            Tuple of (cli, model) for cheapest model, or None if none available.
        """
        filtered = self._filter_order()
        if not filtered:
            return None
        cli, model, _ = filtered[0]
        return (cli, model)

    def get_all_available(self) -> list[tuple[str, str, float]]:
        """Get all available models in escalation order.

        Returns:
            List of (cli, model, capability_score) tuples.
        """
        return self._filter_order()

    def is_at_max_capability(self, cli: str, model: str) -> bool:
        """Check if current model is at max capability.

        Args:
            cli: Current CLI backend.
            model: Current model.

        Returns:
            True if at maximum capability model.
        """
        filtered = self._filter_order()
        if not filtered:
            return True

        current_idx = self.get_current_index(cli, model)
        return current_idx == len(filtered) - 1
