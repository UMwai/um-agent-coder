"""Stuck recovery system for autonomous loop.

This module provides mechanisms to detect when the autonomous loop is stuck
and attempt various recovery strategies:
- Prompt mutation (rephrase, decompose, constrain)
- Model escalation (move to more capable model)
- Branch exploration (try multiple approaches in parallel)

Reference: specs/autonomous-loop-spec.md Section 3
"""

from .branch_explorer import BranchExplorer, ExplorationBranch
from .model_escalator import ESCALATION_ORDER, ModelEscalator
from .prompt_mutator import MutationType, PromptMutator
from .recovery_manager import RecoveryManager, RecoveryResult, RecoveryStrategy
from .stuck_detector import StuckDetector, StuckState

__all__ = [
    "StuckDetector",
    "StuckState",
    "PromptMutator",
    "MutationType",
    "ModelEscalator",
    "ESCALATION_ORDER",
    "BranchExplorer",
    "ExplorationBranch",
    "RecoveryManager",
    "RecoveryResult",
    "RecoveryStrategy",
]
