"""IntelligenceRouter — heuristic model selection based on prompt complexity.

Zero-latency (no LLM call). Scores prompt complexity 0.0–1.0 and routes:
  < 0.3 → Flash (fast, cheap)
  < 0.7 → Pro (balanced)
  >= 0.7 → Pro 3.1 (most capable)
"""

from __future__ import annotations

import re

from um_agent_coder.daemon.routes.gemini.models import GEMINI_MODEL_MAP

# Keywords that signal complex reasoning
REASONING_KEYWORDS = re.compile(
    r"\b(analyze|compare|contrast|evaluate|synthesize|derive|prove|explain why|"
    r"trade-?offs?|implications?|consequences?|advantages?\s+and\s+disadvantages?|"
    r"step[- ]by[- ]step|reasoning|mathematical|theorem|algorithm|complexity|"
    r"architecture|design pattern|refactor|optimize|debug|security|vulnerability)\b",
    re.IGNORECASE,
)

# Patterns indicating multi-step tasks
MULTI_STEP_PATTERNS = re.compile(
    r"\b(first.*then|step\s*\d|1\)|2\)|phase\s*\d|stage\s*\d|"
    r"after that|next,?|finally,?|additionally|furthermore|moreover)\b",
    re.IGNORECASE,
)

CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")


def score_complexity(prompt: str) -> float:
    """Score prompt complexity from 0.0 to 1.0."""
    score = 0.0

    # Word count factor (0–0.25)
    words = len(prompt.split())
    if words > 500:
        score += 0.25
    elif words > 200:
        score += 0.15
    elif words > 50:
        score += 0.05

    # Reasoning keywords (0–0.3)
    reasoning_matches = len(REASONING_KEYWORDS.findall(prompt))
    score += min(reasoning_matches * 0.06, 0.3)

    # Multi-step patterns (0–0.2)
    multi_step_matches = len(MULTI_STEP_PATTERNS.findall(prompt))
    score += min(multi_step_matches * 0.05, 0.2)

    # Code blocks (0–0.15)
    code_blocks = len(CODE_BLOCK_PATTERN.findall(prompt))
    score += min(code_blocks * 0.05, 0.15)

    # Question marks indicating multiple questions (0–0.1)
    questions = prompt.count("?")
    if questions > 3:
        score += 0.1
    elif questions > 1:
        score += 0.05

    return min(score, 1.0)


def select_model(prompt: str, threshold: int = 50) -> str:
    """Select the appropriate model based on prompt complexity.

    Args:
        prompt: The user prompt.
        threshold: Complexity threshold (0–100), mapped to the routing cutoffs.

    Returns:
        Full model name string (e.g., "gemini-3-flash-preview").
    """
    complexity = score_complexity(prompt)

    # Adjust cutoffs based on threshold (higher threshold = more aggressive
    # routing to capable models)
    flash_cutoff = 0.3 * (threshold / 50)
    pro_cutoff = 0.7 * (threshold / 50)

    if complexity < flash_cutoff:
        return GEMINI_MODEL_MAP["flash"]
    elif complexity < pro_cutoff:
        return GEMINI_MODEL_MAP["pro"]
    else:
        return GEMINI_MODEL_MAP["pro-3.1"]
