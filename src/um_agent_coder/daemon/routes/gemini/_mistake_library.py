"""Persistent mistake library — learns from evaluation failures.

Records failing checks from evaluations, normalizes them to patterns,
and retrieves relevant mistakes to prepend to generation prompts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

from ._evaluator import AccuracyCheck

logger = logging.getLogger(__name__)


@dataclass
class MistakeEntry:
    """A normalized mistake pattern from the library."""

    pattern: str
    severity: str
    example_detail: str
    occurrences: int
    last_seen: str


def normalize_check_description(check: str) -> str:
    """Normalize a check description to group similar mistakes.

    Strips specific identifiers (function names, variable names, file paths)
    to create a reusable pattern.

    Examples:
        "Function foo() has wrong params" → "function_wrong_params"
        "Function bar() has wrong params" → "function_wrong_params"
        "Missing import for src/auth/jwt.py" → "missing_import"
        "Undefined variable 'user_id'" → "undefined_variable"
    """
    text = check.lower().strip()

    # Remove quoted strings
    text = re.sub(r"['\"`][^'\"`]*['\"`]", "", text)
    # Remove paths (anything with /)
    text = re.sub(r"[\w./]+/[\w./]+", "", text)
    # Remove function calls like foo(), bar.baz()
    text = re.sub(r"\w+(?:\.\w+)*\(\)", "", text)
    # Remove specific identifiers in backticks
    text = re.sub(r"`[^`]+`", "", text)

    # Extract key phrases
    text = re.sub(r"\s+", " ", text).strip()

    # Convert to snake_case pattern
    words = re.findall(r"[a-z]+", text)

    # Keep meaningful words only
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "and",
        "or",
        "not",
        "it",
        "its",
        "this",
        "that",
        "be",
        "been",
    }
    words = [w for w in words if w not in stop_words and len(w) > 1]

    if not words:
        return check.lower().strip()[:80]

    return "_".join(words[:5])


async def record_failures(
    db,
    checks: List[AccuracyCheck],
    dimension: str,
) -> int:
    """Record failing checks as mistake patterns in the DB.

    Returns number of patterns recorded.
    """
    recorded = 0
    for check in checks:
        if check.status != "fail":
            continue

        pattern = normalize_check_description(check.check)
        if not pattern:
            continue

        try:
            await db.upsert_mistake(
                pattern=pattern,
                severity=check.severity,
                dimension=dimension,
                example_detail=check.detail[:500] if check.detail else "",
            )
            recorded += 1
        except Exception as e:
            logger.warning("Failed to record mistake '%s': %s", pattern, e)

    return recorded


async def get_relevant_mistakes(
    db,
    prompt: str,
    top_k: int = 10,
    min_occurrences: int = 2,
) -> List[MistakeEntry]:
    """Get mistakes relevant to the current prompt.

    Uses keyword overlap + frequency + recency scoring.
    """
    # Extract keywords from prompt
    words = re.findall(r"[a-z]{3,}", prompt.lower())
    # Deduplicate and take most common
    word_counts: dict[str, int] = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    keywords = sorted(word_counts, key=word_counts.get, reverse=True)[:20]

    try:
        rows = await db.search_mistakes(
            keywords=keywords,
            top_k=top_k,
            min_occurrences=min_occurrences,
        )
    except Exception as e:
        logger.warning("Failed to search mistakes: %s", e)
        # Fallback to top mistakes
        try:
            rows = await db.get_top_mistakes(top_k, min_occurrences)
        except Exception:
            return []

    return [
        MistakeEntry(
            pattern=r["pattern"],
            severity=r.get("severity", "breaking"),
            example_detail=r.get("example_detail", ""),
            occurrences=r.get("occurrences", 1),
            last_seen=r.get("last_seen", ""),
        )
        for r in rows
    ]


def build_mistake_preamble(mistakes: List[MistakeEntry]) -> str:
    """Format mistakes as a preamble to prepend to generation prompts."""
    if not mistakes:
        return ""

    lines = ["[COMMON MISTAKES — Avoid these known issues:]"]
    for m in mistakes:
        detail = f" — {m.example_detail}" if m.example_detail else ""
        lines.append(f"- [{m.severity.upper()}] {m.pattern} " f"(seen {m.occurrences}x){detail}")
    lines.append("[END COMMON MISTAKES]\n")

    return "\n".join(lines)
