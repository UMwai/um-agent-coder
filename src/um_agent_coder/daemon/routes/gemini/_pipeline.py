"""PromptEnhancementPipeline — 4-stage prompt enhancement.

Stages:
  1. ChainOfThoughtInjector  — step-by-step reasoning directive
  2. ContextEnricher         — domain context framing
  3. ConstraintClarifier     — precision/accuracy constraints
  4. OutputFormatSpecifier   — output format instructions
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# --- Domain detection patterns ---

DOMAIN_PATTERNS = {
    "code": re.compile(
        r"(```|def\s+\w+|class\s+\w+|function\s+\w+|import\s+\w+|"
        r"\b(python|javascript|typescript|java|rust|go|sql|html|css|react|api|endpoint|bug|error|exception)\b)",
        re.IGNORECASE,
    ),
    "math": re.compile(
        r"(\b(equation|formula|integral|derivative|matrix|vector|probability|"
        r"statistics|theorem|proof|calculate|solve|compute)\b|[=∫∑∏√±])",
        re.IGNORECASE,
    ),
    "science": re.compile(
        r"\b(experiment|hypothesis|molecule|protein|gene|quantum|relativity|"
        r"thermodynamics|evolution|ecology|chemistry|physics|biology|neuroscience)\b",
        re.IGNORECASE,
    ),
    "writing": re.compile(
        r"\b(essay|article|story|blog|write|draft|summarize|paraphrase|"
        r"rewrite|tone|style|narrative|outline|paragraph)\b",
        re.IGNORECASE,
    ),
}

# Check if user already included CoT
COT_PATTERN = re.compile(
    r"\b(step[- ]by[- ]step|think.*through|chain[- ]of[- ]thought|"
    r"let'?s think|reason.*carefully|show.*work|explain.*reasoning)\b",
    re.IGNORECASE,
)

# Desired output format detection
FORMAT_PATTERNS = {
    "json": re.compile(r"\b(json|JSON)\b"),
    "table": re.compile(r"\b(table|tabular|csv|spreadsheet)\b", re.IGNORECASE),
    "code": re.compile(r"\b(code|function|class|script|program|implement)\b", re.IGNORECASE),
    "list": re.compile(r"\b(list|bullet|numbered|enumerate)\b", re.IGNORECASE),
}


@dataclass
class EnhancementResult:
    """Result of running the enhancement pipeline."""
    original: str
    enhanced: str
    stages_applied: List[str] = field(default_factory=list)


def _inject_chain_of_thought(prompt: str) -> tuple[str, bool]:
    """Stage 1: Add CoT directive if not already present."""
    if COT_PATTERN.search(prompt):
        return prompt, False
    return prompt + "\n\nThink step by step before answering.", True


def _enrich_context(prompt: str, domain_hint: Optional[str] = None) -> tuple[str, bool]:
    """Stage 2: Add domain context framing."""
    domain = domain_hint
    if not domain:
        for name, pattern in DOMAIN_PATTERNS.items():
            if pattern.search(prompt):
                domain = name
                break

    if not domain:
        return prompt, False

    context_map = {
        "code": "You are an expert software engineer. Focus on correctness, best practices, and clean code.",
        "math": "You are an expert mathematician. Show your work and verify your calculations.",
        "science": "You are a rigorous scientist. Cite established knowledge and distinguish fact from hypothesis.",
        "writing": "You are a skilled writer. Focus on clarity, coherence, and appropriate tone.",
    }

    context = context_map.get(domain)
    if not context:
        return prompt, False

    return f"[Context: {context}]\n\n{prompt}", True


def _clarify_constraints(prompt: str) -> tuple[str, bool]:
    """Stage 3: Add precision/accuracy constraints."""
    constraint = (
        "\n\nBe precise and accurate. If you are uncertain about something, "
        "explicitly say so rather than guessing. Distinguish between facts and opinions."
    )
    return prompt + constraint, True


def _specify_output_format(prompt: str) -> tuple[str, bool]:
    """Stage 4: Detect and specify output format."""
    for fmt, pattern in FORMAT_PATTERNS.items():
        if pattern.search(prompt):
            directives = {
                "json": "\n\nReturn your answer as valid JSON.",
                "table": "\n\nFormat your answer as a markdown table.",
                "code": "\n\nProvide your answer as clean, well-commented code.",
                "list": "\n\nFormat your answer as a clear, organized list.",
            }
            directive = directives.get(fmt)
            if directive:
                return prompt + directive, True
    return prompt, False


def enhance_prompt(
    prompt: str,
    *,
    enable_cot: bool = True,
    enable_context: bool = True,
    enable_constraints: bool = True,
    enable_format: bool = True,
    domain_hint: Optional[str] = None,
) -> EnhancementResult:
    """Run the full enhancement pipeline on a prompt.

    Each stage can be individually enabled/disabled.
    """
    enhanced = prompt
    stages = []

    if enable_cot:
        enhanced, applied = _inject_chain_of_thought(enhanced)
        if applied:
            stages.append("chain_of_thought")

    if enable_context:
        enhanced, applied = _enrich_context(enhanced, domain_hint=domain_hint)
        if applied:
            stages.append("context_enrichment")

    if enable_constraints:
        enhanced, applied = _clarify_constraints(enhanced)
        if applied:
            stages.append("constraint_clarification")

    if enable_format:
        enhanced, applied = _specify_output_format(enhanced)
        if applied:
            stages.append("output_format")

    return EnhancementResult(
        original=prompt,
        enhanced=enhanced,
        stages_applied=stages,
    )
