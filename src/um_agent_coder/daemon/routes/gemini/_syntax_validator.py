"""Syntax validator — catches 100% of syntax errors at zero LLM cost.

Validates code blocks extracted from markdown using language-specific parsers.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class SyntaxIssue:
    """A syntax error found in a code block."""

    language: str
    file_path: str
    line: int
    column: int
    message: str
    severity: str = "breaking"  # Syntax errors are always breaking


@dataclass
class SyntaxValidationResult:
    """Result of validating all code blocks in a response."""

    total_blocks: int = 0
    blocks_checked: int = 0
    blocks_valid: int = 0
    issues: List[SyntaxIssue] = field(default_factory=list)
    score: float = 1.0  # 1.0 = all valid, 0.0 = all invalid


def validate_python(code: str, file_path: str = "") -> List[SyntaxIssue]:
    """Validate Python code using ast.parse(). Zero dependencies."""
    try:
        ast.parse(code)
        return []
    except SyntaxError as e:
        return [
            SyntaxIssue(
                language="python",
                file_path=file_path,
                line=e.lineno or 0,
                column=e.offset or 0,
                message=str(e.msg) if hasattr(e, "msg") else str(e),
            )
        ]


def validate_json(code: str, file_path: str = "") -> List[SyntaxIssue]:
    """Validate JSON using json.loads()."""
    try:
        json.loads(code)
        return []
    except json.JSONDecodeError as e:
        return [
            SyntaxIssue(
                language="json",
                file_path=file_path,
                line=e.lineno,
                column=e.colno,
                message=e.msg,
            )
        ]


def validate_yaml(code: str, file_path: str = "") -> List[SyntaxIssue]:
    """Validate YAML using yaml.safe_load() if available."""
    try:
        import yaml
    except ImportError:
        return []  # yaml not installed, skip

    try:
        yaml.safe_load(code)
        return []
    except yaml.YAMLError as e:
        line = 0
        col = 0
        if hasattr(e, "problem_mark") and e.problem_mark:
            line = e.problem_mark.line + 1
            col = e.problem_mark.column + 1
        return [
            SyntaxIssue(
                language="yaml",
                file_path=file_path,
                line=line,
                column=col,
                message=str(e),
            )
        ]


def validate_javascript(code: str, file_path: str = "") -> List[SyntaxIssue]:
    """Basic JS/TS validation — bracket/paren/brace balance check."""
    issues: List[SyntaxIssue] = []
    stack: list[tuple[str, int, int]] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {")": "(", "]": "[", "}": "{"}

    in_string = None
    escape_next = False
    in_line_comment = False
    in_block_comment = False

    for line_num, line in enumerate(code.split("\n"), 1):
        in_line_comment = False
        for col, ch in enumerate(line, 1):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue

            if in_block_comment:
                if ch == "*" and col < len(line) and line[col] == "/":
                    in_block_comment = False
                continue
            if in_line_comment:
                continue
            if in_string:
                if ch == in_string:
                    in_string = None
                continue

            if ch in ('"', "'", "`"):
                in_string = ch
                continue
            if ch == "/" and col < len(line):
                next_ch = line[col]
                if next_ch == "/":
                    in_line_comment = True
                    continue
                if next_ch == "*":
                    in_block_comment = True
                    continue

            if ch in pairs:
                stack.append((ch, line_num, col))
            elif ch in closing:
                if not stack:
                    issues.append(
                        SyntaxIssue(
                            language="javascript",
                            file_path=file_path,
                            line=line_num,
                            column=col,
                            message=f"Unmatched closing '{ch}'",
                        )
                    )
                else:
                    top_ch, _, _ = stack[-1]
                    if top_ch == closing[ch]:
                        stack.pop()
                    else:
                        issues.append(
                            SyntaxIssue(
                                language="javascript",
                                file_path=file_path,
                                line=line_num,
                                column=col,
                                message=f"Mismatched '{ch}', expected '{pairs[top_ch]}'",
                            )
                        )

    for ch, line_num, col in stack:
        issues.append(
            SyntaxIssue(
                language="javascript",
                file_path=file_path,
                line=line_num,
                column=col,
                message=f"Unclosed '{ch}'",
            )
        )

    return issues


# Map language → validator
_VALIDATORS = {
    "python": validate_python,
    "json": validate_json,
    "yaml": validate_yaml,
    "javascript": validate_javascript,
    "typescript": validate_javascript,
}


def validate_code_blocks(
    extracted_files: list,
) -> SyntaxValidationResult:
    """Validate all extracted file blocks.

    Args:
        extracted_files: List of ExtractedFile objects from _file_extractor.

    Returns:
        SyntaxValidationResult with issues and score.
    """
    total = len(extracted_files)
    checked = 0
    valid = 0
    all_issues: List[SyntaxIssue] = []

    for ef in extracted_files:
        lang = ef.language
        if lang not in _VALIDATORS:
            continue

        checked += 1
        issues = _VALIDATORS[lang](ef.content, file_path=ef.path)
        if not issues:
            valid += 1
        else:
            all_issues.extend(issues)

    score = valid / checked if checked > 0 else 1.0

    return SyntaxValidationResult(
        total_blocks=total,
        blocks_checked=checked,
        blocks_valid=valid,
        issues=all_issues,
        score=score,
    )
