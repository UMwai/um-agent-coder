"""File extractor — pulls structured files from markdown code blocks.

Handles common patterns:
- ```python:src/auth/jwt.py  (language:path)
- ### File: src/auth/jwt.py + fence  (header + fence)
- **`src/auth/jwt.py`** + fence  (bold backtick path)
- Plain ```python  (language only, no path)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExtractedFile:
    """A single file extracted from markdown."""

    path: str
    language: str
    content: str
    start_line: int = 0
    end_line: int = 0
    is_complete: bool = True
    syntax_valid: Optional[bool] = None


@dataclass
class FileExtractionResult:
    """Result of extracting files from a markdown response."""

    files: List[ExtractedFile] = field(default_factory=list)
    total_files: int = 0
    languages: List[str] = field(default_factory=list)
    truncated_files: int = 0


# Language aliases to normalize
_LANG_MAP = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "jsx": "javascript",
    "yml": "yaml",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
}


def _normalize_language(lang: str) -> str:
    lang = lang.strip().lower()
    return _LANG_MAP.get(lang, lang)


def _infer_language_from_path(path: str) -> str:
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".sql": "sql",
        ".html": "html",
        ".css": "css",
        ".md": "markdown",
        ".sh": "bash",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".rb": "ruby",
        ".xml": "xml",
    }
    for ext, lang in ext_map.items():
        if path.endswith(ext):
            return lang
    return ""


# Pattern: ```language:path/to/file
_FENCE_WITH_PATH = re.compile(
    r"```(\w+):([^\n`]+)\n(.*?)```",
    re.DOTALL,
)

# Pattern: ### File: path/to/file  (or ## or # or **File:)
_HEADER_FILE = re.compile(
    r"(?:#{1,4}\s*(?:File|file):\s*|(?:\*\*)?`?)([^\n`*]+?)(?:`?\*\*)?[ \t]*\n"
    r"```(\w*)\n(.*?)```",
    re.DOTALL,
)

# Pattern: **`path/to/file`**\n```
_BOLD_BACKTICK = re.compile(
    r"\*\*`([^`]+)`\*\*[ \t]*\n```(\w*)\n(.*?)```",
    re.DOTALL,
)

# Pattern: plain ```language (no path)
_PLAIN_FENCE = re.compile(
    r"```(\w+)\n(.*?)```",
    re.DOTALL,
)


def extract_files(response_text: str) -> FileExtractionResult:
    """Extract structured files from a markdown response.

    Tries multiple patterns in order of specificity.
    Returns FileExtractionResult with extracted files.
    """
    files: List[ExtractedFile] = []
    seen_ranges: set[tuple[int, int]] = set()
    response_text.split("\n")

    def _line_number(char_pos: int) -> int:
        return response_text[:char_pos].count("\n") + 1

    def _overlaps(start: int, end: int) -> bool:
        for s, e in seen_ranges:
            if start < e and end > s:
                return True
        return False

    # Pass 1: ```language:path patterns
    for m in _FENCE_WITH_PATH.finditer(response_text):
        start, end = m.start(), m.end()
        if _overlaps(start, end):
            continue
        seen_ranges.add((start, end))
        lang = _normalize_language(m.group(1))
        path = m.group(2).strip()
        content = m.group(3)
        files.append(
            ExtractedFile(
                path=path,
                language=lang,
                content=content,
                start_line=_line_number(start),
                end_line=_line_number(end),
                is_complete=not content.rstrip().endswith("..."),
            )
        )

    # Pass 2: header + fence patterns
    for m in _HEADER_FILE.finditer(response_text):
        start, end = m.start(), m.end()
        if _overlaps(start, end):
            continue
        seen_ranges.add((start, end))
        path = m.group(1).strip()
        lang = _normalize_language(m.group(2)) if m.group(2) else _infer_language_from_path(path)
        content = m.group(3)
        files.append(
            ExtractedFile(
                path=path,
                language=lang,
                content=content,
                start_line=_line_number(start),
                end_line=_line_number(end),
                is_complete=not content.rstrip().endswith("..."),
            )
        )

    # Pass 3: **`path`** + fence patterns
    for m in _BOLD_BACKTICK.finditer(response_text):
        start, end = m.start(), m.end()
        if _overlaps(start, end):
            continue
        seen_ranges.add((start, end))
        path = m.group(1).strip()
        lang = _normalize_language(m.group(2)) if m.group(2) else _infer_language_from_path(path)
        content = m.group(3)
        files.append(
            ExtractedFile(
                path=path,
                language=lang,
                content=content,
                start_line=_line_number(start),
                end_line=_line_number(end),
                is_complete=not content.rstrip().endswith("..."),
            )
        )

    # Pass 4: plain ```language (no path) — only if not already matched
    for m in _PLAIN_FENCE.finditer(response_text):
        start, end = m.start(), m.end()
        if _overlaps(start, end):
            continue
        seen_ranges.add((start, end))
        lang = _normalize_language(m.group(1))
        content = m.group(2)
        files.append(
            ExtractedFile(
                path="",
                language=lang,
                content=content,
                start_line=_line_number(start),
                end_line=_line_number(end),
                is_complete=not content.rstrip().endswith("..."),
            )
        )

    # Compute summary
    languages = sorted({f.language for f in files if f.language})
    truncated = sum(1 for f in files if not f.is_complete)

    return FileExtractionResult(
        files=files,
        total_files=len(files),
        languages=languages,
        truncated_files=truncated,
    )
