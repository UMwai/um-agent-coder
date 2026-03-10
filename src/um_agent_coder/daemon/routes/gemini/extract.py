"""File extraction endpoint — extract structured files from markdown text.

POST /api/gemini/extract-files
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from um_agent_coder.daemon.auth import verify_api_key

from ._file_extractor import extract_files
from ._syntax_validator import validate_code_blocks

router = APIRouter()


class ExtractFilesRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2_000_000)
    validate_syntax: bool = True


class ExtractedFileInfo(BaseModel):
    path: str = ""
    language: str = ""
    content: str = ""
    start_line: int = 0
    end_line: int = 0
    is_complete: bool = True
    syntax_valid: Optional[bool] = None


class SyntaxIssueInfo(BaseModel):
    language: str = ""
    file_path: str = ""
    line: int = 0
    column: int = 0
    message: str = ""
    severity: str = "breaking"


class FileExtractionInfo(BaseModel):
    files: List[ExtractedFileInfo] = []
    total_files: int = 0
    languages: List[str] = []
    truncated_files: int = 0
    syntax_issues: List[SyntaxIssueInfo] = []
    syntax_score: float = 1.0


@router.post("/extract-files", response_model=FileExtractionInfo)
async def extract_files_endpoint(
    req: ExtractFilesRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Extract structured files from markdown text and optionally validate syntax."""
    result = extract_files(req.text)

    syntax_issues: List[SyntaxIssueInfo] = []
    syntax_score = 1.0

    if req.validate_syntax and result.files:
        validation = validate_code_blocks(result.files)
        syntax_score = validation.score
        syntax_issues = [
            SyntaxIssueInfo(
                language=i.language,
                file_path=i.file_path,
                line=i.line,
                column=i.column,
                message=i.message,
                severity=i.severity,
            )
            for i in validation.issues
        ]
        # Update extracted files with syntax validity
        invalid_paths = {i.file_path for i in validation.issues}
        for f in result.files:
            if f.path in invalid_paths:
                f.syntax_valid = False
            elif f.language in ("python", "json", "yaml", "javascript", "typescript"):
                f.syntax_valid = True

    files_info = [
        ExtractedFileInfo(
            path=f.path,
            language=f.language,
            content=f.content,
            start_line=f.start_line,
            end_line=f.end_line,
            is_complete=f.is_complete,
            syntax_valid=f.syntax_valid,
        )
        for f in result.files
    ]

    return FileExtractionInfo(
        files=files_info,
        total_files=result.total_files,
        languages=result.languages,
        truncated_files=result.truncated_files,
        syntax_issues=syntax_issues,
        syntax_score=syntax_score,
    )
