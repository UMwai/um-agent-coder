"""Auto eval_context generator — extracts API signatures, models, enums from Python source.

Uses ast.parse() + ast.unparse() to extract structured context from source files,
eliminating the need to hand-craft eval_context for each task.

POST /api/gemini/extract-context
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from um_agent_coder.daemon.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


@dataclass
class ExtractedSignature:
    """A function/method signature extracted from source."""

    module: str
    class_name: str = ""
    name: str = ""
    params: str = ""
    return_type: str = ""
    is_async: bool = False


@dataclass
class ExtractedModel:
    """A data model (Pydantic, dataclass, SQLAlchemy) extracted from source."""

    module: str
    name: str = ""
    base_classes: List[str] = field(default_factory=list)
    fields: List[str] = field(default_factory=list)


@dataclass
class ExtractedEnum:
    """An enum extracted from source."""

    module: str
    name: str = ""
    values: List[str] = field(default_factory=list)


@dataclass
class RepoContext:
    """Aggregated context from multiple source files."""

    signatures: List[ExtractedSignature] = field(default_factory=list)
    models: List[ExtractedModel] = field(default_factory=list)
    enums: List[ExtractedEnum] = field(default_factory=list)
    import_patterns: List[str] = field(default_factory=list)
    file_structure: List[str] = field(default_factory=list)


def _unparse_safe(node) -> str:
    """ast.unparse with fallback for older Python."""
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


def _extract_param_string(args: ast.arguments) -> str:
    """Extract parameter string from function arguments."""
    parts = []

    # Regular args (skip 'self' and 'cls')
    all_args = args.args
    defaults = args.defaults
    default_offset = len(all_args) - len(defaults)

    for i, arg in enumerate(all_args):
        if arg.arg in ("self", "cls"):
            continue
        param = arg.arg
        if arg.annotation:
            param += f": {_unparse_safe(arg.annotation)}"
        if i >= default_offset and defaults:
            default = defaults[i - default_offset]
            param += f" = {_unparse_safe(default)}"
        parts.append(param)

    # *args
    if args.vararg:
        p = f"*{args.vararg.arg}"
        if args.vararg.annotation:
            p += f": {_unparse_safe(args.vararg.annotation)}"
        parts.append(p)

    # keyword-only args
    for i, arg in enumerate(args.kwonlyargs):
        param = arg.arg
        if arg.annotation:
            param += f": {_unparse_safe(arg.annotation)}"
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            param += f" = {_unparse_safe(args.kw_defaults[i])}"
        parts.append(param)

    # **kwargs
    if args.kwarg:
        p = f"**{args.kwarg.arg}"
        if args.kwarg.annotation:
            p += f": {_unparse_safe(args.kwarg.annotation)}"
        parts.append(p)

    return ", ".join(parts)


def extract_python_signatures(
    source: str,
    file_path: str = "",
) -> List[ExtractedSignature]:
    """Extract function/method signatures from Python source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    signatures: List[ExtractedSignature] = []
    module = file_path.replace("/", ".").replace(".py", "").lstrip(".")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Determine class context
            class_name = ""
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    for child in ast.iter_child_nodes(parent):
                        if child is node:
                            class_name = parent.name
                            break

            # Skip private methods (but keep __init__, __call__)
            if node.name.startswith("_") and not node.name.startswith("__"):
                continue

            params = _extract_param_string(node.args)
            return_type = _unparse_safe(node.returns) if node.returns else ""

            signatures.append(
                ExtractedSignature(
                    module=module,
                    class_name=class_name,
                    name=node.name,
                    params=params,
                    return_type=return_type,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                )
            )

    return signatures


def extract_python_models(
    source: str,
    file_path: str = "",
) -> List[ExtractedModel]:
    """Extract data model definitions (Pydantic, dataclass, SQLAlchemy)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    models: List[ExtractedModel] = []
    module = file_path.replace("/", ".").replace(".py", "").lstrip(".")

    # Known model base classes
    model_bases = {
        "BaseModel",
        "BaseSettings",
        "Base",
        "DeclarativeBase",
        "Model",
    }

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        bases = [_unparse_safe(b) for b in node.bases]
        is_model = any(any(mb in b for mb in model_bases) for b in bases)

        # Also check for @dataclass decorator
        is_dataclass = any(
            (isinstance(d, ast.Name) and d.id == "dataclass")
            or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
            or (
                isinstance(d, ast.Call)
                and isinstance(d.func, ast.Name)
                and d.func.id == "dataclass"
            )
            for d in node.decorator_list
        )

        if not is_model and not is_dataclass:
            continue

        fields_list: List[str] = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                f = item.target.id
                if item.annotation:
                    f += f": {_unparse_safe(item.annotation)}"
                if item.value:
                    f += f" = {_unparse_safe(item.value)}"
                fields_list.append(f)

        models.append(
            ExtractedModel(
                module=module,
                name=node.name,
                base_classes=bases,
                fields=fields_list,
            )
        )

    return models


def extract_python_enums(
    source: str,
    file_path: str = "",
) -> List[ExtractedEnum]:
    """Extract Enum subclass definitions."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    enums: List[ExtractedEnum] = []
    module = file_path.replace("/", ".").replace(".py", "").lstrip(".")

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        bases = [_unparse_safe(b) for b in node.bases]
        is_enum = any("Enum" in b for b in bases)
        if not is_enum:
            continue

        values: List[str] = []
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        val = _unparse_safe(item.value)
                        values.append(f"{target.id} = {val}")

        enums.append(
            ExtractedEnum(
                module=module,
                name=node.name,
                values=values,
            )
        )

    return enums


def generate_eval_context(file_contents: Dict[str, str]) -> str:
    """Generate eval_context from a dict of file_path → source_code.

    Returns markdown-formatted reference material.
    """
    ctx = RepoContext()

    for path, source in sorted(file_contents.items()):
        if not path.endswith(".py"):
            continue

        ctx.file_structure.append(path)
        ctx.signatures.extend(extract_python_signatures(source, path))
        ctx.models.extend(extract_python_models(source, path))
        ctx.enums.extend(extract_python_enums(source, path))

        # Extract import patterns
        try:
            tree = ast.parse(source)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    names = ", ".join(a.name for a in node.names[:5])
                    ctx.import_patterns.append(f"from {node.module} import {names}")
        except SyntaxError:
            pass

    return to_eval_context(ctx)


def to_eval_context(ctx: RepoContext) -> str:
    """Render RepoContext as markdown eval_context."""
    sections: List[str] = []

    # File structure
    if ctx.file_structure:
        sections.append("## File Structure\n```\n" + "\n".join(ctx.file_structure) + "\n```")

    # API Signatures
    if ctx.signatures:
        lines = ["## API Signatures\n"]
        for sig in ctx.signatures:
            prefix = "async " if sig.is_async else ""
            cls = f"{sig.class_name}." if sig.class_name else ""
            ret = f" -> {sig.return_type}" if sig.return_type else ""
            lines.append(f"- `{prefix}def {cls}{sig.name}({sig.params}){ret}`")
            if sig.module:
                lines.append(f"  - Module: `{sig.module}`")
        sections.append("\n".join(lines))

    # Data Models
    if ctx.models:
        lines = ["## Data Models\n"]
        for model in ctx.models:
            bases = ", ".join(model.base_classes)
            lines.append(f"### {model.name}({bases})")
            if model.module:
                lines.append(f"Module: `{model.module}`")
            lines.append("```python")
            for f in model.fields:
                lines.append(f"    {f}")
            lines.append("```")
        sections.append("\n".join(lines))

    # Enums
    if ctx.enums:
        lines = ["## Enums\n"]
        for enum in ctx.enums:
            lines.append(f"### {enum.name}")
            if enum.module:
                lines.append(f"Module: `{enum.module}`")
            for v in enum.values:
                lines.append(f"- `{v}`")
        sections.append("\n".join(lines))

    # Import patterns
    if ctx.import_patterns:
        # Deduplicate and limit
        unique = sorted(set(ctx.import_patterns))[:30]
        sections.append("## Import Patterns\n```python\n" + "\n".join(unique) + "\n```")

    return "\n\n".join(sections)


# --- FastAPI endpoint ---


class ExtractContextRequest(BaseModel):
    files: Dict[str, str] = Field(..., description="Dict of file_path → source_content.")
    focus_on: Optional[List[str]] = Field(
        default=None, description="Limit extraction to these file paths."
    )


class ExtractContextResponse(BaseModel):
    eval_context: str = ""
    file_count: int = 0
    signatures_count: int = 0
    models_count: int = 0
    enums_count: int = 0


@router.post("/extract-context", response_model=ExtractContextResponse)
async def extract_context_endpoint(
    req: ExtractContextRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Extract eval_context from source files."""
    files = req.files

    if req.focus_on:
        files = {k: v for k, v in files.items() if k in req.focus_on}

    eval_context = generate_eval_context(files)

    # Count extracted elements
    ctx = RepoContext()
    for path, source in files.items():
        if path.endswith(".py"):
            ctx.signatures.extend(extract_python_signatures(source, path))
            ctx.models.extend(extract_python_models(source, path))
            ctx.enums.extend(extract_python_enums(source, path))

    return ExtractContextResponse(
        eval_context=eval_context,
        file_count=len(files),
        signatures_count=len(ctx.signatures),
        models_count=len(ctx.models),
        enums_count=len(ctx.enums),
    )
