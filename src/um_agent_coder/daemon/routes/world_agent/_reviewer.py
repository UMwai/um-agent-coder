"""Repo reviewer: analyzes a repository against a goal and produces gap analysis.

Scans repo locally (file tree, docs, git history, tests, packages) and
calls Gemini Pro to assess KPI coverage, identify gaps, and recommend tasks.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from um_agent_coder.daemon.routes.world_agent.models import Goal

logger = logging.getLogger(__name__)

REVIEW_SYSTEM_PROMPT = """\
You are a senior engineering reviewer analysing a repository against a goal.

Given REPO CONTEXT (file tree, docs, git history, tests) and a GOAL with KPIs,
produce a structured gap analysis.

Return a JSON object:
{
  "review_summary": "2-3 sentence overall assessment",
  "kpi_assessment": [
    {
      "metric": "KPI metric name",
      "target": "target value from the goal",
      "current_estimate": "your best estimate of current state",
      "status": "met|partial|unmet"
    }
  ],
  "gaps": [
    {
      "description": "what is missing or incomplete",
      "severity": "critical|high|medium|low",
      "category": "feature|test|infra|docs"
    }
  ],
  "recommended_tasks": [
    {
      "title": "short imperative title",
      "description": "detailed description of what to do",
      "priority": 1-10,
      "estimated_effort": "small|medium|large",
      "cli": "codex|gemini|claude",
      "timeout": "30min|1h|2h",
      "success_criteria": "how to verify completion"
    }
  ]
}

Rules:
- Be specific: reference actual files, directories, and code patterns
- Prioritise gaps that block KPI targets
- Each recommended task should be independently actionable
- Maximum 10 recommended tasks
- If the repo is well-aligned with the goal, gaps list can be short
"""


def _run_cmd(args: List[str], cwd: str, timeout: int = 10) -> str:
    """Run a subprocess and return stdout, empty string on failure."""
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except Exception as e:
        logger.debug("Command %s failed: %s", args[:3], e)
        return ""


def _read_file_head(path: Path, max_lines: int = 200) -> str:
    """Read first N lines of a file, return empty string if missing."""
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(errors="replace").splitlines()[:max_lines]
        return "\n".join(lines)
    except Exception:
        return ""


def _collect_file_tree(repo: Path) -> str:
    """Collect top-level listing plus key subdirectories."""
    parts: List[str] = []

    # Top-level
    try:
        entries = sorted(
            p.name + ("/" if p.is_dir() else "")
            for p in repo.iterdir()
            if not p.name.startswith(".")
        )
        parts.append("## Top-level\n" + "\n".join(entries[:60]))
    except Exception:
        pass

    # Key subdirectories
    for subdir in ("src", "tests", "specs", "lib", "cmd", "pkg", "app"):
        sub_path = repo / subdir
        if not sub_path.is_dir():
            continue
        try:
            tree_output = _run_cmd(
                ["find", subdir, "-type", "f", "-not", "-path", "*/.*"],
                cwd=str(repo),
                timeout=10,
            )
            lines = tree_output.splitlines()
            if lines:
                truncated = lines[:80]
                suffix = f"\n... ({len(lines) - 80} more)" if len(lines) > 80 else ""
                parts.append(f"## {subdir}/\n" + "\n".join(truncated) + suffix)
        except Exception:
            pass

    return "\n\n".join(parts)


def _collect_test_count(repo: Path) -> str:
    """Collect test count via pytest --co without running tests."""
    output = _run_cmd(
        ["python3", "-m", "pytest", "--co", "-q", "--no-header"],
        cwd=str(repo),
        timeout=30,
    )
    if not output:
        return "Could not collect test info."
    # Last line is usually "N tests collected"
    lines = output.splitlines()
    summary = [
        ln
        for ln in lines
        if "test" in ln.lower() and ("collected" in ln.lower() or "selected" in ln.lower())
    ]
    if summary:
        return summary[-1]
    # Fallback: count lines that look like test items
    test_items = [ln for ln in lines if "::" in ln]
    return f"{len(test_items)} tests discovered" if test_items else output[-500:]


def _collect_specs_content(repo: Path) -> str:
    """Read spec files for deep review."""
    specs_dir = repo / "specs"
    if not specs_dir.is_dir():
        return ""
    parts: List[str] = []
    spec_files = sorted(specs_dir.rglob("*.md"))[:10]
    for sf in spec_files:
        rel = sf.relative_to(repo)
        content = _read_file_head(sf, max_lines=100)
        if content:
            parts.append(f"### {rel}\n{content}")
    return "\n\n".join(parts)


def _collect_source_sample(repo: Path) -> str:
    """Sample key source files for deep review."""
    parts: List[str] = []
    src_dir = repo / "src"
    if not src_dir.is_dir():
        src_dir = repo

    py_files = sorted(src_dir.rglob("*.py"))
    # Pick a few representative files (not __init__.py)
    candidates = [f for f in py_files if f.name != "__init__.py"][:8]
    for cf in candidates:
        rel = cf.relative_to(repo)
        content = _read_file_head(cf, max_lines=60)
        if content:
            parts.append(f"### {rel}\n```python\n{content}\n```")
    return "\n\n".join(parts[:5])


def _build_review_prompt(
    repo_path: str,
    goal: Goal,
    depth: str,
) -> str:
    """Build the user prompt with collected repo context."""
    repo = Path(repo_path)
    sections: List[str] = []

    # --- Always collected (quick + standard + deep) ---

    # README / CLAUDE.md
    for doc_name in ("README.md", "CLAUDE.md"):
        content = _read_file_head(repo / doc_name, max_lines=200)
        if content:
            sections.append(f"## {doc_name}\n{content}")

    # Roadmap
    for roadmap_name in ("specs/roadmap.md", "specs/ROADMAP.md"):
        content = _read_file_head(repo / roadmap_name, max_lines=200)
        if content:
            sections.append(f"## {roadmap_name}\n{content}")
            break

    # Git log
    git_log = _run_cmd(
        ["git", "log", "--oneline", "-20"],
        cwd=repo_path,
    )
    if git_log:
        sections.append(f"## Recent commits (last 20)\n{git_log}")

    # Git status
    git_status = _run_cmd(
        ["git", "status", "--porcelain"],
        cwd=repo_path,
    )
    if git_status:
        lines = git_status.splitlines()[:30]
        sections.append(
            f"## Uncommitted changes ({len(git_status.splitlines())} files)\n" + "\n".join(lines)
        )

    # --- Standard depth additions ---

    if depth in ("standard", "deep"):
        tree = _collect_file_tree(repo)
        if tree:
            sections.append(f"## File tree\n{tree}")

        # Package info
        for pkg_file in ("pyproject.toml", "package.json", "Cargo.toml", "go.mod"):
            content = _read_file_head(repo / pkg_file, max_lines=50)
            if content:
                sections.append(f"## {pkg_file}\n{content}")
                break

        test_info = _collect_test_count(repo)
        if test_info:
            sections.append(f"## Test summary\n{test_info}")

    # --- Deep depth additions ---

    if depth == "deep":
        specs_content = _collect_specs_content(repo)
        if specs_content:
            sections.append(f"## Spec files\n{specs_content}")

        source_sample = _collect_source_sample(repo)
        if source_sample:
            sections.append(f"## Source samples\n{source_sample}")

    # Build goal section
    goal_text = f"**{goal.id}** (priority={goal.priority}): {goal.name}\n{goal.description}\n"
    if goal.constraints:
        goal_text += "\nConstraints:\n" + "\n".join(f"- {c}" for c in goal.constraints)
    if goal.kpis:
        goal_text += "\n\nKPIs:\n"
        for kpi in goal.kpis:
            current = f" (current: {kpi.current})" if kpi.current else ""
            goal_text += f"- {kpi.metric}: target={kpi.target}{current}\n"
    if goal.projects:
        goal_text += "\nProjects:\n"
        for p in goal.projects:
            goal_text += f"- {p.repo} ({p.role})\n"

    repo_context = "\n\n".join(sections)

    prompt = (
        f"## GOAL\n{goal_text}\n\n"
        f"## REPO: {repo_path}\n\n{repo_context}\n\n"
        "Analyse the repo against the goal. Produce the JSON gap analysis."
    )
    return prompt


def _parse_review_response(text: str) -> Dict[str, Any]:
    """Parse the LLM review response into structured data."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("Failed to parse review response")
                return {}
        else:
            return {}
    return data


async def review_repo(
    repo_path: str,
    goal: Goal,
    depth: str = "standard",
) -> dict:
    """Analyse a repo against a goal and produce a gap analysis.

    Args:
        repo_path: Absolute path to the repository root.
        goal: The Goal to evaluate against.
        depth: Review depth — "quick", "standard", or "deep".

    Returns:
        Dict with repo, goal_id, kpi_assessment, gaps, recommended_tasks,
        review_summary.
    """
    if depth not in ("quick", "standard", "deep"):
        depth = "standard"

    repo = Path(repo_path)
    if not repo.is_dir():
        return {
            "repo": repo_path,
            "goal_id": goal.id,
            "error": f"Repo path not found: {repo_path}",
            "kpi_assessment": [],
            "gaps": [],
            "recommended_tasks": [],
            "review_summary": "Repository path does not exist.",
        }

    from um_agent_coder.daemon.app import get_llm_router, get_settings

    settings = get_settings()
    model = settings.gemini_model_pro

    user_prompt = _build_review_prompt(repo_path, goal, depth)

    try:
        router = get_llm_router()

        llm_result = await router.generate(
            prompt=user_prompt,
            system_prompt=REVIEW_SYSTEM_PROMPT,
            model=model,
            temperature=0.3,
            max_tokens=4096,
            provider=settings.world_agent_llm_provider or None,
        )

        data = _parse_review_response(llm_result["text"])
        if not data:
            logger.warning("Review LLM returned unparseable response")
            return {
                "repo": repo_path,
                "goal_id": goal.id,
                "error": "Failed to parse LLM review response",
                "kpi_assessment": [],
                "gaps": [],
                "recommended_tasks": [],
                "review_summary": "",
            }

        # Cap recommended tasks at 10
        tasks = data.get("recommended_tasks", [])[:10]

        result = {
            "repo": repo_path,
            "goal_id": goal.id,
            "kpi_assessment": data.get("kpi_assessment", []),
            "gaps": data.get("gaps", []),
            "recommended_tasks": tasks,
            "review_summary": data.get("review_summary", ""),
        }

        logger.info(
            "Review of %s against goal %s: %d gaps, %d tasks recommended",
            repo_path,
            goal.id,
            len(result["gaps"]),
            len(result["recommended_tasks"]),
        )
        return result

    except Exception as e:
        logger.error("Repo review failed for %s: %s", repo_path, e)
        return {
            "repo": repo_path,
            "goal_id": goal.id,
            "error": str(e),
            "kpi_assessment": [],
            "gaps": [],
            "recommended_tasks": [],
            "review_summary": "",
        }
