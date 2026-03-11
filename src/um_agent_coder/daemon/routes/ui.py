"""Web dashboard routes - serves htmx UI and partials."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ui", tags=["ui"])

STATIC_DIR = Path(__file__).parent.parent / "static"


def get_db():
    from um_agent_coder.daemon.app import get_db as _get

    return _get()


def _get_gemini_db():
    from um_agent_coder.daemon.routes.gemini.iterate import _get_db

    return _get_db()


@router.get("", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard page."""
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text())


@router.get("/chat", response_class=HTMLResponse)
async def chat():
    """Serve the chat interface."""
    chat_path = STATIC_DIR / "chat.html"
    return HTMLResponse(chat_path.read_text())


@router.get("/kb", response_class=HTMLResponse)
async def knowledge_base():
    """Serve the knowledge base management page."""
    kb_path = STATIC_DIR / "kb.html"
    return HTMLResponse(kb_path.read_text())


@router.get("/partials/stats", response_class=HTMLResponse)
async def stats_partial():
    """Return stats HTML partial for htmx swap.

    Counts both worker tasks and Gemini iterations.
    """
    db = get_db()
    # Worker task counts
    t_pending = await db.count_tasks(status="pending")
    t_running = await db.count_tasks(status="running")
    t_completed = await db.count_tasks(status="completed")
    t_failed = await db.count_tasks(status="failed")

    # Iteration counts from SQLite
    gdb = _get_gemini_db()
    all_iters = await gdb.list_gemini_iterations(limit=500)

    # Merge Firestore iterations
    from um_agent_coder.daemon.app import get_settings

    settings = get_settings()
    if settings.gemini_firestore_enabled:
        try:
            from um_agent_coder.daemon.routes.gemini._firestore import (
                list_iterations_from_firestore,
            )

            fs_iters = await list_iterations_from_firestore(
                collection=settings.gemini_firestore_collection,
                limit=500,
            )
            sqlite_ids = {it["id"] for it in all_iters}
            for fs_it in fs_iters:
                if fs_it.get("id") not in sqlite_ids:
                    all_iters.append(fs_it)
        except Exception:
            pass

    done_statuses = {"completed", "threshold_met", "max_iterations_reached"}
    i_running = sum(1 for it in all_iters if it.get("status") == "running")
    i_completed = sum(1 for it in all_iters if it.get("status") in done_statuses)
    i_failed = sum(1 for it in all_iters if it.get("status") == "failed")

    # Combine
    running = t_running + i_running
    completed = t_completed + i_completed
    failed = t_failed + i_failed
    pending = t_pending

    return HTMLResponse(
        f"""
    <div class="stat stat-pending">
      <div class="stat-value">{pending}</div>
      <div class="stat-label">Pending</div>
    </div>
    <div class="stat stat-running">
      <div class="stat-value">{running}</div>
      <div class="stat-label">Running</div>
    </div>
    <div class="stat stat-completed">
      <div class="stat-value">{completed}</div>
      <div class="stat-label">Completed</div>
    </div>
    <div class="stat stat-failed">
      <div class="stat-value">{failed}</div>
      <div class="stat-label">Failed</div>
    </div>
    """
    )


@router.get("/partials/tasks", response_class=HTMLResponse)
async def tasks_partial(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """Return tasks table HTML partial for htmx swap."""
    db = get_db()
    tasks = await db.list_tasks(status=status, limit=limit)

    if not tasks:
        return HTMLResponse('<div class="empty">No tasks found.</div>')

    rows = []
    for t in tasks:
        badge_class = f"badge-{t['status']}"
        prompt_short = (t["prompt"] or "")[:80]
        if len(t["prompt"] or "") > 80:
            prompt_short += "..."
        created = _format_time(t.get("created_at", ""))

        rows.append(
            f"""
        <tr>
          <td><code style="color:var(--accent)">{t['id']}</code></td>
          <td class="prompt-cell" title="{_escape(t['prompt'] or '')}">{_escape(prompt_short)}</td>
          <td><span class="badge {badge_class}">{t['status']}</span></td>
          <td class="source-cell">{t['source']}</td>
          <td class="time-cell">{created}</td>
        </tr>
        """
        )

    return HTMLResponse(
        f"""
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Prompt</th>
          <th>Status</th>
          <th>Source</th>
          <th>Created</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """
    )


@router.get("/partials/iterations", response_class=HTMLResponse)
async def iterations_partial(
    limit: int = Query(20, ge=1, le=100),
):
    """Return iterations table HTML partial for htmx swap.

    Merges live SQLite data (running iterations) with Firestore history
    (completed iterations that survive deploys).
    """
    from datetime import datetime

    from um_agent_coder.daemon.app import get_settings

    settings = get_settings()

    # Always get live iterations from SQLite (includes running ones)
    db = _get_gemini_db()
    iterations = await db.list_gemini_iterations(limit=limit)

    # Merge Firestore history if enabled
    if settings.gemini_firestore_enabled:
        try:
            from um_agent_coder.daemon.routes.gemini._firestore import (
                list_iterations_from_firestore,
            )

            fs_iterations = await list_iterations_from_firestore(
                collection=settings.gemini_firestore_collection,
                limit=limit,
            )
            # Merge: SQLite wins for any ID that exists in both
            sqlite_ids = {it["id"] for it in iterations}
            for fs_it in fs_iterations:
                if fs_it.get("id") not in sqlite_ids:
                    iterations.append(fs_it)
            # Re-sort by created_at descending
            iterations.sort(key=lambda x: x.get("created_at") or "", reverse=True)
            iterations = iterations[:limit]
        except Exception as e:
            logger.warning("Failed to fetch Firestore iterations: %s", e)

    if not iterations:
        return HTMLResponse('<div class="empty">No iterations yet.</div>')

    rows = []
    status_labels = {
        "threshold_met": "complete",
        "max_iterations_reached": "max iters",
        "running": "running",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
    }

    for it in iterations:
        status = it["status"]
        badge_class = f"badge-{status}"
        # Map to green badge for success statuses
        if status == "threshold_met":
            badge_class = "badge-completed"
        elif status == "max_iterations_reached":
            badge_class = "badge-pending"
        label = status_labels.get(status, status)

        # Pulsing dot for running
        pulse = ""
        if status == "running":
            pulse = '<span class="iter-pulse"></span>'

        prompt_raw = it.get("original_prompt", "") or ""
        prompt_short = _escape(prompt_raw[:60])
        if len(prompt_raw) > 60:
            prompt_short += "..."

        score = it.get("best_score", 0.0) or 0.0
        score_display = f"{score * 100:.0f}%" if score > 0 else "—"

        steps = it.get("total_iterations", 0) or 0

        # Elapsed time
        elapsed = ""
        if it.get("started_at"):
            try:
                started = datetime.fromisoformat(it["started_at"])
                if it.get("completed_at"):
                    ended = datetime.fromisoformat(it["completed_at"])
                else:
                    ended = datetime.utcnow()
                secs = int((ended - started).total_seconds())
                if secs < 60:
                    elapsed = f"{secs}s"
                else:
                    elapsed = f"{secs // 60}m {secs % 60}s"
            except (ValueError, TypeError):
                pass

        # Chat link
        iter_id = it["id"]
        if status == "running":
            link = f'<a href="/ui/chat#iter={iter_id}" class="iter-link">Resume in Chat &rarr;</a>'
        else:
            link = f'<a href="/ui/chat#iter={iter_id}" class="iter-link">View in Chat &rarr;</a>'

        rows.append(
            f"""
        <tr>
          <td>{pulse}<span class="badge {badge_class}">{label}</span></td>
          <td class="prompt-cell" title="{_escape(prompt_raw)}">{prompt_short}</td>
          <td>{score_display}</td>
          <td>{steps}</td>
          <td class="time-cell">{elapsed}</td>
          <td>{link}</td>
        </tr>
        """
        )

    return HTMLResponse(
        f"""
    <table>
      <thead>
        <tr>
          <th>Status</th>
          <th>Prompt</th>
          <th>Score</th>
          <th>Steps</th>
          <th>Elapsed</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """
    )


def _format_time(iso_str: str) -> str:
    """Format ISO timestamp to a shorter display format."""
    if not iso_str:
        return ""
    try:
        # Show just date and time without microseconds
        return iso_str[:19].replace("T", " ")
    except Exception:
        return iso_str


def _escape(text: str) -> str:
    """Basic HTML escaping."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )
