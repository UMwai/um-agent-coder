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


@router.get("/partials/stats", response_class=HTMLResponse)
async def stats_partial():
    """Return stats HTML partial for htmx swap."""
    db = get_db()
    pending = await db.count_tasks(status="pending")
    running = await db.count_tasks(status="running")
    completed = await db.count_tasks(status="completed")
    failed = await db.count_tasks(status="failed")

    return HTMLResponse(f"""
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
    """)


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

        rows.append(f"""
        <tr>
          <td><code style="color:var(--accent)">{t['id']}</code></td>
          <td class="prompt-cell" title="{_escape(t['prompt'] or '')}">{_escape(prompt_short)}</td>
          <td><span class="badge {badge_class}">{t['status']}</span></td>
          <td class="source-cell">{t['source']}</td>
          <td class="time-cell">{created}</td>
        </tr>
        """)

    return HTMLResponse(f"""
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
    """)


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
