"""Task CRUD API endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from um_agent_coder.daemon.auth import verify_api_key
from um_agent_coder.daemon.models import (
    CancelTaskRequest,
    CreateTaskRequest,
    TaskListResponse,
    TaskLogEntry,
    TaskLogsResponse,
    TaskResponse,
)

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


def get_db():
    from um_agent_coder.daemon.app import get_db as _get

    return _get()


def get_worker():
    from um_agent_coder.daemon.app import get_worker as _get

    return _get()


@router.post("", response_model=TaskResponse, status_code=201)
async def create_task(
    req: CreateTaskRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Submit a new task for processing."""
    db = get_db()
    worker = get_worker()
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    task = await db.create_task(
        task_id=task_id,
        prompt=req.prompt,
        source=req.source,
        spec=req.spec,
        source_meta={"priority": req.priority, "webhook_url": req.webhook_url},
    )
    await db.add_log(task_id, f"Task created via {req.source}")
    # Queue for processing
    await worker.enqueue(task_id)
    return TaskResponse(**task)


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _key: Optional[str] = Depends(verify_api_key),
):
    """List tasks with optional status filter."""
    db = get_db()
    tasks = await db.list_tasks(status=status, limit=limit, offset=offset)
    total = await db.count_tasks(status=status)
    return TaskListResponse(
        tasks=[TaskResponse(**t) for t in tasks],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Get a single task by ID."""
    db = get_db()
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskResponse(**task)


@router.post("/{task_id}/cancel", response_model=TaskResponse)
async def cancel_task(
    task_id: str,
    req: CancelTaskRequest = None,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Cancel a pending or running task."""
    db = get_db()
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Task already {task['status']}")
    worker = get_worker()
    await worker.cancel(task_id)
    task = await db.update_task(
        task_id,
        status="cancelled",
        completed_at=datetime.now(timezone.utc).isoformat(),
        error=req.reason if req else "Cancelled by user",
    )
    await db.add_log(task_id, f"Task cancelled: {req.reason if req else 'user request'}")
    return TaskResponse(**task)


@router.get("/{task_id}/logs", response_model=TaskLogsResponse)
async def get_task_logs(
    task_id: str,
    limit: int = Query(100, ge=1, le=1000),
    _key: Optional[str] = Depends(verify_api_key),
):
    """Get logs for a specific task."""
    db = get_db()
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    logs = await db.get_logs(task_id, limit=limit)
    return TaskLogsResponse(
        logs=[TaskLogEntry(**log) for log in logs],
        task_id=task_id,
    )
