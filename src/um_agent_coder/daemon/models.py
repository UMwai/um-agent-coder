"""Pydantic request/response models for the daemon API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# -- Requests --


class CreateTaskRequest(BaseModel):
    """Request body for POST /api/tasks."""

    prompt: str = Field(..., min_length=1, description="Task prompt or instruction")
    spec: Optional[Dict[str, Any]] = Field(
        None, description="Optional TaskSpec dict for structured tasks"
    )
    source: str = Field("api", description="Task source identifier")
    webhook_url: Optional[str] = Field(None, description="Notification webhook URL")
    priority: int = Field(5, ge=1, le=10, description="Task priority (1=low, 10=high)")


class CancelTaskRequest(BaseModel):
    """Request body for POST /api/tasks/{id}/cancel."""

    reason: Optional[str] = None


# -- Responses --


class TaskResponse(BaseModel):
    """Response model for a single task."""

    id: str
    prompt: str
    spec: Optional[Dict[str, Any]] = None
    status: str
    source: str
    source_meta: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class TaskListResponse(BaseModel):
    """Response model for listing tasks."""

    tasks: List[TaskResponse]
    total: int
    limit: int
    offset: int


class TaskLogEntry(BaseModel):
    """A single log entry for a task."""

    id: int
    task_id: str
    level: str
    message: str
    data: Optional[Dict[str, Any]] = None
    created_at: str


class TaskLogsResponse(BaseModel):
    """Response model for task logs."""

    logs: List[TaskLogEntry]
    task_id: str


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    version: str
    tasks_pending: int = 0
    tasks_running: int = 0


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
