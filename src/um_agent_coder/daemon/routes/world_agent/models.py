"""Pydantic models for the World-Aware Agent."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# --- Enums ---


class EventSeverity(str, Enum):
    info = "info"
    notable = "notable"
    urgent = "urgent"
    critical = "critical"


class EventCategory(str, Enum):
    financial = "financial"
    dev = "dev"
    news = "news"
    system = "system"


class SignalUrgency(str, Enum):
    immediate = "immediate"
    today = "today"
    this_week = "this_week"
    backlog = "backlog"


class GoalStatus(str, Enum):
    active = "active"
    paused = "paused"
    completed = "completed"
    archived = "archived"


class CycleSource(str, Enum):
    heartbeat = "heartbeat"
    manual = "manual"
    self_scheduled = "self"


# --- Core Data Models ---


class Event(BaseModel):
    id: str = Field(..., description="Unique event ID")
    source: str = Field(..., description="Collector source_id, e.g. 'dev.github_events'")
    timestamp: datetime = Field(..., description="When the event occurred")
    category: EventCategory = Field(..., description="Event category")
    severity: EventSeverity = Field(
        default=EventSeverity.info, description="Event severity level"
    )
    title: str = Field(..., description="Human-readable summary")
    body: str = Field(default="", description="Full event data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source-specific metadata")
    related_goals: List[str] = Field(
        default_factory=list, description="Goal IDs this might relate to"
    )


class KPI(BaseModel):
    metric: str = Field(..., description="KPI metric name")
    target: str = Field(..., description="Target value expression")
    current: Optional[str] = Field(default=None, description="Current value")


class GoalProject(BaseModel):
    repo: str = Field(..., description="Repository name")
    role: str = Field(default="", description="Role of this project for the goal")


class GoalSchedule(BaseModel):
    active_hours: Optional[str] = Field(default=None, description="Active hours, e.g. '09:00-16:30 ET'")
    frequency_active: str = Field(default="15min", description="Cycle frequency during active hours")
    frequency_idle: str = Field(default="1h", description="Cycle frequency during idle hours")


class Goal(BaseModel):
    id: str = Field(..., description="Unique goal identifier")
    name: str = Field(..., description="Human-readable goal name")
    description: str = Field(default="", description="Natural language goal description")
    priority: int = Field(default=5, ge=1, le=10, description="Priority (1=highest)")
    status: GoalStatus = Field(default=GoalStatus.active)
    constraints: List[str] = Field(default_factory=list, description="Hard constraints")
    kpis: List[KPI] = Field(default_factory=list, description="Key performance indicators")
    projects: List[GoalProject] = Field(default_factory=list, description="Related projects")
    event_sources: List[str] = Field(default_factory=list, description="Event source IDs to monitor")
    schedule: Optional[GoalSchedule] = Field(default=None, description="Scheduling preferences")
    created_at: Optional[str] = Field(default=None)
    updated_at: Optional[str] = Field(default=None)


class Signal(BaseModel):
    event_id: str = Field(..., description="Source event ID")
    goal_id: str = Field(default="", description="Related goal ID")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="LLM-assessed relevance")
    interpretation: str = Field(default="", description="What this means for the goal")
    suggested_action: str = Field(default="", description="Suggested next action")
    urgency: SignalUrgency = Field(default=SignalUrgency.backlog)


class WorldState(BaseModel):
    last_updated: Optional[str] = Field(default=None)
    summary: str = Field(default="", description="LLM-generated narrative summary")
    active_signals: List[Signal] = Field(default_factory=list)
    cycle_count: int = Field(default=0)
    total_events_collected: int = Field(default=0)


class PlannedTask(BaseModel):
    id: str = Field(..., description="Unique task ID")
    goal_id: str = Field(default="", description="Parent goal ID")
    project: str = Field(default="", description="Target repo/project")
    title: str = Field(..., description="Task title")
    description: str = Field(default="", description="Detailed task description")
    priority: int = Field(default=5, ge=1, le=10)
    estimated_effort: str = Field(default="medium", description="small/medium/large")
    dependencies: List[str] = Field(default_factory=list)
    cli: str = Field(default="codex", description="codex/gemini/claude")
    model: Optional[str] = Field(default=None)
    timeout: str = Field(default="1h")
    success_criteria: str = Field(default="")
    context: Dict[str, Any] = Field(default_factory=dict)


# --- Request / Response Models ---


class CycleRequest(BaseModel):
    source: CycleSource = Field(default=CycleSource.manual, description="What triggered this cycle")


class CycleResponse(BaseModel):
    cycle_id: str
    events_collected: int = 0
    signals_generated: int = 0
    tasks_created: int = 0
    duration_ms: int = 0
    error: Optional[str] = None


class StatusResponse(BaseModel):
    world_state: Optional[WorldState] = None
    goals: List[Goal] = []
    cycle_count: int = 0
    enabled: bool = False


class GoalCreateRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=200, description="Unique goal ID")
    name: str = Field(..., min_length=1, max_length=500)
    description: str = Field(default="")
    priority: int = Field(default=5, ge=1, le=10)
    constraints: List[str] = Field(default_factory=list)
    kpis: List[KPI] = Field(default_factory=list)
    projects: List[GoalProject] = Field(default_factory=list)
    event_sources: List[str] = Field(default_factory=list)
    schedule: Optional[GoalSchedule] = Field(default=None)


class EventsQueryParams(BaseModel):
    since: Optional[str] = Field(default=None, description="ISO timestamp to query events from")
    source: Optional[str] = Field(default=None, description="Filter by event source")
    limit: int = Field(default=50, ge=1, le=500)


# --- GitHub Write Models ---


class CreateBranchRequest(BaseModel):
    branch_name: str = Field(..., min_length=1, max_length=200, description="New branch name")
    base_branch: str = Field(default="main", description="Branch to fork from")


class FileChange(BaseModel):
    path: str = Field(..., min_length=1, description="File path in the repo")
    content: str = Field(..., description="Full file content")
    message: str = Field(default="", description="Commit message for this file")


class CreatePRRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500, description="PR title")
    body: str = Field(default="", description="PR description")
    head_branch: str = Field(..., description="Source branch")
    base_branch: str = Field(default="main", description="Target branch")
    files: List[FileChange] = Field(default_factory=list, description="Files to push before opening PR")


class PostCommentRequest(BaseModel):
    issue_number: int = Field(..., ge=1, description="Issue or PR number")
    body: str = Field(..., min_length=1, description="Comment body")


# --- Cycle History Models ---


class CycleRecord(BaseModel):
    """Append-only record of a single OODA cycle run."""

    cycle_id: str = Field(..., description="Unique cycle ID")
    timestamp: str = Field(..., description="ISO timestamp when cycle started")
    source: str = Field(default="manual", description="What triggered: heartbeat/manual/self")
    events_collected: int = Field(default=0)
    signals_generated: int = Field(default=0)
    tasks_created: int = Field(default=0)
    duration_ms: int = Field(default=0)
    error: Optional[str] = Field(default=None, description="Error message if cycle failed")
    summary: str = Field(default="", description="Orient summary from this cycle")
    signals: List[Signal] = Field(default_factory=list, description="Signals produced this cycle")
    planned_tasks: List[Dict[str, Any]] = Field(
        default_factory=list, description="Tasks planned this cycle"
    )
    event_ids: List[str] = Field(
        default_factory=list, description="IDs of events collected this cycle"
    )
    goal_ids_touched: List[str] = Field(
        default_factory=list, description="Goal IDs referenced by signals"
    )


# --- Journal Models ---


class JournalEntry(BaseModel):
    date: str = Field(..., description="ISO date string (YYYY-MM-DD)")
    summary: str = Field(default="", description="LLM-generated narrative of the day's work")
    cycles_run: int = Field(default=0, description="Number of OODA cycles run")
    events_collected: int = Field(default=0, description="Total events collected")
    signals_generated: int = Field(default=0, description="Total signals generated")
    tasks_created: int = Field(default=0, description="Tasks planned/created")
    goals_progressed: List[str] = Field(default_factory=list, description="Goal IDs that saw progress")
    key_decisions: List[str] = Field(default_factory=list, description="Notable decisions made")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    highlights: List[str] = Field(default_factory=list, description="Key accomplishments")
    created_at: Optional[str] = Field(default=None)
    updated_at: Optional[str] = Field(default=None)


class JournalGenerateRequest(BaseModel):
    date: Optional[str] = Field(default=None, description="Date to generate journal for (default: today)")


class JournalResponse(BaseModel):
    entry: JournalEntry
    generated: bool = Field(default=False, description="Whether this was freshly generated")
