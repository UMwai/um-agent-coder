"""Pydantic request/response schemas for the Gemini intelligence layer."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# --- Enums ---


class GeminiModelTier(str, Enum):
    flash = "flash"
    pro = "pro"
    pro_3_1 = "pro-3.1"
    auto = "auto"


def _build_model_map() -> dict[str, str]:
    """Build model map from config. Called lazily to avoid import-time settings load."""
    from um_agent_coder.daemon.config import DaemonSettings

    s = DaemonSettings()
    return {
        "flash": s.gemini_model_flash,
        "pro": s.gemini_model_pro,
        "pro-3.1": s.gemini_model_pro_latest,
    }


# Lazy-loaded singleton — use get_model_map() instead of GEMINI_MODEL_MAP directly
_model_map_cache: dict[str, str] | None = None


def get_model_map() -> dict[str, str]:
    global _model_map_cache
    if _model_map_cache is None:
        _model_map_cache = _build_model_map()
    return _model_map_cache


# Keep for backward compat — but code should migrate to get_model_map()
GEMINI_MODEL_MAP = {
    "flash": "gemini-3-flash-preview",
    "pro": "gemini-3-pro-preview",
    "pro-3.1": "gemini-3.1-pro-preview",
}


class BatchStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class AgentStatus(str, Enum):
    running = "running"
    completed = "completed"
    failed = "failed"
    max_steps_reached = "max_steps_reached"


class IterationStatus(str, Enum):
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    threshold_met = "threshold_met"
    max_iterations_reached = "max_iterations_reached"


# --- Common ---


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""


class EnhancementInfo(BaseModel):
    original_prompt: str
    enhanced_prompt: str
    stages_applied: List[str] = []
    model_selected: str = ""
    complexity_score: float = 0.0


class AccuracyCheckInfo(BaseModel):
    check: str = ""
    status: str = ""  # "pass" or "fail"
    severity: str = ""  # "breaking", "foreign", "style"
    detail: str = ""


class FulfillmentCheckInfo(BaseModel):
    check: str = ""
    status: str = ""  # "pass" or "fail"
    severity: str = ""  # "breaking", "foreign", "style"
    detail: str = ""


class CompletenessCheckInfo(BaseModel):
    check: str = ""
    status: str = ""
    severity: str = ""
    detail: str = ""


class ClarityCheckInfo(BaseModel):
    check: str = ""
    status: str = ""
    severity: str = ""
    detail: str = ""


class ActionabilityCheckInfo(BaseModel):
    check: str = ""
    status: str = ""
    severity: str = ""
    detail: str = ""


class PreGenCheckInfo(BaseModel):
    dimension: str = ""  # "accuracy", "fulfillment", "completeness"
    check: str = ""
    status: str = ""  # "pass" or "fail"
    severity: str = "breaking"
    detail: str = ""
    source: str = "pre_gen"  # "pre_gen" or "evaluator"


class EvalInfo(BaseModel):
    score: float = 0.0
    accuracy: float = 0.0
    completeness: float = 0.0
    clarity: float = 0.0
    actionability: float = 0.0
    fulfillment: float = 0.0
    issues: List[str] = []
    retry_count: int = 0
    accuracy_passed: Optional[bool] = None
    parse_failed: Optional[bool] = None
    accuracy_checks: List[AccuracyCheckInfo] = []
    fulfillment_checks: List[FulfillmentCheckInfo] = []
    completeness_checks: List[CompletenessCheckInfo] = []
    clarity_checks: List[ClarityCheckInfo] = []
    actionability_checks: List[ActionabilityCheckInfo] = []
    pre_gen_checks: List[PreGenCheckInfo] = []


# --- Evaluate endpoint ---


class EvaluateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=200_000,
        description="The original prompt/task that produced the response.",
    )
    response: str = Field(
        ..., min_length=1, max_length=500_000, description="The response text to evaluate."
    )
    eval_context: Optional[str] = Field(
        default=None,
        max_length=100_000,
        description="Reference material (API signatures, schemas, etc.) to check against.",
    )
    model: GeminiModelTier = GeminiModelTier.flash


class EvaluateResponse(BaseModel):
    id: str
    eval_model: str
    duration_ms: int
    evaluation: EvalInfo
    usage: UsageInfo = UsageInfo()


# --- Enhance endpoint ---


class EnhanceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100_000)
    model: GeminiModelTier = GeminiModelTier.auto
    system_prompt: Optional[str] = Field(default=None, max_length=50_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, ge=1, le=65536)
    enable_enhancement: bool = True
    enable_self_eval: bool = True
    eval_model: Optional[GeminiModelTier] = Field(
        default=None,
        description="Model for self-eval. None = use config default ('auto' matches generation model).",
    )
    eval_context: Optional[str] = Field(
        default=None,
        max_length=50_000,
        description="Reference material (API signatures, schemas, etc.) the evaluator should check the response against.",
    )
    domain_hint: Optional[str] = Field(
        default=None, description="Hint for domain context: code, math, science, etc."
    )


class EnhanceResponse(BaseModel):
    id: str
    model: str
    response: str
    duration_ms: int
    usage: UsageInfo = UsageInfo()
    enhancement: Optional[EnhancementInfo] = None
    evaluation: Optional[EvalInfo] = None


# --- Sessions ---


class CreateSessionRequest(BaseModel):
    system_prompt: Optional[str] = Field(default=None, max_length=50_000)
    model: GeminiModelTier = GeminiModelTier.auto
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, ge=1, le=65536)
    metadata: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    id: str
    system_prompt: Optional[str] = None
    model: str
    temperature: float
    max_tokens: int
    metadata: Optional[Dict[str, Any]] = None
    total_tokens: int = 0
    turn_count: int = 0
    created_at: str
    updated_at: str
    expires_at: Optional[str] = None


class SessionListResponse(BaseModel):
    sessions: List[SessionResponse]
    total: int


class MessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=100_000)
    enable_enhancement: bool = True


class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    token_count: int = 0
    enhancement_applied: bool = False
    model: str = ""
    duration_ms: int = 0
    usage: UsageInfo = UsageInfo()


class SessionDetailResponse(BaseModel):
    session: SessionResponse
    messages: List[MessageResponse]


# --- Batch ---


class BatchQueryItem(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100_000)
    system_prompt: Optional[str] = None
    model: GeminiModelTier = GeminiModelTier.auto


class BatchRequest(BaseModel):
    queries: List[BatchQueryItem] = Field(..., min_length=1, max_length=100)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, ge=1, le=65536)
    enable_enhancement: bool = True


class BatchResultItem(BaseModel):
    index: int
    prompt: str
    model: str = ""
    response: Optional[str] = None
    error: Optional[str] = None
    duration_ms: int = 0
    usage: UsageInfo = UsageInfo()


class BatchResponse(BaseModel):
    id: str
    status: BatchStatus
    total_queries: int
    completed_queries: int = 0
    failed_queries: int = 0
    model: str = ""
    results: Optional[List[BatchResultItem]] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# --- Agent ---


class AgentToolCall(BaseModel):
    tool: str
    args: Dict[str, Any] = {}
    result: Optional[str] = None


class AgentStep(BaseModel):
    step: int
    thought: str = ""
    action: Optional[AgentToolCall] = None
    observation: Optional[str] = None


class AgentRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=100_000)
    tools: Optional[List[str]] = Field(default=None, description="Subset of tools to enable")
    max_steps: int = Field(default=10, ge=1, le=50)
    model: GeminiModelTier = GeminiModelTier.pro_3_1
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class AgentResponse(BaseModel):
    id: str
    task: str
    status: AgentStatus
    answer: Optional[str] = None
    steps: List[AgentStep] = []
    total_steps: int = 0
    duration_ms: int = 0
    usage: UsageInfo = UsageInfo()
    error: Optional[str] = None


# --- Iteration runner ---


class IterateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500_000)
    system_prompt: Optional[str] = Field(default=None, max_length=50_000)
    eval_context: Optional[str] = Field(
        default=None,
        max_length=200_000,
        description="Reference material (API signatures, schemas) for accuracy checking.",
    )
    model: GeminiModelTier = GeminiModelTier.pro_3_1
    eval_models: Optional[List[str]] = Field(
        default=None, description="Models for evaluation. None = use config default."
    )
    max_iterations: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=65536, ge=1, le=1_000_000)
    enable_enhancement: bool = True
    use_multi_turn: bool = True
    domain_hint: Optional[str] = None
    source_files: Optional[Dict[str, str]] = Field(
        default=None, description="Source files (path→content) for auto eval_context generation."
    )
    webhook_url: Optional[str] = Field(
        default=None, description="URL to POST webhook notifications to."
    )
    webhook_headers: Optional[Dict[str, str]] = Field(
        default=None, description="Additional headers for webhook requests."
    )
    webhook_events: List[str] = Field(
        default=["completed", "failed"],
        description="Events to notify: completed, failed, cancelled, step_complete, all.",
    )


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


class IterationStepInfo(BaseModel):
    step: int
    prompt_sent: str = ""
    response: str = ""
    generation_model: str = ""
    generation_duration_ms: int = 0
    generation_tokens: int = 0
    evaluation: Optional[EvalInfo] = None
    eval_models_used: List[str] = []
    strategies_applied: List[str] = []
    finish_reason: str = ""
    extracted_files: Optional[FileExtractionInfo] = None


class IterateResponse(BaseModel):
    id: str
    status: IterationStatus
    original_prompt: str = ""
    best_response: Optional[str] = None
    best_score: float = 0.0
    best_iteration: int = 0
    total_iterations: int = 0
    total_tokens: int = 0
    duration_ms: int = 0
    config: Optional[Dict[str, Any]] = None
    steps: List[IterationStepInfo] = []
    error: Optional[str] = None
    created_at: Optional[str] = None


class IterationSummaryResponse(BaseModel):
    id: str
    status: IterationStatus
    original_prompt: str = ""
    best_score: float = 0.0
    total_iterations: int = 0
    total_tokens: int = 0
    duration_ms: int = 0
    created_at: Optional[str] = None


# --- Batch iteration runner ---


class BatchIterateItem(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500_000)
    system_prompt: Optional[str] = Field(default=None, max_length=50_000)
    eval_context: Optional[str] = Field(default=None, max_length=200_000)
    model: Optional[GeminiModelTier] = Field(default=None, description="Per-item model override.")
    label: Optional[str] = Field(
        default=None, max_length=200, description="Human-readable label for this item."
    )


class BatchIterateRequest(BaseModel):
    items: List[BatchIterateItem] = Field(..., min_length=1, max_length=50)
    model: GeminiModelTier = GeminiModelTier.pro_3_1
    eval_models: Optional[List[str]] = None
    max_iterations: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=65536, ge=1, le=1_000_000)
    enable_enhancement: bool = True
    use_multi_turn: bool = True
    domain_hint: Optional[str] = None
    webhook_url: Optional[str] = None
    webhook_headers: Optional[Dict[str, str]] = None
    webhook_events: List[str] = Field(default=["completed", "failed"])
    max_concurrent: Optional[int] = Field(
        default=None, ge=1, le=20, description="Override batch concurrency limit."
    )


class BatchIterateItemStatus(BaseModel):
    index: int
    label: Optional[str] = None
    iteration_id: Optional[str] = None
    status: IterationStatus = IterationStatus.running
    best_score: float = 0.0
    total_iterations: int = 0
    error: Optional[str] = None


class BatchIterateResponse(BaseModel):
    batch_id: str
    status: BatchStatus
    total_items: int
    completed_items: int = 0
    failed_items: int = 0
    items: List[BatchIterateItemStatus] = []
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
