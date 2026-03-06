"""Pydantic request/response schemas for the Gemini intelligence layer."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --- Enums ---

class GeminiModelTier(str, Enum):
    flash = "flash"
    pro = "pro"
    pro_3_1 = "pro-3.1"
    auto = "auto"


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


class EvalInfo(BaseModel):
    score: float = 0.0
    accuracy: float = 0.0
    completeness: float = 0.0
    clarity: float = 0.0
    actionability: float = 0.0
    issues: List[str] = []
    retry_count: int = 0
    accuracy_checks: List[AccuracyCheckInfo] = []


# --- Evaluate endpoint ---

class EvaluateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=200_000, description="The original prompt/task that produced the response.")
    response: str = Field(..., min_length=1, max_length=500_000, description="The response text to evaluate.")
    eval_context: Optional[str] = Field(default=None, max_length=100_000, description="Reference material (API signatures, schemas, etc.) to check against.")
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
    eval_model: Optional[GeminiModelTier] = Field(default=None, description="Model for self-eval. None = use config default ('auto' matches generation model).")
    eval_context: Optional[str] = Field(default=None, max_length=50_000, description="Reference material (API signatures, schemas, etc.) the evaluator should check the response against.")
    domain_hint: Optional[str] = Field(default=None, description="Hint for domain context: code, math, science, etc.")


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
    eval_context: Optional[str] = Field(default=None, max_length=200_000, description="Reference material (API signatures, schemas) for accuracy checking.")
    model: GeminiModelTier = GeminiModelTier.pro_3_1
    eval_models: Optional[List[str]] = Field(default=None, description="Models for evaluation. None = use config default.")
    max_iterations: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=65536, ge=1, le=1_000_000)
    enable_enhancement: bool = True
    use_multi_turn: bool = True
    domain_hint: Optional[str] = None


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
    best_score: float = 0.0
    total_iterations: int = 0
    total_tokens: int = 0
    duration_ms: int = 0
    created_at: Optional[str] = None
