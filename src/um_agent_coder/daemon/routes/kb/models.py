"""Pydantic models for the Knowledge Base API."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class KBItemType(str, Enum):
    idea = "idea"
    task = "task"
    insight = "insight"
    decision = "decision"


class KBStatus(str, Enum):
    active = "active"
    archived = "archived"


class KBPriority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class KBSource(str, Enum):
    manual = "manual"
    auto_extracted = "auto_extracted"
    chat = "chat"


# --- Request models ---


class KBItemCreate(BaseModel):
    type: KBItemType
    title: str = Field(max_length=200)
    content: str = Field(max_length=50_000)
    tags: List[str] = Field(default_factory=list, max_length=20)
    priority: KBPriority = KBPriority.medium
    source: KBSource = KBSource.manual
    source_ref: Optional[str] = None


class KBItemUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=200)
    content: Optional[str] = Field(None, max_length=50_000)
    tags: Optional[List[str]] = Field(None, max_length=20)
    status: Optional[KBStatus] = None
    priority: Optional[KBPriority] = None


class KBSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


class KBExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)


# --- Response models ---


class KBItemResponse(BaseModel):
    id: str
    type: KBItemType
    title: str
    content: str
    tags: List[str]
    status: KBStatus
    priority: KBPriority
    source: KBSource
    source_ref: Optional[str] = None
    created_at: str
    updated_at: str


class KBSearchResponse(BaseModel):
    items: List[KBItemResponse]
    query_tokens: List[str]


class KBExtractCandidate(BaseModel):
    type: KBItemType
    title: str
    content: str
    tags: List[str]


class KBExtractResponse(BaseModel):
    candidates: List[KBExtractCandidate]
