"""Knowledge Base API endpoints."""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from um_agent_coder.daemon.routes.kb import _store as store
from um_agent_coder.daemon.routes.kb.models import (
    KBExtractCandidate,
    KBExtractRequest,
    KBExtractResponse,
    KBItemCreate,
    KBItemResponse,
    KBItemUpdate,
    KBSearchRequest,
    KBSearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --- CRUD ---


@router.post("/items", response_model=KBItemResponse)
async def create_item(body: KBItemCreate):
    item = await store.create_item(body.model_dump())
    if not item:
        raise HTTPException(503, "Firestore unavailable")
    return item


@router.get("/items", response_model=list[KBItemResponse])
async def list_items(
    type: Optional[str] = Query(None),
    status: Optional[str] = Query(None, pattern="^(active|archived)$"),
    tag: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    return await store.list_items(item_type=type, status=status, tag=tag, limit=limit)


@router.get("/items/{item_id}", response_model=KBItemResponse)
async def get_item(item_id: str):
    item = await store.get_item(item_id)
    if not item:
        raise HTTPException(404, "Item not found")
    return item


@router.put("/items/{item_id}", response_model=KBItemResponse)
async def update_item(item_id: str, body: KBItemUpdate):
    item = await store.update_item(item_id, body.model_dump(exclude_none=True))
    if not item:
        raise HTTPException(404, "Item not found")
    return item


@router.delete("/items/{item_id}")
async def delete_item(item_id: str):
    ok = await store.archive_item(item_id)
    if not ok:
        raise HTTPException(404, "Item not found")
    return {"status": "archived", "id": item_id}


# --- Search ---


@router.post("/search", response_model=KBSearchResponse)
async def search_items(body: KBSearchRequest):
    items, tokens = await store.search_items(body.query, limit=body.limit)
    return KBSearchResponse(items=items, query_tokens=tokens)


# --- Extract ---

_EXTRACT_PROMPT = """\
You are a knowledge extraction assistant. Analyze the following text and extract
any noteworthy ideas, actionable tasks, insights, or decisions.

For each extracted item, provide:
- type: one of "idea", "task", "insight", "decision"
- title: concise title (max 200 chars)
- content: brief description in markdown (1-3 sentences)
- tags: 3-5 lowercase keyword tags

Return ONLY a JSON array of objects. No markdown fences, no explanation.
If nothing worth extracting, return an empty array [].

Text to analyze:
"""


@router.post("/extract", response_model=KBExtractResponse)
async def extract_items(body: KBExtractRequest):
    from um_agent_coder.daemon.app import get_settings

    settings = get_settings()
    if not settings.kb_auto_extract_enabled:
        return KBExtractResponse(candidates=[])

    try:
        from um_agent_coder.daemon.app import get_gemini_client

        client = get_gemini_client()
        result = await client.generate(
            prompt=_EXTRACT_PROMPT + body.text[:20_000],
            model=settings.kb_extract_model,
            temperature=0.3,
            max_tokens=4096,
        )

        raw = result.get("text", "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

        candidates_raw = json.loads(raw)
        if not isinstance(candidates_raw, list):
            candidates_raw = []

        valid_types = {"idea", "task", "insight", "decision"}
        candidates = []
        for c in candidates_raw[:10]:
            if not isinstance(c, dict):
                continue
            c_type = c.get("type", "").lower()
            if c_type not in valid_types:
                continue
            candidates.append(
                KBExtractCandidate(
                    type=c_type,
                    title=(c.get("title") or "")[:200],
                    content=(c.get("content") or "")[:5000],
                    tags=[t.lower() for t in (c.get("tags") or []) if isinstance(t, str)][:5],
                )
            )

        return KBExtractResponse(candidates=candidates)

    except json.JSONDecodeError:
        logger.warning("KB extract: failed to parse JSON from model response")
        return KBExtractResponse(candidates=[])
    except Exception as e:
        logger.error("KB extract failed: %s", e)
        return KBExtractResponse(candidates=[])
