"""Session CRUD + message endpoint for multi-turn conversations."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from um_agent_coder.daemon.auth import verify_api_key

from ._pipeline import enhance_prompt
from .models import (
    GEMINI_MODEL_MAP,
    CreateSessionRequest,
    GeminiModelTier,
    MessageRequest,
    MessageResponse,
    SessionDetailResponse,
    SessionListResponse,
    SessionResponse,
    UsageInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_db():
    from um_agent_coder.daemon.app import get_db

    return get_db()


def _get_client():
    from um_agent_coder.daemon.app import get_gemini_client

    return get_gemini_client()


def _get_settings():
    from um_agent_coder.daemon.app import get_settings

    return get_settings()


def _resolve_model(model: GeminiModelTier, settings) -> str:
    if model == GeminiModelTier.auto:
        return GEMINI_MODEL_MAP["pro"]
    return GEMINI_MODEL_MAP[model.value]


def _build_contents(
    messages: list[dict],
    system_prompt: Optional[str] = None,
) -> list[dict]:
    """Build a Code Assist API contents list from stored messages."""
    contents = []
    if system_prompt:
        contents.append({"role": "user", "parts": [{"text": system_prompt}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})

    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    return contents


async def _summarize_messages(client, messages: list[dict], keep_recent: int = 10) -> list[dict]:
    """Summarize older messages to fit context window, keeping recent turns verbatim."""
    if len(messages) <= keep_recent:
        return messages

    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

    # Build summary prompt
    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:500]}" for m in old_messages
    )
    summary_prompt = (
        f"Summarize this conversation concisely, preserving key facts and decisions:\n\n"
        f"{conversation_text}"
    )

    try:
        result = await client.generate(
            prompt=summary_prompt,
            model="gemini-3-flash-preview",
            temperature=0.2,
            max_tokens=1024,
            timeout=30.0,
        )
        summary_text = result.get("text", "")
    except Exception:
        # On failure, just truncate
        summary_text = f"[Summary of {len(old_messages)} earlier messages]"

    summary_msg = {"role": "user", "content": f"[Conversation summary: {summary_text}]"}
    ack_msg = {
        "role": "assistant",
        "content": "I understand the context from our previous conversation.",
    }

    return [summary_msg, ack_msg] + recent_messages


# --- Endpoints ---


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    req: CreateSessionRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Create a new conversation session."""
    db = _get_db()
    settings = _get_settings()
    session_id = f"gs-{uuid.uuid4().hex[:12]}"
    model_name = _resolve_model(req.model, settings)

    expires_at = (
        datetime.now(timezone.utc) + timedelta(hours=settings.gemini_session_ttl_hours)
    ).isoformat()

    session = await db.create_gemini_session(
        session_id=session_id,
        model=model_name,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        system_prompt=req.system_prompt,
        metadata=req.metadata,
        expires_at=expires_at,
    )
    return SessionResponse(**session)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    limit: int = 50,
    _key: Optional[str] = Depends(verify_api_key),
):
    """List all conversation sessions."""
    db = _get_db()
    sessions = await db.list_gemini_sessions(limit=limit)
    return SessionListResponse(
        sessions=[SessionResponse(**s) for s in sessions],
        total=len(sessions),
    )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Get session details with message history."""
    db = _get_db()
    session = await db.get_gemini_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await db.get_session_messages(session_id)
    msg_responses = []
    for m in messages:
        msg_responses.append(
            MessageResponse(
                id=m["id"],
                session_id=session_id,
                role=m["role"],
                content=m["content"],
                token_count=m.get("token_count", 0),
                enhancement_applied=bool(m.get("enhancement_applied", 0)),
            )
        )
    return SessionDetailResponse(
        session=SessionResponse(**session),
        messages=msg_responses,
    )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Delete a session and all its messages."""
    db = _get_db()
    deleted = await db.delete_gemini_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.post("/sessions/{session_id}/message", response_model=MessageResponse)
async def send_message(
    session_id: str,
    req: MessageRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Send a message in a session and get a response."""
    import time

    db = _get_db()
    client = _get_client()
    settings = _get_settings()

    session = await db.get_gemini_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check token limit
    if session["total_tokens"] >= settings.gemini_session_max_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"Session token limit ({settings.gemini_session_max_tokens}) reached",
        )

    start = time.monotonic()

    # Enhance prompt if enabled
    content_to_store = req.content
    content_to_send = req.content
    enhancement_applied = False

    if req.enable_enhancement and settings.gemini_enhance_enabled:
        result = enhance_prompt(req.content)
        if result.stages_applied:
            content_to_send = result.enhanced
            enhancement_applied = True

    # Store user message
    user_msg_id = f"gm-{uuid.uuid4().hex[:12]}"
    await db.add_gemini_message(
        message_id=user_msg_id,
        session_id=session_id,
        role="user",
        content=content_to_store,
    )

    # Build conversation history
    messages = await db.get_session_messages(session_id)

    # Context window management — summarize if >80% of token limit
    if session["total_tokens"] > settings.gemini_session_max_tokens * 0.8:
        messages = await _summarize_messages(client, messages)

    # Replace last user message content with enhanced version for the API call
    msg_dicts = [dict(m) for m in messages]
    if msg_dicts and msg_dicts[-1]["role"] == "user":
        msg_dicts[-1] = {**msg_dicts[-1], "content": content_to_send}

    contents = _build_contents(msg_dicts, system_prompt=session.get("system_prompt"))

    # Generate response
    try:
        gen_result = await client.generate_multi_turn(
            contents=contents,
            model=session["model"],
            temperature=session["temperature"],
            max_tokens=session["max_tokens"],
        )
    except Exception as e:
        from um_agent_coder.daemon.gemini_client import RateLimitError

        if isinstance(e, RateLimitError):
            raise HTTPException(status_code=429, detail=str(e))
        logger.error("Gemini API error in session %s: %s", session_id, e)
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")

    response_text = gen_result.get("text", "")
    usage = gen_result.get("usage", {})
    total_tokens = usage.get("total_tokens", 0)

    # Store assistant message
    asst_msg_id = f"gm-{uuid.uuid4().hex[:12]}"
    await db.add_gemini_message(
        message_id=asst_msg_id,
        session_id=session_id,
        role="assistant",
        content=response_text,
        token_count=total_tokens,
        enhancement_applied=enhancement_applied,
    )

    duration_ms = int((time.monotonic() - start) * 1000)

    return MessageResponse(
        id=asst_msg_id,
        session_id=session_id,
        role="assistant",
        content=response_text,
        token_count=total_tokens,
        enhancement_applied=enhancement_applied,
        model=session["model"],
        duration_ms=duration_ms,
        usage=UsageInfo(**usage) if usage else UsageInfo(),
    )
