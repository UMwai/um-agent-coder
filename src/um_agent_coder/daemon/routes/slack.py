"""Slack event handler — supports app_mention with two modes:

- Quick Q&A: @bot question → single Gemini response
- Iteration:  @bot /iterate task description → full generate→evaluate→retry loop

Results are posted back to the channel/thread via chat.postMessage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from um_agent_coder.daemon.auth import verify_slack_signature

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slack", tags=["slack"])


def get_db():
    from um_agent_coder.daemon.app import get_db as _get

    return _get()


def get_settings():
    from um_agent_coder.daemon.app import get_settings as _get

    return _get()


@router.post("/events")
async def slack_events(request: Request):
    """Handle Slack Events API (url_verification + app_mention)."""
    settings = get_settings()
    body = await request.body()

    if settings.slack_signing_secret:
        timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
        sig = request.headers.get("X-Slack-Signature", "")
        if not verify_slack_signature(body, timestamp, sig, settings.slack_signing_secret):
            raise HTTPException(status_code=401, detail="Invalid Slack signature")

    payload = json.loads(body)

    if payload.get("type") == "url_verification":
        return JSONResponse({"challenge": payload["challenge"]})

    if payload.get("type") == "event_callback":
        event = payload.get("event", {})
        if event.get("type") == "app_mention":
            return await _handle_app_mention(event)

    return {"status": "ok"}


async def _handle_app_mention(event: dict):
    """Handle app_mention — dispatch to Q&A or iteration mode."""
    text = event.get("text", "")
    channel = event.get("channel", "")
    thread_ts = event.get("thread_ts") or event.get("ts")

    # Strip bot mention
    prompt = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()
    if not prompt:
        return {"status": "ignored", "reason": "empty prompt"}

    settings = get_settings()
    if not settings.slack_bot_token:
        logger.warning("SLACK_BOT_TOKEN not set — cannot reply")
        return {"status": "error", "reason": "slack_bot_token not configured"}

    # Check for /iterate command
    iterate_match = re.match(r"^/iterate\s+(.+)", prompt, re.DOTALL)
    if iterate_match:
        task_prompt = iterate_match.group(1).strip()
        asyncio.create_task(_run_iterate_and_reply(settings, channel, thread_ts, task_prompt))
    else:
        asyncio.create_task(_run_ask_and_reply(settings, channel, thread_ts, prompt))

    # Return immediately (Slack needs response within 3s)
    return {"status": "accepted"}


async def _run_ask_and_reply(settings, channel: str, thread_ts: str, prompt: str):
    """Quick Q&A: single Gemini call, post result back."""
    from ._chat_responder import slack_post_message

    try:
        from um_agent_coder.daemon.gemini_client import gemini_chat

        response = await asyncio.get_event_loop().run_in_executor(None, gemini_chat, prompt)
        await slack_post_message(
            bot_token=settings.slack_bot_token,
            channel=channel,
            text=response,
            thread_ts=thread_ts,
        )
    except Exception as e:
        logger.exception("Slack Q&A failed: %s", e)
        await slack_post_message(
            bot_token=settings.slack_bot_token,
            channel=channel,
            text=f"Error: {e}",
            thread_ts=thread_ts,
        )


async def _run_iterate_and_reply(settings, channel: str, thread_ts: str, prompt: str):
    """Full iteration loop, post progress + final result back."""
    from ._chat_responder import format_iteration_result, slack_post_message
    from .gemini.iterate import _build_iterate_response, _get_db, _run_iteration
    from .gemini.models import IterateRequest

    db = _get_db()
    iteration_id = f"gi-{uuid.uuid4().hex[:12]}"

    req = IterateRequest(
        prompt=prompt,
        max_iterations=5,
        score_threshold=settings.gemini_iterate_score_threshold,
    )

    config = {
        "model": req.model.value,
        "eval_models": req.eval_models,
        "max_iterations": req.max_iterations,
        "score_threshold": req.score_threshold,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "enable_enhancement": req.enable_enhancement,
        "use_multi_turn": req.use_multi_turn,
        "domain_hint": req.domain_hint,
    }

    await db.create_gemini_iteration(
        iteration_id=iteration_id,
        original_prompt=prompt,
        system_prompt=req.system_prompt,
        eval_context=req.eval_context,
        config=config,
    )

    await slack_post_message(
        bot_token=settings.slack_bot_token,
        channel=channel,
        text=f"Starting iteration `{iteration_id}`...",
        thread_ts=thread_ts,
    )

    try:
        await _run_iteration(iteration_id, req)
        result = await _build_iterate_response(iteration_id)
        formatted = format_iteration_result(result.model_dump())
        await slack_post_message(
            bot_token=settings.slack_bot_token,
            channel=channel,
            text=formatted,
            thread_ts=thread_ts,
        )
    except Exception as e:
        logger.exception("Slack iteration failed: %s", e)
        await slack_post_message(
            bot_token=settings.slack_bot_token,
            channel=channel,
            text=f"Iteration `{iteration_id}` failed: {e}",
            thread_ts=thread_ts,
        )
