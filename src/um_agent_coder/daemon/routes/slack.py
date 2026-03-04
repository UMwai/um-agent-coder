"""Slack event handler - processes app_mention events and url_verification."""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from um_agent_coder.daemon.auth import verify_slack_signature

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slack", tags=["slack"])


def get_db():
    from um_agent_coder.daemon.app import get_db as _get
    return _get()


def get_worker():
    from um_agent_coder.daemon.app import get_worker as _get
    return _get()


def get_settings():
    from um_agent_coder.daemon.app import get_settings as _get
    return _get()


@router.post("/events")
async def slack_events(request: Request):
    """Handle Slack Events API (url_verification + app_mention)."""
    settings = get_settings()
    body = await request.body()

    # Verify signature if configured
    if settings.slack_signing_secret:
        timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
        sig = request.headers.get("X-Slack-Signature", "")
        if not verify_slack_signature(body, timestamp, sig, settings.slack_signing_secret):
            raise HTTPException(status_code=401, detail="Invalid Slack signature")

    payload = json.loads(body)

    # URL verification challenge
    if payload.get("type") == "url_verification":
        return JSONResponse({"challenge": payload["challenge"]})

    # Event callback
    if payload.get("type") == "event_callback":
        event = payload.get("event", {})
        event_type = event.get("type")

        if event_type == "app_mention":
            return await _handle_app_mention(event, payload)

    return {"status": "ok"}


async def _handle_app_mention(event: dict, payload: dict):
    """Handle app_mention - bot was mentioned in a channel."""
    text = event.get("text", "")
    user = event.get("user", "")
    channel = event.get("channel", "")

    # Strip the bot mention from the text
    # Slack formats mentions as <@BOTID> text
    import re

    prompt = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()
    if not prompt:
        return {"status": "ignored", "reason": "empty prompt after mention"}

    db = get_db()
    worker = get_worker()
    task_id = f"slack-{uuid.uuid4().hex[:12]}"

    source_meta = {
        "slack_event": "app_mention",
        "channel": channel,
        "user": user,
        "team": payload.get("team_id"),
        "thread_ts": event.get("thread_ts") or event.get("ts"),
    }

    await db.create_task(
        task_id=task_id,
        prompt=prompt,
        source="slack",
        source_meta=source_meta,
    )
    await db.add_log(task_id, f"Created from Slack mention by {user} in {channel}")
    await worker.enqueue(task_id)

    # Return immediately - Slack requires response within 3 seconds
    return {"status": "accepted", "task_id": task_id}
