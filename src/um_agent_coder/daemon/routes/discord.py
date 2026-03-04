"""Discord interaction handler - processes PING and slash commands."""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from um_agent_coder.daemon.auth import verify_discord_signature

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def get_db():
    from um_agent_coder.daemon.app import get_db as _get
    return _get()


def get_worker():
    from um_agent_coder.daemon.app import get_worker as _get
    return _get()


def get_settings():
    from um_agent_coder.daemon.app import get_settings as _get
    return _get()


# Discord interaction types
PING = 1
APPLICATION_COMMAND = 2

# Discord response types
PONG = 1
CHANNEL_MESSAGE = 4
DEFERRED_CHANNEL_MESSAGE = 5


@router.post("/discord")
async def discord_interaction(request: Request):
    """Handle Discord interaction webhooks (PING + slash commands)."""
    settings = get_settings()
    body = await request.body()

    # Verify signature if configured
    if settings.discord_public_key:
        sig = request.headers.get("X-Signature-Ed25519", "")
        timestamp = request.headers.get("X-Signature-Timestamp", "")
        if not verify_discord_signature(body, timestamp, sig, settings.discord_public_key):
            raise HTTPException(status_code=401, detail="Invalid Discord signature")

    payload = json.loads(body)
    interaction_type = payload.get("type")

    # PING - Discord verification
    if interaction_type == PING:
        return JSONResponse({"type": PONG})

    # APPLICATION_COMMAND - Slash command
    if interaction_type == APPLICATION_COMMAND:
        return await _handle_slash_command(payload)

    return JSONResponse({"type": PONG})


async def _handle_slash_command(payload: dict):
    """Handle Discord slash command interaction."""
    data = payload.get("data", {})
    command_name = data.get("name", "")

    # We expect a /agent command with a prompt option
    if command_name != "agent":
        return JSONResponse({
            "type": CHANNEL_MESSAGE,
            "data": {"content": f"Unknown command: /{command_name}"},
        })

    # Extract prompt from options
    options = data.get("options", [])
    prompt = ""
    for opt in options:
        if opt.get("name") == "prompt":
            prompt = opt.get("value", "")
            break

    if not prompt:
        return JSONResponse({
            "type": CHANNEL_MESSAGE,
            "data": {"content": "Please provide a prompt: `/agent prompt:your task here`"},
        })

    db = get_db()
    worker = get_worker()
    task_id = f"discord-{uuid.uuid4().hex[:12]}"

    user = payload.get("member", {}).get("user", {})
    channel_id = payload.get("channel_id")
    guild_id = payload.get("guild_id")

    source_meta = {
        "discord_event": "slash_command",
        "channel_id": channel_id,
        "guild_id": guild_id,
        "user_id": user.get("id"),
        "username": user.get("username"),
        "interaction_id": payload.get("id"),
        "interaction_token": payload.get("token"),
    }

    await db.create_task(
        task_id=task_id,
        prompt=prompt,
        source="discord",
        source_meta=source_meta,
    )
    await db.add_log(task_id, f"Created from Discord slash command by {user.get('username')}")
    await worker.enqueue(task_id)

    # Return deferred response - we'll follow up when the task completes
    return JSONResponse({
        "type": DEFERRED_CHANNEL_MESSAGE,
        "data": {
            "content": f"Task `{task_id}` queued. I'll process your request and respond when done."
        },
    })
