"""Discord interaction handler — supports slash commands:

/ask prompt:question    → single Gemini response
/iterate prompt:task    → full generate→evaluate→retry loop

Results are sent back as followup messages via the interaction webhook.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from um_agent_coder.daemon.auth import verify_discord_signature

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


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

    if settings.discord_public_key:
        sig = request.headers.get("X-Signature-Ed25519", "")
        timestamp = request.headers.get("X-Signature-Timestamp", "")
        if not verify_discord_signature(body, timestamp, sig, settings.discord_public_key):
            raise HTTPException(status_code=401, detail="Invalid Discord signature")

    payload = json.loads(body)
    interaction_type = payload.get("type")

    if interaction_type == PING:
        return JSONResponse({"type": PONG})

    if interaction_type == APPLICATION_COMMAND:
        return await _handle_slash_command(payload)

    return JSONResponse({"type": PONG})


def _get_option(options: list, name: str) -> str:
    """Extract a named option value from Discord command options."""
    for opt in options:
        if opt.get("name") == name:
            return opt.get("value", "")
    return ""


async def _handle_slash_command(payload: dict):
    """Dispatch /ask and /iterate commands."""
    data = payload.get("data", {})
    command_name = data.get("name", "")
    options = data.get("options", [])
    interaction_token = payload.get("token", "")

    settings = get_settings()
    app_id = settings.discord_application_id
    if not app_id:
        return JSONResponse(
            {
                "type": CHANNEL_MESSAGE,
                "data": {"content": "Error: `DISCORD_APPLICATION_ID` not configured."},
            }
        )

    if command_name == "ask":
        prompt = _get_option(options, "prompt")
        if not prompt:
            return JSONResponse(
                {
                    "type": CHANNEL_MESSAGE,
                    "data": {"content": "Usage: `/ask prompt:your question here`"},
                }
            )
        # Defer, then send followup
        asyncio.create_task(_run_ask_and_followup(settings, app_id, interaction_token, prompt))
        return JSONResponse({"type": DEFERRED_CHANNEL_MESSAGE})

    elif command_name == "iterate":
        prompt = _get_option(options, "prompt")
        if not prompt:
            return JSONResponse(
                {
                    "type": CHANNEL_MESSAGE,
                    "data": {"content": "Usage: `/iterate prompt:your task here`"},
                }
            )
        asyncio.create_task(_run_iterate_and_followup(settings, app_id, interaction_token, prompt))
        return JSONResponse({"type": DEFERRED_CHANNEL_MESSAGE})

    # Legacy /agent command — treat as /ask
    elif command_name == "agent":
        prompt = _get_option(options, "prompt")
        if not prompt:
            return JSONResponse(
                {
                    "type": CHANNEL_MESSAGE,
                    "data": {"content": "Usage: `/agent prompt:your task here`"},
                }
            )
        asyncio.create_task(_run_ask_and_followup(settings, app_id, interaction_token, prompt))
        return JSONResponse({"type": DEFERRED_CHANNEL_MESSAGE})

    else:
        return JSONResponse(
            {
                "type": CHANNEL_MESSAGE,
                "data": {"content": f"Unknown command: `/{command_name}`"},
            }
        )


async def _run_ask_and_followup(settings, app_id: str, token: str, prompt: str):
    """Quick Q&A via Gemini, send followup to Discord."""
    from ._chat_responder import discord_followup

    try:
        from um_agent_coder.daemon.gemini_client import gemini_chat

        response = await asyncio.get_event_loop().run_in_executor(None, gemini_chat, prompt)
        await discord_followup(token, app_id, response)
    except Exception as e:
        logger.exception("Discord Q&A failed: %s", e)
        await discord_followup(token, app_id, f"Error: {e}")


async def _run_iterate_and_followup(settings, app_id: str, token: str, prompt: str):
    """Full iteration loop, send progress + result as Discord followups."""
    from ._chat_responder import discord_followup, format_iteration_result
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

    await discord_followup(token, app_id, f"Starting iteration `{iteration_id}`...")

    try:
        await _run_iteration(iteration_id, req)
        result = await _build_iterate_response(iteration_id)
        formatted = format_iteration_result(result.model_dump())
        await discord_followup(token, app_id, formatted)
    except Exception as e:
        logger.exception("Discord iteration failed: %s", e)
        await discord_followup(token, app_id, f"Iteration `{iteration_id}` failed: {e}")
