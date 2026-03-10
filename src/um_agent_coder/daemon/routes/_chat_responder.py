"""Send responses back to Slack channels and Discord interactions.

Uses httpx to call platform APIs directly — no SDK dependencies.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Discord message limit is 2000 chars, Slack is 4000 for text blocks
DISCORD_MAX_LEN = 1900
SLACK_MAX_LEN = 3900


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n\n... (truncated)"


async def slack_post_message(
    bot_token: str,
    channel: str,
    text: str,
    thread_ts: Optional[str] = None,
) -> bool:
    """Post a message to a Slack channel via chat.postMessage."""
    import httpx

    payload: Dict[str, Any] = {
        "channel": channel,
        "text": _truncate(text, SLACK_MAX_LEN),
    }
    if thread_ts:
        payload["thread_ts"] = thread_ts

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://slack.com/api/chat.postMessage",
                json=payload,
                headers={"Authorization": f"Bearer {bot_token}"},
            )
            data = resp.json()
            if not data.get("ok"):
                logger.error("Slack post failed: %s", data.get("error"))
                return False
            return True
    except Exception as e:
        logger.exception("Slack post error: %s", e)
        return False


async def discord_followup(
    interaction_token: str,
    application_id: str,
    content: str,
) -> bool:
    """Send a followup message to a Discord interaction (deferred response)."""
    import httpx

    url = f"https://discord.com/api/v10/webhooks/{application_id}/{interaction_token}"

    # Split long messages into multiple followups
    chunks = _split_discord(content)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            for chunk in chunks:
                resp = await client.post(url, json={"content": chunk})
                if resp.status_code >= 400:
                    logger.error(
                        "Discord followup failed: status=%d body=%s",
                        resp.status_code,
                        resp.text[:200],
                    )
                    return False
                # Small delay between chunks to avoid rate limits
                if len(chunks) > 1:
                    await asyncio.sleep(0.5)
            return True
    except Exception as e:
        logger.exception("Discord followup error: %s", e)
        return False


def _split_discord(text: str) -> list[str]:
    """Split text into Discord-safe chunks (max 2000 chars)."""
    if len(text) <= DISCORD_MAX_LEN:
        return [text]

    chunks = []
    lines = text.split("\n")
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 > DISCORD_MAX_LEN:
            if current:
                chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks or [_truncate(text, DISCORD_MAX_LEN)]


def format_iteration_result(result: Dict[str, Any]) -> str:
    """Format an iteration result for chat display."""
    status = result.get("status", "unknown")
    score = result.get("best_score")
    iterations = result.get("total_steps", 0)
    response = result.get("best_response", "")

    header = f"**Status**: {status}"
    if score is not None:
        header += f" | **Score**: {score:.3f}"
    header += f" | **Iterations**: {iterations}"

    # Include dimension scores if available
    dims = result.get("best_dimensions", {})
    if dims:
        dim_parts = [f"{k}={v:.2f}" for k, v in dims.items() if isinstance(v, (int, float))]
        if dim_parts:
            header += f"\n**Dimensions**: {', '.join(dim_parts)}"

    if response:
        return f"{header}\n\n---\n\n{response}"
    return header
