"""Webhook notifier — sends iteration events to external URLs.

Supports events: threshold_met, max_iterations_reached, failed, cancelled, step_complete.
Async httpx POST with retries + exponential backoff.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def send_iteration_webhook(
    iteration_id: str,
    event: str,
    webhook_url: str,
    webhook_headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
    *,
    timeout_seconds: int = 30,
    max_retries: int = 3,
) -> bool:
    """Send a webhook notification for an iteration event.

    Args:
        iteration_id: The iteration ID.
        event: Event type (threshold_met, max_iterations_reached, failed, etc.).
        webhook_url: URL to POST to.
        webhook_headers: Additional headers for the request.
        payload: Optional payload to include (e.g., IterateResponse summary).
        timeout_seconds: HTTP timeout per attempt.
        max_retries: Number of retry attempts.

    Returns:
        True if webhook was delivered successfully, False otherwise.
    """
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed — webhook notifications disabled")
        return False

    body: Dict[str, Any] = {
        "iteration_id": iteration_id,
        "event": event,
    }
    if payload:
        body["data"] = payload

    headers = {"Content-Type": "application/json"}
    if webhook_headers:
        headers.update(webhook_headers)

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                resp = await client.post(
                    webhook_url,
                    json=body,
                    headers=headers,
                )
                if resp.status_code < 300:
                    logger.info(
                        "Webhook delivered: %s event=%s status=%d",
                        iteration_id,
                        event,
                        resp.status_code,
                    )
                    return True
                elif resp.status_code >= 500:
                    logger.warning(
                        "Webhook server error: %s event=%s status=%d attempt=%d",
                        iteration_id,
                        event,
                        resp.status_code,
                        attempt + 1,
                    )
                else:
                    logger.warning(
                        "Webhook client error: %s event=%s status=%d (not retrying)",
                        iteration_id,
                        event,
                        resp.status_code,
                    )
                    return False

        except Exception as e:
            logger.warning(
                "Webhook failed: %s event=%s attempt=%d error=%s",
                iteration_id,
                event,
                attempt + 1,
                e,
            )

        # Exponential backoff: 1s, 2s, 4s
        if attempt < max_retries - 1:
            await asyncio.sleep(2**attempt)

    logger.error(
        "Webhook exhausted retries: %s event=%s after %d attempts",
        iteration_id,
        event,
        max_retries,
    )
    return False


def should_notify(event: str, webhook_events: List[str]) -> bool:
    """Check if an event should trigger a webhook based on subscribed events.

    Special event aliases:
    - "completed" matches both "threshold_met" and "max_iterations_reached"
    - "all" matches everything
    """
    if "all" in webhook_events:
        return True
    if event in webhook_events:
        return True
    if event in ("threshold_met", "max_iterations_reached") and "completed" in webhook_events:
        return True
    return False
