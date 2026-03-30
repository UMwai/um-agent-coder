"""Push bridge — sends trade recommendations to the Command Center webhook.

After generate_trade_recs() produces recs, this module POSTs them
to the Command Center's /webhook/signal endpoint so they enter the
5-Gate pipeline immediately (instead of waiting for the 5-min poll).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

PUSH_TIMEOUT = 10.0  # seconds


async def push_recs_to_command_center(
    recs: dict,
    cycle_id: str,
    command_center_url: str,
) -> int:
    """Push trade recommendations to the Command Center webhook.

    Returns the number of signals accepted, or 0 on failure.
    """
    recommendations = recs.get("recommendations", [])
    if not recommendations:
        return 0

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Tag each rec with rec_id + rec_date so outcomes can flow back
    signals = []
    for i, rec in enumerate(recommendations):
        tagged = dict(rec)
        tagged["rec_id"] = f"{cycle_id}--{i}"
        tagged["rec_date"] = date_str
        tagged["market_regime"] = recs.get("market_regime", "unknown")
        signals.append(tagged)

    # Build headers
    headers = {"Content-Type": "application/json"}
    ecosystem_token = os.environ.get("ECOSYSTEM_TOKEN", "").strip()
    if ecosystem_token:
        headers["X-Ecosystem-Token"] = ecosystem_token

    try:
        async with httpx.AsyncClient(timeout=PUSH_TIMEOUT) as client:
            resp = await client.post(
                f"{command_center_url}/webhook/signal",
                json={"signals": signals},
                headers=headers,
            )
            if resp.status_code == 200:
                accepted = resp.json().get("accepted", 0)
                logger.info(
                    "Pushed %d trade recs to command center (%s)",
                    accepted,
                    command_center_url,
                )
                return accepted
            else:
                logger.warning(
                    "Command center push failed: %d %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return 0
    except Exception as exc:
        logger.warning("Push bridge error: %s", exc)
        return 0
