"""Batch processing endpoint with rate limiting and background execution."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from um_agent_coder.daemon.auth import verify_api_key

from ._pipeline import enhance_prompt
from ._router import select_model
from .models import (
    GEMINI_MODEL_MAP,
    BatchRequest,
    BatchResponse,
    BatchResultItem,
    GeminiModelTier,
    UsageInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Track running batch tasks so we can cancel them
_batch_tasks: dict[str, asyncio.Task] = {}


def _get_db():
    from um_agent_coder.daemon.app import get_db

    return get_db()


def _get_client():
    from um_agent_coder.daemon.app import get_gemini_client

    return get_gemini_client()


def _get_settings():
    from um_agent_coder.daemon.app import get_settings

    return get_settings()


class TokenBucketLimiter:
    """Simple async token bucket rate limiter."""

    def __init__(self, rate: float = 30.0, burst: int = 5):
        self._rate = rate  # tokens per minute
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._burst,
                self._tokens + elapsed * (self._rate / 60.0),
            )
            self._last_refill = now

            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / (self._rate / 60.0)
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


# Per-model rate limiters
_limiters: dict[str, TokenBucketLimiter] = {}


def _get_limiter(model: str) -> TokenBucketLimiter:
    if model not in _limiters:
        _limiters[model] = TokenBucketLimiter(rate=30.0, burst=5)
    return _limiters[model]


def _resolve_model(model: GeminiModelTier, prompt: str, settings) -> str:
    if model == GeminiModelTier.auto:
        return select_model(prompt, threshold=settings.gemini_complexity_threshold)
    return GEMINI_MODEL_MAP[model.value]


async def _process_batch(batch_id: str, req: BatchRequest):
    """Background task that processes all queries in a batch."""
    db = _get_db()
    client = _get_client()
    settings = _get_settings()

    now = datetime.now(timezone.utc).isoformat()
    await db.update_gemini_batch(batch_id, status="running", started_at=now)

    semaphore = asyncio.Semaphore(settings.gemini_batch_max_concurrent)
    results: list[Optional[BatchResultItem]] = [None] * len(req.queries)
    completed = 0
    failed = 0

    async def process_one(index: int, query):
        nonlocal completed, failed

        model_name = _resolve_model(query.model, query.prompt, settings)
        limiter = _get_limiter(model_name)

        async with semaphore:
            await limiter.acquire()
            start = time.monotonic()
            try:
                # Enhance prompt if enabled
                prompt = query.prompt
                if req.enable_enhancement and settings.gemini_enhance_enabled:
                    enhanced = enhance_prompt(prompt)
                    prompt = enhanced.enhanced

                gen_result = await client.generate(
                    prompt=prompt,
                    model=model_name,
                    system_prompt=query.system_prompt,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                )
                duration_ms = int((time.monotonic() - start) * 1000)
                usage = gen_result.get("usage", {})

                results[index] = BatchResultItem(
                    index=index,
                    prompt=query.prompt,
                    model=model_name,
                    response=gen_result.get("text", ""),
                    duration_ms=duration_ms,
                    usage=UsageInfo(**usage) if usage else UsageInfo(),
                )
                completed += 1
            except Exception as e:
                duration_ms = int((time.monotonic() - start) * 1000)
                results[index] = BatchResultItem(
                    index=index,
                    prompt=query.prompt,
                    model=model_name,
                    error=str(e),
                    duration_ms=duration_ms,
                )
                failed += 1

            # Update progress
            await db.update_gemini_batch(
                batch_id,
                completed_queries=completed,
                failed_queries=failed,
            )

    # Run all queries concurrently (bounded by semaphore)
    tasks = [asyncio.create_task(process_one(i, q)) for i, q in enumerate(req.queries)]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        # Batch was cancelled
        for t in tasks:
            t.cancel()
        await db.update_gemini_batch(batch_id, status="cancelled")
        return

    # Serialize results
    result_dicts = [
        r.model_dump() if r else {"index": i, "error": "skipped"} for i, r in enumerate(results)
    ]

    now = datetime.now(timezone.utc).isoformat()
    final_status = "completed" if failed == 0 else ("failed" if completed == 0 else "completed")
    await db.update_gemini_batch(
        batch_id,
        status=final_status,
        completed_queries=completed,
        failed_queries=failed,
        results=result_dicts,
        completed_at=now,
    )
    logger.info("Batch %s completed: %d/%d succeeded", batch_id, completed, len(req.queries))


# --- Endpoints ---


@router.post("/batch", response_model=BatchResponse)
async def submit_batch(
    req: BatchRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Submit a batch of queries for background processing."""
    db = _get_db()
    batch_id = f"gb-{uuid.uuid4().hex[:12]}"

    config = {
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "enable_enhancement": req.enable_enhancement,
    }

    batch = await db.create_gemini_batch(
        batch_id=batch_id,
        total_queries=len(req.queries),
        config=config,
    )

    # Launch background task
    task = asyncio.create_task(_process_batch(batch_id, req))
    _batch_tasks[batch_id] = task

    # Clean up reference when done
    def _cleanup(t):
        _batch_tasks.pop(batch_id, None)

    task.add_done_callback(_cleanup)

    return _batch_to_response(batch)


def _batch_to_response(batch: dict) -> BatchResponse:
    """Safely convert a DB batch dict to BatchResponse."""
    results = None
    if batch.get("results"):
        results = [BatchResultItem(**r) for r in batch["results"]]
    return BatchResponse(
        id=batch["id"],
        status=batch["status"],
        total_queries=batch["total_queries"],
        completed_queries=batch.get("completed_queries", 0),
        failed_queries=batch.get("failed_queries", 0),
        model=batch.get("model") or "",
        results=results,
        error=batch.get("error"),
        created_at=batch["created_at"],
        started_at=batch.get("started_at"),
        completed_at=batch.get("completed_at"),
    )


@router.get("/batch/{batch_id}", response_model=BatchResponse)
async def get_batch(
    batch_id: str,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Get batch job status and results."""
    db = _get_db()
    batch = await db.get_gemini_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch job not found")

    return _batch_to_response(batch)


@router.delete("/batch/{batch_id}")
async def cancel_batch(
    batch_id: str,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Cancel a running batch job."""
    db = _get_db()
    batch = await db.get_gemini_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch job not found")

    if batch["status"] not in ("pending", "running"):
        raise HTTPException(
            status_code=400, detail=f"Cannot cancel batch in '{batch['status']}' state"
        )

    # Cancel the background task
    task = _batch_tasks.get(batch_id)
    if task and not task.done():
        task.cancel()

    await db.update_gemini_batch(batch_id, status="cancelled")
    return {"status": "cancelled", "batch_id": batch_id}
