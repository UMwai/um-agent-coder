"""Firestore persistence for iteration runs.

Persists completed iteration data to Firestore so history survives
Cloud Run instance recycles. Uses ADC (Application Default Credentials)
which are automatically available on Cloud Run.

Enable via UM_DAEMON_GEMINI_FIRESTORE_ENABLED=true.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy singleton
_client = None
_init_failed = False


def _get_client():
    """Get or create async Firestore client (lazy singleton)."""
    global _client, _init_failed
    if _init_failed:
        return None
    if _client is not None:
        return _client
    try:
        from google.cloud.firestore import AsyncClient

        _client = AsyncClient()
        logger.info("Firestore async client initialized")
        return _client
    except Exception as e:
        _init_failed = True
        logger.error("Failed to initialize Firestore client: %s", e)
        return None


async def persist_iteration_to_firestore(
    iteration_id: str,
    data: Dict[str, Any],
    steps: List[Dict[str, Any]],
    collection: str = "iteration_runs",
) -> bool:
    """Persist a completed iteration run to Firestore.

    Creates a document in the collection and stores steps in a subcollection.

    Args:
        iteration_id: The iteration ID (used as document ID).
        data: Iteration metadata (status, scores, config, etc.).
        steps: List of step dicts from SQLite.
        collection: Firestore collection name.

    Returns:
        True if persisted successfully, False otherwise.
    """
    client = _get_client()
    if not client:
        return False

    try:
        doc_ref = client.collection(collection).document(iteration_id)

        # Store main iteration doc (exclude large fields to keep doc lean)
        doc_data = {
            "id": iteration_id,
            "status": data.get("status"),
            "original_prompt": (data.get("original_prompt") or "")[:10000],
            "best_score": data.get("best_score", 0.0),
            "best_iteration": data.get("best_iteration", 0),
            "total_iterations": data.get("total_iterations", 0),
            "total_tokens": data.get("total_tokens", 0),
            "config": data.get("config"),
            "error": data.get("error"),
            "created_at": data.get("created_at"),
            "started_at": data.get("started_at"),
            "completed_at": data.get("completed_at"),
            "persisted_at": datetime.now(timezone.utc).isoformat(),
        }
        await doc_ref.set(doc_data)

        # Store steps in subcollection
        steps_coll = doc_ref.collection("steps")
        for step in steps:
            step_id = f"step-{step.get('step_number', 0):03d}"
            step_data = {
                "step_number": step.get("step_number"),
                "generation_model": step.get("generation_model"),
                "generation_duration_ms": step.get("generation_duration_ms", 0),
                "generation_tokens": step.get("generation_tokens", 0),
                "eval_scores": step.get("eval_scores"),
                "eval_models": step.get("eval_models"),
                "eval_duration_ms": step.get("eval_duration_ms", 0),
                "strategies_applied": step.get("strategies_applied"),
                "finish_reason": step.get("finish_reason"),
                # Truncate large text fields
                "response": (step.get("response") or "")[:50000],
            }
            await steps_coll.document(step_id).set(step_data)

        logger.info(
            "Persisted iteration %s to Firestore (%d steps)",
            iteration_id, len(steps),
        )
        return True

    except Exception as e:
        logger.error("Failed to persist iteration %s to Firestore: %s", iteration_id, e)
        return False


async def list_iterations_from_firestore(
    collection: str = "iteration_runs",
    limit: int = 50,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List iteration runs from Firestore.

    Args:
        collection: Firestore collection name.
        limit: Max results to return.
        status: Optional status filter.

    Returns:
        List of iteration summary dicts.
    """
    client = _get_client()
    if not client:
        return []

    try:
        query = client.collection(collection)
        if status:
            query = query.where("status", "==", status)
        query = query.order_by("created_at", direction="DESCENDING").limit(limit)

        docs = query.stream()
        results = []
        async for doc in docs:
            results.append(doc.to_dict())
        return results

    except Exception as e:
        logger.error("Failed to list iterations from Firestore: %s", e)
        return []


async def get_iteration_from_firestore(
    iteration_id: str,
    collection: str = "iteration_runs",
    include_steps: bool = False,
) -> Optional[Dict[str, Any]]:
    """Get a single iteration run from Firestore.

    Args:
        iteration_id: The iteration ID.
        collection: Firestore collection name.
        include_steps: Whether to also fetch steps subcollection.

    Returns:
        Iteration dict, or None if not found.
    """
    client = _get_client()
    if not client:
        return None

    try:
        doc_ref = client.collection(collection).document(iteration_id)
        doc = await doc_ref.get()
        if not doc.exists:
            return None

        data = doc.to_dict()

        if include_steps:
            steps = []
            steps_query = doc_ref.collection("steps").order_by("step_number")
            async for step_doc in steps_query.stream():
                steps.append(step_doc.to_dict())
            data["steps"] = steps

        return data

    except Exception as e:
        logger.error("Failed to get iteration %s from Firestore: %s", iteration_id, e)
        return None
