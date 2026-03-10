"""Firestore CRUD for World Agent state.

Uses the same lazy singleton pattern as routes/kb/_store.py.
Collections: world_agent_goals, world_agent_events/{date},
world_agent_state/current, world_agent_system/scheduler_state.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy singleton (same pattern as kb/_store.py)
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
        logger.info("World Agent Firestore async client initialized")
        return _client
    except Exception as e:
        _init_failed = True
        logger.error("Failed to initialize World Agent Firestore client: %s", e)
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Goals ---

GOALS_COLLECTION = "world_agent_goals"


async def save_goal(goal_data: Dict[str, Any]) -> bool:
    """Save or update a goal document. Uses goal['id'] as document ID."""
    client = _get_client()
    if not client:
        return False
    try:
        goal_id = goal_data["id"]
        doc_data = {**goal_data, "updated_at": _now_iso()}
        if "created_at" not in doc_data or not doc_data["created_at"]:
            doc_data["created_at"] = _now_iso()
        await client.collection(GOALS_COLLECTION).document(goal_id).set(doc_data)
        logger.info("Saved goal %s", goal_id)
        return True
    except Exception as e:
        logger.error("Failed to save goal: %s", e)
        return False


async def get_goal(goal_id: str) -> Optional[Dict[str, Any]]:
    """Get a single goal by ID."""
    client = _get_client()
    if not client:
        return None
    try:
        doc = await client.collection(GOALS_COLLECTION).document(goal_id).get()
        if not doc.exists:
            return None
        return doc.to_dict()
    except Exception as e:
        logger.error("Failed to get goal %s: %s", goal_id, e)
        return None


async def list_goals(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all goals, optionally filtered by status."""
    client = _get_client()
    if not client:
        return []
    try:
        query = client.collection(GOALS_COLLECTION)
        if status:
            query = query.where("status", "==", status)
        query = query.order_by("priority")
        results = []
        async for doc in query.stream():
            results.append(doc.to_dict())
        return results
    except Exception as e:
        logger.error("Failed to list goals: %s", e)
        return []


async def delete_goal(goal_id: str) -> bool:
    """Delete a goal document."""
    client = _get_client()
    if not client:
        return False
    try:
        await client.collection(GOALS_COLLECTION).document(goal_id).delete()
        logger.info("Deleted goal %s", goal_id)
        return True
    except Exception as e:
        logger.error("Failed to delete goal %s: %s", goal_id, e)
        return False


# --- Events ---

EVENTS_COLLECTION = "world_agent_events"


async def save_events(events: List[Dict[str, Any]]) -> int:
    """Save events to Firestore, partitioned by date. Returns count saved."""
    client = _get_client()
    if not client:
        return 0
    saved = 0
    try:
        batch = client.batch()
        for event in events:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            doc_ref = client.collection(EVENTS_COLLECTION).document(date_str).collection(
                "items"
            ).document(event["id"])
            batch.set(doc_ref, event)
            saved += 1
        await batch.commit()
        logger.info("Saved %d events", saved)
    except Exception as e:
        logger.error("Failed to save events: %s", e)
    return saved


async def list_events(
    since: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List recent events. Queries today's partition by default."""
    client = _get_client()
    if not client:
        return []
    try:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        query = client.collection(EVENTS_COLLECTION).document(date_str).collection("items")
        if source:
            query = query.where("source", "==", source)
        query = query.order_by("timestamp", direction="DESCENDING").limit(limit)
        results = []
        async for doc in query.stream():
            results.append(doc.to_dict())
        return results
    except Exception as e:
        logger.error("Failed to list events: %s", e)
        return []


# --- World State ---

STATE_COLLECTION = "world_agent_state"
STATE_DOC_ID = "current"


async def save_world_state(state: Dict[str, Any]) -> bool:
    """Save the current world state."""
    client = _get_client()
    if not client:
        return False
    try:
        state["last_updated"] = _now_iso()
        await client.collection(STATE_COLLECTION).document(STATE_DOC_ID).set(state)
        return True
    except Exception as e:
        logger.error("Failed to save world state: %s", e)
        return False


async def get_world_state() -> Optional[Dict[str, Any]]:
    """Get the current world state."""
    client = _get_client()
    if not client:
        return None
    try:
        doc = await client.collection(STATE_COLLECTION).document(STATE_DOC_ID).get()
        if not doc.exists:
            return None
        return doc.to_dict()
    except Exception as e:
        logger.error("Failed to get world state: %s", e)
        return None


# --- Scheduler State ---

SYSTEM_COLLECTION = "world_agent_system"
SCHEDULER_DOC_ID = "scheduler_state"


async def save_scheduler_state(state: Dict[str, Any]) -> bool:
    """Save scheduler state (last cycle time, interval, etc.)."""
    client = _get_client()
    if not client:
        return False
    try:
        state["updated_at"] = _now_iso()
        await client.collection(SYSTEM_COLLECTION).document(SCHEDULER_DOC_ID).set(state)
        return True
    except Exception as e:
        logger.error("Failed to save scheduler state: %s", e)
        return False


async def get_scheduler_state() -> Optional[Dict[str, Any]]:
    """Get scheduler state."""
    client = _get_client()
    if not client:
        return None
    try:
        doc = await client.collection(SYSTEM_COLLECTION).document(SCHEDULER_DOC_ID).get()
        if not doc.exists:
            return None
        return doc.to_dict()
    except Exception as e:
        logger.error("Failed to get scheduler state: %s", e)
        return None
