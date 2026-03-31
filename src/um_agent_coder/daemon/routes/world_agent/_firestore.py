"""Firestore CRUD for World Agent state.

Uses the same lazy singleton pattern as routes/kb/_store.py.
Collections: world_agent_goals, world_agent_events/{date},
world_agent_state/current, world_agent_system/scheduler_state,
world_agent_cycles/{date}/runs/{cycle_id}, world_agent_journal,
world_agent_trade_recs/{date}/recs/{rec_id}.
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
    """List all goals, optionally filtered by status.

    Note: Firestore compound queries (where + order_by on different fields)
    require composite indexes. We fetch all goals and filter in Python to
    avoid the missing-index silent failure.
    """
    client = _get_client()
    if not client:
        return []
    try:
        query = client.collection(GOALS_COLLECTION).order_by("priority")
        results = []
        async for doc in query.stream():
            data = doc.to_dict()
            if status and data.get("status") != status:
                continue
            results.append(data)
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
            doc_ref = (
                client.collection(EVENTS_COLLECTION)
                .document(date_str)
                .collection("items")
                .document(event["id"])
            )
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


# --- Cycle History ---

CYCLES_COLLECTION = "world_agent_cycles"


async def save_cycle_record(record: Dict[str, Any]) -> bool:
    """Save a cycle record. Append-only, partitioned by date.

    Path: world_agent_cycles/{date}/runs/{cycle_id}
    """
    client = _get_client()
    if not client:
        return False
    try:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cycle_id = record["cycle_id"]
        record["saved_at"] = _now_iso()
        doc_ref = (
            client.collection(CYCLES_COLLECTION)
            .document(date_str)
            .collection("runs")
            .document(cycle_id)
        )
        await doc_ref.set(record)
        logger.info("Saved cycle record %s for %s", cycle_id, date_str)
        return True
    except Exception as e:
        logger.error("Failed to save cycle record: %s", e)
        return False


async def list_cycle_records(
    date_str: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """List cycle records for a given date (default: today), newest first."""
    client = _get_client()
    if not client:
        return []
    try:
        if not date_str:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        query = (
            client.collection(CYCLES_COLLECTION)
            .document(date_str)
            .collection("runs")
            .order_by("timestamp", direction="DESCENDING")
            .limit(limit)
        )
        results = []
        async for doc in query.stream():
            results.append(doc.to_dict())
        return results
    except Exception as e:
        logger.error("Failed to list cycle records for %s: %s", date_str, e)
        return []


async def get_cycle_record(
    cycle_id: str, date_str: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Get a single cycle record by ID."""
    client = _get_client()
    if not client:
        return None
    try:
        if not date_str:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        doc = (
            await client.collection(CYCLES_COLLECTION)
            .document(date_str)
            .collection("runs")
            .document(cycle_id)
            .get()
        )
        if not doc.exists:
            return None
        return doc.to_dict()
    except Exception as e:
        logger.error("Failed to get cycle record %s: %s", cycle_id, e)
        return None


async def get_cycle_stats(date_str: Optional[str] = None) -> Dict[str, Any]:
    """Get aggregate stats for all cycles on a given date."""
    records = await list_cycle_records(date_str=date_str, limit=500)
    if not records:
        return {
            "date": date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "total_cycles": 0,
            "successful_cycles": 0,
            "failed_cycles": 0,
            "total_events": 0,
            "total_signals": 0,
            "total_tasks": 0,
            "total_duration_ms": 0,
            "goal_ids_touched": [],
        }

    successful = [r for r in records if not r.get("error")]
    failed = [r for r in records if r.get("error")]
    all_goal_ids = set()
    for r in records:
        all_goal_ids.update(r.get("goal_ids_touched", []))

    return {
        "date": date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "total_cycles": len(records),
        "successful_cycles": len(successful),
        "failed_cycles": len(failed),
        "total_events": sum(r.get("events_collected", 0) for r in records),
        "total_signals": sum(r.get("signals_generated", 0) for r in records),
        "total_tasks": sum(r.get("tasks_created", 0) for r in records),
        "total_duration_ms": sum(r.get("duration_ms", 0) for r in records),
        "goal_ids_touched": sorted(all_goal_ids),
    }


# --- Journal ---

JOURNAL_COLLECTION = "world_agent_journal"


async def save_journal_entry(entry: Dict[str, Any]) -> bool:
    """Save a daily journal entry. Uses the date string as document ID."""
    client = _get_client()
    if not client:
        return False
    try:
        date_str = entry["date"]
        entry["updated_at"] = _now_iso()
        if not entry.get("created_at"):
            # Preserve created_at if updating an existing entry
            existing = await get_journal_entry(date_str)
            entry["created_at"] = (existing or {}).get("created_at") or _now_iso()
        await client.collection(JOURNAL_COLLECTION).document(date_str).set(entry)
        logger.info("Saved journal entry for %s", date_str)
        return True
    except Exception as e:
        logger.error("Failed to save journal entry: %s", e)
        return False


async def get_journal_entry(date_str: str) -> Optional[Dict[str, Any]]:
    """Get a journal entry by date (YYYY-MM-DD)."""
    client = _get_client()
    if not client:
        return None
    try:
        doc = await client.collection(JOURNAL_COLLECTION).document(date_str).get()
        if not doc.exists:
            return None
        return doc.to_dict()
    except Exception as e:
        logger.error("Failed to get journal entry for %s: %s", date_str, e)
        return None


async def list_journal_entries(limit: int = 30) -> List[Dict[str, Any]]:
    """List recent journal entries, newest first."""
    client = _get_client()
    if not client:
        return []
    try:
        query = (
            client.collection(JOURNAL_COLLECTION)
            .order_by("date", direction="DESCENDING")
            .limit(limit)
        )
        results = []
        async for doc in query.stream():
            results.append(doc.to_dict())
        return results
    except Exception as e:
        logger.error("Failed to list journal entries: %s", e)
        return []


# --- Trade Recommendations ---

TRADE_RECS_COLLECTION = "world_agent_trade_recs"


async def save_trade_recs(
    cycle_id: str,
    recs: Dict[str, Any],
    model: str,
    sw_version: str,
    market_context_summary: str = "",
) -> int:
    """Persist trade recommendations to Firestore.

    Stores each individual recommendation as its own document for easy querying,
    plus a summary document for the full batch.

    Path: world_agent_trade_recs/{date}/recs/{rec_id}
          world_agent_trade_recs/{date}/batches/{cycle_id}

    Returns count of recs saved.
    """
    client = _get_client()
    if not client:
        return 0

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now = _now_iso()
    saved = 0

    try:
        batch = client.batch()

        # Save the full batch as a summary doc
        batch_doc = {
            "cycle_id": cycle_id,
            "timestamp": now,
            "model": model,
            "sw_version": sw_version,
            "market_regime": recs.get("market_regime", "unknown"),
            "regime_reasoning": recs.get("regime_reasoning", ""),
            "market_summary": recs.get("market_summary", ""),
            "market_context_summary": market_context_summary[:2000],
            "rec_count": len(recs.get("recommendations", [])),
            "watchlist": recs.get("watchlist", []),
        }
        batch_ref = (
            client.collection(TRADE_RECS_COLLECTION)
            .document(date_str)
            .collection("batches")
            .document(cycle_id)
        )
        batch.set(batch_ref, batch_doc)

        # Save each recommendation individually
        for i, rec in enumerate(recs.get("recommendations", [])):
            rec_id = f"{cycle_id}--{i}"
            rec_doc = {
                "rec_id": rec_id,
                "cycle_id": cycle_id,
                "timestamp": now,
                "date": date_str,
                # Model / version lineage
                "model": model,
                "sw_version": sw_version,
                # Trade details
                "symbol": rec.get("symbol", "?"),
                "direction": rec.get("direction", "?"),
                "conviction": rec.get("conviction", "?"),
                "strategy": rec.get("strategy", "?"),
                "entry": rec.get("entry"),
                "stop_loss": rec.get("stop_loss"),
                "target": rec.get("target"),
                "size_pct": rec.get("size_pct"),
                "risk_reward": rec.get("risk_reward", "?"),
                "timeframe": rec.get("timeframe", "?"),
                "options_detail": rec.get("options_detail"),
                # Reasoning / rationale
                "reasoning": rec.get("reasoning", ""),
                "risks": rec.get("risks", ""),
                # Context
                "market_regime": recs.get("market_regime", "unknown"),
                # Outcome tracking (filled in later by review process)
                "outcome": None,  # "win" | "loss" | "scratch" | "expired"
                "outcome_pnl_pct": None,
                "outcome_notes": None,
                "reviewed_at": None,
            }
            doc_ref = (
                client.collection(TRADE_RECS_COLLECTION)
                .document(date_str)
                .collection("recs")
                .document(rec_id)
            )
            batch.set(doc_ref, rec_doc)
            saved += 1

        await batch.commit()
        logger.info("Saved %d trade recs for cycle %s", saved, cycle_id)

    except Exception as e:
        logger.error("Failed to save trade recs: %s", e)

    return saved


async def list_trade_recs(
    date_str: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List trade recs for a date, optionally filtered by symbol."""
    client = _get_client()
    if not client:
        return []
    try:
        if not date_str:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        query = client.collection(TRADE_RECS_COLLECTION).document(date_str).collection("recs")
        if symbol:
            query = query.where("symbol", "==", symbol.upper())
        query = query.order_by("timestamp", direction="DESCENDING").limit(limit)
        results = []
        async for doc in query.stream():
            results.append(doc.to_dict())
        return results
    except Exception as e:
        logger.error("Failed to list trade recs: %s", e)
        return []


async def update_trade_rec_outcome(
    date_str: str,
    rec_id: str,
    outcome: str,
    pnl_pct: Optional[float] = None,
    notes: Optional[str] = None,
) -> bool:
    """Update the outcome of a trade rec (for tracking accuracy).

    outcome: "win" | "loss" | "scratch" | "expired"
    """
    client = _get_client()
    if not client:
        return False
    try:
        doc_ref = (
            client.collection(TRADE_RECS_COLLECTION)
            .document(date_str)
            .collection("recs")
            .document(rec_id)
        )
        await doc_ref.update(
            {
                "outcome": outcome,
                "outcome_pnl_pct": pnl_pct,
                "outcome_notes": notes,
                "reviewed_at": _now_iso(),
            }
        )
        logger.info("Updated trade rec %s outcome: %s", rec_id, outcome)
        return True
    except Exception as e:
        logger.error("Failed to update trade rec outcome: %s", e)
        return False


async def append_outcome_to_journal(
    date_str: str,
    rec_id: str,
    outcome: str,
    pnl_pct: float | None = None,
) -> bool:
    """Append a trade outcome to today's journal entry for reflection.

    Updates the journal document with an outcomes array tracking
    win/loss/scratch counts and cumulative PnL.
    """
    client = _get_client()
    if not client:
        return False
    try:
        from google.cloud.firestore_v1 import ArrayUnion, Increment  # type: ignore[attr-defined]

        journal_ref = client.collection("world_agent_journal").document(date_str)

        # Try to update existing journal, create if absent
        doc = await journal_ref.get()
        outcome_entry = {
            "rec_id": rec_id,
            "outcome": outcome,
            "pnl_pct": pnl_pct,
            "timestamp": _now_iso(),
        }

        if doc.exists:
            await journal_ref.update({
                "trade_outcomes": ArrayUnion([outcome_entry]),
                f"outcome_counts.{outcome}": Increment(1),
                "outcome_counts.total": Increment(1),
                "outcome_counts.cumulative_pnl_pct": Increment(pnl_pct or 0.0),
            })
        else:
            await journal_ref.set({
                "date": date_str,
                "trade_outcomes": [outcome_entry],
                "outcome_counts": {
                    outcome: 1,
                    "total": 1,
                    "cumulative_pnl_pct": pnl_pct or 0.0,
                },
            })

        logger.info("Appended outcome to journal %s: %s pnl=%.1f%%", date_str, outcome, pnl_pct or 0)
        return True
    except Exception as e:
        logger.warning("Failed to append outcome to journal: %s", e)
        return False


# --- Position Snapshots ---

POSITIONS_COLLECTION = "position_snapshots"


async def save_positions_snapshot(positions: List[Dict[str, Any]]) -> str:
    """Save a position snapshot to Firestore with timestamp.

    Stores as: position_snapshots/{date}/snapshots/{snapshot_id}
    Also updates position_snapshots/latest as a quick-access pointer.

    Returns the snapshot document ID, or empty string on failure.
    """
    client = _get_client()
    if not client:
        return ""
    try:
        now = _now_iso()
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        snapshot_id = f"snap-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

        doc_data = {
            "snapshot_id": snapshot_id,
            "timestamp": now,
            "date": date_str,
            "position_count": len(positions),
            "positions": positions,
            "symbols": sorted({p.get("symbol", "?") for p in positions}),
        }

        # Save dated snapshot for history
        doc_ref = (
            client.collection(POSITIONS_COLLECTION)
            .document(date_str)
            .collection("snapshots")
            .document(snapshot_id)
        )
        await doc_ref.set(doc_data)

        # Upsert the 'latest' pointer for quick retrieval
        latest_ref = client.collection(POSITIONS_COLLECTION).document("latest")
        await latest_ref.set(doc_data)

        logger.info("Saved position snapshot %s with %d positions", snapshot_id, len(positions))
        return snapshot_id
    except Exception as e:
        logger.error("Failed to save position snapshot: %s", e)
        return ""


async def get_positions_snapshot() -> List[Dict[str, Any]]:
    """Retrieve the latest position snapshot from Firestore.

    Returns the list of position dicts, or empty list if none found.
    """
    client = _get_client()
    if not client:
        return []
    try:
        doc = await client.collection(POSITIONS_COLLECTION).document("latest").get()
        if not doc.exists:
            return []
        data = doc.to_dict()
        return data.get("positions", [])
    except Exception as e:
        logger.error("Failed to get latest position snapshot: %s", e)
        return []
