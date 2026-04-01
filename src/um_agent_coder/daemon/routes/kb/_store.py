"""Firestore CRUD for Knowledge Base items.

Uses the same lazy singleton pattern as routes/gemini/_firestore.py.
Collection name configured via UM_DAEMON_KB_FIRESTORE_COLLECTION.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy singleton (same pattern as gemini/_firestore.py)
_client = None
_init_failed = False

# Common English stop words to strip from search queries
_STOP_WORDS = frozenset(
    "a an and are as at be by for from has have i in is it its of on or that "
    "the this to was were what when where which who will with you".split()
)


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
        logger.info("KB Firestore async client initialized")
        return _client
    except Exception as e:
        _init_failed = True
        logger.error("Failed to initialize KB Firestore client: %s", e)
        return None


def _collection_name() -> str:
    from um_agent_coder.daemon.app import get_settings

    return get_settings().kb_firestore_collection


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tokenize_query(text: str) -> List[str]:
    """Tokenize text into lowercase keywords, stripping stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) >= 2]


# --- CRUD ---


async def create_item(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create a KB item. Returns the item dict with generated ID."""
    client = _get_client()
    if not client:
        return None

    now = _now_iso()
    # Normalize tags to lowercase
    tags = [t.lower().strip() for t in data.get("tags", []) if t.strip()][:20]

    doc_data = {
        "type": data["type"],
        "title": data["title"],
        "content": data["content"],
        "tags": tags,
        "status": data.get("status", "active"),
        "priority": data.get("priority", "medium"),
        "source": data.get("source", "manual"),
        "source_ref": data.get("source_ref"),
        "created_at": now,
        "updated_at": now,
    }

    try:
        _, doc_ref = await client.collection(_collection_name()).add(doc_data)
        doc_data["id"] = doc_ref.id
        logger.info("Created KB item %s: %s", doc_ref.id, data["title"])
        return doc_data
    except Exception as e:
        logger.error("Failed to create KB item: %s", e)
        return None


async def get_item(item_id: str) -> Optional[Dict[str, Any]]:
    """Get a single KB item by ID."""
    client = _get_client()
    if not client:
        return None

    try:
        doc = await client.collection(_collection_name()).document(item_id).get()
        if not doc.exists:
            return None
        data = doc.to_dict()
        data["id"] = doc.id
        return data
    except Exception as e:
        logger.error("Failed to get KB item %s: %s", item_id, e)
        return None


async def list_items(
    item_type: Optional[str] = None,
    status: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List KB items with optional filters."""
    client = _get_client()
    if not client:
        return []

    try:
        query = client.collection(_collection_name())

        if status:
            query = query.where("status", "==", status)
        if item_type:
            query = query.where("type", "==", item_type)
        if tag:
            query = query.where("tags", "array_contains", tag.lower())

        query = query.order_by("updated_at", direction="DESCENDING").limit(limit)

        results = []
        async for doc in query.stream():
            data = doc.to_dict()
            data["id"] = doc.id
            results.append(data)
        return results
    except Exception as e:
        logger.error("Failed to list KB items: %s", e)
        return []


async def update_item(item_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a KB item. Returns the updated item."""
    client = _get_client()
    if not client:
        return None

    try:
        doc_ref = client.collection(_collection_name()).document(item_id)
        doc = await doc_ref.get()
        if not doc.exists:
            return None

        # Only update provided fields
        update_data = {k: v for k, v in updates.items() if v is not None}
        if "tags" in update_data:
            update_data["tags"] = [t.lower().strip() for t in update_data["tags"] if t.strip()][:20]
        update_data["updated_at"] = _now_iso()

        await doc_ref.update(update_data)

        # Return full item
        updated = await doc_ref.get()
        data = updated.to_dict()
        data["id"] = updated.id
        return data
    except Exception as e:
        logger.error("Failed to update KB item %s: %s", item_id, e)
        return None


async def archive_item(item_id: str) -> bool:
    """Soft-delete (archive) a KB item."""
    result = await update_item(item_id, {"status": "archived"})
    return result is not None


async def search_items(query_text: str, limit: int = 5) -> tuple[List[Dict[str, Any]], List[str]]:
    """Search KB items by keyword matching against tags.

    Tokenizes the query, strips stop words, then uses Firestore
    array_contains_any on the tags field. Returns (items, query_tokens).
    """
    client = _get_client()
    if not client:
        return [], []

    tokens = tokenize_query(query_text)
    if not tokens:
        return [], []

    # Firestore array_contains_any supports up to 30 values
    search_tokens = tokens[:30]

    try:
        query = (
            client.collection(_collection_name())
            .where("status", "==", "active")
            .where("tags", "array_contains_any", search_tokens)
            .limit(limit * 3)  # over-fetch then rank
        )

        candidates = []
        async for doc in query.stream():
            data = doc.to_dict()
            data["id"] = doc.id
            # Score by number of matching tags
            match_count = len(set(data.get("tags", [])) & set(search_tokens))
            candidates.append((match_count, data))

        # Sort by match count descending, take top N
        candidates.sort(key=lambda x: x[0], reverse=True)
        items = [c[1] for c in candidates[:limit]]

        return items, search_tokens
    except Exception as e:
        logger.error("Failed to search KB items: %s", e)
        return [], search_tokens
