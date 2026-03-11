"""Knowledge Base router — Firestore-backed project brain."""

from __future__ import annotations

from fastapi import APIRouter

from um_agent_coder.daemon.routes.kb.endpoints import router as endpoints_router

router = APIRouter(prefix="/api/kb", tags=["knowledge-base"])
router.include_router(endpoints_router)
