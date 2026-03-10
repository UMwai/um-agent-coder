"""World Agent router — event-driven autonomous agent."""

from __future__ import annotations

from fastapi import APIRouter

from um_agent_coder.daemon.routes.world_agent.endpoints import router as endpoints_router

router = APIRouter(prefix="/api/world-agent", tags=["world-agent"])
router.include_router(endpoints_router)
