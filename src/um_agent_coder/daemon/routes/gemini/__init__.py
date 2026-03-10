"""Gemini Intelligence Layer — combined APIRouter mounting all sub-routers."""

from fastapi import APIRouter

from .agent import router as agent_router
from .batch import router as batch_router
from .context_extractor import router as context_extractor_router
from .enhance import router as enhance_router
from .evaluate import router as evaluate_router
from .extract import router as extract_router
from .iterate import router as iterate_router
from .models_endpoint import router as models_endpoint_router
from .sessions import router as sessions_router

router = APIRouter(prefix="/api/gemini", tags=["gemini"])

router.include_router(enhance_router)
router.include_router(evaluate_router)
router.include_router(sessions_router)
router.include_router(batch_router)
router.include_router(agent_router)
router.include_router(context_extractor_router)
router.include_router(extract_router)
router.include_router(iterate_router)
router.include_router(models_endpoint_router)
