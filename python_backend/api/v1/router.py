"""
Main API Router
Main router that integrates all endpoints
"""

from fastapi import APIRouter

# Import individual endpoint routers
from .endpoints import conversion, health, profile, quality, feedback, rag, finetune

# Create main API router
api_router = APIRouter()

# Health check (root level)
api_router.include_router(health.router, tags=["health"])

# API v1 endpoints (removed duplicate "/api" prefix)
api_router.include_router(conversion.router, prefix="/conversion", tags=["conversion"])
api_router.include_router(profile.router, prefix="/profile", tags=["profile"])
api_router.include_router(quality.router, prefix="/quality", tags=["quality"])
api_router.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
api_router.include_router(finetune.router, prefix="/finetune", tags=["finetune"])

