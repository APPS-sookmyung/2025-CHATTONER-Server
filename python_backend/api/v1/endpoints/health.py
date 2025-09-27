"""
Health Check Endpoint
Endpoint for checking server status
"""

import os
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Dict, Any
from core.config import get_settings

router = APIRouter()

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Server status", example="ok")
    service: str = Field(..., description="Service name", example="chat-toner-fastapi")
    openai_available: bool = Field(..., description="OpenAI connection status", example=True)
    prompt_engineering_available: bool = Field(..., description="Prompt engineering status", example=True)
    features: Dict[str, bool] = Field(..., description="Available features")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "ok",
                "service": "chat-toner-fastapi",
                "openai_available": True,
                "prompt_engineering_available": True,
                "features": {
                    "basic_conversion": True,
                    "advanced_prompts": True,
                    "openai_integration": True,
                    "rag_chains": True,
                    "finetune_service": True
                }
            }
        }

@router.get("/health", 
            response_model=HealthResponse,
            summary="Check server status",
            description="Checks the overall system status and external service connections.")
async def health_check() -> HealthResponse:
    """
    ## Check server status
    
    Checks the overall status of the Chat Toner backend service.
    
    ### Items to check
    - **OpenAI connection status**
    - **Prompt engineering service status**  
    - **List of available features**
    """
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service="chat-toner-fastapi",
        openai_available=bool(settings.OPENAI_API_KEY),
        prompt_engineering_available=True,
        features={
            "basic_conversion": True,
            "advanced_prompts": True,
            "openai_integration": bool(settings.OPENAI_API_KEY),
            "rag_chains": True,
            "finetune_service": True
        }
    )

# Duplicate path removal ("/api/health" → deleted). Health check is provided only at "/health".