"""Common API dependencies"""

from typing import Optional, Annotated
from fastapi import Depends, HTTPException, Header
from dependency_injector.wiring import inject, Provide

from core.container import Container
from services.user_service import UserService

async def get_current_user_optional(
    x_user_id: Annotated[Optional[str], Header()] = None
) -> Optional[dict]:
    """Optional user authentication"""
    if not x_user_id:
        return None
    
    # Actual user verification logic
    return {"user_id": x_user_id}

@inject
async def get_user_service(
    user_service: Annotated[
        UserService,
        Depends(Provide[Container.user_service])
    ]
) -> UserService:
    """User service dependency"""
    return user_service