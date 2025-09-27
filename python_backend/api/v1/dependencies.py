"""Common API dependencies"""

from typing import Optional, Annotated
from fastapi import Depends, HTTPException, Header
from core.container import Container
from services.conversion_service import ConversionService # Added import

async def get_current_user_optional(
    x_user_id: Annotated[Optional[str], Header()] = None
) -> Optional[dict]:
    """Optional user authentication"""
    if not x_user_id:
        return None
    
    # Actual user verification logic
    return {"user_id": x_user_id}

def get_conversion_service() -> ConversionService: # Changed type hint
    """Provides a ConversionService instance."""
    return Container.conversion_service()