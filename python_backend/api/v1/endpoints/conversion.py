"""Text Conversion API Endpoint
Things to note: 
- Router separation + DI configuration 
- Dependency injection -> Using Depends + Provide

Things to be aware of: 
1. Define request/response models for the feedback route
2. Improve exception handling 
3. Define common response messages as models 
4. Separate meaningful status codes 
5. Ensure that the pydantic model is explicit in the request body 
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from services.conversion_service import ConversionService
from ..schemas.conversion import ConversionRequest, ConversionResponse
from ..dependencies import get_conversion_service
import logging 

logger=logging.getLogger('chattoner')

router = APIRouter()

@router.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    print("[DEBUG] Test endpoint called")
    return {"message": "Test successful", "status": "ok"}

@router.post("/convert")
async def convert_text(request: ConversionRequest, 
                      conversion_service: ConversionService = Depends(get_conversion_service)):
    """Text style conversion using actual AI service"""
    try:
        # Use the actual ConversionService with camelCase preservation
        user_profile_dict = request.user_profile.model_dump(by_alias=True, exclude_none=True)
        negative_preferences_dict = request.negative_preferences.model_dump(by_alias=True, exclude_none=True) if request.negative_preferences else None
        
        result = await conversion_service.convert_text(
            input_text=request.text,
            user_profile=user_profile_dict,
            context=request.context,
            negative_preferences=negative_preferences_dict
        )
        
        return ConversionResponse(
            success=result.get("success", True),
            original_text=request.text,
            converted_texts=result.get("converted_texts", {}),
            context=request.context,
            sentiment_analysis=result.get("sentiment_analysis"),
            metadata=result.get("metadata", {})
        )
        
    except Exception as e:
        import logging, traceback
        logger = logging.getLogger(__name__)
        logger.error("convert failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail="An error occurred on the server during text conversion.")

