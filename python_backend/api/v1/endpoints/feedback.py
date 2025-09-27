"""
Feedback Endpoints
Feedback processing endpoint
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from pydantic import BaseModel

from services.user_preferences import UserPreferencesService
from database.storage import DatabaseStorage

router = APIRouter()

# Request/Response Models
class FeedbackRequest(BaseModel):
    conversionId: int
    selectedVersion: str  # direct, gentle, neutral
    rating: int  # 1-5 score
    userId: str = "default"
    feedback_text: str = ""

class FeedbackResponse(BaseModel):
    status: str
    message: str
    adjustments_made: Dict[str, Any] = {}

# Dependency injection
def get_database_storage():
    return DatabaseStorage()

def get_user_preferences_service(db: DatabaseStorage = Depends(get_database_storage)):
    from services.openai_services import OpenAIService
    openai_service = OpenAIService()
    return UserPreferencesService(db, openai_service)

@router.post("")
async def submit_feedback(
    feedback: FeedbackRequest,
    user_service: UserPreferencesService = Depends(get_user_preferences_service)
) -> FeedbackResponse:
    """Process and learn from user feedback"""
    try:
        # Adjust user preferences based on feedback
        adjustments = {}
        
        # Logic for adjusting preferences based on selected version and rating
        if feedback.selectedVersion == "direct" and feedback.rating >= 4:
            adjustments["directness_preference"] = "increased"
        elif feedback.selectedVersion == "gentle" and feedback.rating >= 4:
            adjustments["politeness_preference"] = "increased"
        elif feedback.selectedVersion == "neutral" and feedback.rating >= 4:
            adjustments["balance_preference"] = "maintained"
        
        # Adjust in the opposite direction for low ratings
        if feedback.rating <= 2:
            if feedback.selectedVersion == "direct":
                adjustments["directness_preference"] = "decreased"
            elif feedback.selectedVersion == "gentle":
                adjustments["politeness_preference"] = "decreased"
        
        # Actual learning logic (activated)
        user_service.process_feedback(feedback.userId, adjustments)
        
        # Save feedback to database (activated)
        was_saved = user_service.save_feedback(feedback)

        if not was_saved:
            raise HTTPException(status_code=400, detail="Failed to save feedback.")
        
        return FeedbackResponse(
            status="success",
            message="Feedback has been processed successfully. Preferences have been updated.",
            adjustments_made=adjustments
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process feedback: {str(e)}")

@router.get("/stats/{user_id}")
async def get_feedback_stats(user_id: str) -> Dict[str, Any]:
    """Retrieve user's feedback statistics"""
    try:
        # Actual statistics retrieval (activated)
        stats = user_service.get_feedback_stats(user_id)
        
        if stats is not None:
            return stats

        raise HTTPException(status_code=404, detail=f"Statistics for user '{user_id}' not found.")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")