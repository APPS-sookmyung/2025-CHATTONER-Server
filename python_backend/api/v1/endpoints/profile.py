"""
User Profile Endpoints
User profile management endpoint
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from services.user_preferences import UserPreferencesService
from database.storage import DatabaseStorage

router = APIRouter()

# Request/Response Models
class ProfileRequest(BaseModel):
    userId: str
    baseFormalityLevel: int = Field(default=5, ge=1, le=10, description="Formality level (1-10)", example=5)
    baseFriendlinessLevel: int = Field(default=5, ge=1, le=10, description="Friendliness level (1-10)", example=5)
    baseEmotionLevel: int = Field(default=5, ge=1, le=10, description="Emotion expression level (1-10)", example=5)
    baseDirectnessLevel: int = Field(default=5, ge=1, le=10, description="Directness level (1-10)", example=5)
    sessionFormalityLevel: Optional[int] = Field(None, ge=1, le=10, description="Session-specific formality level")
    sessionFriendlinessLevel: Optional[int] = Field(None, ge=1, le=10, description="Session-specific friendliness level")
    sessionEmotionLevel: Optional[int] = Field(None, ge=1, le=10, description="Session-specific emotion expression level")
    sessionDirectnessLevel: Optional[int] = Field(None, ge=1, le=10, description="Session-specific directness level")
    responses: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional response data")

    class Config:
        schema_extra = {
            "example": {
                "userId": "user456",
                "baseFormalityLevel": 4,
                "baseFriendlinessLevel": 3,
                "baseEmotionLevel": 5,
                "baseDirectnessLevel": 2,
                "sessionFormalityLevel": 3,
                "sessionFriendlinessLevel": 4,
                "sessionEmotionLevel": 5,
                "sessionDirectnessLevel": 3,
                "responses": {"greeting": "Hello!"}
            }
        }

class ProfileResponse(BaseModel):
    id: int
    userId: str
    baseFormalityLevel: int
    baseFriendlinessLevel: int
    baseEmotionLevel: int
    baseDirectnessLevel: int
    sessionFormalityLevel: int
    sessionFriendlinessLevel: int
    sessionEmotionLevel: int
    sessionDirectnessLevel: int
    responses: Dict[str, Any]
    completedAt: str

    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "userId": "user456",
                "baseFormalityLevel": 4,
                "baseFriendlinessLevel": 3,
                "baseEmotionLevel": 5,
                "baseDirectnessLevel": 2,
                "sessionFormalityLevel": 3,
                "sessionFriendlinessLevel": 4,
                "sessionEmotionLevel": 5,
                "sessionDirectnessLevel": 3,
                "responses": {"greeting": "Hello!"},
                "completedAt": "2025-08-10T00:00:00Z"
            }
        }

# Dependency injection
def get_database_storage():
    return DatabaseStorage()

def get_user_preferences_service(db: DatabaseStorage = Depends(get_database_storage)):
    from services.openai_services import OpenAIService
    openai_service = OpenAIService()
    return UserPreferencesService(db, openai_service)

@router.get("/{user_id}", response_model=ProfileResponse, summary="Retrieve user profile", description="Retrieves the user's personalization settings.")
async def get_user_profile(
    user_id: str,
    user_service: UserPreferencesService = Depends(get_user_preferences_service)
) -> ProfileResponse:
    """
    ## Retrieve user profile

    Returns the user's text style personalization settings.
    """
    try:
        profile_data = user_service.get_user_profile(user_id)

        if not profile_data:
            raise HTTPException(status_code=404, detail=f"Profile for user '{user_id}' not found.")

        # Maps the database response to the API response model (ProfileResponse).
        return ProfileResponse(
            id=profile_data.get("id", 1),  # Use a temporary ID because 'id' does not exist
            userId=profile_data.get("userId"),
            baseFormalityLevel=profile_data.get("baseFormalityLevel"),
            baseFriendlinessLevel=profile_data.get("baseFriendlinessLevel"),
            baseEmotionLevel=profile_data.get("baseEmotionLevel"),
            baseDirectnessLevel=profile_data.get("baseDirectnessLevel"),
            # Use the default value if the session value does not exist.
            sessionFormalityLevel=profile_data.get("sessionFormalityLevel") or profile_data.get("baseFormalityLevel"),
            sessionFriendlinessLevel=profile_data.get("sessionFriendlinessLevel") or profile_data.get("baseFriendlinessLevel"),
            sessionEmotionLevel=profile_data.get("sessionEmotionLevel") or profile_data.get("baseEmotionLevel"),
            sessionDirectnessLevel=profile_data.get("sessionDirectnessLevel") or profile_data.get("baseDirectnessLevel"),
            responses=profile_data.get("questionnaireResponses", {}),
            completedAt=profile_data.get("updatedAt") or profile_data.get("createdAt")
        )
    except HTTPException:
        raise
    except Exception as e:
        # Other exception handling such as Pydantic validation error
        raise HTTPException(status_code=500, detail=f"Failed to retrieve profile: {str(e)}")

@router.post("", response_model=ProfileResponse, summary="Save user profile", description="Saves the user's personalization settings.")
async def save_user_profile(
    profile: ProfileRequest,
    user_service: UserPreferencesService = Depends(get_user_preferences_service)
) -> ProfileResponse:
    """
    ## Save user profile

    Saves new user profile settings or updates existing settings.

    ### Request body
    - `profile`: User profile data to be saved
    """
    try:
        # Save profile in actual service (activated)
        # Note: Depending on the arguments or return value of user_service.save_user_profile, modification may be necessary.
        was_saved = user_service.save_user_profile(profile.userId, profile.model_dump())

        if not was_saved:
            raise HTTPException(status_code=400, detail="Failed to save profile.")

        # After successful saving, it is ideal to retrieve and return the saved profile information again.
        # Here, a response is generated based on the requested data.
        return ProfileResponse(
            id=1, # Temporary ID
            userId=profile.userId,
            baseFormalityLevel=profile.baseFormalityLevel,
            baseFriendlinessLevel=profile.baseFriendlinessLevel,
            baseEmotionLevel=profile.baseEmotionLevel,
            baseDirectnessLevel=profile.baseDirectnessLevel,
            sessionFormalityLevel=profile.sessionFormalityLevel or profile.baseFormalityLevel,
            sessionFriendlinessLevel=profile.sessionFriendlinessLevel or profile.baseFriendlinessLevel,
            sessionEmotionLevel=profile.sessionEmotionLevel or profile.baseEmotionLevel,
            sessionDirectnessLevel=profile.sessionDirectnessLevel or profile.baseDirectnessLevel,
            responses=profile.responses or {},
            completedAt="2025-08-10T00:00:00Z" # Needs to be changed to the save time
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save profile: {str(e)}")