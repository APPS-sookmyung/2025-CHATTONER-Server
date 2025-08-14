"""
텍스트 변환 API 라우트
- 이 API는 텍스트 스타일 변환을 위한 엔드포인트를 제공합
- 즉 FastAPI 기반의 라우터 파일로, 외부에서 요청을 받을 실제 API endpoint를 정의
- 사용자 피드백을 처리하는 엔드포인트도 포함
- 변환 요청과 피드백 요청을 처리하는 메소드 포함

"""
import os
import sys
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict,  Any, Optional
from services.conversion_service import ConversionService

router = APIRouter(prefix="/api/conversion", tags=["conversion"])

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_preferences: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "message": "안녕하세요! 비즈니스 이메일 작성을 도와주세요.",
                "conversation_id": "conv_123", 
                "user_preferences": {"tone": "formal"}
            }
        }

class ConversionRequest(BaseModel):
    text: str
    user_profile: Dict[str, Any]
    context: str = "personal"
    negative_preferences: Optional[Dict[str, str]] = None

class FeedbackRequest(BaseModel):
    feedback_text: str
    user_profile: Dict[str, Any]

# 서비스 인스턴스
conversion_service = ConversionService()

@router.post("/chat")
async def chat(request: ChatRequest):
    """채팅 메시지 처리"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="메시지를 입력해주세요.")
        
        result = await conversion_service.handle_chat_message(
            message=request.message,
            conversation_id=request.conversation_id,
            user_preferences=request.user_preferences
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "채팅 처리 중 오류가 발생했습니다."))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}") from e

@router.post("/convert")
async def convert_text(request: ConversionRequest):
    """텍스트 스타일 변환"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="변환할 텍스트를 입력해주세요.")
        
        if len(request.text) > 5000:
            raise HTTPException(status_code=400, detail="텍스트가 너무 깁니다. 5000자 이하로 입력해주세요.")
        
        result = await conversion_service.convert_text(
            input_text=request.text,
            user_profile=request.user_profile,
            context=request.context,
            negative_preferences=request.negative_preferences
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "변환 중 오류가 발생했습니다."))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

@router.post("/feedback")
async def process_feedback(request: FeedbackRequest):
    """사용자 피드백 처리"""
    try:
        if not request.feedback_text.strip():
            raise HTTPException(status_code=400, detail="피드백을 입력해주세요.")
        
        result = await conversion_service.process_user_feedback(
            feedback_text=request.feedback_text,
            user_profile=request.user_profile
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "피드백 처리 중 오류가 발생했습니다."))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")
    
openai_client = None 
PROMPT_ENGINEERING_AVAILABLE = False
finetune_service = None

@router.get("/health")
@app.get("/health", tags=["Health Check"], summary="서버 상태 확인")
async def health_check():
    """
    시스템 전체 상태 및 외부 서비스 연결 확인
    
    - **OpenAI 연결 상태**
    - **프롬프트 엔지니어링 서비스 상태**
    - **사용 가능한 기능 목록**
    """
    return {
        "status": "ok",
        "service": "chat-toner-fastapi",
        "openai_available": openai_client is not None,
        "openai_key_exists": bool(os.getenv("OPENAI_API_KEY")),
        "prompt_engineering_available": PROMPT_ENGINEERING_AVAILABLE,
        "python_version": sys.version,
        "features": {
            "basic_conversion": True,
            "advanced_prompts": PROMPT_ENGINEERING_AVAILABLE,
            "openai_integration": openai_client is not None,
            "rag_chains": PROMPT_ENGINEERING_AVAILABLE,
            "finetune_service": finetune_service is not None
        }
    }
