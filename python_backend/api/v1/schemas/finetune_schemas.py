"""
파인튜닝 API 요청/응답 스키마
공식 문서 변환을 위한 Pydantic 모델들
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class FinetuneRequest(BaseModel):
    """파인튜닝 변환 요청 모델"""
    text: str = Field(..., description="변환할 원본 텍스트", min_length=1, max_length=5000)
    user_profile: Dict[str, Any] = Field(..., description="사용자 프로필 정보")
    context: str = Field(default="business", description="변환 컨텍스트 (business, report, personal)")
    force_convert: bool = Field(default=False, description="강제 변환 여부 (사용자 명시적 요청)")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "안녕하세요! 회의 참석 부탁드립니다.",
                "user_profile": {
                    "sessionFormalityLevel": 5,
                    "baseFormalityLevel": 4,
                    "formal_document_mode": False,
                    "negativePreferences": {}
                },
                "context": "business",
                "force_convert": False
            }
        }

class FinetuneResponse(BaseModel):
    """파인튜닝 변환 응답 모델"""
    success: bool = Field(..., description="변환 성공 여부")
    converted_text: str = Field(default="", description="변환된 텍스트")
    lora_output: Optional[str] = Field(default=None, description="LoRA 모델 1차 변환 결과")
    method: str = Field(..., description="사용된 변환 방법 (lora_gpt, gpt_only, none, error)")
    reason: str = Field(..., description="변환 수행/미수행 이유")
    forced: bool = Field(default=False, description="강제 변환 여부")
    error: Optional[str] = Field(default=None, description="오류 메시지 (실패 시)")
    timestamp: Optional[str] = Field(default=None, description="변환 완료 시간")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "converted_text": "안녕하십니까. 회의 참석을 요청드립니다.",
                "lora_output": "안녕하세요. 회의 참석 부탁드립니다.",
                "method": "lora_gpt",
                "reason": "auto_condition",
                "forced": False,
                "error": None,
                "timestamp": "2025-01-15T10:30:00"
            }
        }

class FinetuneStatusResponse(BaseModel):
    """파인튜닝 서비스 상태 응답 모델"""
    lora_status: str = Field(..., description="LoRA 모델 상태")
    lora_model_path: str = Field(..., description="LoRA 모델 경로")
    services_available: bool = Field(..., description="Services 사용 가능 여부")
    base_model_loaded: bool = Field(..., description="베이스 모델 로드 상태")
    device: str = Field(..., description="사용 중인 디바이스 (cuda/cpu)")
    model_name: str = Field(..., description="사용 중인 모델명")
    
    class Config:
        schema_extra = {
            "example": {
                "lora_status": "ready",
                "lora_model_path": "/content/langchain_pipeline/chains/finetuned_models",
                "services_available": True,
                "base_model_loaded": True,
                "device": "cuda",
                "model_name": "gemma-2-2b-it"
            }
        }