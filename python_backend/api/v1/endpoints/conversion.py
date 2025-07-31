"""텍스트 변환 API 엔드포인트
신경 쓴 점: 
- 라우터 분리 + DI 구성 
- 의존성 주입 -> Depends + Provide 사용

신경 써야 하는점: 
1. 피드백 route 에 요청/응답 모델 정의
2. exception 처리 개선 
3. 공통 응답 메세지는 모델로 정의 
4. 의미 있는 status code 분리 
5. pydantic 모델이 request body 에서 명시적임을 보장 
"""

from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from dependency_injector.wiring import inject, Provide

from core.container import Container
from services.conversion_service import ConversionService
from services.conversion_service import ConversionService
from api.v1.schemas.conversion import (
    ConversionRequest,
    ConversionResponse,
    FeedbackRequest,
    FeedbackResponse
)
from api.dependencies import get_current_user_optional
import logging 

logger=logging.getLogger('chattoner')

router = APIRouter()

@router.post("/convert", response_model=ConversionResponse)
@inject
async def convert_text(
    request: ConversionRequest,
    conversion_service: Annotated[
        ConversionService, 
        Depends(Provide[Container.conversion_service])
    ],
    current_user = Depends(get_current_user_optional)
) -> ConversionResponse:
    """텍스트 스타일 변환"""
    try:
        result = await conversion_service.convert_text(
            input_text=request.text,
            user_profile=request.user_profile,
            context=request.context,
            negative_preferences=request.negative_preferences
        )
        
        return ConversionResponse(**result)
        
    # 내가 신경쓴 오류 체킹 
    except ValueError as e:
        logger.warning(f"입력 값 오류: {ve}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.error(f"변환 실패: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="텍스트 변환 중 서버 오류가 발생했습니다.")

@router.post("/feedback")
@inject
async def process_feedback(
     request: FeedbackRequest,
    conversion_service: Annotated[
        ConversionService, Depends(Provide[Container.conversion_service])
    ]
) -> FeedbackResponse:
    """사용자 피드백 처리"""
    try:
        result = await conversion_service.process_user_feedback(
            feedback_text=request.feedback_text,
            user_profile=request.user_profile
        )
        return FeedbackResponse(success=True, message="피드백이 반영되었습니다.", data=result)
    
    except Exception as e:
        logger.error(f"피드백 처리 실패: {e}")
        raise HTTPException(status_code=500, detail="피드백 처리 중 오류가 발생했습니다.")