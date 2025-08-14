"""파인튜닝 API 엔드포인트
공식 문서 변환을 위한 LoRA 파인튜닝 모델 API

주요 기능:
- LoRA + ChatGPT 파이프라인을 통한 공식 문서 변환
- 사용자 프로필 기반 자동/수동 변환 판단
- DI 컨테이너를 통한 서비스 주입
- 체계적인 예외 처리 및 로깅
"""

from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from dependency_injector.wiring import inject, Provide

from core.container import Container
from services.finetune_service import FinetuneService
from api.v1.schemas.finetune_schemas import (
    FinetuneRequest,
    FinetuneResponse,
    FinetuneStatusResponse
)
from api.v1.dependencies import get_current_user_optional
import logging

logger = logging.getLogger('chattoner')

router = APIRouter()

@router.post("/generate-only", response_model=FinetuneResponse)
@inject
async def generate_with_model_only(
    request: FinetuneRequest,
    finetune_service: Annotated[
        FinetuneService, 
        Depends(Provide[Container.finetune_service])
    ],
    current_user = Depends(get_current_user_optional)
) -> FinetuneResponse:
    """순수 파인튜닝된 모델만 사용하여 텍스트 생성 (ChatGPT 없이)"""
    try:
        logger.info(f"순수 모델 생성 요청 - 텍스트 길이: {len(request.text)}")
        
        if not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="입력 텍스트를 입력해주세요."
            )
        
        result = await finetune_service.generate_with_finetuned_model_only(
            input_text=request.text
        )
        
        return FinetuneResponse(**result)
        
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"순수 모델 생성 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="모델 생성 중 오류가 발생했습니다."
        )
    
@router.post("/convert", response_model=FinetuneResponse)
@inject
async def convert_text(
    request: FinetuneRequest,
    finetune_service: Annotated[
        FinetuneService, 
        Depends(Provide[Container.finetune_service]) # noqa: B008
    ],
    current_user = Depends(get_current_user_optional) # noqa: B008
) -> FinetuneResponse:
    """텍스트를 공식 문서 스타일로 변환"""
    try:
        logger.info(f"공식 문서 변환 요청 - 텍스트 길이: {len(request.text)}, 강제변환: {request.force_convert}")
        
        # 입력 검증
        if not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="변환할 텍스트를 입력해주세요."
            )
        
        if len(request.text) > 5000:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="텍스트가 너무 깁니다. 5000자 이하로 입력해주세요."
            )
        
        # 파인튜닝 서비스 사용 가능 여부 확인
        if not finetune_service.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="파인튜닝 서비스를 사용할 수 없습니다. 서비스 상태를 확인해주세요."
            )
        
        # 변환 수행
        result = await finetune_service.convert_to_formal(
            input_text=request.text,
            user_profile=request.user_profile,
            context=request.context,
            force_convert=request.force_convert
        )
        
        return FinetuneResponse(**result)
        
    except HTTPException:
        raise
    except ValueError as ve:
        logger.warning(f"입력 값 오류: {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail=str(ve)
        ) from ve
    except Exception as e:
        logger.error(f"공식 문서 변환 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="공식 문서 변환 중 서버 오류가 발생했습니다."
        ) from e

@router.post("/convert-forced", response_model=FinetuneResponse)
@inject
async def convert_by_user_request(
    request: FinetuneRequest,
    finetune_service: Annotated[
        FinetuneService, 
        Depends(Provide[Container.finetune_service])
    ],
    current_user = Depends(get_current_user_optional)
) -> FinetuneResponse:
    """사용자 명시적 요청으로 강제 변환 (버튼 클릭 등)"""
    try:
        logger.info(f"사용자 명시적 요청 - 강제 공식 문서 변환: {len(request.text)}자")
        
        # 입력 검증
        if not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="변환할 텍스트를 입력해주세요."
            )
        
        if len(request.text) > 5000:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="텍스트가 너무 깁니다. 5000자 이하로 입력해주세요."
            )
        
        # 파인튜닝 서비스 사용 가능 여부 확인
        if not finetune_service.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="파인튜닝 서비스를 사용할 수 없습니다."
            )
        
        # 강제 변환 수행
        result = await finetune_service.convert_by_user_request(
            input_text=request.text,
            user_profile=request.user_profile,
            context=request.context
        )
        
        return FinetuneResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"강제 공식 문서 변환 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="강제 변환 중 서버 오류가 발생했습니다."
        ) from e

@router.get("/status", response_model=FinetuneStatusResponse)
@inject
async def get_finetune_status(
    finetune_service: Annotated[
        FinetuneService, 
        Depends(Provide[Container.finetune_service])
    ]
) -> FinetuneStatusResponse:
    """파인튜닝 서비스 상태 확인"""
    try:
        status_info = finetune_service.get_status()
        return FinetuneStatusResponse(**status_info)
        
    except Exception as e:
        logger.error(f"상태 확인 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서비스 상태 확인 중 오류가 발생했습니다."
        ) from e

@router.get("/health")
async def health_check():
    """파인튜닝 서비스 헬스체크"""
    return {
        "status": "ok",
        "service": "finetune-service",
        "message": "파인튜닝 서비스가 정상 작동 중입니다."
    }