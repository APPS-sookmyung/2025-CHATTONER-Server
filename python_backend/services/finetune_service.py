"""
파인튜닝 서비스 
기존 ConversionService와 동일한 패턴으로 직접 인스턴스 생성
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import requests

logger = logging.getLogger('chattoner')

class FinetuneService:
    """파인튜닝 모델을 활용한 공식 문서 변환 서비스"""
    
    def __init__(self, user_preferences_service=None):
        self.inference_url = "http://localhost:8010"
        """파인튜닝 서비스 초기화 - ConversionService와 동일한 패턴"""
        # 직접 생성 (ConversionService 패턴)
        from services.prompt_engineering import PromptEngineer
        from services.openai_services import OpenAIService
        
        self.prompt_engineer = PromptEngineer()
        self.openai_service = OpenAIService()
        
        # 선택적 주입 (향후 확장용)
        self.user_preferences_service = user_preferences_service
        
        # FinetuneChain 초기화
        self.finetune_chain = None
        self._initialize_chain()
    
    def _initialize_chain(self):
        """FinetuneChain 초기화"""
        try:
            from langchain_pipeline.chains.finetune_chain import FinetuneChain
            # FinetuneChain에 서비스들 전달
            self.finetune_chain = FinetuneChain(
                prompt_engineer=self.prompt_engineer,
                openai_service=self.openai_service
            )
            logger.info("FinetuneChain 초기화 완료")
        except ImportError as e:
            logger.error(f"FinetuneChain import 실패: {e}")
            self.finetune_chain = None
        except Exception as e:
            logger.error(f"FinetuneChain 초기화 실패: {e}")
            self.finetune_chain = None
    
    async def convert_to_formal(self, 
                               input_text: str, 
                               user_profile: Dict[str, Any], 
                               context: str = "business",
                               force_convert: bool = False) -> Dict[str, Any]:
        """
        텍스트를 공식 문서 스타일로 변환
        
        Args:
            input_text: 변환할 텍스트
            user_profile: 사용자 프로필
            context: 변환 컨텍스트
            force_convert: 강제 변환 여부
            
        Returns:
            변환 결과 딕셔너리 (ConversionService와 동일한 구조)
        """
        if not self.finetune_chain:
            return {
                "success": False,
                "error": "파인튜닝 서비스가 초기화되지 않았습니다.",
                "original_text": input_text,
                "converted_text": input_text,
                "method": "error",
                "reason": "service_not_initialized",
                "metadata": {
                    "conversion_timestamp": self._get_timestamp(),
                    "model_used": "none"
                }
            }
        
        try:
            logger.info(f"공식 문서 변환 요청: force_convert={force_convert}, context={context}")
            
            result = await self.finetune_chain.convert_to_formal(
                input_text=input_text,
                user_profile=user_profile,
                context=context,
                force_convert=force_convert
            )
            
            # ConversionService와 호환되는 형태로 결과 구성
            if result.get("success"):
                formatted_result = {
                    "success": True,
                    "original_text": input_text,
                    "converted_text": result.get("converted_text", ""),
                    "context": context,
                    "user_profile": user_profile,
                    "lora_output": result.get("lora_output"),
                    "method": result.get("method", "lora_gpt"),
                    "reason": result.get("reason", "auto_condition"),
                    "forced": result.get("forced", force_convert),
                    "metadata": {
                        "prompts_used": ["formal_document_conversion"],
                        "conversion_timestamp": result.get("timestamp", self._get_timestamp()),
                        "model_used": "gemma-2-2b-it + gpt-4o"
                    }
                }
            else:
                formatted_result = {
                    "success": False,
                    "error": result.get("error", "변환 실패"),
                    "original_text": input_text,
                    "converted_text": input_text,
                    "context": context,
                    "user_profile": user_profile,
                    "method": result.get("method", "error"),
                    "reason": result.get("reason", "unknown"),
                    "metadata": {
                        "conversion_timestamp": self._get_timestamp(),
                        "model_used": "none"
                    }
                }
            
            logger.info(f"변환 완료: success={formatted_result.get('success')}, method={formatted_result.get('method')}")
            return formatted_result
            
        except Exception as e:
            logger.error(f"공식 문서 변환 중 오류: {e}")
            return {
                "success": False,
                "error": f"변환 중 서버 오류가 발생했습니다: {str(e)}",
                "original_text": input_text,
                "converted_text": input_text,
                "context": context,
                "user_profile": user_profile,
                "method": "error",
                "reason": "service_error",
                "metadata": {
                    "conversion_timestamp": self._get_timestamp(),
                    "model_used": "none"
                }
            }
    
    async def convert_by_user_request(self, 
                                     input_text: str, 
                                     user_profile: Dict[str, Any], 
                                     context: str = "business") -> Dict[str, Any]:
        """
        사용자 명시적 요청으로 공식 문서 변환
        (버튼 클릭 등으로 강제 변환하는 경우)
        """
        return await self.convert_to_formal(
            input_text=input_text,
            user_profile=user_profile,
            context=context,
            force_convert=True
        )
    
    async def process_user_feedback(self,
                                   feedback_text: str,
                                   user_profile: Dict[str, Any],
                                   rating: Optional[int] = None,
                                   selected_version: str = "neutral") -> Dict[str, Any]:
        """
        사용자 피드백 처리 (ConversionService와 호환)
        
        Args:
            feedback_text: 사용자 피드백
            user_profile: 현재 사용자 프로필
            rating: 피드백 평점 (1-5)
            selected_version: 선택된 변환 버전
        
        Returns:
            업데이트된 사용자 프로필과 피드백 분석 결과
        """
        try:
            if self.user_preferences_service and rating:
                # UserPreferencesService를 통한 고급 피드백 처리
                user_id = user_profile.get('userId', 'unknown')
                success = await self.user_preferences_service.adapt_user_style(
                    user_id=user_id,
                    feedback_text=feedback_text,
                    rating=rating,
                    selected_version=selected_version
                )
                
                if success:
                    # 업데이트된 프로필 조회
                    # 실제 구현시 storage에서 가져와야 함
                    updated_profile = user_profile.copy()
                    return {
                        "success": True,
                        "updated_profile": updated_profile,
                        "style_adjustments": {"advanced_learning": True},
                        "feedback_processed": feedback_text
                    }
            
            # 기본 피드백 처리 (OpenAIService 사용)
            style_deltas = await self.openai_service.analyze_style_feedback(feedback_text)
            
            # 사용자 프로필 업데이트 (기존 ConversionService 로직)
            updated_profile = user_profile.copy()
            
            current_formality = updated_profile.get('sessionFormalityLevel', 
                                                  updated_profile.get('baseFormalityLevel', 3))
            current_friendliness = updated_profile.get('sessionFriendlinessLevel',
                                                     updated_profile.get('baseFriendlinessLevel', 3))
            current_emotion = updated_profile.get('sessionEmotionLevel',
                                                updated_profile.get('baseEmotionLevel', 3))
            current_directness = updated_profile.get('sessionDirectnessLevel',
                                                   updated_profile.get('baseDirectnessLevel', 3))
            
            updated_profile['sessionFormalityLevel'] = max(1, min(5, 
                current_formality + style_deltas.get('formalityDelta', 0) * 2))
            updated_profile['sessionFriendlinessLevel'] = max(1, min(5,
                current_friendliness + style_deltas.get('friendlinessDelta', 0) * 2))
            updated_profile['sessionEmotionLevel'] = max(1, min(5,
                current_emotion + style_deltas.get('emotionDelta', 0) * 2))
            updated_profile['sessionDirectnessLevel'] = max(1, min(5,
                current_directness + style_deltas.get('directnessDelta', 0) * 2))
            
            return {
                "success": True,
                "updated_profile": updated_profile,
                "style_adjustments": style_deltas,
                "feedback_processed": feedback_text
            }
            
        except Exception as e:
            logger.error(f"피드백 처리 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "updated_profile": user_profile
            }
    
    def get_status(self) -> Dict[str, Any]:
        """파인튜닝 서비스 상태 확인"""
        if not self.finetune_chain:
            return {
                "lora_status": "not_ready",
                "lora_model_path": "N/A",
                "services_available": False,
                "base_model_loaded": False,
                "device": "N/A",
                "model_name": "N/A",
                "service_initialized": False
            }
        
        try:
            status = self.finetune_chain.get_status()
            status["service_initialized"] = True
            status["openai_service_ready"] = self.openai_service is not None
            status["prompt_engineer_ready"] = self.prompt_engineer is not None
            status["user_preferences_available"] = self.user_preferences_service is not None
            return status
        except Exception as e:
            logger.error(f"상태 확인 중 오류: {e}")
            return {
                "lora_status": "error",
                "lora_model_path": "N/A",
                "services_available": False,
                "base_model_loaded": False,
                "device": "N/A",
                "model_name": "N/A",
                "service_initialized": False,
                "error": str(e)
            }
    
    def is_available(self) -> bool:
        """파인튜닝 서비스 사용 가능 여부"""
        return self.finetune_chain is not None
    
    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    # 순수 파인튜닝 모델만 사용하여 텍스트 생성(파인튜닝 테스트 용도)
    async def generate_with_finetuned_model_only(
        self, 
        input_text: str,
        max_tokens: int = 256
    ) -> Dict[str, Any]:
        """순수 파인튜닝된 모델만 사용하여 텍스트 생성"""
        try:
            response = requests.post(
                f"{self.inference_url}/generate",
                json={
                    "prompt": input_text,
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "do_sample": True
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            return {
                "success": True,
                "original_text": input_text,
                "converted_text": result["result"],
                "method": "finetuned_model_only",
                "metadata": {
                    "prompt_length": result["prompt_length"],
                    "generated_length": result["generated_length"],
                    "conversion_timestamp": self._get_timestamp(),
                    "model_used": "gemma-2-2b-it-lora"
                }
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"추론 서버 연결 실패: {e}")
            raise ValueError(f"추론 서버에 연결할 수 없습니다: {e}")
        except Exception as e:
            logger.error(f"모델 추론 실패: {e}")
            raise ValueError(f"모델 추론 중 오류가 발생했습니다: {e}")