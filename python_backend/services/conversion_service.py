"""
텍스트 변환 통합 서비스
프롬프트 엔지니어링과 OpenAI API를 연결하는 메인 서비스

전체 텍스트 스타일 변환 프로세스를 통합하는 메인 서비스 클래스
> gpt 보낼 프롬프트 생성
> OpenAI API 호출로 텍스트 변환
> 변환된 텍스트와 메타데이터 반환
> 사용자 피드백 처리 및 프로필 업데이트
"""

from typing import Dict, List, Any, Optional
from .prompt_engineering import PromptEngineer
from .openai_service import OpenAIService

class ConversionService:
    """텍스트 변환 메인 서비스 클래스"""
    
    def __init__(self):
        self.prompt_engineer = PromptEngineer()
        self.openai_service = OpenAIService()
    
    async def convert_text(self, 
                          input_text: str,
                          user_profile: Dict[str, Any],
                          context: str = "personal",
                          negative_preferences: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        메인 텍스트 변환 함수
        
        Args:
            input_text: 변환할 원본 텍스트
            user_profile: 사용자 스타일 프로필
            context: 변환 컨텍스트 ("business", "report", "personal")
            negative_preferences: 사용자 네거티브 프롬프트 선호도
        
        Returns:
            변환 결과와 메타데이터
        """
        try:
            # 1. 프롬프트 생성
            prompts = self.prompt_engineer.generate_conversion_prompts(
                user_profile=user_profile,
                context=context,
                negative_preferences=negative_preferences
            )
            
            # 2. OpenAI API 호출하여 텍스트 변환
            converted_texts = self.openai_service.convert_text_styles(
                input_text=input_text,
                prompts=prompts
            )
            
            # 3. 원본 텍스트 감정 분석
            sentiment_analysis = self.openai_service.analyze_sentiment(input_text)
            
            # 4. 결과 구성
            result = {
                "success": True,
                "original_text": input_text,
                "converted_texts": converted_texts,
                "context": context,
                "user_profile": user_profile,
                "sentiment_analysis": sentiment_analysis,
                "metadata": {
                    "prompts_used": list(prompts.keys()),
                    "conversion_timestamp": self._get_timestamp(),
                    "model_used": self.openai_service.model
                }
            }
            
            return result
            
        except Exception as e:
            print(f"텍스트 변환 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_text": input_text,
                "converted_texts": {
                    "direct": input_text,
                    "gentle": input_text,
                    "neutral": input_text
                }
            }
    
    async def process_user_feedback(self, 
                                   feedback_text: str,
                                   user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자 피드백 처리 및 프로필 업데이트
        
        Args:
            feedback_text: 사용자 피드백
            user_profile: 현재 사용자 프로필
        
        Returns:
            업데이트된 사용자 프로필과 피드백 분석 결과
        """
        try:
            # 피드백 분석
            style_deltas = self.openai_service.analyze_style_feedback(feedback_text)
            
            # 사용자 프로필 업데이트 (세션 레벨)
            updated_profile = user_profile.copy()
            
            # 기존 값에 델타 적용 (1-5 범위로 제한)
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
            print(f"피드백 처리 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "updated_profile": user_profile
            }
    
    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()