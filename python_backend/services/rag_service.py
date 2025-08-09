"""
RAG (Retrieval-Augmented Generation) 서비스
RAG Chain을 래핑하여 FastAPI와 연동하는 서비스 레이어
ConversionService와 동일한 패턴으로 구현
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger('chattoner')

class RAGService:
    """RAG를 활용한 문서 검색 및 질의응답 서비스"""
    
    def __init__(self, user_preferences_service=None):
        """RAG 서비스 초기화 - ConversionService와 동일한 패턴"""
        # 직접 생성 (ConversionService 패턴)
        from services.prompt_engineering import PromptEngineer
        from services.openai_services import OpenAIService
        from services.conversion_service import ConversionService
        
        self.prompt_engineer = PromptEngineer()
        self.openai_service = OpenAIService()
        self.conversion_service = ConversionService()
        
        # 선택적 주입 (향후 확장용)
        self.user_preferences_service = user_preferences_service
        
        # RAG Chain 초기화
        self.rag_chain = None
        self._initialize_chain()
    
    def _initialize_chain(self):
        """RAG Chain 초기화"""
        try:
            from langchain_pipeline.chains.rag_chain import RAGChain
            # RAG Chain에 서비스들 전달
            self.rag_chain = RAGChain(
                prompt_engineer=self.prompt_engineer,
                openai_service=self.openai_service,
                conversion_service=self.conversion_service
            )
            logger.info("RAG Chain 초기화 완료")
        except ImportError as e:
            logger.error(f"RAG Chain import 실패: {e}")
            self.rag_chain = None
        except Exception as e:
            logger.error(f"RAG Chain 초기화 실패: {e}")
            self.rag_chain = None
    
    def ingest_documents(self, folder_path: str) -> Dict[str, Any]:
        """
        문서 폴더에서 벡터 DB 생성
        
        Args:
            folder_path: 문서가 있는 폴더 경로
            
        Returns:
            처리 결과 딕셔너리
        """
        if not self.rag_chain:
            return {
                "success": False,
                "error": "RAG 서비스가 초기화되지 않았습니다.",
                "documents_processed": 0
            }
        
        try:
            logger.info(f"문서 인덱싱 요청: {folder_path}")
            
            result = self.rag_chain.ingest_documents(folder_path)
            
            if result.get("success"):
                logger.info(f"문서 인덱싱 완료: {result.get('documents_processed', 0)}개 처리")
            else:
                logger.error(f"문서 인덱싱 실패: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"문서 인덱싱 중 오류: {e}")
            return {
                "success": False,
                "error": f"문서 인덱싱 중 서버 오류가 발생했습니다: {str(e)}",
                "documents_processed": 0
            }
    
    async def ask_question(self, 
                          query: str, 
                          context: Optional[str] = None) -> Dict[str, Any]:
        """
        단일 질의응답 (스타일 변환 없음)
        
        Args:
            query: 질문
            context: 추가 컨텍스트
            
        Returns:
            질의응답 결과
        """
        if not self.rag_chain:
            return {
                "success": False,
                "answer": "RAG 서비스가 초기화되지 않았습니다.",
                "sources": [],
                "metadata": {
                    "query_timestamp": self._get_timestamp(),
                    "model_used": "none"
                }
            }
        
        try:
            logger.info(f"단일 질의응답 요청: {len(query)}자")
            
            result = self.rag_chain.ask(query, context)
            
            # ConversionService와 호환되는 형태로 결과 구성
            formatted_result = {
                "success": result.get("success", False),
                "answer": result.get("answer", ""),
                "original_query": query,
                "context": context,
                "sources": result.get("sources", []),
                "metadata": {
                    "query_timestamp": result.get("timestamp", self._get_timestamp()),
                    "model_used": "gpt-4o + faiss",
                    "query_type": "single_answer"
                }
            }
            
            if not result.get("success"):
                formatted_result["error"] = result.get("answer", "알 수 없는 오류")
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"단일 질의응답 중 오류: {e}")
            return {
                "success": False,
                "error": f"질의응답 중 서버 오류가 발생했습니다: {str(e)}",
                "answer": "",
                "original_query": query,
                "sources": [],
                "metadata": {
                    "query_timestamp": self._get_timestamp(),
                    "model_used": "none"
                }
            }
    
    async def ask_with_styles(self, 
                             query: str, 
                             user_profile: Dict[str, Any], 
                             context: str = "personal") -> Dict[str, Any]:
        """
        3가지 스타일로 질의응답 (ConversionService와 동일한 구조)
        
        Args:
            query: 질문
            user_profile: 사용자 프로필
            context: 변환 컨텍스트
            
        Returns:
            3가지 스타일 변환 결과
        """
        if not self.rag_chain:
            return {
                "success": False,
                "error": "RAG 서비스가 초기화되지 않았습니다.",
                "original_query": query,
                "converted_texts": {
                    "direct": "",
                    "gentle": "",
                    "neutral": ""
                },
                "sources": [],
                "metadata": {
                    "query_timestamp": self._get_timestamp(),
                    "model_used": "none"
                }
            }
        
        try:
            logger.info(f"스타일별 질의응답 요청: {len(query)}자, context={context}")
            
            result = await self.rag_chain.ask_with_styles(
                query=query,
                user_profile=user_profile,
                context=context
            )
            
            # ConversionService와 호환되는 형태로 결과 구성
            if result.get("success"):
                formatted_result = {
                    "success": True,
                    "original_query": query,
                    "converted_texts": result.get("converted_texts", {}),
                    "context": context,
                    "user_profile": user_profile,
                    "sources": result.get("sources", []),
                    "rag_context": result.get("rag_context", ""),
                    "sentiment_analysis": result.get("sentiment_analysis", {}),
                    "metadata": {
                        "prompts_used": result.get("metadata", {}).get("prompts_used", ["rag_conversion"]),
                        "query_timestamp": self._get_timestamp(),
                        "model_used": "gpt-4o + faiss",
                        "query_type": "style_conversion"
                    }
                }
            else:
                formatted_result = {
                    "success": False,
                    "error": result.get("error", "스타일 변환 실패"),
                    "original_query": query,
                    "converted_texts": result.get("converted_texts", {
                        "direct": "",
                        "gentle": "",
                        "neutral": ""
                    }),
                    "context": context,
                    "user_profile": user_profile,
                    "sources": [],
                    "metadata": {
                        "query_timestamp": self._get_timestamp(),
                        "model_used": "none"
                    }
                }
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"스타일별 질의응답 중 오류: {e}")
            return {
                "success": False,
                "error": f"스타일 변환 중 서버 오류가 발생했습니다: {str(e)}",
                "original_query": query,
                "converted_texts": {
                    "direct": "",
                    "gentle": "",
                    "neutral": ""
                },
                "context": context,
                "user_profile": user_profile,
                "sources": [],
                "metadata": {
                    "query_timestamp": self._get_timestamp(),
                    "model_used": "none"
                }
            }
    
    async def process_user_feedback(self,
                                   feedback_text: str,
                                   user_profile: Dict[str, Any],
                                   rating: Optional[int] = None,
                                   selected_version: str = "neutral") -> Dict[str, Any]:
        """
        사용자 피드백 처리 (ConversionService와 호환)
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
                    updated_profile = user_profile.copy()
                    return {
                        "success": True,
                        "updated_profile": updated_profile,
                        "style_adjustments": {"advanced_learning": True},
                        "feedback_processed": feedback_text
                    }
            
            # 기본 피드백 처리 (OpenAIService 사용) - ConversionService 로직
            style_deltas = await self.openai_service.analyze_style_feedback(feedback_text)
            
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
        """RAG 서비스 상태 확인"""
        if not self.rag_chain:
            return {
                "rag_status": "not_ready",
                "doc_count": 0,
                "services_available": False,
                "service_initialized": False
            }
        
        try:
            status = self.rag_chain.get_status()
            status["service_initialized"] = True
            status["openai_service_ready"] = self.openai_service is not None
            status["prompt_engineer_ready"] = self.prompt_engineer is not None
            status["conversion_service_ready"] = self.conversion_service is not None
            status["user_preferences_available"] = self.user_preferences_service is not None
            return status
        except Exception as e:
            logger.error(f"상태 확인 중 오류: {e}")
            return {
                "rag_status": "error",
                "doc_count": 0,
                "services_available": False,
                "service_initialized": False,
                "error": str(e)
            }
    
    def is_available(self) -> bool:
        """RAG 서비스 사용 가능 여부"""
        return self.rag_chain is not None
    
    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()