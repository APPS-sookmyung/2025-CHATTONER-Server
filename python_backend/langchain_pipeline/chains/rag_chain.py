"""
FAISS 벡터스토어를 기반으로 문서를 인덱싱
GPT 기반 LLM을 통해 문서 기반 질의응답 기능을 제공
사용자의 말투 스타일에 맞춘 3가지 버전의 응답 적용

주요 기능:
- 문서 폴더에서 .txt 및 .pdf 파일을 불러와 벡터화 및 저장
- 저장된 벡터 인덱스를 로드하여 검색 기반 질의응답 수행
- 사용자 말투 프로필에 따른 스타일 변환 응답 제공
- 시스템 상태 및 문서 인덱스 현황 확인 기능 포함
"""

import sys
import logging
from pathlib import Path
import os
from typing import Dict, Optional, Any
from datetime import datetime

# 프로젝트 경로 설정
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

from langchain_pipeline.retriever.vector_db import ingest_documents_from_folder, FAISS_INDEX_PATH, get_embedding

# 로거 설정
logger = logging.getLogger(__name__)

class RAGChain:
    """RAG (Retrieval-Augmented Generation) 체인"""
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7):
        """RAG Chain 초기화"""
        dotenv_path = Path(__file__).resolve().parents[3] / ".env"
        load_dotenv(dotenv_path=dotenv_path)
        
        # Services 초기화
        try:
            from services.prompt_engineering import PromptEngineer
            from services.openai_services import OpenAIService
            from services.conversion_service import ConversionService
            from services.user_preferences import UserPreferencesService

            self.prompt_engineer = PromptEngineer()
            self.openai_service = OpenAIService()
            self.conversion_service = ConversionService()
            self.user_preferences_service = UserPreferencesService()
            self.services_available = True
            logger.info("Services 초기화 완료")
        except ImportError as e:
            self.services_available = False
            logger.warning(f"Services import 실패: {e}")
        except Exception as e:
            self.services_available = False
            logger.error(f"Services 인스턴스 생성 실패: {e}")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
        
        self.vectorstore = None
        self.retriever = None
        self.is_initialized = False
        
        # 기본 프롬프트
        self.default_rag_prompt = PromptTemplate.from_template("""
다음 문서 내용을 바탕으로 질문에 답변해주세요.

문서:
{context}

질문: {question}

답변:
""")
        
        # 자동 초기화 시도
        self._load_vectorstore()
    
    def _load_vectorstore(self):
        """기존 인덱스 로드"""
        try:
            if FAISS_INDEX_PATH.exists() and any(FAISS_INDEX_PATH.iterdir()):
                self.vectorstore = FAISS.load_local(
                    str(FAISS_INDEX_PATH),
                    embeddings=get_embedding(),
                    # WARNING: 이 옵션은 악의적으로 조작된 인덱스를 통해 임의 코드 실행이 발생할 수 있습니다.
                    # 반드시 FAISS 인덱스 파일이 신뢰된 경로에서 생성되었는지 확인하세요.
                    allow_dangerous_deserialization=True
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
                self.is_initialized = True
                logger.info(f"RAG Chain 준비 완료: {self.vectorstore.index.ntotal}개 문서")
        except Exception as e:
            logger.warning(f"기존 인덱스 로드 실패: {e}")
    
    def ingest_documents(self, folder_path: str) -> Dict:
        """문서 폴더에서 벡터 DB 생성"""
        try:
            result = ingest_documents_from_folder(Path(folder_path))
            if result and result[0] is not None:
                self._load_vectorstore()  # 재로드
                return {"success": True, "documents_processed": len(result[1])}
            return {"success": False, "error": "문서 처리 실패"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def ask(self, query: str, context: Optional[str] = None) -> Dict:
        """질문하기"""
        if not self.is_initialized:
            return {"success": False, "answer": "문서가 인덱싱되지 않았습니다."}
        
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": self.default_rag_prompt},
                return_source_documents=True
            )
            
            # 쿼리 실행
            if context and context.strip():
                enhanced_query = f"문맥: {context.strip()}\n\n질문: {query}"
            else:
                enhanced_query = query

            result = qa_chain.invoke({"query": enhanced_query})
            
            # 결과 정리
            source_docs = result.get("source_documents", [])
            chunks = [
                {
                    "content": doc.page_content[:100] + "...",
                    "source": doc.metadata.get("source", "Unknown")
                }
                for doc in source_docs
            ]
            
            return {
                "success": True,
                "answer": result["result"],
                "sources": chunks,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "answer": f"오류 발생: {e}",
                "sources": []
            }
    
    async def ask_with_styles(self, query: str, user_profile: Dict, context: str = "personal") -> Dict:
        """3가지 스타일 RAG 답변"""
        if not self.services_available:
            return {
                "success": False,
                "error": "Services가 로드되지 않았습니다.",
                "converted_texts": {"direct": "", "gentle": "", "neutral": ""}
            }
        
        if not self.is_initialized:
            return {
                "success": False,
                "error": "관련 문서를 찾을 수 없습니다.",
                "converted_texts": {"direct": "", "gentle": "", "neutral": ""}
            }
        
        try:
            # 문서 검색 (인라인화)
            docs = self.retriever.get_relevant_documents(query)
            if not docs:
                return {
                    "success": False,
                    "error": "관련 문서를 찾을 수 없습니다.",
                    "converted_texts": {"direct": "", "gentle": "", "neutral": ""}
                }
            
            # 검색된 문서를 텍스트로 결합
            context_parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                content = doc.page_content
                context_parts.append(f"[문서 {i}] ({source}):\n{content}")
            
            retrieved_docs = "\n\n".join(context_parts)
            enhanced_input = f"질문: {query}\n\n참고문서:\n{retrieved_docs}"
            
            # 기존 ConversionService 활용
            result = await self.conversion_service.convert_text(
                input_text=enhanced_input,
                user_profile=user_profile,
                context=context
            )
            
            # RAG 소스 문서 정보 추가
            if result.get("success"):
                sources = [
                    {
                        "content": doc.page_content[:100] + "...",
                        "source": doc.metadata.get("source", "Unknown")
                    }
                    for doc in docs
                ]
                result["sources"] = sources
                result["rag_context"] = retrieved_docs[:200] + "..." if len(retrieved_docs) > 200 else retrieved_docs
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"스타일 변환 중 오류 발생: {str(e)}",
                "converted_texts": {"direct": "", "gentle": "", "neutral": ""}
            }
        
    async def process_user_feedback(self,
                                   feedback_text: str,
                                   user_profile: Dict[str, Any],
                                   rating: Optional[int] = None,
                                   selected_version: str = "neutral") -> Dict[str, Any]:
        """사용자 피드백 처리 - 고급/기본 처리 선택"""
        try:
            # 고급 처리 (UserPreferencesService 사용)
            if self.user_preferences_service and rating:
                user_id = user_profile.get('userId', 'unknown')
                success = await self.user_preferences_service.adapt_user_style(
                    user_id=user_id,
                    feedback_text=feedback_text,
                    rating=rating,
                    selected_version=selected_version
                )
                
                if success:
                    updated_profile = user_profile.copy()
                    logger.info(f"고급 피드백 처리 완료: user_id={user_id}, rating={rating}")
                    return {
                        "success": True,
                        "updated_profile": updated_profile,
                        "style_adjustments": {"advanced_learning": True},
                        "feedback_processed": feedback_text,
                        "processing_method": "user_preferences_service"
                    }
            
            # 기본 처리 (ConversionService 사용)
            if not self.services_available or not self.conversion_service:
                return {
                    "success": False,
                    "error": "피드백 처리 서비스가 초기화되지 않았습니다.",
                    "updated_profile": user_profile
                }
            
            # ConversionService의 기본 피드백 처리 활용
            return await self.conversion_service.process_user_feedback(
                feedback_text=feedback_text,
                user_profile=user_profile
            )
            
        except Exception as e:
            logger.error(f"피드백 처리 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "updated_profile": user_profile,
                "processing_method": "error"
            }
            
    def get_status(self) -> Dict:
        """상태 정보"""
        return {
            "rag_status": "ready" if self.is_initialized else "not_ready",
            "doc_count": self.vectorstore.index.ntotal if self.is_initialized else 0,
            "services_available": self.services_available
        }