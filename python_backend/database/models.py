"""
SQLAlchemy 모델 정의:
1. User - 기본 사용자 정보
2. UserProfile - 스타일 선호도
3. ConversionHistory - 변환 기록
4. NegativePreferences - 네거티브 프롬프트 설정
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()

class User(Base):
    """사용자 기본 정보 모델"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 관계 설정 (외래키 제약조건 제거)
    # profile = relationship("UserProfile", back_populates="user", uselist=False)
    # conversion_history = relationship("ConversionHistory", back_populates="user")
    # negative_preferences = relationship("NegativePreferences", back_populates="user", uselist=False)

class UserProfile(Base):
    """사용자 스타일 선호도 프로필"""
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    
    # 기본 스타일 레벨 (1-5 스케일)
    base_formality_level = Column(Integer, default=3)
    base_friendliness_level = Column(Integer, default=3)
    base_emotion_level = Column(Integer, default=3)
    base_directness_level = Column(Integer, default=3)
    
    # 세션별 학습된 스타일 레벨
    session_formality_level = Column(Float, default=None)
    session_friendliness_level = Column(Float, default=None)
    session_emotion_level = Column(Float, default=None)
    session_directness_level = Column(Float, default=None)
    
    # 설문 응답 데이터 (JSON 형태)
    questionnaire_responses = Column(JSON, default={})
    
    # 프로필 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 관계 설정
    # user = relationship("User", back_populates="profile")

class ConversionHistory(Base):
    """텍스트 변환 기록"""
    __tablename__ = "conversion_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    
    # 변환 데이터
    original_text = Column(Text, nullable=False)
    converted_texts = Column(JSON, nullable=False)  # {direct: text, gentle: text, neutral: text}
    context = Column(String(50), default="personal")  # business, report, personal
    
    # 피드백 데이터
    user_rating = Column(Integer, default=None)  # 1-5 스케일
    selected_version = Column(String(20), default=None)  # direct, gentle, neutral
    feedback_text = Column(Text, default=None)
    
    # 감정 분석 결과
    sentiment_analysis = Column(JSON, default={})
    
    # 메타데이터
    prompts_used = Column(JSON, default={})
    model_used = Column(String(50), default="gpt-4o")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 관계 설정
    # user = relationship("User", back_populates="conversion_history")

class NegativePreferences(Base):
    """사용자 네거티브 프롬프트 선호도"""
    __tablename__ = "negative_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)

    # 6가지 네거티브 프롬프트 카테고리 (strict, moderate, lenient)
    avoid_flowery_language = Column(String(20), default="moderate")
    avoid_repetitive_words = Column(String(20), default="moderate")
    comma_usage_style = Column(String(20), default="moderate")
    content_over_format = Column(String(20), default="moderate")
    bullet_point_usage = Column(String(20), default="moderate")
    emoticon_usage = Column(String(20), default="strict")

    # 커스텀 네거티브 프롬프트
    custom_negative_prompts = Column(JSON, default=[])

    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 관계 설정
    # user = relationship("User", back_populates="negative_preferences")

class CompanyDocument(Base):
    """기업 문서 정보"""
    __tablename__ = "company_documents"

    id = Column(Integer, primary_key=True, index=True)

    # 문서 식별 정보
    document_id = Column(String(255), unique=True, nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)

    # 파일 정보
    file_hash = Column(String(64), unique=True, nullable=False, index=True)  # SHA-256 해시
    file_size_bytes = Column(Integer, nullable=False)
    content_type = Column(String(50), default="application/pdf")

    # 추출된 텍스트 정보
    extracted_text = Column(Text, nullable=False)
    text_length = Column(Integer, nullable=False)

    # 저장 경로
    text_file_path = Column(Text, nullable=False)

    # 처리 상태
    status = Column(String(20), default="active")  # active, deleted, processing, error
    processing_error = Column(Text, default=None)

    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class VectorDocumentMetadata(Base):
    """벡터 데이터베이스 문서 메타데이터"""
    __tablename__ = "vector_document_metadata"

    id = Column(Integer, primary_key=True, index=True)

    # 문서 정보
    document_hash = Column(String(64), unique=True, nullable=False, index=True)  # SHA-256 해시
    file_name = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    content_type = Column(String(50), default="text/plain")

    # 임베딩 정보
    embedding_model = Column(String(100), nullable=False)
    chunk_count = Column(Integer, nullable=False)
    chunk_size = Column(Integer, nullable=False)
    chunk_overlap = Column(Integer, nullable=False)

    # FAISS 인덱스 정보
    faiss_index_path = Column(Text, nullable=False)
    vector_dimension = Column(Integer, nullable=False)

    # 상태 정보
    status = Column(String(20), default="active")  # active, deleted, error
    last_accessed = Column(DateTime, default=datetime.utcnow)

    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class RAGQueryHistory(Base):
    """RAG 질의 응답 기록"""
    __tablename__ = "rag_query_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)

    # 질의 정보
    query_text = Column(Text, nullable=False)
    query_hash = Column(String(64), nullable=False, index=True)  # 중복 질의 추적용
    context_type = Column(String(50), default="general")

    # 검색 결과
    retrieved_documents = Column(JSON, default=[])  # 검색된 문서 청크 정보
    similarity_scores = Column(JSON, default=[])  # 유사도 점수들
    total_search_time_ms = Column(Integer, default=0)

    # 응답 정보
    generated_answer = Column(Text)
    answer_quality_score = Column(Float)  # 0-1 사이 품질 점수
    model_used = Column(String(50), default="gpt-4")
    total_generation_time_ms = Column(Integer, default=0)

    # 사용자 피드백
    user_rating = Column(Integer)  # 1-5 점수
    user_feedback = Column(Text)
    was_helpful = Column(Boolean, default=None)

    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow)

# 데이터베이스 엔진 및 세션 설정
def create_database_engine():
    """데이터베이스 엔진 생성"""
    database_url = os.getenv("DATABASE_URL", "sqlite:///./chat_toner.db")
    
    if database_url.startswith("sqlite"):
        engine = create_engine(
            database_url, connect_args={"check_same_thread": False}
        )
    else:
        engine = create_engine(database_url)
    
    return engine

# 글로벌 엔진 및 세션 생성
engine = create_database_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 테이블 생성
Base.metadata.create_all(bind=engine)

def get_db():
    """데이터베이스 세션 의존성"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()