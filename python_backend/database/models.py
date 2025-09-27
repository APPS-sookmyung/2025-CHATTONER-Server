"""
SQLAlchemy Model Definitions:
1. User - Basic user information
2. UserProfile - Style preferences
3. ConversionHistory - Conversion records
4. NegativePreferences - Negative prompt settings
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()

class User(Base):
    """Basic user information model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship settings (foreign key constraints removed)
    # profile = relationship("UserProfile", back_populates="user", uselist=False)
    # conversion_history = relationship("ConversionHistory", back_populates="user")
    # negative_preferences = relationship("NegativePreferences", back_populates="user", uselist=False)

class UserProfile(Base):
    """User style preference profile"""
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Basic style levels (1-5 scale)
    base_formality_level = Column(Integer, default=3)
    base_friendliness_level = Column(Integer, default=3)
    base_emotion_level = Column(Integer, default=3)
    base_directness_level = Column(Integer, default=3)
    
    # Session-specific learned style levels
    session_formality_level = Column(Float, default=None)
    session_friendliness_level = Column(Float, default=None)
    session_emotion_level = Column(Float, default=None)
    session_directness_level = Column(Float, default=None)
    
    # Questionnaire response data (JSON format)
    questionnaire_responses = Column(JSON, default={})
    
    # Profile metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship settings
    # user = relationship("User", back_populates="profile")

class ConversionHistory(Base):
    """Text conversion records"""
    __tablename__ = "conversion_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Conversion data
    original_text = Column(Text, nullable=False)
    converted_texts = Column(JSON, nullable=False)  # {direct: text, gentle: text, neutral: text}
    context = Column(String(50), default="personal")  # business, report, personal
    
    # Feedback data
    user_rating = Column(Integer, default=None)  # 1-5 스케일
    selected_version = Column(String(20), default=None)  # direct, gentle, neutral
    feedback_text = Column(Text, default=None)
    
    # Sentiment analysis results
    sentiment_analysis = Column(JSON, default={})
    
    # Metadata
    prompts_used = Column(JSON, default={})
    model_used = Column(String(50), default="gpt-4o")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship settings
    # user = relationship("User", back_populates="conversion_history")

class NegativePreferences(Base):
    """User negative prompt preferences"""
    __tablename__ = "negative_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)

    # 6 negative prompt categories (strict, moderate, lenient)
    avoid_flowery_language = Column(String(20), default="moderate")
    avoid_repetitive_words = Column(String(20), default="moderate")
    comma_usage_style = Column(String(20), default="moderate")
    content_over_format = Column(String(20), default="moderate")
    bullet_point_usage = Column(String(20), default="moderate")
    emoticon_usage = Column(String(20), default="strict")

    # Custom negative prompts
    custom_negative_prompts = Column(JSON, default=[])

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship settings
    # user = relationship("User", back_populates="negative_preferences")

class VectorDocumentMetadata(Base):
    """Vector database document metadata"""
    __tablename__ = "vector_document_metadata"

    id = Column(Integer, primary_key=True, index=True)

    # Document information
    document_hash = Column(String(64), unique=True, nullable=False, index=True)  # SHA-256 해시
    file_name = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    content_type = Column(String(50), default="text/plain")

    # Embedding information
    embedding_model = Column(String(100), nullable=False)
    chunk_count = Column(Integer, nullable=False)
    chunk_size = Column(Integer, nullable=False)
    chunk_overlap = Column(Integer, nullable=False)

    # FAISS index information
    faiss_index_path = Column(Text, nullable=False)
    vector_dimension = Column(Integer, nullable=False)

    # Status information
    status = Column(String(20), default="active")  # active, deleted, error
    last_accessed = Column(DateTime, default=datetime.utcnow)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class RAGQueryHistory(Base):
    """RAG query response records"""
    __tablename__ = "rag_query_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)

    # Query information
    query_text = Column(Text, nullable=False)
    query_hash = Column(String(64), nullable=False, index=True)  # 중복 질의 추적용
    context_type = Column(String(50), default="general")

    # Search results
    retrieved_documents = Column(JSON, default=[])  # 검색된 문서 청크 정보
    similarity_scores = Column(JSON, default=[])  # 유사도 점수들
    total_search_time_ms = Column(Integer, default=0)

    # Response information
    generated_answer = Column(Text)
    answer_quality_score = Column(Float)  # 0-1 사이 품질 점수
    model_used = Column(String(50), default="gpt-4")
    total_generation_time_ms = Column(Integer, default=0)

    # User feedback
    user_rating = Column(Integer)  # 1-5 점수
    user_feedback = Column(Text)
    was_helpful = Column(Boolean, default=None)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

# Database engine and session configuration
def create_database_engine():
    """Create database engine"""
    database_url = os.getenv("DATABASE_URL", "sqlite:///./chat_toner.db")
    
    if database_url.startswith("sqlite"):
        engine = create_engine(
            database_url, connect_args={"check_same_thread": False}
        )
    else:
        engine = create_engine(database_url)
    
    return engine

# Global engine and session creation
engine = create_database_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()