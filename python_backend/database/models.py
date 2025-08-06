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

# 데이터베이스 설정
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_toner.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserProfile(Base):
    """사용자 프로필 모델"""
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    
    # 기본 스타일 레벨 (1-5)
    base_formality_level = Column(Integer, default=3)
    base_friendliness_level = Column(Integer, default=3)
    base_emotion_level = Column(Integer, default=3)
    base_directness_level = Column(Integer, default=3)
    
    # 세션 스타일 레벨 (동적 조정)
    session_formality_level = Column(Integer, default=3)
    session_friendliness_level = Column(Integer, default=3)
    session_emotion_level = Column(Integer, default=3)
    session_directness_level = Column(Integer, default=3)
    
    # 네거티브 프롬프트 선호도
    negative_prompt_preferences = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ConversionHistory(Base):
    """변환 기록 모델"""
    __tablename__ = "conversion_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    
    original_text = Column(Text, nullable=False)
    converted_texts = Column(JSON, nullable=False)  # {"direct": "...", "gentle": "...", "neutral": "..."}
    context = Column(String, nullable=False)  # "business", "report", "personal"
    
    selected_version = Column(String, nullable=True)  # 사용자가 선택한 버전
    user_rating = Column(Integer, nullable=True)  # 1-5 평점
    user_feedback = Column(Text, nullable=True)
    
    sentiment_analysis = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class NegativePreferences(Base):
    """네거티브 프롬프트 선호도 모델"""
    __tablename__ = "negative_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    
    avoid_flowery_language = Column(String, default="moderate")  # strict, moderate, lenient
    avoid_repetitive_words = Column(String, default="moderate")
    comma_usage_style = Column(String, default="moderate")
    content_over_format = Column(String, default="moderate")
    bullet_point_usage = Column(String, default="moderate")
    emoticon_usage = Column(String, default="strict")
    
    custom_negative_prompts = Column(JSON, nullable=True)  # List[str]
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)