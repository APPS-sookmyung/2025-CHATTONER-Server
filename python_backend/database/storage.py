"""
데이터베이스 스토리지 클래스
사용자 프로필, 변환 기록, 선호도 관리
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from database.models import (
    SessionLocal, UserProfile, ConversionHistory, 
    NegativePreferences, Base, engine
)


class DatabaseStorage:
    """데이터베이스 스토리지 관리 클래스"""
    
    def __init__(self, database_url: str = None):
        # 테이블 생성
        Base.metadata.create_all(bind=engine)
    
    def get_db(self) -> Session:
        """데이터베이스 세션 생성"""
        db = SessionLocal()
        try:
            return db
        finally:
            pass  # 호출자가 close 해야 함
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """사용자 프로필 조회"""
        db = self.get_db()
        try:
            profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
            if profile:
                return {
                    "userId": profile.user_id,
                    "baseFormalityLevel": profile.base_formality_level,
                    "baseFriendlinessLevel": profile.base_friendliness_level,
                    "baseEmotionLevel": profile.base_emotion_level,
                    "baseDirectnessLevel": profile.base_directness_level,
                    "sessionFormalityLevel": profile.session_formality_level,
                    "sessionFriendlinessLevel": profile.session_friendliness_level,
                    "sessionEmotionLevel": profile.session_emotion_level,
                    "sessionDirectnessLevel": profile.session_directness_level,
                    "negativePromptPreferences": profile.negative_prompt_preferences
                }
            return None
        finally:
            db.close()
    
    async def create_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """사용자 프로필 생성"""
        db = self.get_db()
        try:
            profile = UserProfile(
                user_id=user_id,
                base_formality_level=profile_data.get("baseFormalityLevel", 3),
                base_friendliness_level=profile_data.get("baseFriendlinessLevel", 3),
                base_emotion_level=profile_data.get("baseEmotionLevel", 3),
                base_directness_level=profile_data.get("baseDirectnessLevel", 3)
            )
            db.add(profile)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            return False
        finally:
            db.close()
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """사용자 프로필 업데이트"""
        db = self.get_db()
        try:
            profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
            if profile:
                for key, value in updates.items():
                    if hasattr(profile, key):
                        setattr(profile, key, value)
                profile.updated_at = datetime.utcnow()
                db.commit()
                return True
            return False
        except Exception as e:
            db.rollback()
            return False
        finally:
            db.close()
    
    async def update_session_style(self, user_id: str, style_levels: Dict[str, float]) -> bool:
        """세션 스타일 레벨 업데이트"""
        db = self.get_db()
        try:
            profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
            if profile:
                for key, value in style_levels.items():
                    if hasattr(profile, key):
                        setattr(profile, key, value)
                profile.updated_at = datetime.utcnow()
                db.commit()
                return True
            return False
        except Exception as e:
            db.rollback()
            return False
        finally:
            db.close()
    
    async def get_negative_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """네거티브 선호도 조회"""
        db = self.get_db()
        try:
            prefs = db.query(NegativePreferences).filter(NegativePreferences.user_id == user_id).first()
            if prefs:
                return {
                    "user_id": prefs.user_id,
                    "avoid_flowery_language": prefs.avoid_flowery_language,
                    "avoid_repetitive_words": prefs.avoid_repetitive_words,
                    "comma_usage_style": prefs.comma_usage_style,
                    "content_over_format": prefs.content_over_format,
                    "bullet_point_usage": prefs.bullet_point_usage,
                    "emoticon_usage": prefs.emoticon_usage,
                    "custom_negative_prompts": prefs.custom_negative_prompts or []
                }
            return None
        finally:
            db.close()
    
    async def save_negative_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """네거티브 선호도 저장"""
        db = self.get_db()
        try:
            # 기존 선호도 조회
            existing = db.query(NegativePreferences).filter(NegativePreferences.user_id == user_id).first()
            
            if existing:
                # 업데이트
                for key, value in preferences.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
            else:
                # 새로 생성
                new_prefs = NegativePreferences(user_id=user_id, **preferences)
                db.add(new_prefs)
            
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            return False
        finally:
            db.close()
    
    async def save_conversion_history(self, conversion_data: Dict[str, Any]) -> bool:
        """변환 기록 저장"""
        db = self.get_db()
        try:
            history = ConversionHistory(
                user_id=conversion_data["user_id"],
                original_text=conversion_data["original_text"],
                converted_texts=conversion_data["converted_texts"],
                context=conversion_data["context"],
                sentiment_analysis=conversion_data.get("sentiment_analysis")
            )
            db.add(history)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            return False
        finally:
            db.close()
