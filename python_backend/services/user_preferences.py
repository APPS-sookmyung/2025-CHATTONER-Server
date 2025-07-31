"""
사용자 선호도 관리 서비스
네거티브 프롬프트 선호도 및 스타일 학습 관리
"""

from typing import Dict, List, Any, Optional
from database.storage import DatabaseStorage
from .openai_services import OpenAIService  # and actually use it, or delete