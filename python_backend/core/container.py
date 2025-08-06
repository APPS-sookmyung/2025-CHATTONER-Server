"""의존성 주입 컨테이너
- DI 원칙 적용: factory 와 singleton 패턴을 사용하여 서비스 인스턴스를 관리
- gpt, prompt, User 로 서비스 계층 분리 목적 
- 유닛 테스트 및 주입 변경 가능 
"""

from dependency_injector import containers, providers
#from dependency_injector.wiring import Provide

from services.prompt_engineering import PromptEngineer
from services.openai_services import OpenAIService
from services.conversion_service import ConversionService
from services.user_preferences import UserPreferencesService
from database.storage import DatabaseStorage

class Container(containers.DeclarativeContainer):
    """애플리케이션 의존성 컨테이너"""
    
    # 설정
    config = providers.Configuration()
    
    # 데이터베이스
    database = providers.Singleton(
        DatabaseStorage,
        database_url=config.DATABASE_URL
    )
    
    # 서비스들
    prompt_engineer = providers.Factory(
        PromptEngineer
    )
    
    openai_service = providers.Factory(
        OpenAIService,
        api_key=config.OPENAI_API_KEY,
        model=config.OPENAI_MODEL
    )
    
    conversion_service = providers.Factory(
        ConversionService,
        prompt_engineer=prompt_engineer,
        openai_service=openai_service
    )
    
    user_preferences_service = providers.Factory(
        UserPreferencesService,
        storage=database,
        openai_service=openai_service
    )