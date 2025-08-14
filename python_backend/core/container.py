"""
Dependency Injection Container
GitHub 기반 의존성 주입 컨테이너
"""

from dependency_injector import containers, providers
from .config import Settings, get_settings
from services.conversion_service import ConversionService
from services.prompt_engineering import PromptEngineer
from services.openai_services import OpenAIService
from services.user_preferences import UserPreferencesService
from services.finetune_service import FinetuneService

class Container(containers.DeclarativeContainer):
    """의존성 주입 컨테이너"""
    
    # 설정
    config = providers.Configuration()
    
    # 설정 프로바이더
    settings = providers.Singleton(get_settings)
    
    # 코어 서비스들
    prompt_engineer = providers.Singleton(PromptEngineer)
    
    openai_service = providers.Singleton(
        OpenAIService,
        api_key=config.OPENAI_API_KEY,
        model=config.OPENAI_MODEL
    )
    
    user_preferences_service = providers.Singleton(UserPreferencesService)
    
    # 메인 변환 서비스
    conversion_service = providers.Singleton(
        ConversionService,
        prompt_engineer=prompt_engineer,
        openai_service=openai_service,
        user_preferences_service=user_preferences_service
    )

    # 파인튜닝 서비스
    finetune_service = providers.Singleton(
      FinetuneService,
      prompt_engineer=prompt_engineer,
      openai_service=openai_service
    )