"""
Dependency Injection Container
GitHub-based dependency injection container
"""

from dependency_injector import containers, providers
from .config import Settings, get_settings
from services.conversion_service import ConversionService
from services.prompt_engineering import PromptEngineer
from services.openai_services import OpenAIService
from services.user_preferences import UserPreferencesService
from services.finetune_service import FinetuneService
from database.storage import DatabaseStorage

class Container(containers.DeclarativeContainer):
    """Dependency injection container"""
    
    # Configuration
    config = providers.Configuration()
    
    # Configuration provider
    settings = providers.Singleton(get_settings)
    
    # Core services
    prompt_engineer = providers.Singleton(PromptEngineer)
    
    openai_service = providers.Singleton(
        OpenAIService,
        api_key=config.OPENAI_API_KEY,
        model=config.OPENAI_MODEL
    )
    
    user_preferences_service = providers.Singleton(
        UserPreferencesService,
        storage=DatabaseStorage,
        openai_service=openai_service
    )
    
    # Main conversion service
    conversion_service = providers.Singleton(
        ConversionService,
        prompt_engineer=prompt_engineer,
        openai_service=openai_service
    )

    # Fine-tuning service
    finetune_service = providers.Singleton(
        FinetuneService,
        prompt_engineer=prompt_engineer,
        openai_service=openai_service,
        user_preferences_service=user_preferences_service
    )