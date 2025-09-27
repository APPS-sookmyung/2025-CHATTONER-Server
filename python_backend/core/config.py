# python_backend/core/config.py
# BaseSettings + .env + @lru_cache 
from pydantic_settings import BaseSettings
import os
from functools import lru_cache
from pydantic import Field
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    PROJECT_NAME: str = "Chat Toner API"
    DESCRIPTION: str = "AI-based Korean text style personalization API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server configuration
    #HOST: str = "0.0.0.0"
    HOST: str = "127.0.0.1"
    PORT: int = 5001
    
    # Fine-tuning inference server configuration
    FINETUNE_INFERENCE_HOST: str = Field(default="localhost", validation_alias="RUNPOD_IP")
    FINETUNE_INFERENCE_PORT: int = 8010
    FINETUNE_URL_OVERRIDE: Optional[str] = Field(default=None)

    @property
    def FINETUNE_URL(self) -> str:
        if self.FINETUNE_URL_OVERRIDE:
            return self.FINETUNE_URL_OVERRIDE
        return f"http://{self.FINETUNE_INFERENCE_HOST}:{self.FINETUNE_INFERENCE_PORT}"
    
    # OpenAI configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4o"
    
    # Database configuration
    DATABASE_URL: str = "sqlite:///./chat_toner.db"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "chattoner_db"
    DB_USER: str = "username"
    DB_PASSWORD: str = "password"
    
    # CORS configuration
    CORS_ORIGINS: list = ["*"]
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]
    
    class Config:
        env_file = ".env"  # Reference to .env file in current directory
        case_sensitive = True
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()