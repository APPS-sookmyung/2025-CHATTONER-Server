"""
RAG-related Common Configuration Management
Ensures consistency of environment settings and eliminates duplication
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from functools import lru_cache

logger = logging.getLogger(__name__)

class RAGConfig:
    """RAG service-specific configuration manager"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._load_environment()
            self._setup_paths()
            self._validate_config()
            self.__class__._initialized = True

    def _load_environment(self):
        """Load environment variables (based on project root)"""
        try:
            # Find .env file from project root
            current_file = Path(__file__).resolve()

            # Starting from python_backend/core/rag_config.py
            project_root = current_file.parents[2]  # 2025-CHATTONER-Server

            # Check multiple possible .env locations
            env_locations = [
                project_root / "python_backend" / ".env",
                project_root / ".env",
                Path("python_backend/.env"),
                Path(".env")
            ]

            env_loaded = False
            for env_path in env_locations:
                if env_path.exists():
                    load_dotenv(dotenv_path=env_path)
                    logger.info(f"Environment configuration loaded: {env_path}")
                    env_loaded = True
                    break

            if not env_loaded:
                logger.warning("Environment configuration file not found")

        except Exception as e:
            logger.error(f"Failed to load environment configuration: {e}")

    def _setup_paths(self):
        """Path configuration"""
        try:
            # Project root path
            self.project_root = Path(__file__).resolve().parents[2]

            # RAG-related paths
            self.faiss_index_path = self.project_root / "python_backend/langchain_pipeline/data/faiss_index"
            self.documents_path = self.project_root / "python_backend/langchain_pipeline/data/documents"

            # Create directories
            self.faiss_index_path.mkdir(parents=True, exist_ok=True)
            self.documents_path.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            logger.error(f"Path configuration failed: {e}")
            raise

    def _validate_config(self):
        """Configuration validation"""
        try:
            # OpenAI API key validation
            api_key = self.get_openai_api_key()
            if not api_key:
                logger.warning("OpenAI API key is not configured")
            elif len(api_key) < 20:
                logger.warning("OpenAI API key is too short")
            else:
                logger.info("OpenAI API key validation completed")

            # Path access permission validation
            if not self.faiss_index_path.exists():
                logger.warning(f"FAISS index path does not exist: {self.faiss_index_path}")

            if not self.documents_path.exists():
                logger.warning(f"Document path does not exist: {self.documents_path}")

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")

    @lru_cache(maxsize=1)
    def get_openai_api_key(self) -> Optional[str]:
        """Return OpenAI API key (cached)"""
        # Check API key from multiple sources
        api_key = (
            os.getenv("OPENAI_API_KEY") or
            os.getenv("OPENAI_API_KEY_RAG") or
            None
        )

        if api_key and api_key.startswith("sk-"):
            return api_key

        logger.warning("Valid OpenAI API key not found")
        return None

    def get_database_url(self) -> str:
        """Return database URL"""
        return os.getenv("DATABASE_URL", "postgresql://postgres:geenie@localhost:5432/chat_toner_db")

    def get_embedding_model(self) -> str:
        """Return embedding model name"""
        return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    def get_max_file_size_mb(self) -> int:
        """Return maximum file size (MB)"""
        try:
            return int(os.getenv("MAX_FAISS_FILE_SIZE_MB", "500"))
        except ValueError:
            return 500

    def is_debug_mode(self) -> bool:
        """Debug mode status"""
        return os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    def get_chunk_size(self) -> int:
        """Document chunk size"""
        try:
            return int(os.getenv("RAG_CHUNK_SIZE", "1000"))
        except ValueError:
            return 1000

    def get_chunk_overlap(self) -> int:
        """Document chunk overlap"""
        try:
            return int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
        except ValueError:
            return 200

# Singleton instance
@lru_cache(maxsize=1)
def get_rag_config() -> RAGConfig:
    """Return RAG configuration singleton instance"""
    return RAGConfig()

def validate_openai_connection() -> bool:
    """OpenAI connection test"""
    try:
        config = get_rag_config()
        api_key = config.get_openai_api_key()

        if not api_key:
            return False

        # Simple test with OpenAI client
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Connection test by requesting model list
        models = client.models.list()
        if models:
            logger.info("OpenAI connection test successful")
            return True

    except Exception as e:
        logger.error(f"OpenAI connection test failed: {e}")

    return False