"""
Configuration management for the RAG API
Loads configuration from environment variables with validation
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration"""

    # API Keys
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")

    # Database
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "../frontend/database.sqlite")

    # Server
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

    # CORS
    ALLOWED_ORIGINS: list = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

    # File Upload
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "../frontend/uploads")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # RAG Settings
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "700"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # DeepSeek API
    DEEPSEEK_API_URL: str = os.getenv(
        "DEEPSEEK_API_URL",
        "https://api.deepseek.com/chat/completions"
    )
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration"""
        errors = []

        if not cls.DEEPSEEK_API_KEY:
            errors.append("DEEPSEEK_API_KEY is required")

        if not Path(cls.DATABASE_PATH).parent.exists():
            errors.append(f"Database directory does not exist: {cls.DATABASE_PATH}")

        if errors:
            raise ValueError(
                "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    @classmethod
    def get_database_path(cls) -> str:
        """Get absolute database path"""
        return str(Path(cls.DATABASE_PATH).resolve())

# Validate configuration on import
try:
    Config.validate()
except ValueError as e:
    print(f"⚠️  Configuration Warning: {e}")
    print("Please check your .env file or environment variables")
