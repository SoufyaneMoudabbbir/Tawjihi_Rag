#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application Configuration Management
Validates and manages all environment variables and settings
"""
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import validator, Field
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application settings with validation"""

    # API Configuration
    APP_NAME: str = "Tawjihi RAG API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")

    # Server Configuration
    HOST: str = Field(default="0.0.0.0", env="BACKEND_HOST")
    PORT: int = Field(default=8000, env="BACKEND_PORT")

    # Security
    SECRET_KEY: str = Field(default="development-secret-key-change-in-production", env="SECRET_KEY")
    ALLOWED_HOSTS: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        env="CORS_ORIGINS"
    )

    # DeepSeek API
    DEEPSEEK_API_KEY: str = Field(..., env="DEEPSEEK_API_KEY")  # Required
    DEEPSEEK_API_URL: str = Field(
        default="https://api.deepseek.com/v1/chat/completions",
        env="DEEPSEEK_API_URL"
    )
    DEEPSEEK_MODEL: str = Field(default="deepseek-chat", env="DEEPSEEK_MODEL")
    DEEPSEEK_TIMEOUT: int = Field(default=30, env="DEEPSEEK_TIMEOUT")

    # Database Configuration
    DATABASE_PATH: str = Field(default="./database.sqlite", env="DATABASE_PATH")
    DATABASE_POOL_SIZE: int = Field(default=5, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")

    # For PostgreSQL migration (future)
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")

    # File Upload Configuration
    UPLOAD_DIR: str = Field(default="./uploads", env="UPLOAD_DIR")
    MAX_FILE_SIZE_MB: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    ALLOWED_FILE_TYPES: List[str] = Field(
        default=["application/pdf"],
        env="ALLOWED_FILE_TYPES"
    )

    # RAG Configuration
    EMBEDDING_MODEL: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        env="EMBEDDING_MODEL"
    )
    CHUNK_SIZE: int = Field(default=700, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=100, env="CHUNK_OVERLAP")
    MAX_SEARCH_RESULTS: int = Field(default=10, env="MAX_SEARCH_RESULTS")

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="./logs/app.log", env="LOG_FILE")
    LOG_ROTATION: str = Field(default="10 MB", env="LOG_ROTATION")
    LOG_RETENTION: str = Field(default="30 days", env="LOG_RETENTION")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")  # json or text

    # Caching (Redis - optional)
    REDIS_URL: Optional[str] = Field(default=None, env="REDIS_URL")
    CACHE_ENABLED: bool = Field(default=False, env="CACHE_ENABLED")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # seconds

    # Monitoring
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")

    @validator("DATABASE_PATH")
    def validate_database_path(cls, v):
        """Ensure database directory exists"""
        db_path = Path(v)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return str(db_path.absolute())

    @validator("UPLOAD_DIR")
    def validate_upload_dir(cls, v):
        """Ensure upload directory exists"""
        upload_path = Path(v)
        upload_path.mkdir(parents=True, exist_ok=True)
        return str(upload_path.absolute())

    @validator("LOG_FILE")
    def validate_log_file(cls, v):
        """Ensure log directory exists"""
        log_path = Path(v)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return str(log_path.absolute())

    @validator("MAX_FILE_SIZE_MB")
    def validate_max_file_size(cls, v):
        """Ensure reasonable file size limit"""
        if v < 1 or v > 500:
            raise ValueError("MAX_FILE_SIZE_MB must be between 1 and 500")
        return v

    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes"""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT.lower() == "development"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Dependency for FastAPI"""
    return settings
