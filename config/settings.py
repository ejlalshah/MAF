"""
FRIDAY-MAF Configuration
All settings loaded from environment variables with safe defaults.
Create a .env file or set env vars before running.
"""
from __future__ import annotations
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM Provider — change this one value to switch providers
    LLM_PROVIDER:  str = "anthropic"          # openai | anthropic | gemini
    LLM_MODEL:     str = "claude-3-5-haiku-20241022"
    LLM_API_KEY:   str = ""                   # REQUIRED — set in .env
    EMBED_MODEL:   str = "text-embedding-3-small"

    # Memory
    REDIS_URL:     str = "redis://localhost:6379/0"

    # Tools
    SEARCH_API_KEY: Optional[str] = None
    GITHUB_TOKEN:   Optional[str] = None

    # API
    API_HOST:      str = "0.0.0.0"
    API_PORT:      int = 8000
    LOG_LEVEL:     str = "DEBUG"

    # Limits
    MAX_RETRIES:   int = 3
    TASK_TIMEOUT:  int = 300   # seconds


settings = Settings()