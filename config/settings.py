"""
FRIDAY-MAF Configuration
All settings loaded from environment variables with safe defaults.
Default provider: Ollama (free, local, no API key required).
Create a .env file or set env vars to override.
"""
from __future__ import annotations
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM Provider — change this one value to switch providers
    # Free default: ollama (local). Options: openai | anthropic | gemini | ollama
    LLM_PROVIDER:  str = "ollama"
    LLM_MODEL:     str = "llama3.2"
    LLM_API_KEY:   str = ""                   # not required for Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBED_MODEL:   str = "all-MiniLM-L6-v2"  # sentence-transformers

    # Memory
    REDIS_URL:     str = "redis://localhost:6379/0"

    # Persistence
    DB_PATH:       str = "friday_maf.db"      # SQLite database path

    # Tools
    SEARCH_API_KEY: Optional[str] = None       # unused — DuckDuckGo is free
    GITHUB_TOKEN:   Optional[str] = None

    # API
    API_HOST:      str = "0.0.0.0"
    API_PORT:      int = 8000
    LOG_LEVEL:     str = "INFO"

    # Limits
    MAX_RETRIES:   int = 3
    TASK_TIMEOUT:  int = 300   # seconds

    # Autonomy loop
    LOOP_INTERVAL: float = 3.0  # seconds between execution loop ticks


settings = Settings()
