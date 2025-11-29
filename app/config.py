import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration settings."""

    STORED_SECRET: str
    OPENAI_API_KEY: str
    QUIZ_TIMEOUT_SECONDS: int = 180
    MAX_RETRIES: int = 3
    BROWSER_HEADLESS: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
