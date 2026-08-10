"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings with defaults suitable for local development."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Anthropic
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-6"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "niftybridge_docs"

    # Embeddings
    embedding_model_name: str = "intfloat/multilingual-e5-base"

    # Retrieval
    retrieval_top_k: int = 4

    # App
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Using lru_cache ensures settings are loaded once per process
    and makes this function usable as a FastAPI dependency.
    """
    return Settings()