from functools import lru_cache

from app.infrastructure.anthropic import ClaudeClient
from app.infrastructure.embeddings import EmbeddingModel
from app.infrastructure.qdrant import QdrantRepository


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()


@lru_cache
def get_qdrant_repository() -> QdrantRepository:
    return QdrantRepository()


@lru_cache
def get_claude_client() -> ClaudeClient:
    return ClaudeClient()