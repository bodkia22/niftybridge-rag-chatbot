import logging

from app.infrastructure.embeddings import EmbeddingModel
from app.infrastructure.qdrant import QdrantRepository
from app.schemas.ingestion import Chunk

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(
    question: str,
    embedding_model: EmbeddingModel,
    qdrant_repository: QdrantRepository,
    top_k: int,
) -> list[Chunk]:
    """Finds the chunks most relevant to the user's question.

    Embeds the question using the query prefix, then searches Qdrant
    for the closest matching document chunks.
    """
    logger.info("Embedding query: %s", question)
    query_vector = embedding_model.embed_query(question)

    logger.info("Searching Qdrant (top_k=%d)...", top_k)
    results = qdrant_repository.search(query_vector, top_k)
    logger.info("Found %d relevant chunks.", len(results))

    return results