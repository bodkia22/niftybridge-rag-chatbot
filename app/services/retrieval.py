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


if __name__ == "__main__":
    from app.core.config import get_settings
    from app.core.logging import setup_logging

    settings = get_settings()
    setup_logging(settings.log_level)

    embedding_model = EmbeddingModel()
    qdrant_repository = QdrantRepository()

    question = "What is the maximum dollar amount Nifty Bridge is liable for?"
    chunks = retrieve_relevant_chunks(
        question=question,
        embedding_model=embedding_model,
        qdrant_repository=qdrant_repository,
        top_k=49,
    )

    for i, chunk in enumerate(chunks, start=1):
        label = f"{chunk.section_number}.{chunk.subsection_letter or ''}".rstrip(".")
        print(f"{i}. Section {label}: {chunk.section_title}")