from pathlib import Path

from app.infrastructure.embeddings import EmbeddingModel
from app.infrastructure.qdrant import QdrantRepository
from app.services.chunker import split_into_sections
from app.services.pdf_parser import extract_text_by_page


def ingest_document(
    pdf_path: Path,
    embedding_model: EmbeddingModel,
    qdrant_repository: QdrantRepository,
) -> int:
    """Runs the full ingestion pipeline for a single PDF document.

    Parses the PDF, splits it into chunks, embeds each chunk, and
    upserts everything into the Qdrant collection. Safe to re-run —
    chunk IDs are deterministic, so existing points get updated rather
    than duplicated.

    Returns the number of chunks written.
    """
    pages = extract_text_by_page(pdf_path)
    chunks = split_into_sections(pages)

    texts = [chunk.text for chunk in chunks]
    vectors = embedding_model.embed_passages(texts)

    qdrant_repository.ensure_collection_exists()
    qdrant_repository.upsert_chunks(chunks, vectors)

    return len(chunks)


if __name__ == "__main__":
    embedding_model = EmbeddingModel()
    qdrant_repository = QdrantRepository()

    chunk_count = ingest_document(
        pdf_path=Path("data/NiftyBridge.pdf"),
        embedding_model=embedding_model,
        qdrant_repository=qdrant_repository,
    )

    print(f"Ingested {chunk_count} chunks into Qdrant.")