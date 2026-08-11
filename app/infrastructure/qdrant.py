import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.core.config import get_settings
from app.schemas.ingestion import Chunk

# The embedding model outputs 768-dimensional vectors (multilingual-e5-base).
VECTOR_SIZE = 768


class QdrantRepository:
    """Thin wrapper around qdrant-client for storing and searching document chunks."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self._collection_name = settings.qdrant_collection_name

    def ensure_collection_exists(self) -> None:
        """Creates the collection if it doesn't already exist.

        Safe to call on every startup — does nothing if the collection
        is already there.
        """
        if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Writes chunks and their vectors to the collection.

        Each chunk gets a deterministic ID derived from its section and
        subsection, so re-running ingestion updates existing points
        instead of creating duplicates.
        """
        points = [
            PointStruct(
                id=_chunk_id(chunk),
                vector=vector,
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(self, query_vector: list[float], top_k: int) -> list[Chunk]:
        """Finds the top_k chunks most similar to the given query vector."""
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
        ).points
        return [Chunk(**point.payload) for point in results]


def _chunk_id(chunk: Chunk) -> str:
    """Derives a deterministic UUID from a chunk's identity.

    Uses section/subsection when available. Falls back to including a
    slice of the chunk text, since chunks produced by paragraph or
    sentence-group splitting share the same section/subsection (None)
    and would otherwise collide, causing later chunks to silently
    overwrite earlier ones during upsert.
    """
    letter = chunk.subsection_letter or "full"
    key = f"{chunk.section_number}_{letter}_{chunk.text[:50]}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


if __name__ == "__main__":
    from app.infrastructure.embeddings import EmbeddingModel

    embedding_model = EmbeddingModel()
    repo = QdrantRepository()

    query = "What is the maximum liability of Nifty Bridge?"
    query_vector = embedding_model.embed_query(query)

    results = repo.search(query_vector, top_k=14)

    print(f"Query: {query}\n")
    for chunk in results:
        print(f"--- Section {chunk.section_number}: {chunk.section_title} ---")
        print(chunk.text[:200])
        print()