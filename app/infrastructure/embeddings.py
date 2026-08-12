from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

# e5 models are trained with these exact prefixes baked into the input —
# omitting them noticeably degrades retrieval quality, especially for
# cross-lingual queries (Ukrainian question -> English document).
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


class EmbeddingModel:
    """Thin wrapper around a sentence-transformers model for text embeddings."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model = SentenceTransformer(settings.embedding_model_name)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of document chunks for storage in the vector database."""
        prefixed_texts = [PASSAGE_PREFIX + text for text in texts]
        embeddings = self._model.encode(prefixed_texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single user query for similarity search."""
        prefixed_text = QUERY_PREFIX + text
        embedding = self._model.encode(prefixed_text, normalize_embeddings=True)
        return embedding.tolist()