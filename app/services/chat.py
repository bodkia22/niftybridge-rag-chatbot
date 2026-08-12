import logging

from app.infrastructure.anthropic import ClaudeClient
from app.infrastructure.embeddings import EmbeddingModel
from app.infrastructure.qdrant import QdrantRepository
from app.schemas.ingestion import Chunk
from app.services.retrieval import retrieve_relevant_chunks

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant answering questions about Nifty Bridge's Terms of Service.
Use ONLY the following document excerpts to answer the question.
If the answer is not contained in the excerpts, say that you don't have enough information — do not make up an answer.
Always answer in the same language as the question.

Answer naturally and directly, as if you simply know the Terms of Service well.
Do not mention "excerpts", "provided context", "the documents given to you", or similar phrases —
just answer the question as if it were common knowledge about the policy.

--- Document excerpts ---
{context}
"""


def answer_question(
    question: str,
    embedding_model: EmbeddingModel,
    qdrant_repository: QdrantRepository,
    claude_client: ClaudeClient,
    top_k: int,
) -> tuple[str, list[Chunk]]:
    """Answers a user's question using retrieval-augmented generation.

    Retrieves the most relevant document chunks, builds a grounded
    system prompt from them, and asks Claude to answer using only that
    context. The raw user question is passed as the sole user message,
    keeping untrusted input separate from trusted instructions.

    Returns the answer text along with the chunks used as sources.
    """
    chunks = retrieve_relevant_chunks(
        question=question,
        embedding_model=embedding_model,
        qdrant_repository=qdrant_repository,
        top_k=top_k,
    )

    system_prompt = _build_system_prompt(chunks)

    logger.info("Sending prompt to Claude...")
    answer_text = claude_client.generate_response(
        system_prompt=system_prompt,
        user_prompt=question,
    )
    logger.info("Received answer from Claude.")

    return answer_text, chunks


def _build_system_prompt(chunks: list[Chunk]) -> str:
    """Formats retrieved chunks into the system prompt template."""
    context = "\n\n".join(
        f"[Section {chunk.section_number}.{chunk.subsection_letter or ''}: {chunk.section_title}]\n{chunk.text}"
        for chunk in chunks
    )
    return SYSTEM_PROMPT_TEMPLATE.format(context=context)