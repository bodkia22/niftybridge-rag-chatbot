from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.dependencies import get_claude_client, get_embedding_model, get_qdrant_repository
from app.infrastructure.anthropic import ClaudeClient
from app.infrastructure.embeddings import EmbeddingModel
from app.infrastructure.qdrant import QdrantRepository
from app.schemas.chat import ChatRequest, ChatResponse, Source
from app.schemas.ingestion import Chunk
from app.services.chat import answer_question

router = APIRouter()


@router.post("/chat")
def chat(
    request: ChatRequest,
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
    qdrant_repository: QdrantRepository = Depends(get_qdrant_repository),
    claude_client: ClaudeClient = Depends(get_claude_client),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """Answers a question about the Terms of Service using RAG."""
    answer_text, chunks = answer_question(
        question=request.question,
        embedding_model=embedding_model,
        qdrant_repository=qdrant_repository,
        claude_client=claude_client,
        top_k=settings.retrieval_top_k,
    )

    sources = [_chunk_to_source(chunk) for chunk in chunks]
    return ChatResponse(answer=answer_text, sources=sources)


def _chunk_to_source(chunk: Chunk) -> Source:
    """Converts a retrieved chunk into an API source reference."""
    section = f"{chunk.section_number}.{chunk.subsection_letter or ''}".rstrip(".")
    return Source(section=section, title=chunk.section_title, page=chunk.page_number)