import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import chat, health
from app.core.config import get_settings
from app.core.dependencies import get_embedding_model, get_qdrant_repository
from app.core.logging import setup_logging
from app.services.ingestion import ingest_document

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensures the document is indexed before the server starts serving requests.

    Runs once at startup. If the Qdrant collection already has data
    (e.g. from a previous run with a persisted volume), ingestion is
    skipped to avoid redundant work.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    qdrant_repository = get_qdrant_repository()
    qdrant_repository.ensure_collection_exists()

    if qdrant_repository.count_points() == 0:
        logger.info("Collection is empty, running ingestion...")
        embedding_model = get_embedding_model()
        chunk_count = ingest_document(
            pdf_path=Path("data/NiftyBridge.pdf"),
            embedding_model=embedding_model,
            qdrant_repository=qdrant_repository,
        )
        logger.info("Ingestion complete: %d chunks indexed.", chunk_count)
    else:
        logger.info("Collection already has data, skipping ingestion.")

    yield


app = FastAPI(title="NiftyBridge RAG Chatbot", lifespan=lifespan)

app.include_router(health.router, prefix="/api")
app.include_router(chat.router, prefix="/api")

app.mount("/", StaticFiles(directory="static", html=True), name="static")