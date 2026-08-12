from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.core.dependencies import get_claude_client, get_embedding_model, get_qdrant_repository
from app.main import app
from app.schemas.chat import StructuredAnswer
from app.schemas.ingestion import Chunk


def test_health_check():
    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_answer_and_sources():
    fake_chunk = Chunk(
        section_number="3",
        section_title="FEES",
        subsection_letter="f",
        text="Fee increases require 14 days notice.",
        page_number=1,
    )

    mock_embedding_model = MagicMock()
    mock_embedding_model.embed_query.return_value = [0.0] * 768

    mock_qdrant_repository = MagicMock()
    mock_qdrant_repository.search.return_value = [fake_chunk]

    mock_claude_client = MagicMock()
    mock_claude_client.generate_structured_answer.return_value = StructuredAnswer(
        answer="Nifty Bridge gives 14 days notice.",
        used_sources=[1],
    )

    app.dependency_overrides[get_embedding_model] = lambda: mock_embedding_model
    app.dependency_overrides[get_qdrant_repository] = lambda: mock_qdrant_repository
    app.dependency_overrides[get_claude_client] = lambda: mock_claude_client

    try:
        client = TestClient(app)
        response = client.post("/api/chat", json={"question": "How much notice for fee increases?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Nifty Bridge gives 14 days notice."
    assert body["sources"] == [{"section": "3.f", "title": "FEES", "page": 1}]