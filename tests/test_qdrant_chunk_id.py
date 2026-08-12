from app.infrastructure.qdrant import _chunk_id
from app.schemas.ingestion import Chunk


def test_chunk_id_differs_for_same_section_different_text():
    """Regression test for a real bug found during development: two
    sentence-grouped chunks from the same section (no subsection letter)
    previously shared an ID key derived only from section number, causing
    the second chunk to silently overwrite the first during upsert."""
    chunk_a = Chunk(
        section_number="12",
        section_title="LIMITATION OF LIABILITY",
        subsection_letter=None,
        text="First group of sentences about liability limits.",
        page_number=1,
    )
    chunk_b = Chunk(
        section_number="12",
        section_title="LIMITATION OF LIABILITY",
        subsection_letter=None,
        text="Second group of sentences about liability limits.",
        page_number=1,
    )

    assert _chunk_id(chunk_a) != _chunk_id(chunk_b)


def test_chunk_id_is_deterministic():
    """Same chunk content should always produce the same ID, so
    re-ingestion updates existing points instead of duplicating them."""
    chunk = Chunk(
        section_number="3",
        section_title="FEES",
        subsection_letter="f",
        text="Fee increase notice text.",
        page_number=1,
    )

    assert _chunk_id(chunk) == _chunk_id(chunk)