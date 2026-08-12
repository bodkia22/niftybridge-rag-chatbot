from app.schemas.ingestion import PageContent
from app.services.chunker import split_into_sections


def test_splits_document_into_numbered_sections():
    """Basic sections without oversized content should map 1:1 to chunks."""
    pages = [
        PageContent(
            page_number=1,
            text=(
                "1. GENERAL\n"
                "Some short introductory text about the agreement.\n"
                "2. FEES\n"
                "Some short text about fees."
            ),
        )
    ]

    chunks = split_into_sections(pages)

    assert len(chunks) == 2
    assert chunks[0].section_number == "1"
    assert chunks[0].section_title == "GENERAL"
    assert chunks[1].section_number == "2"
    assert chunks[1].section_title == "FEES"


def test_oversized_section_splits_by_lettered_subsections():
    """A section longer than the max chunk length should split on a., b., ...
    with each subsection remaining a single chunk, as long as no individual
    subsection is itself oversized."""
    para_a = "Filler text. " * 60       # 780 chars
    para_b = "More filler text. " * 60  # 1140 chars

    pages = [
        PageContent(
            page_number=1,
            text=(
                "3. FEES\n"
                f"a. Service Fees\n{para_a}\n"
                f"b. Minting Fees\n{para_b}"
            ),
        )
    ]

    chunks = split_into_sections(pages)

    assert len(chunks) == 2
    assert chunks[0].subsection_letter == "a"
    assert chunks[1].subsection_letter == "b"
    assert "3. FEES" in chunks[0].text


def test_oversized_subsection_is_split_further():
    """A lettered subsection that is itself still oversized (e.g. a long
    unbroken block of prose) should be recursively split further, rather
    than surviving as a single chunk past MAX_CHUNK_LENGTH_CHARS."""
    long_para_a = "Filler text. " * 200  # comfortably over MAX_CHUNK_LENGTH_CHARS
    short_para_b = "Short minting fees text."

    pages = [
        PageContent(
            page_number=1,
            text=(
                "3. FEES\n"
                f"a. Service Fees\n{long_para_a}\n"
                f"b. Minting Fees\n{short_para_b}"
            ),
        )
    ]

    chunks = split_into_sections(pages)

    assert all(len(chunk.text) <= 1400 for chunk in chunks)
    assert len(chunks) > 2