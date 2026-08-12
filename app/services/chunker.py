import re

from app.schemas.ingestion import Chunk, PageContent

SECTION_HEADER_PATTERN = re.compile(r"^(\d{1,2})\.\s*([A-Z][A-Z\s\-]+)$", re.MULTILINE)
SUBSECTION_PATTERN = re.compile(r"^([a-k])\.\s+(.+)$", re.MULTILINE)

# Sections longer than this are further split to stay within the
# embedding model's 512-token limit (see token measurement).
MAX_CHUNK_LENGTH_CHARS = 1400

# Sub-threshold used when grouping sentences for sections without any
# structural markers (no lettered subsections, no paragraph breaks).
SENTENCE_GROUP_TARGET_CHARS = 1000


def split_into_sections(pages: list[PageContent]) -> list[Chunk]:
    """Splits document text into chunks based on top-level section headers.

    Sections that exceed MAX_CHUNK_LENGTH_CHARS are further split by
    lettered subsections (a., b., c., ...), paragraph breaks, or sentence
    grouping, to avoid exceeding the embedding model's token limit.
    """
    full_text, offset_to_page = _combine_pages(pages)

    section_matches = list(SECTION_HEADER_PATTERN.finditer(full_text))
    chunks = []

    for i, match in enumerate(section_matches):
        section_number = match.group(1)
        section_title = match.group(2).strip()

        section_start = match.start()
        section_end = (
            section_matches[i + 1].start()
            if i + 1 < len(section_matches)
            else len(full_text)
        )
        section_text = full_text[section_start:section_end].strip()
        page_number = _page_for_offset(section_start, offset_to_page)

        if len(section_text) <= MAX_CHUNK_LENGTH_CHARS:
            chunks.append(
                Chunk(
                    section_number=section_number,
                    section_title=section_title,
                    subsection_letter=None,
                    text=section_text,
                    page_number=page_number,
                )
            )
        else:
            chunks.extend(
                _split_oversized_section(
                    section_number, section_title, section_text, page_number
                )
            )

    return chunks


def _split_oversized_section(
    section_number: str, section_title: str, section_text: str, page_number: int
) -> list[Chunk]:
    """Splits an oversized section into smaller chunks.

    Tries, in order: lettered subsections (a., b., c., ...), paragraph
    breaks, and finally sentence grouping — since some sections (e.g.
    LIMITATION OF LIABILITY) are a single unbroken block of prose with
    no structural markers at all.
    """
    subsection_matches = list(SUBSECTION_PATTERN.finditer(section_text))
    if subsection_matches:
        return _split_by_letters(
            section_number, section_title, section_text, page_number, subsection_matches
        )

    paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return _split_by_paragraphs(section_number, section_title, paragraphs, page_number)

    return _split_by_sentence_groups(section_number, section_title, section_text, page_number)


def _split_by_letters(
    section_number: str,
    section_title: str,
    section_text: str,
    page_number: int,
    subsection_matches: list[re.Match],
) -> list[Chunk]:
    """Splits section text into one chunk per lettered subsection.

    Each chunk is prefixed with the parent section header so embeddings
    retain context even when only one subsection is retrieved. Any
    subsection that is itself still oversized (e.g. a single unbroken
    block of prose with no paragraph breaks) is split further.
    """
    header = f"{section_number}. {section_title}"
    chunks = []

    for i, match in enumerate(subsection_matches):
        subsection_letter = match.group(1)

        sub_start = match.start()
        sub_end = (
            subsection_matches[i + 1].start()
            if i + 1 < len(subsection_matches)
            else len(section_text)
        )
        subsection_body = section_text[sub_start:sub_end].strip()

        chunk = Chunk(
            section_number=section_number,
            section_title=section_title,
            subsection_letter=subsection_letter,
            text=f"{header}\n{subsection_body}",
            page_number=page_number,
        )
        chunks.extend(_split_if_oversized(chunk, header))

    return chunks


def _split_by_paragraphs(
    section_number: str, section_title: str, paragraphs: list[str], page_number: int
) -> list[Chunk]:
    """Splits section text into one chunk per paragraph.

    Used when a section has no lettered subsections but does have
    double-newline paragraph breaks to split on. Any paragraph that is
    itself still oversized is split further.
    """
    header = f"{section_number}. {section_title}"
    chunks = []

    for paragraph in paragraphs:
        chunk = Chunk(
            section_number=section_number,
            section_title=section_title,
            subsection_letter=None,
            text=f"{header}\n{paragraph}",
            page_number=page_number,
        )
        chunks.extend(_split_if_oversized(chunk, header))

    return chunks


def _split_if_oversized(chunk: Chunk, header: str) -> list[Chunk]:
    """Splits a chunk further via sentence grouping if it's still oversized.

    Letter- and paragraph-based splitting can still leave a single
    subsection or paragraph that's a large unbroken block of prose (e.g.
    a list of restrictions with no internal breaks). Sentence grouping
    is used directly here rather than the full three-level cascade,
    since a chunk that's already past the letter/paragraph stage has, by
    definition, no further structural markers left to split on.

    Note: chunks produced this way lose their subsection_letter, since
    the original subsection had to be broken up further — the parent
    section number is preserved, but fine-grained navigation for this
    specific edge case is slightly less precise as a result.
    """
    if len(chunk.text) <= MAX_CHUNK_LENGTH_CHARS:
        return [chunk]

    # Strip the header back off before re-grouping, since
    # _split_by_sentence_groups adds its own header prefix.
    body = chunk.text.removeprefix(f"{header}\n")

    return _split_by_sentence_groups(
        chunk.section_number, chunk.section_title, body, chunk.page_number
    )


def _split_by_sentence_groups(
    section_number: str, section_title: str, section_text: str, page_number: int
) -> list[Chunk]:
    """Splits section text into chunks of grouped sentences.

    Used as a last-resort fallback when a section has neither lettered
    subsections nor paragraph breaks to split on. Consecutive sentences
    are grouped until the target size is reached, keeping related
    clauses together rather than cutting at an arbitrary character count.

    A single "sentence" that is itself still oversized (e.g. a long
    semicolon-separated list with only one terminal period, as seen in
    real ToS restriction clauses) is further split on semicolons.
    """
    header = f"{section_number}. {section_title}"

    # Normalize the single-newline line breaks into spaces first, so
    # sentence splitting is not thrown off by mid-sentence line wraps.
    normalized_text = re.sub(r"\s+", " ", section_text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", normalized_text)

    # A sentence with no internal punctuation to split on (e.g. a long
    # semicolon-separated clause list) would otherwise stay as one
    # oversized unit — break it into smaller pieces on semicolons first.
    sentences = [
        piece
        for sentence in sentences
        for piece in _split_on_semicolons_if_oversized(sentence)
    ]

    groups: list[str] = []
    current_group: list[str] = []
    current_length = 0

    for sentence in sentences:
        current_group.append(sentence)
        current_length += len(sentence)

        if current_length >= SENTENCE_GROUP_TARGET_CHARS:
            groups.append(" ".join(current_group))
            current_group = []
            current_length = 0

    if current_group:
        groups.append(" ".join(current_group))

    return [
        Chunk(
            section_number=section_number,
            section_title=section_title,
            subsection_letter=None,
            text=f"{header}\n{group}",
            page_number=page_number,
        )
        for group in groups
    ]


def _split_on_semicolons_if_oversized(sentence: str) -> list[str]:
    """Splits an oversized "sentence" on semicolons as a last resort.

    Legal clause lists (e.g. restrictions "(i) ...; (ii) ...; (iii) ...")
    are often written as a single grammatical sentence with only one
    terminal period, so period-based sentence splitting leaves them
    intact as one oversized unit. Semicolons are the next-most-reliable
    clause boundary in this kind of text.
    """
    if len(sentence) <= MAX_CHUNK_LENGTH_CHARS:
        return [sentence]

    return [clause.strip() for clause in sentence.split(";") if clause.strip()]


def _combine_pages(pages: list[PageContent]) -> tuple[str, list[tuple[int, int]]]:
    """Joins all pages into one text stream and records where each page starts."""
    full_text_parts = []
    offset_to_page = []
    current_offset = 0

    for page in pages:
        offset_to_page.append((current_offset, page.page_number))
        full_text_parts.append(page.text)
        current_offset += len(page.text) + 1

    full_text = "\n".join(full_text_parts)
    return full_text, offset_to_page


def _page_for_offset(offset: int, offset_to_page: list[tuple[int, int]]) -> int:
    """Finds which page a character offset in the combined text belongs to."""
    page_number = offset_to_page[0][1]

    for start_offset, page in offset_to_page:
        if start_offset > offset:
            break
        page_number = page

    return page_number