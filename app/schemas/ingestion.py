from pydantic import BaseModel

class PageContent(BaseModel):
    """Raw text extracted from a single PDF page."""

    page_number: int
    text: str


class Chunk(BaseModel):
    """A single semantically coherent piece of the document, ready for embedding."""

    section_number: str
    section_title: str
    subsection_letter: str | None = None
    text: str
    page_number: int