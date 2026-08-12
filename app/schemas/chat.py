from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    section: str
    title: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class StructuredAnswer(BaseModel):
    """Parsed result of a single tool-use call to Claude.

    `used_sources` holds 1-based indices into the numbered excerpt list
    that was sent in the prompt — not final Source objects. Chat service
    maps these indices back onto the original Chunk list.
    """

    answer: str
    used_sources: list[int]