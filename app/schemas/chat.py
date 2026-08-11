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