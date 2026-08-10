from pydantic import BaseModel

class PageContent(BaseModel):
    """Raw text extracted from a single PDF page."""
    
    page_number: int
    text: str