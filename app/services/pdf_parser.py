from pathlib import Path

import pdfplumber

from app.schemas.ingestion import PageContent


def extract_text_by_page(pdf_path: Path) -> list[PageContent]:
    """Extracts text from a PDF file, returning a list of PageContent objects."""
    page_contents = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            page_contents.append(PageContent(page_number=page_number, text=text))

    return page_contents


if __name__ == "__main__":
    # Example usage
    pdf_path = Path("data/NiftyBridge.pdf")
    page_contents = extract_text_by_page(pdf_path)

    for page_content in page_contents:
        print(page_content.text)