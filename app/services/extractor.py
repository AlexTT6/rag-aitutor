from dataclasses import dataclass
from typing import List

import fitz  # PyMuPDF


@dataclass
class PageContent:
    page: int   # 1-indexed, matches the PDF page number
    text: str


@dataclass
class ExtractionResult:
    pages: List[PageContent]      # only pages that yielded text
    total_page_count: int         # every page in the PDF, including blank/scanned
    extractable_page_count: int   # pages that had at least some text


def extract_pages(pdf_path: str) -> ExtractionResult:
    """
    Opens the PDF and extracts text page by page using PyMuPDF.

    Pages with no extractable text (blank pages, scanned images without OCR)
    are counted in total_page_count but excluded from the returned pages list.

    This function does not perform OCR. If the PDF is fully scanned, the
    returned pages list will be empty and the ingestion pipeline will fail
    the file with a descriptive error message.
    """
    doc = fitz.open(pdf_path)
    total = len(doc)
    pages = []

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(PageContent(page=i, text=text))

    doc.close()

    return ExtractionResult(
        pages=pages,
        total_page_count=total,
        extractable_page_count=len(pages),
    )
