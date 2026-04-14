"""
PDF text extraction with Vision OCR fallback.

Normal pages: text extracted directly via PyMuPDF (fast, free).
Image-only pages: rendered as PNG → sent to OpenAI gpt-4o-mini Vision
                  to extract text and math formulas.

A page is considered "image-only" when PyMuPDF extracts fewer than
OCR_MIN_CHARS characters from it (default: 50).
"""

import base64
import logging
from dataclasses import dataclass
from typing import List

import fitz  # PyMuPDF

from app.config import settings

logger = logging.getLogger(__name__)

# Pages with fewer chars than this get sent to Vision OCR
OCR_MIN_CHARS = 50
# Render resolution for Vision — 150 DPI is enough for GPT-4o to read clearly
OCR_DPI = 150


@dataclass
class PageContent:
    page: int   # 1-indexed, matches the PDF page number
    text: str
    ocr_used: bool = False  # True if Vision OCR was used for this page


@dataclass
class ExtractionResult:
    pages: List[PageContent]      # only pages that yielded text
    total_page_count: int         # every page in the PDF, including blank/scanned
    extractable_page_count: int   # pages that had at least some text
    ocr_page_count: int = 0       # pages that needed Vision OCR


def _ocr_page(fitz_page: fitz.Page, page_num: int) -> str:
    """
    Renders the page as a PNG and sends it to OpenAI Vision.
    Returns extracted text, or empty string on failure.
    """
    if not settings.OPENAI_API_KEY:
        logger.warning(f"[extractor] OCR skipped page {page_num} — no OPENAI_API_KEY")
        return ""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        # Render page to PNG bytes
        mat = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
        pix = fitz_page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        b64 = base64.b64encode(png_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This is a page from a university lecture PDF. "
                                "Extract ALL text exactly as written, including "
                                "math formulas, theorems, definitions, and notes. "
                                "Write formulas in plain text (e.g. f'(x) = 2x). "
                                "Do not add any commentary — only the extracted text."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=2000,
        )
        text = response.choices[0].message.content or ""
        logger.info(f"[extractor] OCR page {page_num} → {len(text)} chars")
        return text

    except Exception as e:
        logger.error(f"[extractor] OCR failed page {page_num}: {e}")
        return ""


def extract_pages(pdf_path: str) -> ExtractionResult:
    """
    Opens the PDF and extracts text page by page.

    - Pages with enough native text → extracted directly (fast).
    - Pages with little/no text → rendered and sent to OpenAI Vision OCR.
    - Pages that yield nothing from either method → skipped.
    """
    doc = fitz.open(pdf_path)
    total = len(doc)
    pages: List[PageContent] = []
    ocr_count = 0

    for i, fitz_page in enumerate(doc, start=1):
        text = fitz_page.get_text("text").strip()

        if len(text) >= OCR_MIN_CHARS:
            # Normal text page
            pages.append(PageContent(page=i, text=text, ocr_used=False))
        else:
            # Image-only page — try Vision OCR
            logger.info(f"[extractor] page {i} has only {len(text)} chars → trying OCR")
            ocr_text = _ocr_page(fitz_page, i)
            if ocr_text.strip():
                pages.append(PageContent(page=i, text=ocr_text, ocr_used=True))
                ocr_count += 1
            elif text:
                # Keep the little text we had rather than losing the page
                pages.append(PageContent(page=i, text=text, ocr_used=False))

    doc.close()

    return ExtractionResult(
        pages=pages,
        total_page_count=total,
        extractable_page_count=len(pages),
        ocr_page_count=ocr_count,
    )
