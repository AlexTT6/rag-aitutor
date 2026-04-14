"""
PDF text extraction with Vision OCR fallback.

Normal pages: text extracted directly via PyMuPDF (fast, free).
Image-only pages: rendered as PNG → sent to OpenAI gpt-4o-mini Vision
                  to extract text and math formulas.

OCR calls run in parallel (up to OCR_MAX_WORKERS at once) so a PDF
with 10 image pages takes ~the same time as 1 image page.

A page is considered "image-only" when PyMuPDF extracts fewer than
OCR_MIN_CHARS characters from it (default: 50).
"""

import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple

import fitz  # PyMuPDF

from app.config import settings

logger = logging.getLogger(__name__)

# Pages with fewer chars than this get sent to Vision OCR
OCR_MIN_CHARS = 50
# Pages where less than this fraction of chars are alphanumeric are garbage text
OCR_MIN_ALPHA_RATIO = 0.40
# Render resolution — 120 DPI is plenty for gpt-4o-mini, keeps image small
OCR_DPI = 120
# Max parallel OCR requests to OpenAI
OCR_MAX_WORKERS = 8


def _is_good_text(text: str) -> bool:
    """
    Returns True if the extracted text looks real.
    Catches cases like garbled scans where PyMuPDF extracts garbage characters.

    Checks:
    - Minimum character count
    - Minimum ratio of alphanumeric chars (letters + digits) to total
    """
    if len(text) < OCR_MIN_CHARS:
        return False
    alphanumeric = sum(1 for c in text if c.isalnum())
    ratio = alphanumeric / len(text)
    return ratio >= OCR_MIN_ALPHA_RATIO


@dataclass
class PageContent:
    page: int        # 1-indexed, matches the PDF page number
    text: str
    ocr_used: bool = False


@dataclass
class ExtractionResult:
    pages: List[PageContent]
    total_page_count: int
    extractable_page_count: int
    ocr_page_count: int = 0


def _render_page_b64(fitz_page: fitz.Page) -> str:
    """Renders a PDF page to a base64-encoded JPEG string.
    JPEG at 85% quality is ~10x smaller than PNG — faster API calls, same OCR accuracy.
    """
    mat = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
    pix = fitz_page.get_pixmap(matrix=mat)
    return base64.b64encode(pix.tobytes("jpeg", jpg_quality=85)).decode("utf-8")


def _ocr_one(page_num: int, b64_image: str) -> Tuple[int, str]:
    """
    Sends a pre-rendered page image to OpenAI Vision.
    Returns (page_num, extracted_text).
    Runs in a thread — fitz_page must NOT be passed here (not thread-safe).
    """
    if not settings.OPENAI_API_KEY:
        logger.warning(f"[extractor] OCR skipped page {page_num} — no OPENAI_API_KEY")
        return page_num, ""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

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
                                "url": f"data:image/jpeg;base64,{b64_image}",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
            max_tokens=1500,
        )
        text = response.choices[0].message.content or ""
        logger.info(f"[extractor] OCR page {page_num} → {len(text)} chars")
        return page_num, text

    except Exception as e:
        logger.error(f"[extractor] OCR failed page {page_num}: {e}")
        return page_num, ""


def extract_pages(pdf_path: str) -> ExtractionResult:
    """
    Extracts text from every PDF page.

    Pass 1 (main thread, sequential):
      - Extract native text from all pages with PyMuPDF.
      - Render image-only pages to PNG bytes (still in main thread — fitz is not thread-safe).

    Pass 2 (parallel threads):
      - Send all image-only pages to OpenAI Vision simultaneously.
      - Up to OCR_MAX_WORKERS concurrent requests.

    Result: same speed for text pages, ~OCR_MAX_WORKERS× faster for image pages.
    """
    doc = fitz.open(pdf_path)
    total = len(doc)

    # page_num → native text (may be "")
    native: Dict[int, str] = {}
    # page_num → base64 PNG  (only for image-only pages)
    to_ocr: Dict[int, str] = {}

    # --- Pass 1: extract text + render bad pages (must be single-threaded) ---
    for i, fitz_page in enumerate(doc, start=1):
        text = fitz_page.get_text("text").strip()
        native[i] = text
        if not _is_good_text(text):
            reason = "too short" if len(text) < OCR_MIN_CHARS else "garbage text"
            logger.info(f"[extractor] page {i} → {reason} ({len(text)} chars), queuing for OCR")
            to_ocr[i] = _render_page_b64(fitz_page)

    doc.close()

    # --- Pass 2: OCR in parallel ---
    ocr_results: Dict[int, str] = {}
    if to_ocr:
        logger.info(f"[extractor] running OCR on {len(to_ocr)} pages in parallel")
        with ThreadPoolExecutor(max_workers=OCR_MAX_WORKERS) as pool:
            futures = {
                pool.submit(_ocr_one, page_num, b64): page_num
                for page_num, b64 in to_ocr.items()
            }
            for future in as_completed(futures):
                page_num, text = future.result()
                ocr_results[page_num] = text

    # --- Assemble final page list in order ---
    pages: List[PageContent] = []
    ocr_count = 0

    for i in range(1, total + 1):
        if i in to_ocr:
            ocr_text = ocr_results.get(i, "").strip()
            if ocr_text:
                pages.append(PageContent(page=i, text=ocr_text, ocr_used=True))
                ocr_count += 1
            elif native[i]:
                # OCR failed but we had a little native text — keep it
                pages.append(PageContent(page=i, text=native[i], ocr_used=False))
        else:
            if native[i]:
                pages.append(PageContent(page=i, text=native[i], ocr_used=False))

    return ExtractionResult(
        pages=pages,
        total_page_count=total,
        extractable_page_count=len(pages),
        ocr_page_count=ocr_count,
    )
