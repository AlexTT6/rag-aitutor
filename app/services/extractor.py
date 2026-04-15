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
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from app.config import settings

logger = logging.getLogger(__name__)

# Pages with fewer chars than this get sent to Vision OCR
OCR_MIN_CHARS = 50
# Pages where less than this fraction of chars are alphanumeric are garbage text
OCR_MIN_ALPHA_RATIO = 0.40
# Pages with images AND fewer than this many chars also get OCR (catches theorem boxes)
OCR_IMAGE_PAGE_THRESHOLD = 300
# Render resolution — 96 DPI keeps image small = faster API call, quality fine for text
OCR_DPI = 96
# Hard pages (image-heavy, near-zero native text) get 2× resolution + detail=high
OCR_HARD_DPI = 192
# Max parallel OCR requests to OpenAI
OCR_MAX_WORKERS = 16


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


def _render_page_b64(fitz_page: fitz.Page, dpi: int = OCR_DPI, quality: int = 85) -> str:
    """Renders a PDF page to a base64-encoded JPEG string.
    Normal pages: quality=85 (~10x smaller than PNG — fast API calls).
    Hard pages:   quality=95 (less JPEG artifacting on dense slide text).
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = fitz_page.get_pixmap(matrix=mat)
    return base64.b64encode(pix.tobytes("jpeg", jpg_quality=quality)).decode("utf-8")


def _load_ocr_cache(cache_path: str) -> Dict[int, str]:
    """Loads cached OCR results from disk. Returns empty dict if not found."""
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {}


def _save_ocr_cache(cache_path: str, cache: Dict[int, str]) -> None:
    """Persists OCR cache to disk. Failures are logged but never crash ingestion."""
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in cache.items()}, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[extractor] OCR cache write failed: {e}")


def clear_ocr_cache_page(cache_path: str, page_num: int) -> bool:
    """
    Removes one page's entry from the OCR cache so it will be re-sent to OCR
    on the next ingestion. Returns True if the entry existed and was removed.
    """
    cache = _load_ocr_cache(cache_path)
    if page_num not in cache:
        logger.info(f"[extractor] clear_ocr_cache_page page={page_num} — not in cache, nothing to do")
        return False
    del cache[page_num]
    _save_ocr_cache(cache_path, cache)
    logger.info(f"[extractor] clear_ocr_cache_page page={page_num} — entry removed, will retry on next index")
    return True


# --- OCR prompt constants ---
# Normal pages: generic academic PDF prompt.
_OCR_PROMPT_NORMAL = (
    "This is a page from a university lecture PDF. "
    "Extract ALL text exactly as written, including "
    "math formulas, theorems, definitions, and notes. "
    "Write formulas in plain text (e.g. f'(x) = 2x). "
    "Do not add any commentary — only the extracted text."
)

# Hard pages (image-heavy, near-zero native text, detail=high):
# Explicitly mentions boxes/diagrams and slide layout — targets gpt-4o-mini's
# tendency to ignore text embedded in visual elements on dense slides.
_OCR_PROMPT_HARD = (
    "This is a lecture slide or image-heavy PDF page. "
    "Extract every piece of visible text including headings, body text, "
    "bullet points, labels, text inside boxes or diagrams, math expressions, "
    "and any handwritten annotations. "
    "Write each item on its own line. "
    "Do not describe the images — output only the text you can read."
)

# Fallback: used on a single retry when the first hard-page OCR returns
# empty or very short text. Maximally direct — removes any abstraction.
_OCR_PROMPT_FALLBACK = (
    "List every word and number visible anywhere in this image, line by line. "
    "Include text inside boxes, arrows, diagrams, and overlaid labels. "
    "Output only the text, nothing else."
)


_OCR_RATE_LIMIT_MAX_RETRIES = 3
_OCR_RATE_LIMIT_BASE_WAIT = 2.0   # seconds; doubles each retry (2s → 4s → 8s)


def _call_ocr_api(client, page_num: int, b64_image: str, detail: str, prompt: str, attempt: int) -> str:
    """
    Single OpenAI Vision call with automatic 429 retry (exponential backoff).
    Returns extracted text, or empty string on unrecoverable failure.
    """
    last_exc = None
    for rate_retry in range(_OCR_RATE_LIMIT_MAX_RETRIES):
        try:
            logger.info(
                f"[extractor] OCR_REQUEST page={page_num} attempt={attempt} "
                f"rate_retry={rate_retry} "
                f"image_b64_len={len(b64_image)} "
                f"model=gpt-4o-mini detail={detail!r} max_tokens=1500"
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                    "detail": detail,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1500,
                timeout=settings.OCR_TIMEOUT,
            )
            raw = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            text = raw or ""
            logger.info(
                f"[extractor] OCR_RESPONSE page={page_num} attempt={attempt} "
                f"finish_reason={finish_reason!r} "
                f"raw_len={len(text)} "
                f"raw_preview={text[:120]!r}"
            )
            return text

        except Exception as exc:
            exc_str = str(exc)
            is_rate_limit = "429" in exc_str or "rate_limit" in exc_str.lower() or "rate limit" in exc_str.lower()
            if is_rate_limit and rate_retry < _OCR_RATE_LIMIT_MAX_RETRIES - 1:
                wait = _OCR_RATE_LIMIT_BASE_WAIT * (2 ** rate_retry)
                logger.warning(
                    f"[extractor] OCR_RATELIMIT page={page_num} attempt={attempt} "
                    f"rate_retry={rate_retry} — waiting {wait:.0f}s before retry"
                )
                time.sleep(wait)
                last_exc = exc
                continue
            raise exc

    raise last_exc  # unreachable, but satisfies type checkers


def _ocr_one(page_num: int, b64_image: str, detail: str = "auto") -> Tuple[int, str]:
    """
    Sends a pre-rendered page image to OpenAI Vision.
    Returns (page_num, extracted_text).
    Runs in a thread — fitz_page must NOT be passed here (not thread-safe).

    detail="auto"  → normal pages, uses _OCR_PROMPT_NORMAL (single attempt).
    detail="high"  → hard pages (image-heavy, <50 chars native text).
                     Uses _OCR_PROMPT_HARD on attempt 1.
                     If result is empty or weak (<OCR_MIN_CHARS chars),
                     retries ONCE with _OCR_PROMPT_FALLBACK + detail="high".
    """
    if not settings.OPENAI_API_KEY:
        logger.warning(f"[extractor] OCR skipped page {page_num} — no OPENAI_API_KEY")
        return page_num, ""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        is_hard = (detail == "high")
        prompt = _OCR_PROMPT_HARD if is_hard else _OCR_PROMPT_NORMAL

        text = _call_ocr_api(client, page_num, b64_image, detail, prompt, attempt=1)

        # --- Refusal / empty checks ---
        refusal_phrases = ["unable to extract", "can't extract", "cannot extract", "i'm unable"]
        if any(p in text.lower() for p in refusal_phrases):
            logger.warning(f"[extractor] OCR_REFUSED page={page_num} attempt=1 — model refused, result discarded")
            text = ""

        if not text.strip():
            logger.warning(f"[extractor] OCR_EMPTY page={page_num} attempt=1 — model returned empty text")
            text = ""

        # --- Fallback retry for hard pages only ---
        if is_hard and len(text.strip()) < OCR_MIN_CHARS:
            logger.warning(
                f"[extractor] OCR_WEAK page={page_num} attempt=1 "
                f"chars={len(text.strip())} — retrying with fallback prompt"
            )
            text2 = _call_ocr_api(client, page_num, b64_image, "high", _OCR_PROMPT_FALLBACK, attempt=2)

            if any(p in text2.lower() for p in refusal_phrases):
                logger.warning(f"[extractor] OCR_REFUSED page={page_num} attempt=2 — model refused fallback")
                text2 = ""

            if text2.strip():
                # Use whichever attempt produced more text
                if len(text2.strip()) > len(text.strip()):
                    logger.info(
                        f"[extractor] OCR_FALLBACK_USED page={page_num} "
                        f"attempt2_chars={len(text2.strip())} > attempt1_chars={len(text.strip())}"
                    )
                    text = text2
                else:
                    logger.info(
                        f"[extractor] OCR_FALLBACK_KEPT_ORIGINAL page={page_num} "
                        f"attempt1_chars={len(text.strip())} >= attempt2_chars={len(text2.strip())}"
                    )
            else:
                logger.warning(f"[extractor] OCR_EMPTY page={page_num} attempt=2 — fallback also returned empty")

        if not text.strip():
            logger.warning(f"[extractor] OCR_FAILED_FINAL page={page_num} — all attempts returned no text")
            return page_num, ""

        logger.info(f"[extractor] OCR_SUCCESS page={page_num} chars={len(text)}")
        return page_num, text

    except Exception as e:
        logger.error(f"[extractor] OCR_FAILED page={page_num}: {e}")
        return page_num, ""


def extract_pages(pdf_path: str, ocr_cache_path: Optional[str] = None, ocr_enabled: bool = True) -> ExtractionResult:
    """
    Extracts text from every PDF page.

    Pass 1 (main thread, sequential):
      - Extract native text from all pages with PyMuPDF.
      - Render image-only pages to JPEG bytes (still in main thread — fitz is not thread-safe).
      - Pages already in the OCR cache are skipped (no API call needed).

    Pass 2 (parallel threads):
      - Send all uncached image-only pages to OpenAI Vision simultaneously.
      - Up to OCR_MAX_WORKERS concurrent requests.
      - Results are written back to the cache after all calls complete.

    Result: re-indexing the same file never calls OpenAI Vision again.
    """
    doc = fitz.open(pdf_path)
    total = len(doc)

    # page_num → native text (may be "")
    native: Dict[int, str] = {}
    # page_num → (base64 JPEG, detail level) — only for image-only pages not in cache
    to_ocr: Dict[int, Tuple[str, str]] = {}

    if not ocr_enabled:
        # OCR disabled — extract native text only, no rendering, no API calls
        logger.info("[extractor] OCR disabled — native text only")
        for i, fitz_page in enumerate(doc, start=1):
            native[i] = fitz_page.get_text("text").strip()
        doc.close()
        ocr_cache: Dict[int, str] = {}
        ocr_results: Dict[int, str] = {}
    else:
        # Load existing OCR cache (empty dict if first time)
        ocr_cache = _load_ocr_cache(ocr_cache_path) if ocr_cache_path else {}
        cache_hits = 0

        # --- Pass 1: extract text + render bad pages (must be single-threaded) ---
        for i, fitz_page in enumerate(doc, start=1):
            text = fitz_page.get_text("text").strip()
            native[i] = text
            has_images = len(fitz_page.get_images()) > 0

            needs_ocr = (
                not _is_good_text(text)                                     # too short or garbage
                or (has_images and len(text) < OCR_IMAGE_PAGE_THRESHOLD)    # has image boxes + little text
            )
            if needs_ocr:
                if i in ocr_cache:
                    if ocr_cache[i].strip():
                        # Valid cache hit — skip rendering and API call entirely
                        cache_hits += 1
                        logger.debug(f"[extractor] page {i} → OCR cache hit")
                    else:
                        # Stale empty cache entry — treat as uncached and re-queue
                        logger.warning(f"[extractor] OCR_CACHE_EMPTY page={i} — cached value is empty, re-queuing for OCR")
                        _is_hard = has_images and len(text) < OCR_MIN_CHARS
                        to_ocr[i] = (
                            _render_page_b64(
                                fitz_page,
                                dpi=OCR_HARD_DPI if _is_hard else OCR_DPI,
                                quality=95 if _is_hard else 85,
                            ),
                            "high" if _is_hard else "auto",
                        )
                else:
                    is_hard = has_images and len(text) < OCR_MIN_CHARS
                    dpi = OCR_HARD_DPI if is_hard else OCR_DPI
                    ocr_detail = "high" if is_hard else "auto"
                    reason = (
                        "too short" if len(text) < OCR_MIN_CHARS
                        else "garbage text" if not _is_good_text(text)
                        else f"has images + only {len(text)} chars"
                    )
                    logger.info(
                        f"[extractor] page {i} → {reason}, queuing for OCR "
                        f"[dpi={dpi} detail={ocr_detail!r}]"
                    )
                    to_ocr[i] = (
                        _render_page_b64(fitz_page, dpi=dpi, quality=95 if is_hard else 85),
                        ocr_detail,
                    )

        doc.close()

        if cache_hits:
            logger.info(f"[extractor] {cache_hits} pages served from OCR cache (no API call)")

        # --- Pass 2: OCR in parallel (only uncached pages) ---
        ocr_results = {}
        if to_ocr:
            logger.info(f"[extractor] running OCR on {len(to_ocr)} pages in parallel")
            with ThreadPoolExecutor(max_workers=OCR_MAX_WORKERS) as pool:
                futures = {
                    pool.submit(_ocr_one, page_num, b64, detail): page_num
                    for page_num, (b64, detail) in to_ocr.items()
                }
                for future in as_completed(futures):
                    page_num, text = future.result()
                    if text.strip():                    # only store successful results
                        ocr_results[page_num] = text

            # Persist new results to cache — empty strings are never written
            if ocr_cache_path:
                valid_ocr = {k: v for k, v in ocr_results.items() if v.strip()}
                ocr_cache.update(valid_ocr)
                _save_ocr_cache(ocr_cache_path, ocr_cache)
                logger.info(f"[extractor] OCR cache updated ({len(ocr_cache)} pages total)")

    # --- Assemble final page list in order ---
    pages: List[PageContent] = []
    ocr_count = 0

    for i in range(1, total + 1):
        needs_ocr_page = i in to_ocr or i in ocr_cache
        if needs_ocr_page:
            # Prefer freshly fetched result; fall back to cache for cache-hit pages
            ocr_text = ocr_results.get(i, ocr_cache.get(i, "")).strip()
            nat = native[i].strip()
            if ocr_text:
                # Merge: prefer OCR but prepend any unique native text
                # (OCR already sees the full page, so it usually contains native text too)
                combined = ocr_text if not nat else f"{nat}\n{ocr_text}"
                pages.append(PageContent(page=i, text=combined, ocr_used=True))
                ocr_count += 1
            elif nat:
                logger.warning(
                    f"[extractor] PAGE_OCR_FAILED page={i} — "
                    f"OCR produced no text, falling back to native ({len(nat)} chars)"
                )
                pages.append(PageContent(page=i, text=nat, ocr_used=False))
            else:
                logger.warning(f"[extractor] PAGE_DROPPED page={i} — OCR empty and no native text")
        else:
            if native[i]:
                pages.append(PageContent(page=i, text=native[i], ocr_used=False))

    return ExtractionResult(
        pages=pages,
        total_page_count=total,
        extractable_page_count=len(pages),
        ocr_page_count=ocr_count,
    )
