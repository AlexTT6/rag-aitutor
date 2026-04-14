"""
Page-level extraction debugger — read-only, no API calls, nothing written.

Usage (run from project root):
    python -m scripts.debug_page --pdf path/to/lecture.pdf --page 78

What it does:
  1. Opens the PDF with PyMuPDF and reads native text for the target page.
  2. Reads the OCR cache file (if it exists) for that page.
  3. Simulates the extractor's needs_ocr decision so you can see WHY
     a page was or wasn't sent to OCR.
  4. Shows the final merged text (native + OCR) exactly as the extractor
     would produce it.
  5. Runs chunk_document on the full document and shows which chunks
     came from the target page.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # PyMuPDF

from app.services.extractor import (
    OCR_IMAGE_PAGE_THRESHOLD,
    OCR_MIN_ALPHA_RATIO,
    OCR_MIN_CHARS,
    _is_good_text,
)
from app.services.chunker import chunk_document
from app.services.extractor import PageContent

SEP = "-" * 60
PREVIEW_LEN = 300


def preview(text: str, n: int = PREVIEW_LEN) -> str:
    if not text:
        return "(empty)"
    snip = text[:n]
    return repr(snip) + (" …" if len(text) > n else "")


def debug_page(pdf_path: str, target_page: int) -> None:
    ocr_cache_path = pdf_path.replace(".pdf", "_ocr_cache.json")

    print(f"\n{'='*60}")
    print(f"PAGE EXTRACTION DEBUGGER")
    print(f"  pdf        : {pdf_path}")
    print(f"  page       : {target_page}  (1-indexed)")
    print(f"  ocr cache  : {ocr_cache_path}")
    print(f"{'='*60}\n")

    # ── Open PDF ──────────────────────────────────────────────────
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found: {pdf_path}")
        return

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Total pages in PDF: {total_pages}")

    if target_page < 1 or target_page > total_pages:
        print(f"ERROR: page {target_page} is out of range (1–{total_pages})")
        doc.close()
        return

    # ── 1. NATIVE TEXT ────────────────────────────────────────────
    print(f"\n[1/4] NATIVE TEXT  (PyMuPDF)")
    print(SEP)

    fitz_page  = doc[target_page - 1]   # fitz is 0-indexed
    native_raw = fitz_page.get_text("text")
    native     = native_raw.strip()
    has_images = len(fitz_page.get_images()) > 0

    print(f"  raw length (before strip) : {len(native_raw)}")
    print(f"  stripped length           : {len(native)}")
    print(f"  has embedded images       : {has_images}")
    print(f"  preview                   : {preview(native)}")

    # ── 2. needs_ocr DECISION ─────────────────────────────────────
    print(f"\n[2/4] needs_ocr DECISION  (extractor logic)")
    print(SEP)

    too_short    = len(native) < OCR_MIN_CHARS
    garbage_text = not _is_good_text(native)
    image_sparse = has_images and len(native) < OCR_IMAGE_PAGE_THRESHOLD
    needs_ocr    = too_short or garbage_text or image_sparse

    alphanumeric = sum(1 for c in native if c.isalnum())
    alpha_ratio  = alphanumeric / len(native) if native else 0.0

    print(f"  OCR_MIN_CHARS             : {OCR_MIN_CHARS}")
    print(f"  OCR_MIN_ALPHA_RATIO       : {OCR_MIN_ALPHA_RATIO}")
    print(f"  OCR_IMAGE_PAGE_THRESHOLD  : {OCR_IMAGE_PAGE_THRESHOLD}")
    print(f"  ---")
    print(f"  native char count         : {len(native)}")
    print(f"  alphanumeric ratio        : {alpha_ratio:.2f}")
    print(f"  too_short                 : {too_short}")
    print(f"  garbage_text              : {garbage_text}")
    print(f"  image_sparse              : {image_sparse}")
    print(f"  → needs_ocr               : {needs_ocr}")

    doc.close()

    # ── 3. OCR CACHE ──────────────────────────────────────────────
    print(f"\n[3/4] OCR CACHE")
    print(SEP)

    ocr_text = ""
    if os.path.exists(ocr_cache_path):
        try:
            with open(ocr_cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            cache = {int(k): v for k, v in cache.items()}
            total_cached = len(cache)
            print(f"  cache file found          : yes")
            print(f"  total cached pages        : {total_cached}")

            if target_page in cache:
                ocr_text = cache[target_page].strip()
                print(f"  page {target_page} in cache       : yes")
                print(f"  ocr text length           : {len(ocr_text)}")
                print(f"  ocr preview               : {preview(ocr_text)}")
            else:
                print(f"  page {target_page} in cache       : no")
                if needs_ocr:
                    print(f"  → page needs OCR but has no cached result")
                    print(f"     (it was either never indexed, or OCR was disabled)")
                else:
                    print(f"  → page does not need OCR (native text is sufficient)")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  cache file found but unreadable: {e}")
    else:
        print(f"  cache file found          : no")
        print(f"  path checked              : {ocr_cache_path}")
        if needs_ocr:
            print(f"  → page needs OCR but no cache exists yet")

    # ── 4. FINAL MERGED TEXT ──────────────────────────────────────
    print(f"\n[4/4] FINAL MERGED TEXT  (what extractor returns)")
    print(SEP)

    # Replicate extractor assembly logic exactly
    if needs_ocr:
        if ocr_text:
            final_text = ocr_text if not native else f"{native}\n{ocr_text}"
            ocr_used   = True
        elif native:
            final_text = native
            ocr_used   = False
        else:
            final_text = ""
            ocr_used   = False
    else:
        final_text = native
        ocr_used   = False

    print(f"  ocr_used                  : {ocr_used}")
    print(f"  final text length         : {len(final_text)}")
    print(f"  final preview             : {preview(final_text)}")

    if not final_text:
        print(f"\n  WARNING: page {target_page} produces EMPTY final text")
        print(f"           It will be SKIPPED by the extractor (not indexed).")

    # ── 5. CHUNKS FROM THIS PAGE ──────────────────────────────────
    print(f"\n[5/5] CHUNKS FROM PAGE {target_page}")
    print(SEP)

    if not final_text:
        print(f"  No final text → no chunks produced for page {target_page}")
        return

    # Build a minimal PageContent list with just this one page so we can
    # chunk it in isolation and see exactly what comes out.
    page_content = PageContent(page=target_page, text=final_text, ocr_used=ocr_used)
    chunks       = chunk_document([page_content])

    if not chunks:
        print(f"  WARNING: chunker produced 0 chunks for page {target_page}")
        print(f"  (text length {len(final_text)} may be below MIN_CHARS={50})")
    else:
        print(f"  chunks produced for page {target_page}: {len(chunks)}")
        print()
        for i, c in enumerate(chunks):
            print(f"  chunk [{i}]  chunk_index={c.chunk_index}  page={c.page}")
            print(f"    length  : {len(c.text)} chars")
            print(f"    preview : {preview(c.text)}")
            print()

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug extraction for one PDF page")
    parser.add_argument("--pdf",  required=True,         help="Path to the PDF file")
    parser.add_argument("--page", type=int, default=78,  help="1-indexed page number to debug (default: 78)")
    args = parser.parse_args()

    debug_page(args.pdf, args.page)
