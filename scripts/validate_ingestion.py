"""
Ingestion validation script — dry run only, nothing is written.

Usage (run from project root):
    python -m scripts.validate_ingestion \
        --pdf  path/to/lecture.pdf \
        --file_id <uuid-of-already-indexed-file> \
        --course_id <course_id>

What it does:
  1. Runs extraction   → reports total chars, OCR page count
  2. Runs chunking     → reports chunk count, avg size, 3 sample chunks
  3. Runs embedding    → reports vector count and dimension
  4. Queries Qdrant    → reports how many vectors are stored for this file_id

Nothing is written to Postgres or Qdrant.
"""

import argparse
import sys
import os

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tiktoken
from app.config import settings
from app.services.extractor import extract_pages
from app.services.chunker import chunk_document
from app.services.embedder import get_embeddings
from app.services.vector_store import count_by_file_id, get_client

_enc = tiktoken.get_encoding("cl100k_base")

SEP = "-" * 60


def validate_ingestion(pdf_path: str, file_id: str, course_id: str) -> None:
    print(f"\n{'='*60}")
    print(f"INGESTION VALIDATION")
    print(f"  pdf       : {pdf_path}")
    print(f"  file_id   : {file_id}")
    print(f"  course_id : {course_id}")
    print(f"  ocr       : {'enabled' if settings.OCR_ENABLED else 'disabled'}")
    print(f"{'='*60}\n")

    # ── 1. EXTRACTION ────────────────────────────────────────────
    print(f"[1/4] EXTRACTION")
    print(SEP)
    extraction = extract_pages(
        pdf_path,
        ocr_cache_path=pdf_path.replace(".pdf", "_ocr_cache.json"),
        ocr_enabled=settings.OCR_ENABLED,
    )

    total_chars = sum(len(p.text) for p in extraction.pages)
    ocr_chars   = sum(len(p.text) for p in extraction.pages if p.ocr_used)

    print(f"  total pages        : {extraction.total_page_count}")
    print(f"  extractable pages  : {extraction.extractable_page_count}")
    print(f"  ocr pages          : {extraction.ocr_page_count}")
    print(f"  ocr used           : {'yes' if extraction.ocr_page_count > 0 else 'no'}")
    print(f"  total chars        : {total_chars}")
    print(f"  ocr chars          : {ocr_chars}")
    print()

    # ── 2. CHUNKING ──────────────────────────────────────────────
    print(f"[2/4] CHUNKING")
    print(SEP)
    chunks = chunk_document(extraction.pages)

    if not chunks:
        print("  ERROR: no chunks produced — aborting")
        return

    token_counts = [len(_enc.encode(c.text)) for c in chunks]
    avg_tokens   = sum(token_counts) / len(token_counts)

    print(f"  total chunks  : {len(chunks)}")
    print(f"  avg tokens    : {avg_tokens:.0f}")
    print(f"  min tokens    : {min(token_counts)}")
    print(f"  max tokens    : {max(token_counts)}")
    print()

    sample_indices = [0, len(chunks) // 2, len(chunks) - 1]
    for si in sample_indices:
        c = chunks[si]
        print(f"  SAMPLE chunk idx={si}  page={c.page}  tokens={token_counts[si]}")
        print(f"    {repr(c.text[:200])}")
        print()

    # ── 3. EMBEDDING ─────────────────────────────────────────────
    print(f"[3/4] EMBEDDING")
    print(SEP)
    texts      = [c.text for c in chunks]
    embeddings = get_embeddings(texts)

    dim = len(embeddings[0]) if embeddings else 0
    print(f"  model          : text-embedding-3-small")
    print(f"  embeddings     : {len(embeddings)}")
    print(f"  dimension      : {dim}")
    print(f"  match chunks   : {'yes' if len(embeddings) == len(chunks) else 'NO — MISMATCH'}")
    print()

    # ── 4. QDRANT CHECK ──────────────────────────────────────────
    print(f"[4/4] QDRANT (existing vectors for this file_id)")
    print(SEP)
    try:
        stored = count_by_file_id(file_id)
        print(f"  collection     : {settings.QDRANT_COLLECTION}")
        print(f"  file_id        : {file_id}")
        print(f"  vectors stored : {stored}")
        if stored == 0:
            print("  WARNING: 0 vectors found — file may not be indexed yet")
        elif stored != len(chunks):
            print(f"  WARNING: stored={stored} but dry-run produced {len(chunks)} chunks")
        else:
            print(f"  OK: vector count matches chunk count")
    except Exception as e:
        print(f"  ERROR querying Qdrant: {e}")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ingestion pipeline on a PDF (dry run)")
    parser.add_argument("--pdf",       required=True,  help="Path to the PDF file")
    parser.add_argument("--file_id",   required=True,  help="UUID of the already-indexed file (for Qdrant check)")
    parser.add_argument("--course_id", required=True,  help="Course ID")
    args = parser.parse_args()

    validate_ingestion(args.pdf, args.file_id, args.course_id)
