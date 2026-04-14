"""
Token-based chunking using tiktoken.

Why token-based instead of sentence-based:
  The old regex sentence splitter broke on abbreviations ("Dr.", "Fig.", "e.g."),
  numbered lists, and math formulas — common in academic PDFs.
  Token counting is deterministic, predictable, and immune to these edge cases.

Chunk size: 400 tokens  (~300 words, enough context for one concept)
Overlap:     50 tokens  (continuity between adjacent chunks)
Min length:  50 chars   (skip headers and page artifacts)

Each chunk preserves its source page number for citation purposes.
"""

import logging
import time
from dataclasses import dataclass
from typing import List

import tiktoken

from app.services.extractor import PageContent

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE = 400    # tokens per chunk — OpenAI text-embedding-3-small handles up to 8191
OVERLAP = 50        # tokens shared between adjacent chunks
MIN_CHARS = 50      # shorter chunks are page artifacts — skip them


@dataclass
class Chunk:
    page: int
    chunk_index: int   # global order across the whole document
    text: str


def _token_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Splits text into overlapping token windows.
    Decodes each window back to a string.
    """
    tokens = _enc.encode(text)
    results = []
    step = chunk_size - overlap
    i = 0
    while i < len(tokens):
        window = tokens[i : i + chunk_size]
        results.append(_enc.decode(window))
        i += step
    return results


def chunk_document(pages: List[PageContent]) -> List[Chunk]:
    """
    Chunks every page independently to preserve page metadata,
    then assigns globally unique chunk_index values across the document.
    """
    t_total_start = time.monotonic()

    # Per-substep accumulators (seconds)
    t_strip   = 0.0   # text prep: page_content.text.strip()
    t_encode  = 0.0   # tiktoken encode: text → token ids
    t_window  = 0.0   # windowing: slice token list into overlapping windows
    t_decode  = 0.0   # tiktoken decode: token windows → strings
    t_filter  = 0.0   # filter MIN_CHARS + build Chunk objects

    total_input_chars   = 0
    total_tokens_in     = 0   # raw tokens encoded (before windowing)

    all_chunks: List[Chunk] = []

    for page_content in pages:

        # --- substep 1: text prep ---
        t0 = time.monotonic()
        text = page_content.text.strip()
        t_strip += time.monotonic() - t0

        if not text:
            continue

        total_input_chars += len(text)

        # --- substep 2: encode ---
        t0 = time.monotonic()
        tokens = _enc.encode(text)
        t_encode += time.monotonic() - t0
        total_tokens_in += len(tokens)

        # --- substep 3: windowing (split + overlap) ---
        t0 = time.monotonic()
        step = CHUNK_SIZE - OVERLAP
        windows: List[List[int]] = []
        i = 0
        while i < len(tokens):
            windows.append(tokens[i : i + CHUNK_SIZE])
            i += step
        t_window += time.monotonic() - t0

        # --- substep 4: decode ---
        t0 = time.monotonic()
        decoded = [_enc.decode(w) for w in windows]
        t_decode += time.monotonic() - t0

        # --- substep 5: filter + metadata construction ---
        t0 = time.monotonic()
        for raw in decoded:
            cleaned = raw.strip()
            if len(cleaned) >= MIN_CHARS:
                all_chunks.append(
                    Chunk(
                        page=page_content.page,
                        chunk_index=len(all_chunks),
                        text=cleaned,
                    )
                )
        t_filter += time.monotonic() - t0

    t_total = time.monotonic() - t_total_start

    avg_tokens = total_tokens_in / len(all_chunks) if all_chunks else 0

    logger.info(
        f"[chunker:debug] INPUT   pages={len(pages)} "
        f"total_chars={total_input_chars} "
        f"total_tokens={total_tokens_in}"
    )
    logger.info(
        f"[chunker:debug] OUTPUT  chunks={len(all_chunks)} "
        f"avg_tokens={avg_tokens:.0f}"
    )
    logger.info(
        f"[chunker:debug] TIMING  "
        f"strip={t_strip*1000:.1f}ms  "
        f"encode={t_encode*1000:.1f}ms  "
        f"window={t_window*1000:.1f}ms  "
        f"decode={t_decode*1000:.1f}ms  "
        f"filter_metadata={t_filter*1000:.1f}ms  "
        f"total={t_total*1000:.1f}ms"
    )

    return all_chunks
