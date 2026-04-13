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

from dataclasses import dataclass
from typing import List

import tiktoken

from app.services.extractor import PageContent

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
    all_chunks: List[Chunk] = []

    for page_content in pages:
        text = page_content.text.strip()
        if not text:
            continue

        raw_chunks = _token_chunks(text, CHUNK_SIZE, OVERLAP)
        for raw in raw_chunks:
            cleaned = raw.strip()
            if len(cleaned) >= MIN_CHARS:
                all_chunks.append(
                    Chunk(
                        page=page_content.page,
                        chunk_index=len(all_chunks),
                        text=cleaned,
                    )
                )

    return all_chunks
