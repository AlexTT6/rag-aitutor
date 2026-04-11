import re
from dataclasses import dataclass
from typing import List

from app.services.extractor import PageContent


@dataclass
class Chunk:
    page: int
    chunk_index: int   # global order across the whole document
    text: str


# CHUNKING ROBUSTNESS NOTE:
# The regex below is a baseline suitable for clean, well-formatted academic PDFs.
# Known failure modes:
#   - Numbered lists ("1. Next item") — capital after period triggers a split
#   - Bullet points without terminal punctuation
#   - Mathematical equations and formulas (no clear sentence boundaries)
#   - PDFs with line-break artifacts in PyMuPDF output
#   - Abbreviations: "e.g.", "Fig.", "Dr.", "et al." cause false splits
#
# This is intentionally kept simple for the MVP. When these failure modes
# become a real problem in production data, replace split_sentences() with:
#   nltk.sent_tokenize(text)                          — better, needs punkt_tab data
#   [s.text for s in spacy_nlp(text).sents]           — best accuracy, heavier dependency


def split_sentences(text: str) -> List[str]:
    """
    Splits text on sentence boundaries using a simple regex heuristic.
    Splits after '.', '!', or '?' followed by whitespace and a capital letter.
    """
    pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    return [s.strip() for s in re.split(pattern, text) if s.strip()]


def chunk_page(
    page: int,
    text: str,
    chunk_size: int = 5,    # number of sentences per chunk
    overlap: int = 1,       # sentences shared between adjacent chunks
    start_index: int = 0,
) -> List[Chunk]:
    """
    Applies a sliding window of `chunk_size` sentences with `overlap` sentences
    of context carried into the next chunk. Chunks shorter than 50 characters
    are skipped — they are typically headers or page artifacts.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size}); "
            "otherwise the sliding window never advances and loops forever."
        )

    sentences = split_sentences(text)
    chunks = []
    i = 0
    idx = start_index

    while i < len(sentences):
        window = sentences[i : i + chunk_size]
        chunk_text = " ".join(window)
        if len(chunk_text) > 50:
            chunks.append(Chunk(page=page, chunk_index=idx, text=chunk_text))
            idx += 1
        i += chunk_size - overlap

    return chunks


def chunk_document(pages: List[PageContent]) -> List[Chunk]:
    """
    Chunks all pages in order, maintaining a globally unique chunk_index
    across the entire document.
    """
    all_chunks: List[Chunk] = []

    for page_content in pages:
        page_chunks = chunk_page(
            page=page_content.page,
            text=page_content.text,
            start_index=len(all_chunks),
        )
        all_chunks.extend(page_chunks)

    return all_chunks
