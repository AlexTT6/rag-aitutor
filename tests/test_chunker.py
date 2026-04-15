"""
Tests for the token-based chunker service.

The old sentence-based chunker (chunk_page / split_sentences) was removed.
All tests now exercise the public API: chunk_document(pages) → List[Chunk].
"""
import pytest

from app.services.chunker import CHUNK_SIZE, MIN_CHARS, OVERLAP, Chunk, chunk_document
from app.services.extractor import PageContent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _page(text: str, page: int = 1) -> PageContent:
    return PageContent(page=page, text=text)


def _long_text(n_words: int = 500) -> str:
    """Generates a predictable block of text that's long enough to produce chunks."""
    words = [f"word{i}" for i in range(n_words)]
    return " ".join(words)


# ---------------------------------------------------------------------------
# chunk_document: basic contract
# ---------------------------------------------------------------------------

class TestChunkDocumentBasic:
    def test_empty_input_returns_empty_list(self):
        assert chunk_document([]) == []

    def test_blank_page_produces_no_chunks(self):
        assert chunk_document([_page("")]) == []

    def test_whitespace_only_page_produces_no_chunks(self):
        assert chunk_document([_page("   \n\t  ")]) == []

    def test_short_text_below_min_chars_is_dropped(self):
        # Single short sentence — should be filtered by MIN_CHARS
        assert chunk_document([_page("Hi.")]) == []

    def test_returns_list_of_chunk_objects(self):
        chunks = chunk_document([_page(_long_text())])
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)


# ---------------------------------------------------------------------------
# Chunk index ordering
# ---------------------------------------------------------------------------

class TestChunkIndex:
    def test_chunk_index_starts_at_zero(self):
        chunks = chunk_document([_page(_long_text())])
        assert chunks[0].chunk_index == 0

    def test_global_chunk_index_is_contiguous(self):
        pages = [
            _page(_long_text(300), page=1),
            _page(_long_text(300), page=2),
        ]
        chunks = chunk_document(pages)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_single_page_indices_are_contiguous(self):
        chunks = chunk_document([_page(_long_text(600))])
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# Page number preservation
# ---------------------------------------------------------------------------

class TestPageNumbers:
    def test_page_number_preserved_on_single_page(self):
        chunks = chunk_document([_page(_long_text(), page=7)])
        for c in chunks:
            assert c.page == 7

    def test_page_numbers_preserved_across_multiple_pages(self):
        pages = [
            _page(_long_text(300), page=3),
            _page(_long_text(300), page=9),
        ]
        chunks = chunk_document(pages)
        pages_seen = {c.page for c in chunks}
        assert pages_seen == {3, 9}

    def test_chunks_from_page_2_have_correct_page(self):
        pages = [
            _page(_long_text(300), page=1),
            _page(_long_text(300), page=2),
        ]
        chunks = chunk_document(pages)
        p2_chunks = [c for c in chunks if c.page == 2]
        assert len(p2_chunks) > 0


# ---------------------------------------------------------------------------
# Chunk size and overlap
# ---------------------------------------------------------------------------

class TestChunkSizeAndOverlap:
    def test_token_count_never_exceeds_chunk_size(self):
        chunks = chunk_document([_page(_long_text(800))])
        for c in chunks:
            assert c.token_count <= CHUNK_SIZE

    def test_token_count_populated(self):
        chunks = chunk_document([_page(_long_text())])
        for c in chunks:
            assert c.token_count > 0

    def test_long_text_produces_multiple_chunks(self):
        # 800 words ≈ 1000+ tokens, well above CHUNK_SIZE=400
        chunks = chunk_document([_page(_long_text(800))])
        assert len(chunks) > 1

    def test_text_length_never_below_min_chars(self):
        chunks = chunk_document([_page(_long_text(800))])
        for c in chunks:
            assert len(c.text) >= MIN_CHARS

    def test_overlap_means_adjacent_chunks_share_content(self):
        """Adjacent chunks must share text content at the boundary.

        We sample words from the end of chunk[0] and check that they appear
        at the start of chunk[1]. This is more robust than comparing raw token
        ids because `.strip()` on decoded windows shifts the first token.
        """
        chunks = chunk_document([_page(_long_text(600))])
        if len(chunks) < 2:
            pytest.skip("not enough chunks to test overlap")

        # Take a few words from near the END of chunk[0]
        tail_words = chunks[0].text.split()[-5:]
        tail_sample = " ".join(tail_words)

        # They must appear somewhere near the START of chunk[1]
        head = chunks[1].text
        assert tail_sample in head, (
            f"Expected tail of chunk[0] to appear in head of chunk[1].\n"
            f"Tail sample: {tail_sample!r}\n"
            f"Head of chunk[1]: {head[:200]!r}"
        )


# ---------------------------------------------------------------------------
# Empty pages mixed with content pages
# ---------------------------------------------------------------------------

class TestMixedPages:
    def test_empty_pages_between_content_pages_are_skipped(self):
        pages = [
            _page(_long_text(300), page=1),
            _page("", page=2),
            _page(_long_text(300), page=3),
        ]
        chunks = chunk_document(pages)
        pages_seen = {c.page for c in chunks}
        assert 2 not in pages_seen
        assert {1, 3}.issubset(pages_seen)

    def test_all_empty_pages_returns_empty(self):
        pages = [_page("", page=i) for i in range(1, 5)]
        assert chunk_document(pages) == []
