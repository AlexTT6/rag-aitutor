"""
Tests for the chunker service.
Verifies chunk structure, overlap behavior, and edge cases.
"""
from app.services.chunker import chunk_document, chunk_page, split_sentences
from app.services.extractor import PageContent


class TestSplitSentences:
    def test_basic_split(self):
        text = "First sentence. Second sentence. Third one."
        result = split_sentences(text)
        assert len(result) == 3

    def test_single_sentence(self):
        result = split_sentences("Just one sentence.")
        assert result == ["Just one sentence."]

    def test_empty_string(self):
        result = split_sentences("")
        assert result == []


class TestChunkPage:
    def test_produces_chunks(self):
        text = " ".join(
            f"Sentence number {i} on this page." for i in range(1, 21)
        )
        chunks = chunk_page(page=1, text=text, start_index=0)
        assert len(chunks) > 0
        for c in chunks:
            assert c.page == 1
            assert len(c.text) > 0

    def test_chunk_index_starts_at_start_index(self):
        text = " ".join(f"Sentence {i}." for i in range(1, 11))
        chunks = chunk_page(page=2, text=text, start_index=10)
        assert chunks[0].chunk_index == 10

    def test_short_text_below_min_chars_is_dropped(self):
        # A single very short sentence should be dropped (below MIN_CHUNK_CHARS=50)
        chunks = chunk_page(page=1, text="Hi.", start_index=0)
        assert chunks == []


class TestChunkDocument:
    def test_global_chunk_index_is_contiguous(self):
        pages = [
            PageContent(page=1, text=" ".join(f"Sentence {i}." for i in range(1, 16))),
            PageContent(page=2, text=" ".join(f"Sentence {i}." for i in range(16, 31))),
        ]
        chunks = chunk_document(pages)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_pages_produces_no_chunks(self):
        assert chunk_document([]) == []

    def test_page_numbers_preserved(self):
        pages = [
            PageContent(page=5, text=" ".join(f"Sentence {i}." for i in range(1, 11))),
        ]
        chunks = chunk_document(pages)
        for c in chunks:
            assert c.page == 5
