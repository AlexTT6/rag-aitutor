"""
Tests for the ingestion pipeline (services/ingestion.py).

These tests exercise run_ingestion() directly with a real SQLite session
and mocked external calls (PyMuPDF, OpenAI, Qdrant).
"""
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.course import Course
from app.models.file import File, FileStatus
from app.services.ingestion import run_ingestion
from app.services.extractor import ExtractionResult, PageContent

_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
_Session = sessionmaker(bind=_engine)


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.create_all(_engine)
    yield
    Base.metadata.drop_all(_engine)


@pytest.fixture
def db():
    s = _Session()
    try:
        yield s
    finally:
        s.close()


def _make_file(db, total_pages=10, status=FileStatus.uploaded) -> File:
    course = Course(name="Test")
    db.add(course)
    db.commit()

    f = File(
        course_id=course.id,
        filename="test.pdf",
        storage_path="/fake/path.pdf",
        size_bytes=1024,
        status=status,
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


def _good_extraction(n_pages=5) -> ExtractionResult:
    pages = [
        PageContent(page=i, text=f"Sentence one on page {i}. Sentence two here. More text follows.")
        for i in range(1, n_pages + 1)
    ]
    return ExtractionResult(
        pages=pages,
        total_page_count=n_pages,
        extractable_page_count=n_pages,
    )


class TestIngestionSuccess:
    def test_file_status_becomes_indexed(self, db):
        f = _make_file(db)
        extraction = _good_extraction()

        with patch("app.services.ingestion.extract_pages", return_value=extraction), \
             patch("app.services.ingestion.get_embeddings", return_value=[[0.1] * 10] * 20), \
             patch("app.services.ingestion.insert_chunks", return_value=[f"id-{i}" for i in range(20)]), \
             patch("app.services.ingestion.delete_by_file_id"):
            run_ingestion(f.id, db)

        db.refresh(f)
        assert f.status == FileStatus.indexed
        assert f.chunk_count > 0
        assert f.indexed_at is not None
        assert f.total_page_count == 5
        assert f.extractable_page_count == 5

    def test_chunks_written_to_postgres(self, db):
        from app.models.chunk import Chunk
        f = _make_file(db)
        extraction = _good_extraction(n_pages=2)
        fake_ids = [f"qid-{i}" for i in range(20)]

        with patch("app.services.ingestion.extract_pages", return_value=extraction), \
             patch("app.services.ingestion.get_embeddings", return_value=[[0.0] * 10] * 20), \
             patch("app.services.ingestion.insert_chunks", return_value=fake_ids), \
             patch("app.services.ingestion.delete_by_file_id"):
            run_ingestion(f.id, db)

        chunks = db.query(Chunk).filter(Chunk.file_id == f.id).all()
        assert len(chunks) > 0
        for c in chunks:
            assert c.qdrant_id.startswith("qid-")
            assert c.page in (1, 2)


class TestIngestionFailures:
    def test_exceeds_max_pages_sets_failed(self, db):
        f = _make_file(db)
        big_extraction = ExtractionResult(
            pages=[PageContent(page=i, text="text") for i in range(1, 301)],
            total_page_count=400,  # exceeds MAX_PAGES=300
            extractable_page_count=300,
        )
        with patch("app.services.ingestion.extract_pages", return_value=big_extraction), \
             patch("app.services.ingestion.settings") as mock_settings:
            mock_settings.MAX_PAGES = 300
            mock_settings.MIN_EXTRACTABLE_RATIO = 0.5
            run_ingestion(f.id, db)

        db.refresh(f)
        assert f.status == FileStatus.failed
        assert "400" in f.error_message  # page count in message

    def test_low_extractable_ratio_sets_failed(self, db):
        f = _make_file(db)
        mostly_scanned = ExtractionResult(
            pages=[PageContent(page=1, text="Only one page has text.")],
            total_page_count=10,
            extractable_page_count=1,  # 10% < 50% MIN_EXTRACTABLE_RATIO
        )
        with patch("app.services.ingestion.extract_pages", return_value=mostly_scanned), \
             patch("app.services.ingestion.delete_by_file_id"):
            run_ingestion(f.id, db)

        db.refresh(f)
        assert f.status == FileStatus.failed
        assert f.error_message is not None
        assert "scanned" in f.error_message.lower()

    def test_qdrant_rollback_called_on_postgres_failure(self, db):
        from app.models.chunk import Chunk
        f = _make_file(db)
        extraction = _good_extraction()
        fake_ids = ["qid-0", "qid-1"]

        with patch("app.services.ingestion.extract_pages", return_value=extraction), \
             patch("app.services.ingestion.get_embeddings", return_value=[[0.0] * 10] * 2), \
             patch("app.services.ingestion.insert_chunks", return_value=fake_ids), \
             patch("app.services.ingestion.delete_by_file_id") as mock_delete, \
             patch("app.services.ingestion.chunk_document") as mock_chunk:

            # Make chunk_document return two items, but then simulate Postgres failure
            # by making db.add_all raise after Qdrant insert
            mock_chunks = [MagicMock(page=1, chunk_index=i, text="text " * 10) for i in range(2)]
            mock_chunk.return_value = mock_chunks

            original_add_all = db.add_all

            def failing_add_all(items):
                raise RuntimeError("Simulated Postgres failure")

            db.add_all = failing_add_all

            run_ingestion(f.id, db)

            db.add_all = original_add_all  # restore

        mock_delete.assert_called_once_with(f.id)
        db.refresh(f)
        assert f.status == FileStatus.failed

    def test_nonexistent_file_id_is_noop(self, db):
        # Should return without raising
        run_ingestion("does-not-exist", db)


class TestIngestionStateTransitions:
    def test_status_is_processing_during_run(self, db):
        """Verify the file is marked processing before extraction begins."""
        f = _make_file(db)
        observed_statuses = []

        def fake_extract(path):
            db.refresh(f)
            observed_statuses.append(f.status)
            return _good_extraction()

        with patch("app.services.ingestion.extract_pages", side_effect=fake_extract), \
             patch("app.services.ingestion.get_embeddings", return_value=[[0.1]] * 20), \
             patch("app.services.ingestion.insert_chunks", return_value=[f"id-{i}" for i in range(20)]), \
             patch("app.services.ingestion.delete_by_file_id"):
            run_ingestion(f.id, db)

        assert FileStatus.processing in observed_statuses
