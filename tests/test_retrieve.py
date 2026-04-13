"""
Tests for POST /retrieve.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.models.course import Course
from app.models.file import File, FileStatus
from app.models.chunk import Chunk


def _make_course(db, name="Test Course") -> str:
    course = Course(name=name)
    db.add(course)
    db.commit()
    db.refresh(course)
    return course.id


def _make_file(db, course_id: str) -> File:
    f = File(
        course_id=course_id,
        filename="lecture.pdf",
        storage_path="/tmp/lecture.pdf",
        size_bytes=1024,
        status=FileStatus.indexed,
        chunk_count=1,
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


def _make_chunk(db, file: File, text: str, page: int = 1, score_qdrant_id: str = None) -> Chunk:
    import uuid
    qid = score_qdrant_id or str(uuid.uuid4())
    chunk = Chunk(
        file_id=file.id,
        course_id=file.course_id,
        page=page,
        chunk_index=0,
        text=text,
        token_count=len(text.split()),
        qdrant_id=qid,
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk


class TestRetrieve:
    def test_returns_results_with_metadata(self, client, db):
        course_id = _make_course(db)
        f = _make_file(db, course_id)
        qid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        chunk = _make_chunk(db, f, "Gradient descent minimizes the loss function.", score_qdrant_id=qid)

        mock_hit = MagicMock()
        mock_hit.id = qid
        mock_hit.score = 0.92

        with patch("app.routers.retrieve.embed_query", return_value=[0.1] * 1536), \
             patch("app.routers.retrieve.search_chunks", return_value=[mock_hit]):
            response = client.post(
                "/retrieve",
                json={"query": "gradient descent", "course_id": course_id},
            )

        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 1
        r = results[0]
        assert r["chunk_id"] == qid
        assert r["file_id"] == f.id
        assert r["filename"] == "lecture.pdf"
        assert r["course_id"] == course_id
        assert r["page"] == 1
        assert r["score"] == 0.92
        assert r["low_confidence"] is False  # 0.92 > 0.70 threshold

    def test_low_confidence_flag_set_below_threshold(self, client, db):
        course_id = _make_course(db)
        f = _make_file(db, course_id)
        qid = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        _make_chunk(db, f, "Some weakly related text.", score_qdrant_id=qid)

        mock_hit = MagicMock()
        mock_hit.id = qid
        mock_hit.score = 0.45  # below 0.70

        with patch("app.routers.retrieve.embed_query", return_value=[0.1] * 1536), \
             patch("app.routers.retrieve.search_chunks", return_value=[mock_hit]):
            response = client.post(
                "/retrieve",
                json={"query": "something unrelated", "course_id": course_id},
            )

        assert response.status_code == 200
        r = response.json()["results"][0]
        assert r["low_confidence"] is True

    def test_empty_results_when_no_hits(self, client, db):
        course_id = _make_course(db)

        with patch("app.routers.retrieve.embed_query", return_value=[0.1] * 1536), \
             patch("app.routers.retrieve.search_chunks", return_value=[]):
            response = client.post(
                "/retrieve",
                json={"query": "anything", "course_id": course_id},
            )

        assert response.status_code == 200
        assert response.json()["results"] == []

    def test_top_k_exceeds_max_returns_400(self, client, db):
        course_id = _make_course(db)
        response = client.post(
            "/retrieve",
            json={"query": "test", "course_id": course_id, "top_k": 999},
        )
        assert response.status_code == 400

    def test_empty_query_returns_422(self, client, db):
        course_id = _make_course(db)
        response = client.post(
            "/retrieve",
            json={"query": "", "course_id": course_id},
        )
        assert response.status_code == 422
