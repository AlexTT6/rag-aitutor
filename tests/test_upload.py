"""
Tests for POST /files/upload, GET /files/{file_id}, DELETE /files/{file_id}.
"""
import io
import os
from unittest.mock import patch, MagicMock

import pytest

from app.models.course import Course
from app.models.file import FileStatus


def _make_course(db) -> str:
    course = Course(name="Test Course")
    db.add(course)
    db.commit()
    db.refresh(course)
    return course.id


def _minimal_pdf_bytes() -> bytes:
    # A valid single-page PDF with no extractable text (just the structure).
    # Sufficient to pass the file-type check and reach the ingestion task.
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f\n"
        b"0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
    )


class TestUpload:
    def test_upload_returns_202(self, client, db):
        course_id = _make_course(db)
        with patch("app.routers.files.ingest_task"), \
             patch("builtins.open", MagicMock()), \
             patch("os.makedirs"):
            response = client.post(
                "/files/upload",
                data={"course_id": course_id},
                files={"file": ("lecture.pdf", _minimal_pdf_bytes(), "application/pdf")},
            )
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == FileStatus.uploaded
        assert "file_id" in body

    def test_upload_unknown_course_returns_404(self, client):
        response = client.post(
            "/files/upload",
            data={"course_id": "nonexistent-uuid"},
            files={"file": ("lecture.pdf", _minimal_pdf_bytes(), "application/pdf")},
        )
        assert response.status_code == 404

    def test_upload_non_pdf_returns_400(self, client, db):
        course_id = _make_course(db)
        response = client.post(
            "/files/upload",
            data={"course_id": course_id},
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_upload_oversized_file_returns_400(self, client, db):
        course_id = _make_course(db)
        # Patch MAX_FILE_SIZE_MB to 0 to trigger the limit with a small file
        with patch("app.routers.files.settings") as mock_settings:
            mock_settings.MAX_FILE_SIZE_MB = 0
            mock_settings.STORAGE_PATH = "./storage"
            # Recalculate the cap inside the route — patch the module-level constant
        with patch("app.routers.files._MAX_SIZE_BYTES", 1):
            response = client.post(
                "/files/upload",
                data={"course_id": course_id},
                files={"file": ("big.pdf", b"%PDF" + b"x" * 100, "application/pdf")},
            )
        assert response.status_code == 400
        assert "size limit" in response.json()["detail"]


class TestGetFileStatus:
    def test_get_existing_file(self, client, db):
        from app.models.file import File
        course_id = _make_course(db)
        f = File(
            course_id=course_id,
            filename="test.pdf",
            storage_path="/tmp/test.pdf",
            size_bytes=1024,
            status=FileStatus.indexed,
            chunk_count=10,
        )
        db.add(f)
        db.commit()

        response = client.get(f"/files/{f.id}")
        assert response.status_code == 200
        body = response.json()
        assert body["file_id"] == f.id
        assert body["status"] == FileStatus.indexed
        assert body["chunk_count"] == 10

    def test_get_nonexistent_file_returns_404(self, client):
        response = client.get("/files/does-not-exist")
        assert response.status_code == 404


class TestDeleteFile:
    def test_delete_indexed_file(self, client, db, tmp_path):
        from app.models.file import File
        course_id = _make_course(db)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-test")

        f = File(
            course_id=course_id,
            filename="test.pdf",
            storage_path=str(pdf),
            size_bytes=9,
            status=FileStatus.indexed,
            chunk_count=5,
        )
        db.add(f)
        db.commit()

        with patch("app.routers.files.delete_by_file_id"):
            response = client.delete(f"/files/{f.id}")

        assert response.status_code == 200
        assert response.json()["deleted"] is True
        assert not pdf.exists()

    def test_delete_processing_file_returns_409(self, client, db):
        from app.models.file import File
        course_id = _make_course(db)
        f = File(
            course_id=course_id,
            filename="test.pdf",
            storage_path="/tmp/x.pdf",
            size_bytes=100,
            status=FileStatus.processing,
        )
        db.add(f)
        db.commit()

        response = client.delete(f"/files/{f.id}")
        assert response.status_code == 409

    def test_delete_nonexistent_file_returns_404(self, client):
        response = client.delete("/files/nonexistent")
        assert response.status_code == 404
