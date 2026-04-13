"""
Tests for POST /courses and GET /courses/{course_id}/files.
"""
from app.models.course import Course
from app.models.file import File, FileStatus


class TestCreateCourse:
    def test_create_returns_201(self, client):
        response = client.post("/courses", json={"name": "Linear Algebra"})
        assert response.status_code == 201
        body = response.json()
        assert body["name"] == "Linear Algebra"
        assert "course_id" in body

    def test_create_missing_name_returns_422(self, client):
        response = client.post("/courses", json={})
        assert response.status_code == 422


class TestListCourseFiles:
    def test_lists_files_for_course(self, client, db):
        course = Course(name="Calculus")
        db.add(course)
        db.commit()

        f = File(
            course_id=course.id,
            filename="week1.pdf",
            storage_path="/tmp/w1.pdf",
            size_bytes=500,
            status=FileStatus.indexed,
            chunk_count=8,
        )
        db.add(f)
        db.commit()

        response = client.get(f"/courses/{course.id}/files")
        assert response.status_code == 200
        body = response.json()
        assert body["course_id"] == course.id
        assert len(body["files"]) == 1
        assert body["files"][0]["filename"] == "week1.pdf"
        assert body["files"][0]["status"] == FileStatus.indexed

    def test_unknown_course_returns_404(self, client):
        response = client.get("/courses/nonexistent/files")
        assert response.status_code == 404

    def test_empty_course_returns_empty_list(self, client, db):
        course = Course(name="Empty Course")
        db.add(course)
        db.commit()

        response = client.get(f"/courses/{course.id}/files")
        assert response.status_code == 200
        assert response.json()["files"] == []
