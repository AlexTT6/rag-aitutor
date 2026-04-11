from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.course import Course
from app.models.file import File as FileModel
from app.schemas.course import CourseCreateRequest, CourseResponse
from app.schemas.file import CourseFilesResponse, FileStatusResponse

router = APIRouter(prefix="/courses", tags=["courses"])


@router.post("", response_model=CourseResponse, status_code=201)
def create_course(
    req: CourseCreateRequest,
    db: Session = Depends(get_db),
) -> CourseResponse:
    """
    Creates a new course. Course creation is intentionally minimal —
    the agent or an admin tool calls this to register a course before
    uploading files to it.
    """
    course = Course(name=req.name)
    db.add(course)
    db.commit()
    db.refresh(course)
    return CourseResponse(
        course_id=course.id,
        name=course.name,
        created_at=course.created_at,
    )


@router.get("/{course_id}/files", response_model=CourseFilesResponse)
def list_course_files(
    course_id: str,
    db: Session = Depends(get_db),
) -> CourseFilesResponse:
    """
    Returns all files registered under a course, with their current status.
    The agent uses this to verify ingestion state before issuing retrieve calls.
    """
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found.")

    files = db.query(FileModel).filter(FileModel.course_id == course_id).all()

    return CourseFilesResponse(
        course_id=course_id,
        files=[FileStatusResponse.from_orm_file(f) for f in files],
    )
