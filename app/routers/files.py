import os
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.course import Course
from app.models.file import File as FileModel, FileStatus
from app.schemas.file import FileDeleteResponse, FileStatusResponse, FileUploadResponse
from app.services.vector_store import delete_by_file_id
from app.tasks.ingest_task import ingest_task

router = APIRouter(prefix="/files", tags=["files"])

_MAX_SIZE_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


@router.post("/upload", response_model=FileUploadResponse, status_code=202)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    course_id: str = Form(...),
    db: Session = Depends(get_db),
) -> FileUploadResponse:
    """
    Accepts a PDF upload for a given course_id.

    Validates size and type synchronously, saves the file to disk,
    creates a database record, then dispatches ingestion as a background task.
    Returns immediately with status="uploaded" — the caller must poll
    GET /files/{file_id} to observe the transition to "indexed" or "failed".

    Size is checked against Content-Length first (fast, no read required),
    then verified against the actual byte count after reading to guard against
    clients that omit or lie about the header.
    """
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found.")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Fast pre-check: reject immediately if Content-Length is declared and too large.
    # Avoids reading the body at all for obviously oversized uploads.
    content_length = file.size  # set by FastAPI/Starlette from Content-Length header
    if content_length is not None and content_length > _MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the {settings.MAX_FILE_SIZE_MB} MB size limit.",
        )

    contents = await file.read()

    # Authoritative check on actual byte count — guards against missing/wrong header.
    if len(contents) > _MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the {settings.MAX_FILE_SIZE_MB} MB size limit.",
        )

    file_id = str(uuid.uuid4())
    os.makedirs(settings.STORAGE_PATH, exist_ok=True)
    dest_path = os.path.join(settings.STORAGE_PATH, f"{file_id}.pdf")

    with open(dest_path, "wb") as f:
        f.write(contents)

    db_file = FileModel(
        id=file_id,
        course_id=course_id,
        filename=file.filename,
        storage_path=dest_path,
        size_bytes=len(contents),
        status=FileStatus.uploaded,
    )
    db.add(db_file)
    db.commit()

    # Pass only the file_id string — never the db session or ORM object.
    # The task opens its own session (see tasks/ingest_task.py).
    background_tasks.add_task(ingest_task, file_id)

    return FileUploadResponse(
        file_id=file_id,
        course_id=course_id,
        status=FileStatus.uploaded,
        message="File accepted. Indexing in progress. Poll GET /files/{file_id} for status.",
    )


@router.get("/{file_id}", response_model=FileStatusResponse)
def get_file_status(
    file_id: str,
    db: Session = Depends(get_db),
) -> FileStatusResponse:
    """
    Returns the current status and metadata for a file.
    Used by the agent to poll ingestion progress.
    """
    db_file = db.query(FileModel).filter(FileModel.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found.")
    return FileStatusResponse.from_orm_file(db_file)


@router.delete("/{file_id}", response_model=FileDeleteResponse)
def delete_file(
    file_id: str,
    db: Session = Depends(get_db),
) -> FileDeleteResponse:
    """
    Deletes a file and all associated data:
      1. Qdrant vectors (by file_id payload filter)
      2. PDF from disk
      3. File row from Postgres (cascades to chunk rows)

    Rejected while status="processing" to avoid partial-delete races
    with the ingestion task.
    """
    db_file = db.query(FileModel).filter(FileModel.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found.")

    if db_file.status == FileStatus.processing:
        raise HTTPException(
            status_code=409,
            detail="File is currently being processed. Try again after ingestion completes.",
        )

    # Delete Qdrant vectors first. If this succeeds and Postgres delete fails,
    # the file record remains and the next delete attempt will re-run this safely
    # (delete_by_file_id is idempotent — deleting an already-empty filter is a no-op).
    delete_by_file_id(file_id)

    storage_path = db_file.storage_path
    db.delete(db_file)
    db.commit()

    # Delete the PDF from disk after committing the Postgres delete.
    # If this fails, the file record is already gone — the orphaned file on disk
    # is harmless and can be cleaned up by a maintenance job.
    if os.path.exists(storage_path):
        try:
            os.remove(storage_path)
        except OSError:
            pass  # log in production; do not fail the response

    return FileDeleteResponse(file_id=file_id, deleted=True)
