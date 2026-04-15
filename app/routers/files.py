import logging
import os
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.tasks.executor import active_job_count, submit_ingest, MAX_CONCURRENT_JOBS

logger = logging.getLogger(__name__)

from app.config import settings
from app.database import get_db
from app.models.course import Course
from app.models.file import File as FileModel, FileStatus
from app.schemas.file import FileDeleteResponse, FileStatusResponse, FileUploadResponse
from app.services.vector_store import delete_by_file_id

router = APIRouter(prefix="/files", tags=["files"])

_MAX_SIZE_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


@router.post("/upload", response_model=FileUploadResponse, status_code=202)
async def upload_file(
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

    logger.info(
        f"[upload] saved file_id={file_id} "
        f"filename={file.filename} "
        f"size_bytes={len(contents)} "
        f"course_id={course_id}"
    )

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

    # Submit to bounded thread pool — rejects if already at MAX_CONCURRENT_JOBS.
    if not submit_ingest(file_id):
        # Clean up the file we just saved since we can't process it now.
        try:
            os.remove(dest_path)
        except OSError:
            pass
        db.delete(db_file)
        db.commit()
        raise HTTPException(
            status_code=503,
            detail=f"Server is busy processing other files ({MAX_CONCURRENT_JOBS} max). Try again in a moment.",
        )

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


_IN_PROGRESS_STATUSES = {
    FileStatus.processing,
    FileStatus.extracting,
    FileStatus.ocr,
    FileStatus.chunking,
    FileStatus.embedding,
}


@router.post("/{file_id}/reindex", status_code=202)
def reindex_file(
    file_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """Re-runs ingestion on an already-uploaded file. Useful after OCR improvements."""
    from app.services.vector_store import delete_by_file_id
    db_file = db.query(FileModel).filter(FileModel.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found.")
    if db_file.status in _IN_PROGRESS_STATUSES:
        raise HTTPException(status_code=409, detail=f"File is currently being processed (status={db_file.status}).")
    if not os.path.exists(db_file.storage_path):
        raise HTTPException(status_code=404, detail="Original file not found on disk. Please re-upload.")

    # Clean old vectors
    try:
        delete_by_file_id(file_id)
    except Exception:
        pass

    db_file.status = FileStatus.uploaded
    db_file.error_message = None
    db_file.chunk_count = 0
    db_file.indexed_at = None
    db.commit()

    if not submit_ingest(file_id):
        raise HTTPException(status_code=503, detail="Server busy. Try again in a moment.")

    return {"file_id": file_id, "status": "reindexing", "message": "Re-indexing started."}


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

    if db_file.status in _IN_PROGRESS_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"File is currently being processed (status={db_file.status}). Try again after ingestion completes.",
        )

    # Delete Qdrant vectors first. If this succeeds and Postgres delete fails,
    # the file record remains and the next delete attempt will re-run this safely
    # (delete_by_file_id is idempotent — deleting an already-empty filter is a no-op).
    delete_by_file_id(file_id)

    storage_path = db_file.storage_path
    db.delete(db_file)
    db.commit()

    # Delete the PDF and OCR cache from disk after committing the Postgres delete.
    # If this fails, the file record is already gone — orphaned files on disk
    # are harmless and can be cleaned up by a maintenance job.
    for path in (storage_path, storage_path.replace(".pdf", "_ocr_cache.json")):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass  # log in production; do not fail the response

    return FileDeleteResponse(file_id=file_id, deleted=True)
