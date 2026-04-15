import logging
import os
import threading
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.tasks.executor import active_job_count, submit_ingest, MAX_CONCURRENT_JOBS

# Limit concurrent ocr/missing background jobs to the same cap as the main executor.
# Prevents unbounded thread spawning when multiple clients call ocr/missing in parallel.
_ocr_missing_semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)

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
    OCR progress (ocr_pages_done / ocr_pages_total) is served from an
    in-memory store while status="ocr" — no DB round-trip needed.
    """
    from app.services.ingestion import get_live_ocr_progress
    db_file = db.query(FileModel).filter(FileModel.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found.")
    response = FileStatusResponse.from_orm_file(db_file)
    progress = get_live_ocr_progress(file_id)
    if progress:
        response.ocr_pages_done, response.ocr_pages_total = progress
    return response


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

    # Clean old Qdrant vectors
    try:
        delete_by_file_id(file_id)
    except Exception:
        pass

    # Clean old Postgres chunk rows — without this, reindex doubles the row count
    from app.models.chunk import Chunk as ChunkModel
    db.query(ChunkModel).filter(ChunkModel.file_id == file_id).delete()

    db_file.status = FileStatus.uploaded
    db_file.error_message = None
    db_file.chunk_count = 0
    db_file.indexed_at = None
    db_file.empty_pages = None      # cleared by new ingestion run
    db_file.ocr_completed = False   # allow ocr/missing after reindex
    db.commit()

    if not submit_ingest(file_id):
        raise HTTPException(status_code=503, detail="Server busy. Try again in a moment.")

    return {"file_id": file_id, "status": "reindexing", "message": "Re-indexing started."}


@router.post("/{file_id}/ocr/page", status_code=200)
def ocr_page(
    file_id: str,
    body: dict,
    db: Session = Depends(get_db),
) -> dict:
    """
    OCR a single page on demand and merge it into the existing index.

    Body: {"page": 78}

    Synchronous — returns when OCR is complete (~3–5 seconds).
    Removes the page from the file's empty_pages list.
    Safe to call multiple times — re-OCRs the page if called again.
    """
    from app.services.ocr_service import ocr_single_page

    page_num = body.get("page")
    if not isinstance(page_num, int) or page_num < 1:
        raise HTTPException(status_code=400, detail="'page' must be a positive integer.")

    db_file = db.query(FileModel).filter(FileModel.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found.")
    if db_file.status in _IN_PROGRESS_STATUSES:
        raise HTTPException(status_code=409, detail="File is currently being indexed.")

    try:
        text = ocr_single_page(file_id, page_num, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "file_id": file_id,
        "page": page_num,
        "ocr_text_length": len(text),
        "indexed": bool(text.strip()),
        "message": (
            f"Page {page_num} OCR'd and added to index."
            if text.strip()
            else f"Page {page_num} OCR'd but produced no text."
        ),
    }


@router.post("/{file_id}/ocr/missing", status_code=202)
def ocr_missing(
    file_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Trigger background OCR for all pages that had no native text at index time.

    Returns 202 immediately. OCR runs in the background thread pool.
    When complete, file.ocr_completed is set to True.

    This endpoint is a no-op if ocr_completed is already True — call reindex
    first if you want to re-run OCR from scratch.
    """
    from app.services.ocr_service import ocr_missing_pages
    from app.tasks.executor import submit_ingest

    db_file = db.query(FileModel).filter(FileModel.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found.")
    if db_file.status in _IN_PROGRESS_STATUSES:
        raise HTTPException(status_code=409, detail="File is currently being indexed.")
    if db_file.ocr_completed:
        return {"file_id": file_id, "status": "already_completed", "message": "OCR was already run for this file."}

    pages = list(db_file.empty_pages or [])
    if not pages:
        db_file.ocr_completed = True
        db.commit()
        return {"file_id": file_id, "status": "nothing_to_ocr", "pages_count": 0}

    def _run_ocr_missing():
        if not _ocr_missing_semaphore.acquire(blocking=False):
            logger.warning(f"[files] ocr_missing rejected file_id={file_id} — semaphore full")
            return
        from app.database import SessionLocal
        _db = SessionLocal()
        try:
            ocr_missing_pages(file_id, _db)
        except Exception as e:
            logger.error(f"[files] ocr_missing background failed file_id={file_id}: {e}")
        finally:
            _db.close()
            _ocr_missing_semaphore.release()

    t = threading.Thread(target=_run_ocr_missing, daemon=True, name=f"ocr-missing-{file_id[:8]}")
    t.start()

    return {
        "file_id": file_id,
        "status": "started",
        "pages_count": len(pages),
        "message": f"OCR started for {len(pages)} pages. Poll GET /files/{file_id} to monitor.",
    }


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
