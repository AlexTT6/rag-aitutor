import concurrent.futures
import logging

from app.database import SessionLocal
from app.models.file import File, FileStatus
from app.services.ingestion import run_ingestion

logger = logging.getLogger(__name__)

# Max seconds any ingestion job is allowed to run before being killed
INGESTION_TIMEOUT_SECONDS = 120


def _run_with_own_session(file_id: str) -> None:
    """Runs ingestion in its own DB session — safe to execute in a thread."""
    db = SessionLocal()
    try:
        run_ingestion(file_id, db)
    finally:
        db.close()


def _mark_failed(file_id: str, message: str) -> None:
    """Opens a fresh session and marks the file as failed."""
    db = SessionLocal()
    try:
        file = db.query(File).filter(File.id == file_id).first()
        if file:
            file.status = FileStatus.failed
            file.error_message = message[:500]
            db.commit()
    except Exception as e:
        logger.error(f"Could not mark file {file_id} as failed: {e}")
    finally:
        db.close()


def ingest_task(file_id: str) -> None:
    """
    Entry point for FastAPI BackgroundTasks.

    Runs ingestion in a ThreadPoolExecutor with a hard timeout.
    If processing exceeds INGESTION_TIMEOUT_SECONDS the file is marked failed.
    """
    logger.info(f"[ingest_task] START file_id={file_id}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_with_own_session, file_id)
        try:
            future.result(timeout=INGESTION_TIMEOUT_SECONDS)
            logger.info(f"[ingest_task] DONE  file_id={file_id}")
        except concurrent.futures.TimeoutError:
            msg = f"Processing timed out after {INGESTION_TIMEOUT_SECONDS} seconds."
            logger.error(f"[ingest_task] TIMEOUT file_id={file_id} — {msg}")
            _mark_failed(file_id, msg)
        except Exception as e:
            logger.exception(f"[ingest_task] FAILED file_id={file_id}: {e}")
            _mark_failed(file_id, str(e))
