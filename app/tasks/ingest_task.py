import concurrent.futures
import logging

from app.database import SessionLocal
from app.models.file import File, FileStatus
from app.services.ingestion import run_ingestion
from app.tasks.executor import INGESTION_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


def _mark_failed(file_id: str, message: str) -> None:
    db = SessionLocal()
    try:
        f = db.query(File).filter(File.id == file_id).first()
        if f:
            f.status = FileStatus.failed
            f.error_message = message[:500]
            db.commit()
    except Exception as e:
        logger.error(f"[ingest_task] could not mark failed file_id={file_id}: {e}")
    finally:
        db.close()


def _run_ingestion_with_session(file_id: str) -> None:
    """Opens its own DB session — safe to run in a thread."""
    db = SessionLocal()
    try:
        run_ingestion(file_id, db)
    finally:
        db.close()


def ingest_task(file_id: str) -> None:
    """
    Runs ingestion with a hard timeout.
    Called by executor.py — already running in a background thread.
    Uses an inner executor only to enforce the timeout via future.result().
    """
    logger.info(f"[ingest_task] START file_id={file_id}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as inner:
        future = inner.submit(_run_ingestion_with_session, file_id)
        try:
            future.result(timeout=INGESTION_TIMEOUT_SECONDS)
            logger.info(f"[ingest_task] DONE file_id={file_id}")
        except concurrent.futures.TimeoutError:
            msg = f"Processing timed out after {INGESTION_TIMEOUT_SECONDS}s."
            logger.error(f"[ingest_task] TIMEOUT file_id={file_id}")
            _mark_failed(file_id, msg)
        except Exception as e:
            logger.exception(f"[ingest_task] FAILED file_id={file_id}: {e}")
            _mark_failed(file_id, str(e))
