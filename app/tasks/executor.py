"""
Module-level bounded thread pool for ingestion jobs.

Why not BackgroundTasks:
  BackgroundTasks runs inside uvicorn's worker thread — a slow PDF blocks
  all incoming API requests on a single-worker server.

Why not Celery/Redis:
  Overkill for a single Railway container. A ThreadPoolExecutor with
  max_workers=2 gives load protection and true background execution
  with zero extra infrastructure.

Restart safety:
  main.py lifespan re-queues any "processing" or stale "uploaded" files
  on startup, so crashed jobs are automatically recovered.
"""

import concurrent.futures
import logging
import threading

logger = logging.getLogger(__name__)

# Hard cap: at most 2 PDFs process at the same time.
# Prevents CPU/RAM exhaustion on Railway's small container.
MAX_CONCURRENT_JOBS = 2
INGESTION_TIMEOUT_SECONDS = 120

_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_JOBS,
    thread_name_prefix="ingest",
)
_active = 0
_lock = threading.Lock()


def active_job_count() -> int:
    return _active


def submit_ingest(file_id: str) -> bool:
    """
    Submit a file for background ingestion.
    Returns True if accepted, False if the pool is at capacity.
    """
    global _active
    with _lock:
        if _active >= MAX_CONCURRENT_JOBS:
            logger.warning(
                f"[executor] rejected file_id={file_id} — "
                f"{_active}/{MAX_CONCURRENT_JOBS} slots in use"
            )
            return False
        _active += 1

    _executor.submit(_run, file_id)
    logger.info(f"[executor] queued file_id={file_id} active={_active}/{MAX_CONCURRENT_JOBS}")
    return True


def _run(file_id: str) -> None:
    global _active
    try:
        from app.tasks.ingest_task import ingest_task
        ingest_task(file_id)
    except Exception as e:
        logger.exception(f"[executor] unhandled error file_id={file_id}: {e}")
    finally:
        with _lock:
            _active -= 1
        logger.info(f"[executor] finished file_id={file_id} active={_active}/{MAX_CONCURRENT_JOBS}")


def shutdown() -> None:
    _executor.shutdown(wait=False, cancel_futures=True)
