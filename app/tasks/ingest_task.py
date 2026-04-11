# ARCHITECTURE NOTE:
# FastAPI BackgroundTasks is used here for MVP simplicity.
# It requires no additional infrastructure but has real limitations:
#   - Runs in the same process as the web server. A slow ingestion job
#     (large PDF, slow embedding API) ties up a Uvicorn worker thread.
#   - No retry logic, no queue visibility, no task state beyond what we
#     write to the database ourselves.
#   - A process crash loses any queued tasks that have not started yet.
#     The startup recovery in main.py mitigates this by re-queuing files
#     stuck in "processing" status, but it cannot recover tasks that were
#     queued but never started.
#   - Does not scale horizontally without coordination across workers.
#
# Migration path when ready:
#   Replace the background_tasks.add_task(ingest_task, file_id) call in
#   routers/files.py with an enqueue call to Celery + Redis or ARQ.
#   This function's signature stays identical — no other changes needed.

from app.database import SessionLocal
from app.services.ingestion import run_ingestion


def ingest_task(file_id: str) -> None:
    """
    Entry point for FastAPI BackgroundTasks.

    Opens its own DB session. Never reuses the request-scoped session
    from the upload endpoint — that session is closed before this task runs.
    """
    db = SessionLocal()
    try:
        run_ingestion(file_id, db)
    finally:
        db.close()
