# AUTHORIZATION MODEL:
# This service does not enforce caller authorization.
# It assumes all requests originate from a trusted upstream agent that has
# already verified user identity and course enrollment.
# course_id filtering on retrieval ensures a query for course A never returns
# chunks from course B, but this is scoping, not access control.
# This service must not be exposed to the public internet without an API
# gateway or network-level access control in front of it.

# CONSISTENCY MODEL (Postgres + Qdrant):
# There are no cross-system transactions. Postgres is the source of truth
# for file and chunk metadata. Qdrant holds vectors with a payload copy of
# fields needed for retrieval. Consistency is maintained through ordering:
#   Insert: Qdrant first, then Postgres. On Postgres failure, Qdrant is
#           cleaned up as a compensating action.
#   Delete: Qdrant first, then Postgres. delete_by_file_id is idempotent.
# A brief window exists during insert where Qdrant has vectors but Postgres
# does not yet have the chunk rows. Vectors in this state are unreachable
# via the API and are cleaned up by the startup recovery job if the process
# crashes mid-ingestion.

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal, engine
from app.models import Base, File, FileStatus
from app.routers import courses, files, retrieve, ui
from app.services.vector_store import delete_by_file_id, ensure_collection
from app.tasks.ingest_task import ingest_task


def _recover_stuck_files(db: Session) -> list[str]:
    """
    Recovers two classes of stuck files on startup:

    1. "processing" — ingestion task started but the process crashed mid-pipeline.
       Partial Qdrant vectors are cleaned up and status is reset to "uploaded"
       so the file can be re-queued below.

    2. "uploaded" (stale) — background task was queued but the process crashed
       before the task ever started. Only files older than 60 seconds are
       recovered to avoid racing with a task that is legitimately about to run.

    Returns the file_ids that should be re-queued by the caller.
    """
    # Reset processing → uploaded
    stuck = db.query(File).filter(File.status == FileStatus.processing).all()
    for f in stuck:
        try:
            delete_by_file_id(str(f.id))
        except Exception:
            pass
        f.status = FileStatus.uploaded
        f.error_message = "Ingestion interrupted by server restart. Re-queued automatically."
    if stuck:
        db.commit()

    # Collect stale uploaded files (includes any just reset above)
    stale_cutoff = datetime.now(timezone.utc) - timedelta(seconds=60)
    orphaned = (
        db.query(File)
        .filter(File.status == FileStatus.uploaded, File.uploaded_at < stale_cutoff)
        .all()
    )
    return [str(f.id) for f in orphaned]


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    ensure_collection(vector_size=settings.EMBEDDING_DIM)

    db = SessionLocal()
    try:
        to_requeue = _recover_stuck_files(db)
    finally:
        db.close()

    # Re-queue outside the DB session — ingest_task opens its own session.
    import threading
    for file_id in to_requeue:
        threading.Thread(target=ingest_task, args=(file_id,), daemon=True).start()

    yield


app = FastAPI(
    title="RAG Backend Service",
    description=(
        "Document ingestion and retrieval service for an AI tutor system. "
        "Handles PDF upload, text extraction, embedding, and vector search. "
        "Does not generate answers or contain any tutor logic."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(files.router)
app.include_router(retrieve.router)
app.include_router(courses.router)
app.include_router(ui.router)


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}
