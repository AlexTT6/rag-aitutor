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

import logging
import logging.config
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from app.config import settings
from app.database import SessionLocal, engine
from app.models import Base, File, FileStatus
from app.routers import courses, files, retrieve, ui
from app.services.vector_store import delete_by_file_id, ensure_collection, get_client
from app.tasks.executor import shutdown as shutdown_executor, submit_ingest


_IN_PROGRESS_STATUSES = {
    FileStatus.processing,
    FileStatus.extracting,
    FileStatus.ocr,
    FileStatus.chunking,
    FileStatus.embedding,
}


def _recover_stuck_files(db: Session) -> list[str]:
    """
    Recovers two classes of stuck files on startup:

    1. Any in-progress status — ingestion task started but the process crashed.
       Partial Qdrant vectors are cleaned up and status is reset to "uploaded"
       so the file can be re-queued below.

    2. "uploaded" (stale) — background task was queued but the process crashed
       before the task ever started. Only files older than 60 seconds are
       recovered to avoid racing with a task that is legitimately about to run.

    Returns the file_ids that should be re-queued by the caller.
    """
    # Reset all in-progress statuses → uploaded
    stuck = (
        db.query(File)
        .filter(File.status.in_([s.value for s in _IN_PROGRESS_STATUSES]))
        .all()
    )
    for f in stuck:
        try:
            delete_by_file_id(str(f.id))
        except Exception as e:
            logger.warning(f"[startup] Qdrant cleanup failed for file_id={f.id}: {e}")
        f.status = FileStatus.uploaded
        f.error_message = "Ingestion interrupted by server restart. Re-queued automatically."
    if stuck:
        db.commit()
        logger.info(f"[startup] reset {len(stuck)} stuck file(s) to uploaded")

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
    # --- DB schema ---
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("[startup] DB schema ready")
    except Exception as e:
        logger.error(f"[startup] DB schema creation failed: {e}")
        raise

    # --- Qdrant collection ---
    collection_recreated = False
    try:
        collection_recreated = ensure_collection(vector_size=settings.EMBEDDING_DIM)
        logger.info("[startup] Qdrant collection ready")
    except Exception as e:
        logger.error(f"[startup] Qdrant collection setup failed: {e}")
        raise

    # --- Pre-load local embedding model ---
    if settings.EMBEDDING_PROVIDER == "local":
        try:
            from app.services.embedder import _get_local_model
            _get_local_model()
        except Exception as e:
            logger.error(f"[startup] local model pre-load failed: {e}")

    # --- Pre-warm sparse embedder (downloads BM25 model if not cached) ---
    if settings.HYBRID_SEARCH:
        try:
            from app.services.sparse_embedder import _get_model
            _get_model()
            logger.info("[startup] sparse BM25 model ready")
        except Exception as e:
            logger.warning(f"[startup] sparse embedder pre-warm failed (non-fatal): {e}")

    # --- Pre-warm reranker (downloads cross-encoder model ~80MB if not cached) ---
    if settings.RERANKER_ENABLED:
        try:
            from app.services.reranker import _get_model
            _get_model()
            logger.info("[startup] reranker model ready")
        except Exception as e:
            logger.warning(f"[startup] reranker pre-warm failed (non-fatal): {e}")

    # --- Recover stuck files + auto re-index if collection was recreated ---
    to_requeue: list[str] = []
    try:
        db = SessionLocal()
        try:
            to_requeue = _recover_stuck_files(db)
            if collection_recreated:
                # Collection was dropped and recreated (schema migration).
                # All "indexed" files have lost their vectors — reset them to
                # "uploaded" so they are automatically re-indexed now.
                from app.models.file import File as _File
                orphaned = (
                    db.query(_File)
                    .filter(_File.status == FileStatus.indexed)
                    .all()
                )
                for f in orphaned:
                    f.status = FileStatus.uploaded
                    f.error_message = "Auto re-index after schema migration."
                if orphaned:
                    db.commit()
                    logger.info(
                        f"[startup] schema migration: queued {len(orphaned)} "
                        f"file(s) for automatic re-indexing"
                    )
                    to_requeue += [str(f.id) for f in orphaned]
        finally:
            db.close()
    except Exception as e:
        logger.error(f"[startup] file recovery failed (non-fatal): {e}")

    # Re-queue via executor — same bounded pool, respects MAX_CONCURRENT_JOBS.
    for file_id in to_requeue:
        submit_ingest(file_id)
        logger.info(f"[startup] re-queued file_id={file_id}")

    yield

    shutdown_executor()
    logger.info("[shutdown] ingestion executor stopped")


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(files.router)
app.include_router(retrieve.router)
app.include_router(courses.router)
app.include_router(ui.router)


@app.get("/config-check", tags=["debug"])
def config_check() -> dict:
    from app.services.vector_store import get_client
    client = get_client()
    try:
        info = client.get_collection(settings.QDRANT_COLLECTION)
        vector_count = info.points_count
        vector_dim = info.config.params.vectors.size
    except Exception as e:
        vector_count = f"error: {e}"
        vector_dim = "unknown"
    return {
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_dim": settings.EMBEDDING_DIM,
        "threshold": settings.RETRIEVAL_SCORE_THRESHOLD,
        "openai_key_set": bool(settings.OPENAI_API_KEY),
        "qdrant_vector_count": vector_count,
        "qdrant_vector_dim": vector_dim,
    }



@app.get("/health", tags=["health"])
def health() -> dict:
    # Check Postgres
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        return {"status": "unhealthy", "detail": f"Database: {str(e)}"}

    # Check Qdrant
    try:
        get_client().get_collections()
    except Exception as e:
        return {"status": "unhealthy", "detail": f"Qdrant: {str(e)}"}

    return {"status": "ok"}
