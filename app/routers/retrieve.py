import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.schemas.retrieve import ChunkResult, RetrieveRequest, RetrieveResponse
from app.services.embedder import embed_query
from app.services.vector_store import search_chunks

router = APIRouter(tags=["retrieve"])
logger = logging.getLogger(__name__)


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    req: RetrieveRequest,
    db: Session = Depends(get_db),
) -> RetrieveResponse:
    top_k = req.top_k if req.top_k is not None else settings.TOP_K_DEFAULT

    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1.")
    if top_k > settings.TOP_K_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"top_k cannot exceed {settings.TOP_K_MAX}.",
        )

    # Embed the query
    query_vector = embed_query(req.query)
    if not query_vector:
        raise HTTPException(status_code=500, detail="Embedding failed.")

    hits = search_chunks(query_vector, str(req.course_id), top_k=top_k)

    if not hits:
        logger.info(f"[retrieve] no results query='{req.query[:60]}' course={req.course_id}")
        return RetrieveResponse(results=[])

    # If ALL results are below the confidence threshold — return empty.
    # Sending low-quality chunks to the LLM causes hallucinations.
    best_score = max(h.score for h in hits)
    if best_score < settings.RETRIEVAL_SCORE_THRESHOLD:
        logger.info(
            f"[retrieve] all scores below threshold ({best_score:.3f} < "
            f"{settings.RETRIEVAL_SCORE_THRESHOLD}) — returning empty "
            f"query='{req.query[:60]}'"
        )
        return RetrieveResponse(results=[])

    # Build results directly from Qdrant payload — no Postgres JOIN needed.
    # filename, text, page are stored in the payload at index time.
    # Fall back to Postgres only for chunks indexed before this change
    # (they won't have filename in payload).
    results: list[ChunkResult] = []
    missing_from_payload: list[str] = []

    for hit in hits:
        score = hit.score
        if score < settings.RETRIEVAL_SCORE_THRESHOLD:
            continue  # skip individual low-confidence results

        payload = hit.payload or {}
        text = payload.get("text", "")
        filename = payload.get("filename", "")
        page = payload.get("page", 0)
        file_id = payload.get("file_id", "")
        course_id = payload.get("course_id", str(req.course_id))

        if not text:
            # Old vector without text in payload — need DB fallback
            missing_from_payload.append(str(hit.id))
            continue

        results.append(
            ChunkResult(
                chunk_id=str(hit.id),
                file_id=file_id,
                filename=filename,
                course_id=course_id,
                page=page,
                text=text,
                score=round(score, 4),
                low_confidence=False,  # already filtered above
            )
        )

    # Fallback for old vectors without payload text (backwards compatibility)
    if missing_from_payload:
        from sqlalchemy.orm import joinedload
        from app.models.chunk import Chunk

        db_chunks = (
            db.query(Chunk)
            .options(joinedload(Chunk.file))
            .filter(Chunk.qdrant_id.in_(missing_from_payload))
            .all()
        )
        score_map = {str(h.id): h.score for h in hits}
        for chunk in db_chunks:
            score = score_map.get(chunk.qdrant_id, 0.0)
            if score < settings.RETRIEVAL_SCORE_THRESHOLD:
                continue
            results.append(
                ChunkResult(
                    chunk_id=chunk.qdrant_id,
                    file_id=str(chunk.file_id),
                    filename=chunk.file.filename,
                    course_id=str(chunk.course_id),
                    page=chunk.page,
                    text=chunk.text,
                    score=round(score, 4),
                    low_confidence=False,
                )
            )

    # Keep original Qdrant ranking order
    qdrant_order = {str(h.id): i for i, h in enumerate(hits)}
    results.sort(key=lambda r: qdrant_order.get(r.chunk_id, 999))

    logger.info(
        f"[retrieve] returned {len(results)} chunks "
        f"best_score={best_score:.3f} query='{req.query[:60]}'"
    )
    return RetrieveResponse(results=results)
