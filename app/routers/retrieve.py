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

# Dynamic retrieval: if top_k results are all below threshold,
# retry with a larger top_k before giving up.
_TOP_K_LADDER = [5, 10, 15]


def _search_with_fallback(
    query_vector: list,
    course_id: str,
    requested_top_k: int,
) -> tuple[list, bool]:
    """
    Tries top_k values in _TOP_K_LADDER (starting from requested_top_k).
    Returns (hits, all_low_confidence).
    Stops as soon as at least one result is above threshold.
    """
    ladder = sorted(set([requested_top_k] + _TOP_K_LADDER))

    for k in ladder:
        hits = search_chunks(query_vector, course_id, top_k=k)
        if not hits:
            continue

        best = max(h.score for h in hits)
        if best >= settings.RETRIEVAL_SCORE_THRESHOLD:
            logger.info(f"[retrieve] found confident results at top_k={k} best={best:.3f}")
            return hits, False

        logger.info(
            f"[retrieve] top_k={k} best_score={best:.3f} < "
            f"{settings.RETRIEVAL_SCORE_THRESHOLD} — trying larger top_k"
        )

    # All ladder steps exhausted — still low confidence
    return hits if hits else [], True


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

    query_vector = embed_query(req.query)
    if not query_vector:
        raise HTTPException(status_code=500, detail="Embedding failed.")

    hits, all_low = _search_with_fallback(query_vector, str(req.course_id), top_k)

    if not hits or all_low:
        logger.info(
            f"[retrieve] returning empty — all scores below threshold "
            f"query='{req.query[:60]}'"
        )
        return RetrieveResponse(results=[])

    # Build results from Qdrant payload — no Postgres JOIN needed.
    # Chunks indexed before the payload migration fall back to Postgres.
    results: list[ChunkResult] = []
    missing_from_payload: list[str] = []

    for hit in hits:
        if hit.score < settings.RETRIEVAL_SCORE_THRESHOLD:
            continue

        payload = hit.payload or {}
        text = payload.get("text", "")

        if not text:
            missing_from_payload.append(str(hit.id))
            continue

        results.append(
            ChunkResult(
                chunk_id=str(hit.id),
                file_id=payload.get("file_id", ""),
                filename=payload.get("filename", ""),
                course_id=payload.get("course_id", str(req.course_id)),
                page=payload.get("page", 0),
                text=text,
                score=round(hit.score, 4),
                low_confidence=False,
            )
        )

    # Postgres fallback for old vectors without payload text
    if missing_from_payload:
        from sqlalchemy.orm import joinedload
        from app.models.chunk import Chunk

        score_map = {str(h.id): h.score for h in hits}
        db_chunks = (
            db.query(Chunk)
            .options(joinedload(Chunk.file))
            .filter(Chunk.qdrant_id.in_(missing_from_payload))
            .all()
        )
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

    # Preserve Qdrant ranking order
    order = {str(h.id): i for i, h in enumerate(hits)}
    results.sort(key=lambda r: order.get(r.chunk_id, 999))

    logger.info(
        f"[retrieve] {len(results)} results "
        f"best={max((r.score for r in results), default=0):.3f} "
        f"query='{req.query[:60]}'"
    )
    return RetrieveResponse(results=results)
