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

    # Step 1: embed query
    query_vector = embed_query(req.query)
    if not query_vector:
        raise HTTPException(status_code=500, detail="Embedding failed.")

    # Step 2: fetch more candidates than needed (for reranker to pick from)
    fetch_k = max(top_k, settings.RERANKER_FETCH_K) if settings.RERANKER_ENABLED else top_k
    hits = search_chunks(query_vector, str(req.course_id), fetch_k)

    if not hits:
        logger.info(f"[retrieve] no results query='{req.query[:60]}'")
        return RetrieveResponse(results=[])

    # Step 3: rerank candidates with cross-encoder, return top_k
    if settings.RERANKER_ENABLED and len(hits) > top_k:
        try:
            from app.services.reranker import rerank
            hits = rerank(req.query, hits, top_k)
        except Exception as e:
            logger.warning(f"[retrieve] reranker failed, using original order: {e}")
            hits = hits[:top_k]
    else:
        hits = hits[:top_k]

    # Step 4: build results, filter by threshold
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
                low_confidence=hit.score < settings.RETRIEVAL_SCORE_THRESHOLD,
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
                    low_confidence=score < settings.RETRIEVAL_SCORE_THRESHOLD,
                )
            )

    if not results:
        logger.info(f"[retrieve] all below threshold query='{req.query[:60]}'")
        return RetrieveResponse(results=[])

    logger.info(
        f"[retrieve] {len(results)} results "
        f"best={max(r.score for r in results):.3f} "
        f"reranked={settings.RERANKER_ENABLED} "
        f"query='{req.query[:60]}'"
    )
    return RetrieveResponse(results=results)
