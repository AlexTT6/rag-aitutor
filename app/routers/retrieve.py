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

    logger.info(
        f"[retrieve:debug] QUERY "
        f"text={req.query!r} "
        f"course_id={req.course_id} "
        f"top_k={top_k}"
    )

    # Step 1: embed query
    query_vector = embed_query(req.query)
    if not query_vector:
        raise HTTPException(status_code=500, detail="Embedding failed.")

    # Step 2: vector search
    hits = search_chunks(query_vector, str(req.course_id), top_k)

    if not hits:
        logger.info(f"[retrieve] no results query='{req.query[:60]}'")
        return RetrieveResponse(results=[])

    logger.info(f"[retrieve:debug] RAW_HITS count={len(hits)}")

    # Step 3: build results, filter by threshold
    results: list[ChunkResult] = []

    for hit in hits:
        if hit.score < settings.RETRIEVAL_SCORE_THRESHOLD:
            continue

        payload = hit.payload or {}
        text = payload.get("text", "")
        if not text:
            logger.warning(f"[retrieve] hit {hit.id} has no text in payload — skipping")
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
        logger.info(
            f"[retrieve:debug] CHUNK "
            f"id={hit.id} "
            f"score={hit.score:.4f} "
            f"text={text[:200]!r}"
        )

    logger.info(f"[retrieve:debug] RESULTS_RETURNED count={len(results)}")

    if not results:
        logger.info(f"[retrieve] all below threshold query='{req.query[:60]}'")
        return RetrieveResponse(results=[])

    logger.info(
        f"[retrieve] {len(results)} results "
        f"best={max(r.score for r in results):.3f} "
        f"query='{req.query[:60]}'"
    )
    return RetrieveResponse(results=results)
