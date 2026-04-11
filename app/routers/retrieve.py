from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from app.config import settings
from app.database import get_db
from app.models.chunk import Chunk
from app.schemas.retrieve import ChunkResult, RetrieveRequest, RetrieveResponse
from app.services.embedder import embed_query
from app.services.vector_store import search_chunks

router = APIRouter(tags=["retrieve"])


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
    hits = search_chunks(query_vector, str(req.course_id), top_k=top_k)

    if not hits:
        return RetrieveResponse(results=[])

    qdrant_ids = [str(h.id) for h in hits]
    score_map = {str(h.id): h.score for h in hits}

    # joinedload(Chunk.file) fetches all chunk rows and their parent file rows
    # in a single JOIN query, eliminating the N+1 problem that would arise from
    # accessing chunk.file.filename inside the loop without eager loading.
    chunks = (
        db.query(Chunk)
        .options(joinedload(Chunk.file))
        .filter(Chunk.qdrant_id.in_(qdrant_ids))
        .all()
    )
    chunk_map = {c.qdrant_id: c for c in chunks}

    results: list[ChunkResult] = []
    for qid in qdrant_ids:
        chunk = chunk_map.get(qid)
        if not chunk:
            # Qdrant has a vector with no matching Postgres chunk row.
            # Indicates a consistency issue. Skip silently; log in production.
            continue

        score = score_map[qid]
        results.append(
            ChunkResult(
                chunk_id=qid,
                file_id=str(chunk.file_id),
                filename=chunk.file.filename,
                course_id=str(chunk.course_id),
                page=chunk.page,
                text=chunk.text,
                score=round(score, 4),
                low_confidence=score < settings.RETRIEVAL_SCORE_THRESHOLD,
            )
        )

    return RetrieveResponse(results=results)
