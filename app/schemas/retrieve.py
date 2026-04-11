from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    course_id: UUID
    top_k: Optional[int] = None


class ChunkResult(BaseModel):
    chunk_id: str
    file_id: UUID
    filename: str
    course_id: UUID
    page: int
    text: str
    score: float
    # True if score < RETRIEVAL_SCORE_THRESHOLD. The upstream agent decides
    # whether to use low-confidence results. This service does not filter them out.
    low_confidence: bool


class RetrieveResponse(BaseModel):
    results: List[ChunkResult]
