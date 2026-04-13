import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.services.chunker import Chunk

# Module-level singleton. Created once on first call to get_client() and
# reused for every subsequent operation. Avoids per-call connection overhead.
_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if settings.QDRANT_HOST == ":memory:":
            _client = QdrantClient(":memory:", timeout=settings.QDRANT_TIMEOUT)
        else:
            _client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=settings.QDRANT_TIMEOUT,
            )
    return _client


def ensure_collection(vector_size: int) -> None:
    client = get_client()
    existing_names = [c.name for c in client.get_collections().collections]

    if settings.QDRANT_COLLECTION not in existing_names:
        try:
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        except Exception:
            # Another startup call may have won the race and already created
            # the collection. Re-check before propagating the error.
            existing_names = [c.name for c in client.get_collections().collections]
            if settings.QDRANT_COLLECTION not in existing_names:
                raise

    # create_payload_index is idempotent — safe to call on every startup.
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION,
        field_name="course_id",
        field_schema="keyword",
    )
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION,
        field_name="file_id",
        field_schema="keyword",
    )


def insert_chunks(
    file_id: str,
    course_id: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    filename: str = "",
) -> list[str]:
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks and embeddings must have the same length; "
            f"got {len(chunks)} chunks and {len(embeddings)} embeddings."
        )

    ensure_collection(len(embeddings[0]))
    client = get_client()
    points: list[PointStruct] = []
    qdrant_ids: list[str] = []

    for chunk, vector in zip(chunks, embeddings):
        qid = str(uuid.uuid4())
        qdrant_ids.append(qid)
        points.append(
            PointStruct(
                id=qid,
                vector=vector,
                payload={
                    "chunk_id": qid,
                    "file_id": file_id,
                    "course_id": course_id,
                    "filename": filename,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                },
            )
        )

    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points[i : i + batch_size],
        )

    return qdrant_ids


def search_chunks(
    query_vector: list[float],
    course_id: str,
    top_k: int,
) -> list:
    client = get_client()
    existing_names = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in existing_names:
        return []
    result = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="course_id", match=MatchValue(value=course_id))]
        ),
        limit=top_k,
        with_payload=True,
    )
    return result.points


def count_by_file_id(file_id: str) -> int:
    client = get_client()
    result = client.count(
        collection_name=settings.QDRANT_COLLECTION,
        count_filter=Filter(
            must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
        ),
        exact=True,
    )
    return result.count


def delete_by_file_id(file_id: str) -> None:
    # Accepts a plain str. Never pass an ORM object — it may be detached.
    # Idempotent: deleting a file_id with no matching vectors is a no-op.
    # Does not return a count; call count_by_file_id() first if needed.
    client = get_client()
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
            )
        ),
    )
