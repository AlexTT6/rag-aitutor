import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    Fusion,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.config import settings
from app.services.chunker import Chunk

logger = logging.getLogger(__name__)

# Module-level singleton.
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
    """
    Creates the Qdrant collection with named dense + sparse vectors if it doesn't exist.
    Migrates from old single-vector schema automatically (requires re-indexing all files).
    """
    client = get_client()
    existing_names = [c.name for c in client.get_collections().collections]

    if settings.QDRANT_COLLECTION in existing_names:
        info = client.get_collection(settings.QDRANT_COLLECTION)
        vectors_config = info.config.params.vectors

        needs_recreate = False

        if isinstance(vectors_config, dict):
            # New named-vector schema — check "dense" dimension
            if "dense" not in vectors_config:
                logger.warning("[vector_store] 'dense' key missing — recreating collection")
                needs_recreate = True
            elif vectors_config["dense"].size != vector_size:
                logger.warning(
                    f"[vector_store] dimension mismatch "
                    f"(existing={vectors_config['dense'].size}, required={vector_size}) "
                    f"— recreating collection"
                )
                needs_recreate = True
        else:
            # Old unnamed single-vector schema — migrate to hybrid
            logger.warning(
                "[vector_store] old single-vector schema detected — "
                "migrating to hybrid schema. All files must be re-indexed."
            )
            needs_recreate = True

        if needs_recreate:
            client.delete_collection(settings.QDRANT_COLLECTION)
            existing_names = []

    if settings.QDRANT_COLLECTION not in existing_names:
        try:
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config={
                    "dense": VectorParams(size=vector_size, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                },
            )
            logger.info(
                f"[vector_store] created hybrid collection "
                f"dim={vector_size} sparse=BM25"
            )
        except Exception:
            existing_names = [c.name for c in client.get_collections().collections]
            if settings.QDRANT_COLLECTION not in existing_names:
                raise

    # Payload indexes — idempotent
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
    sparse_embeddings: Optional[list[SparseVector]] = None,
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

    for i, (chunk, dense_vec) in enumerate(zip(chunks, embeddings)):
        qid = str(uuid.uuid4())
        qdrant_ids.append(qid)

        if sparse_embeddings and i < len(sparse_embeddings):
            vector = {
                "dense": dense_vec,
                "sparse": sparse_embeddings[i],
            }
        else:
            vector = {"dense": dense_vec}

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
    sparse_query: Optional[SparseVector] = None,
) -> list:
    """
    Searches for relevant chunks.
    - If sparse_query is provided: hybrid search (dense + BM25) fused via RRF.
    - Otherwise: pure dense vector search (fallback).
    """
    client = get_client()
    existing_names = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in existing_names:
        return []

    course_filter = Filter(
        must=[FieldCondition(key="course_id", match=MatchValue(value=course_id))]
    )

    if sparse_query is not None:
        # Hybrid: prefetch from both dense and sparse, then RRF fusion.
        # Falls back to dense-only if server version doesn't support hybrid queries.
        try:
            result = client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                prefetch=[
                    Prefetch(
                        query=query_vector,
                        using="dense",
                        filter=course_filter,
                        limit=top_k * 2,
                    ),
                    Prefetch(
                        query=sparse_query,
                        using="sparse",
                        filter=course_filter,
                        limit=top_k * 2,
                    ),
                ],
                query=Fusion.RRF,
                limit=top_k,
                with_payload=True,
            )
            return result.points
        except Exception as e:
            logger.warning(
                f"[vector_store] hybrid search failed ({e.__class__.__name__}) "
                f"— falling back to dense-only"
            )

    # Dense-only (default or fallback).
    # Try named "dense" vector first; fall back to unnamed vector for old collections.
    try:
        result = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            using="dense",
            query_filter=course_filter,
            limit=top_k,
            with_payload=True,
        )
    except Exception:
        result = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            query_filter=course_filter,
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
    client = get_client()
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
            )
        ),
    )
