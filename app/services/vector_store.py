import logging
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

logger = logging.getLogger(__name__)

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


def ensure_collection(vector_size: int) -> bool:
    """
    Creates or migrates the Qdrant collection.
    Returns True if the collection was recreated (triggers auto re-index).
    """
    client = get_client()
    existing_names = [c.name for c in client.get_collections().collections]

    if settings.QDRANT_COLLECTION in existing_names:
        info = client.get_collection(settings.QDRANT_COLLECTION)
        vectors_config = info.config.params.vectors

        needs_recreate = False

        if isinstance(vectors_config, dict):
            existing_size = vectors_config.get("dense", {})
            size = getattr(existing_size, "size", None)
            if size != vector_size:
                logger.warning(
                    f"[vector_store] dimension mismatch "
                    f"(existing={size}, required={vector_size}) — recreating"
                )
                needs_recreate = True
        else:
            # Old unnamed single-vector schema — migrate
            if vectors_config.size != vector_size:
                logger.warning("[vector_store] dimension mismatch — recreating")
                needs_recreate = True
            else:
                # Same dimension but old unnamed schema — migrate to named
                logger.warning("[vector_store] migrating to named vector schema")
                needs_recreate = True

        if needs_recreate:
            client.delete_collection(settings.QDRANT_COLLECTION)
            existing_names = []

    collection_created = False
    if settings.QDRANT_COLLECTION not in existing_names:
        try:
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config={
                    "dense": VectorParams(size=vector_size, distance=Distance.COSINE),
                },
            )
            collection_created = True
            logger.info(f"[vector_store] created collection dim={vector_size}")
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
    return collection_created


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

    for chunk, dense_vec in zip(chunks, embeddings):
        qid = str(uuid.uuid4())
        qdrant_ids.append(qid)
        points.append(
            PointStruct(
                id=qid,
                vector={"dense": dense_vec},
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

    course_filter = Filter(
        must=[FieldCondition(key="course_id", match=MatchValue(value=course_id))]
    )

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
        # Fallback for collections without named vectors (old schema)
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
    existing_names = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in existing_names:
        return 0
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


def delete_by_qdrant_ids(qdrant_ids: list[str]) -> None:
    """Delete specific Qdrant points by their IDs. Used for targeted page re-OCR cleanup."""
    if not qdrant_ids:
        return
    from qdrant_client.models import PointIdsList
    client = get_client()
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=PointIdsList(points=qdrant_ids),
    )
