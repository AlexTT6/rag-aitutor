import collections
import logging
import threading
import time

from app.config import settings

logger = logging.getLogger(__name__)

_local_model = None

# Query embedding cache — avoids recomputing the same query twice.
# OrderedDict gives us cheap LRU eviction: when over MAX_SIZE,
# the oldest entry is dropped. Thread-safe via _cache_lock.
_cache: collections.OrderedDict = collections.OrderedDict()
_cache_lock = threading.Lock()
_CACHE_MAX_SIZE = 512

# Max texts per encoding batch. all-MiniLM-L6-v2 has a 256-token limit per
# input — sending hundreds of chunks at once spikes RAM. 32 is safe on Railway.
_EMBED_BATCH_SIZE = 32


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a list of texts in batches of _EMBED_BATCH_SIZE.
    Provider is determined by settings.EMBEDDING_PROVIDER.
    """
    if settings.EMBEDDING_PROVIDER == "openai":
        return _openai_embed(texts)
    return _local_embed(texts)


def embed_query(query: str) -> list[float]:
    """
    Embeds a single query string.
    Results are cached — repeated identical queries skip model inference.
    """
    with _cache_lock:
        if query in _cache:
            _cache.move_to_end(query)   # mark as recently used
            return _cache[query]

    vector = get_embeddings([query])[0]

    with _cache_lock:
        _cache[query] = vector
        if len(_cache) > _CACHE_MAX_SIZE:
            _cache.popitem(last=False)  # evict oldest entry

    return vector


def _openai_embed(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        last_err = None
        for attempt in range(3):
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small", input=batch
                )
                all_embeddings.extend([item.embedding for item in response.data])
                break
            except Exception as e:
                last_err = e
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"[embedder] OpenAI attempt {attempt+1} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"OpenAI embedding failed after 3 attempts: {last_err}")

    return all_embeddings


def _local_embed(texts: list[str]) -> list[list[float]]:
    """
    Encodes in batches of _EMBED_BATCH_SIZE to prevent RAM spikes.
    """
    model = _get_local_model()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i : i + _EMBED_BATCH_SIZE]
        all_embeddings.extend(model.encode(batch, show_progress_bar=False).tolist())

    return all_embeddings
