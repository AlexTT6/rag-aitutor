import collections
import logging
import threading
import time

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Query embedding cache — avoids re-embedding identical queries.
# OrderedDict gives cheap LRU eviction: when over MAX_SIZE, oldest entry is dropped.
_cache: collections.OrderedDict = collections.OrderedDict()
_cache_lock = threading.Lock()
_CACHE_MAX_SIZE = 512


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generates embeddings for a list of texts in batches of 100."""
    return _openai_embed(texts)


def embed_query(query: str) -> list[float]:
    """Embeds a single query. Results are cached to skip repeated API calls."""
    with _cache_lock:
        if query in _cache:
            _cache.move_to_end(query)
            return _cache[query]

    vector = get_embeddings([query])[0]

    with _cache_lock:
        _cache[query] = vector
        if len(_cache) > _CACHE_MAX_SIZE:
            _cache.popitem(last=False)

    return vector


_openai_client: OpenAI | None = None
_openai_client_lock = threading.Lock()


def _get_openai_client() -> OpenAI:
    """Returns a module-level singleton OpenAI client.

    Creating OpenAI() on every call sets up a new httpx session + SSL handshake.
    Reusing one client keeps the HTTP/2 connection alive between requests.
    Thread-safe: the underlying httpx session is designed for concurrent use.
    """
    global _openai_client
    if _openai_client is None:
        with _openai_client_lock:
            if _openai_client is None:
                _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


def _openai_embed(texts: list[str]) -> list[list[float]]:
    client = _get_openai_client()
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
                wait = 2 ** attempt
                logger.warning(f"[embedder] attempt {attempt+1} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"OpenAI embedding failed after 3 attempts: {last_err}")

    return all_embeddings
