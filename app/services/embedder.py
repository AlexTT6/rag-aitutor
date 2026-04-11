from app.config import settings

# Module-level cache for the local sentence-transformers model.
# SentenceTransformer("all-MiniLM-L6-v2") loads weights from disk on
# construction — reloading on every call would be extremely slow.
# This is only initialised when EMBEDDING_PROVIDER == "local".
_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a list of texts.
    Provider is determined by settings.EMBEDDING_PROVIDER.

    Both providers must produce vectors of the same dimension as the
    Qdrant collection was created with. Switching providers after the
    collection exists requires dropping the collection and re-indexing
    every file from scratch.
    """
    if settings.EMBEDDING_PROVIDER == "openai":
        return _openai_embed(texts)
    return _local_embed(texts)


def embed_query(query: str) -> list[float]:
    """Convenience wrapper for embedding a single query string."""
    return get_embeddings([query])[0]


def _openai_embed(texts: list[str]) -> list[list[float]]:
    """
    Uses OpenAI text-embedding-3-small (1536 dimensions).
    Batches in groups of 100 to stay within the API per-request input limit.
    Raises openai.APIError on network or auth failures — caller handles these.
    """
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    all_embeddings: list[list[float]] = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


def _local_embed(texts: list[str]) -> list[list[float]]:
    """
    Uses all-MiniLM-L6-v2 via sentence-transformers (384 dimensions).
    No external API calls. Suitable for development or air-gapped environments.
    Significantly slower than OpenAI on CPU for large batches.
    The model is loaded once and cached at module level (_get_local_model).
    """
    model = _get_local_model()
    return model.encode(texts, show_progress_bar=False).tolist()
