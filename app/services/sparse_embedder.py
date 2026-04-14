"""
BM25 sparse embeddings via fastembed.
Used for the keyword-match leg of hybrid search.
Model: Qdrant/bm25 — pure statistical, no neural network, tiny download (~2 MB).
"""
import logging
from typing import List

from qdrant_client.models import SparseVector

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import SparseTextEmbedding
        logger.info("[sparse_embedder] loading BM25 model...")
        _model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("[sparse_embedder] BM25 model ready")
    return _model


def get_sparse_embeddings(texts: List[str]) -> List[SparseVector]:
    """Compute BM25 sparse vectors for a list of texts."""
    model = _get_model()
    result = []
    for emb in model.embed(texts):
        result.append(SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        ))
    return result


def get_sparse_embedding(text: str) -> SparseVector:
    """Compute BM25 sparse vector for a single text."""
    return get_sparse_embeddings([text])[0]
