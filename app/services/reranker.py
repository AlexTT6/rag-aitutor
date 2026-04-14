"""
Cross-encoder reranker.
Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80 MB, included in sentence-transformers).
Takes a query + list of candidate chunks, returns them re-ordered by relevance.
Runs locally — no API cost, adds ~100-200ms per request.
"""
import logging
from typing import Any, List

import numpy as np

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        logger.info("[reranker] loading cross-encoder model...")
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("[reranker] cross-encoder ready")
    return _model


def rerank(query: str, hits: List[Any], top_k: int) -> List[Any]:
    """
    Re-ranks Qdrant hits using a cross-encoder.

    Args:
        query: the user's search query
        hits:  list of Qdrant ScoredPoint objects (must have .payload["text"])
        top_k: how many top results to return

    Returns:
        top_k hits sorted by cross-encoder relevance score (best first).
        Original Qdrant scores on hits are unchanged (only ordering changes).
    """
    if not hits:
        return hits
    if len(hits) <= top_k:
        # No need to rerank if we have fewer candidates than requested
        return hits

    model = _get_model()
    texts = [hit.payload.get("text", "") for hit in hits]
    pairs = [(query, text) for text in texts]

    scores = model.predict(pairs)  # returns numpy array of logit scores

    ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
    top = [hit for _, hit in ranked[:top_k]]

    best = float(np.max(scores))
    logger.info(f"[reranker] reranked {len(hits)} → {len(top)}, best_logit={best:.3f}")
    return top
