"""
reranker.py -- Cross-encoder re-ranking module for retrieval quality improvement.

Uses BAAI/bge-reranker-base to re-rank retrieved documents by relevance.
Placed between retrieve and check_sufficiency in the LangGraph workflow.

Usage:
    from src.retrieval.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    top_docs = reranker.rerank(query, documents)
"""

import logging
import time
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder re-ranker using sentence-transformers.

    Loads a pre-trained cross-encoder model (default: BAAI/bge-reranker-base)
    and re-ranks candidate documents by computing relevance scores for each
    (query, document) pair, returning the top-k most relevant documents.

    Attributes:
        model_name: HuggingFace model identifier for the cross-encoder.
        top_k: Number of documents to return after re-ranking.
        max_length: Maximum characters per document sent to the model.
        enabled: Whether re-ranking is active (configurable via settings).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        top_k: int = 5,
        max_length: int = 512,
        enabled: bool = True,
    ):
        """Initialize the cross-encoder re-ranker.

        Args:
            model_name: HuggingFace cross-encoder model name or local path.
            top_k: Number of top documents to return after re-ranking.
            max_length: Max characters per document (truncated before scoring).
            enabled: If False, rerank() returns documents unchanged.
        """
        self.model_name = model_name
        self.top_k = top_k
        self.max_length = max_length
        self.enabled = enabled
        self._model = None

    @property
    def model(self):
        """Lazy-load the cross-encoder model on first use."""
        if self._model is None and self.enabled:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading cross-encoder model: %s", self.model_name)
                self._model = CrossEncoder(self.model_name)
                logger.info("Cross-encoder model loaded successfully.")
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
                self.enabled = False
            except Exception as e:
                logger.error("Failed to load cross-encoder model: %s", e)
                self.enabled = False
        return self._model

    def rerank(
        self, query: str, documents: List[Dict], return_all_on_fallback: bool = True
    ) -> List[Dict]:
        """Re-rank documents by relevance to the query using the cross-encoder.

        Args:
            query: The user query string.
            documents: List of document dicts. Each dict must have at least
                       'text' and 'score' keys (score is the original retrieval score).
            return_all_on_fallback: If re-ranking fails or is disabled, return
                                    first top_k documents (True) or all (False).

        Returns:
            List of document dicts, re-ordered by cross-encoder relevance,
            limited to top_k. Each dict gains a 'rerank_score' field.
        """
        if not documents:
            return []

        # If re-ranking is disabled, return top_k by original score
        if not self.enabled:
            docs = sorted(documents, key=lambda d: d.get("score", 0), reverse=True)
            return docs[: self.top_k]

        start_time = time.time()

        try:
            # Prepare (query, doc_text) pairs with truncation
            pairs = []
            for doc in documents:
                text = doc.get("text", "")
                if len(text) > self.max_length:
                    text = text[: self.max_length]
                pairs.append([query, text])

            # Compute cross-encoder scores
            scores = self.model.predict(pairs, show_progress_bar=False)

            # Attach scores and re-rank
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)

            ranked = sorted(
                documents, key=lambda d: d.get("rerank_score", 0), reverse=True
            )
            result = ranked[: self.top_k]

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                "Re-ranked %d documents → top-%d in %.0f ms (model: %s)",
                len(documents),
                len(result),
                elapsed_ms,
                self.model_name,
            )

            # Warn if re-ranking is slow
            if elapsed_ms > 500:
                logger.warning(
                    "Re-ranking took %.0f ms (> 500 ms threshold). "
                    "Consider using a smaller model or reducing max_length.",
                    elapsed_ms,
                )

            return result

        except Exception as e:
            logger.error("Re-ranking failed: %s. Returning top documents by original score.", e)
            docs = sorted(documents, key=lambda d: d.get("score", 0), reverse=True)
            return docs if not return_all_on_fallback else docs[: self.top_k]


# Module-level singleton
_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(
    model_name: str = "BAAI/bge-reranker-base",
    top_k: int = 5,
    enabled: bool = True,
) -> CrossEncoderReranker:
    """Return the global CrossEncoderReranker singleton.

    Args:
        model_name: Cross-encoder model name.
        top_k: Number of documents to return after re-ranking.
        enabled: Whether re-ranking is active.

    Returns:
        CrossEncoderReranker instance (created on first call).
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(
            model_name=model_name, top_k=top_k, enabled=enabled
        )
    return _reranker
