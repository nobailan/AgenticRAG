"""
retriever.py -- Hybrid retrieval with FAISS (dense) + BM25 (sparse) + RRF fusion.

Provides:
    hybrid_search(query: str, top_k: int = 5) -> List[RetrievedChunk]

Architecture:
    - Lazy-loads FAISS index, BM25 index, and chunks.jsonl on first call.
    - Vector search: SentenceTransformer embed → FAISS IndexFlatIP (cosine via L2 norm).
    - Sparse search: rank_bm25.BM25Okapi with whitespace tokenization.
    - RRF fusion: score = Σ 1/(k + rank), k=60, across both rankers.
    - Deterministic: same (query, index) pair always returns identical results.

Usage:
    from retriever import hybrid_search
    results = hybrid_search("What was the R&D budget?", top_k=5)
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import config
from workflow import RetrievedChunk

logger = logging.getLogger(__name__)

# Deterministic behavior: fixed random seed for reproducibility
np.random.seed(42)


# =============================================================================
# Module-level cache — loaded lazily on first hybrid_search() call
# =============================================================================

_embedding_model: Optional[SentenceTransformer] = None
"""SentenceTransformer instance, loaded once."""

_faiss_index: Optional[faiss.IndexFlatIP] = None
"""FAISS IndexFlatIP, loaded from faiss.index."""

_bm25: Optional[Any] = None  # BM25Okapi instance
"""BM25Okapi instance, loaded from bm25_index.pkl."""

_chunks: List[Dict[str, Any]] = []
"""All chunks loaded from chunks.jsonl, each a dict with chunk_id, text, metadata."""

_chunk_texts: List[str] = []
"""Parallel list of chunk texts for BM25 tokenization."""

_loaded: bool = False
"""True once _load_indexes() has completed successfully."""


# =============================================================================
# Lazy loader
# =============================================================================

def _load_indexes() -> None:
    """Lazy-load FAISS index, BM25 index, embedding model, and chunks.jsonl.

    Idempotent — only loads once. Thread-safe only for read-after-load
    (no concurrent modification during loading).

    Raises:
        FileNotFoundError: If chunks.jsonl, faiss.index, or bm25_index.pkl is missing.
            Run data_prepare.py first to generate these files.
    """
    global _embedding_model, _faiss_index, _bm25, _chunks, _chunk_texts, _loaded

    if _loaded:
        return

    logger.info("Loading retrieval indexes (first call)...")

    # --- Load chunks.jsonl ---
    chunks_path = config.chunk_jsonl_path
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"chunks.jsonl not found at {chunks_path}. "
            f"Run 'python data_prepare.py' first."
        )
    with open(chunks_path, "r", encoding="utf-8") as f:
        _chunks = [json.loads(line) for line in f if line.strip()]
    _chunk_texts = [c["text"] for c in _chunks]
    logger.info(f"Loaded {len(_chunks)} chunks from {chunks_path}")

    # --- Load FAISS index ---
    faiss_path = config.faiss_index_path
    if not faiss_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {faiss_path}. "
            f"Run 'python data_prepare.py' first."
        )
    _faiss_index = faiss.read_index(str(faiss_path))
    logger.info(f"Loaded FAISS index ({_faiss_index.ntotal} vectors, dim={_faiss_index.d})")

    # --- Load BM25 index ---
    bm25_path = config.bm25_index_path
    if not bm25_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {bm25_path}. "
            f"Run 'python data_prepare.py' first."
        )
    with open(bm25_path, "rb") as f:
        _bm25 = pickle.load(f)
    logger.info(f"Loaded BM25 index ({len(_chunks)} docs)")

    # --- Load embedding model ---
    _embedding_model = SentenceTransformer(
        config.embedding_model_name,
        device=config.embedding_device,
    )
    actual_device = str(_embedding_model.device) if hasattr(_embedding_model, 'device') else 'unknown'
    logger.info(f"Loaded embedding model: {config.embedding_model_name} (device={actual_device})")

    _loaded = True
    logger.info("All retrieval indexes loaded successfully.")


# =============================================================================
# HybridRetriever class — the canonical retrieval interface
# =============================================================================

class HybridRetriever:
    """Hybrid retrieval combining FAISS (dense) + BM25 (sparse) + RRF fusion.

    Singleton pattern — the constructor always returns the same global instance.
    On first construction, eagerly loads all indexes (FAISS, BM25, embedding model)
    and sets the global random seed for deterministic behavior.

    Usage:
        retriever = HybridRetriever()
        results = retriever.search("What was the R&D budget?", top_k=5)

    Raises:
        FileNotFoundError: If chunks.jsonl, faiss.index, or bm25_index.pkl is missing.
            Run 'python data_prepare.py' first to generate these files.
    """

    _instance: Optional["HybridRetriever"] = None

    def __new__(cls) -> "HybridRetriever":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the retriever and eagerly load all indexes.

        Idempotent — subsequent constructions return the same instance
        without re-loading indexes.

        Raises:
            FileNotFoundError: If any index file is missing.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        np.random.seed(42)
        _load_indexes()
        logger.info("HybridRetriever initialized (singleton)")

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Execute hybrid search and return top_k chunks sorted by RRF score descending.

        Args:
            query: Natural language search query.
            top_k: Number of chunks to return (default 5).

        Returns:
            List of RetrievedChunk objects sorted by RRF score descending.
            Empty list if query is empty or no indexes are loaded.
        """
        return hybrid_search(query, top_k)


# =============================================================================
# Tokenizer for BM25
# =============================================================================

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation-splitting tokenizer for BM25.

    Language-agnostic: lowercases, splits on whitespace and punctuation.
    Suitable for European languages (French, English, German, Italian, Spanish).

    Args:
        text: Raw text string to tokenize.

    Returns:
        List of lowercase tokens.
    """
    text = text.lower()
    # Split on non-alphanumeric characters (preserves Unicode letters)
    tokens = re.findall(r"[^\W_]+", text, re.UNICODE)
    return tokens


# =============================================================================
# Vector search (FAISS)
# =============================================================================

def _faiss_search(query: str, top_k: int) -> List[Tuple[int, float]]:
    """Dense vector search via FAISS inner product (cosine similarity).

    Steps:
        1. Embed query with SentenceTransformer.
        2. L2-normalize query vector.
        3. Search FAISS index (vectors are pre-normalized, so IP == cosine).
        4. Return (chunk_index, similarity_score) sorted descending.

    Args:
        query: Natural language query string.
        top_k: Number of results to return.

    Returns:
        List of (chunk_index_in_chunks_list, cosine_similarity) pairs.
    """
    assert _embedding_model is not None
    assert _faiss_index is not None

    # Embed query
    query_vec = _embedding_model.encode(
        [query],
        normalize_embeddings=True,  # L2 normalization
        show_progress_bar=False,
    ).astype(np.float32)

    # Search FAISS (IP = cosine for normalized vectors)
    scores, indices = _faiss_index.search(query_vec, top_k)

    # Build result list, filter invalid indices (-1 from FAISS)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(_chunks):
            results.append((int(idx), float(score)))

    logger.debug(f"FAISS: query='{query[:50]}...' → {len(results)} results, "
                  f"top_score={results[0][1]:.4f}" if results else "no results")
    return results


# =============================================================================
# BM25 sparse search
# =============================================================================

def _bm25_search(query: str, top_k: int) -> List[Tuple[int, float]]:
    """BM25 sparse retrieval.

    Steps:
        1. Tokenize query with _tokenize().
        2. Get BM25 scores for all documents.
        3. Return top_k (chunk_index, bm25_score) sorted descending.

    Args:
        query: Natural language query string.
        top_k: Number of results to return.

    Returns:
        List of (chunk_index_in_chunks_list, bm25_score) pairs.
    """
    assert _bm25 is not None

    query_tokens = _tokenize(query)
    if not query_tokens:
        logger.warning(f"BM25: empty query tokens for '{query[:50]}...'")
        return []

    doc_scores = _bm25.get_scores(query_tokens)

    # Get top_k indices sorted by score descending
    # rank_bm25 scores are non-negative floats
    if len(doc_scores) == 0:
        return []

    # Get indices of top_k scores
    if top_k >= len(doc_scores):
        top_indices = np.argsort(doc_scores)[::-1]
    else:
        # Partial arg-sort for efficiency
        top_indices = np.argpartition(doc_scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(doc_scores[top_indices])][::-1]

    results = [(int(idx), float(doc_scores[idx])) for idx in top_indices if doc_scores[idx] > 0]

    logger.debug(f"BM25: query='{query[:50]}...' → {len(results)} results, "
                  f"top_score={results[0][1]:.4f}" if results else "no results")
    return results


# =============================================================================
# Reciprocal Rank Fusion (RRF)
# =============================================================================

def _rrf_fusion(
    faiss_results: List[Tuple[int, float]],
    bm25_results: List[Tuple[int, float]],
    k: int = config.rrf_k,
) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion: combine FAISS and BM25 result lists.

    Formula:
        RRF_score(d) = Σ_{ranker} 1 / (k + rank_d_in_ranker)

    Where rank starts at 1 (not 0). If a document appears in only one ranker,
    it receives score only from that ranker. Documents appearing in both
    rankers get scores from both.

    Args:
        faiss_results: Pairs of (chunk_idx, score) from FAISS, sorted desc.
        bm25_results: Pairs of (chunk_idx, score) from BM25, sorted desc.
        k: Smoothing constant (default 60, as per TREC RRF literature).

    Returns:
        List of (chunk_idx, fused_rrf_score) sorted descending by fused score.
    """
    rrf_scores: Dict[int, float] = {}

    # FAISS ranker — rank starts at 1
    for rank, (idx, _score) in enumerate(faiss_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

    # BM25 ranker — rank starts at 1
    for rank, (idx, _score) in enumerate(bm25_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

    # Sort by RRF score descending
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    logger.debug(f"RRF: {len(faiss_results)} dense + {len(bm25_results)} sparse "
                  f"→ {len(fused)} fused results")
    return fused


# =============================================================================
# Public API
# =============================================================================

def hybrid_search(
    query: str,
    top_k: int | None = None,
) -> List[RetrievedChunk]:
    """Hybrid retrieval: FAISS vector + BM25 sparse + RRF fusion.

    Deterministic: identical (query, index) always returns identical results.

    Workflow:
        1. Lazy-load indexes on first call.
        2. Run FAISS dense search (cosine similarity via inner product).
        3. Run BM25 sparse search.
        4. Fuse results via Reciprocal Rank Fusion (RRF, k=60).
        5. Return top_k RetrievedChunk objects sorted by fused score descending.

    Args:
        query: Natural language search query (any language in the corpus).
        top_k: Number of chunks to return. Defaults to config.top_k (5).

    Returns:
        List of RetrievedChunk objects sorted by RRF score descending.
        Empty list if no indexes are loaded or query is empty.

    Raises:
        FileNotFoundError: If indexes haven't been built yet.
    """
    if top_k is None:
        top_k = config.top_k

    if not query or not query.strip():
        logger.warning("hybrid_search: empty query")
        return []

    # Lazy-load indexes on first call
    _load_indexes()

    assert _chunks is not None
    assert _faiss_index is not None
    assert _bm25 is not None

    # Run both rankers (use larger internal k for better RRF fusion coverage)
    faiss_internal_k = min(config.faiss_search_k, len(_chunks))
    bm25_internal_k = min(config.bm25_search_k, len(_chunks))

    faiss_results = _faiss_search(query, faiss_internal_k)
    bm25_results = _bm25_search(query, bm25_internal_k)

    # RRF fusion
    fused = _rrf_fusion(faiss_results, bm25_results, k=config.rrf_k)

    # Build RetrievedChunk objects
    results: List[RetrievedChunk] = []
    for chunk_idx, rrf_score in fused[:top_k]:
        chunk_data = _chunks[chunk_idx]
        results.append(RetrievedChunk(
            chunk_id=chunk_data["chunk_id"],
            text=chunk_data["text"],
            score=rrf_score,
            metadata=chunk_data.get("metadata", {}),
        ))

    logger.info(
        f"hybrid_search: '{query[:60]}...' → {len(results)} results "
        f"(FAISS: {len(faiss_results)}, BM25: {len(bm25_results)}, "
        f"RRF fused: {len(fused)})"
    )
    return results


def is_loaded() -> bool:
    """Check whether retrieval indexes have been loaded.

    Returns:
        True if _load_indexes() has completed successfully.
    """
    return _loaded


def get_chunk_count() -> int:
    """Return the total number of chunks in the index.

    Returns:
        Number of chunks, or 0 if indexes haven't been loaded.
    """
    return len(_chunks)


# =============================================================================
# Module self-test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Retriever module test")
    print("=" * 60)

    if not is_loaded():
        try:
            _load_indexes()
            print(f"[OK] Loaded {get_chunk_count()} chunks")
        except FileNotFoundError as e:
            print(f"[WARN] Indexes not built yet: {e}")
            print("  Run 'python data_prepare.py' first, or test with stub.")
            # For testing without real data, create dummy indexes
            print("\n  Creating dummy in-memory indexes for structural test...")
            _chunks = [
                {
                    "chunk_id": f"doc_{i}",
                    "text": f"This is test chunk {i}. It contains sample text for retrieval testing.",
                    "metadata": {"source_file": f"test/file_{i}.pdf", "page": 1, "entity": "test"},
                }
                for i in range(100)
            ]
            _chunk_texts = [c["text"] for c in _chunks]

            # Dummy FAISS index
            dummy_dim = config.embedding_dim
            _faiss_index = faiss.IndexFlatIP(dummy_dim)
            dummy_vecs = np.random.randn(100, dummy_dim).astype(np.float32)
            faiss.normalize_L2(dummy_vecs)
            _faiss_index.add(dummy_vecs)

            # Dummy BM25
            from rank_bm25 import BM25Okapi
            tokenized = [_tokenize(t) for t in _chunk_texts]
            _bm25 = BM25Okapi(tokenized)

            # Dummy embedding model
            _embedding_model = SentenceTransformer(
                config.embedding_model_name,
                device=config.embedding_device,
            )

            _loaded = True
            print(f"  Created {len(_chunks)} dummy chunks with FAISS+BM25 indexes.")

    # Test search
    print("\nTesting hybrid_search...")
    results = hybrid_search("test query about R&D spending")
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r.chunk_id} | score={r.score:.4f} | {r.text[:80]}...")
    print(f"\n[OK] hybrid_search returned {len(results)} results")
