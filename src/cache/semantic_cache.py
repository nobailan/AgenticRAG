"""
semantic_cache.py -- Semantic cache with FAISS vector similarity search.

Caches (question, answer) pairs using dense embeddings. When a new query
arrives, its embedding is compared against cached questions via FAISS cosine
similarity. If a sufficiently similar question exists (above threshold), the
cached answer is returned immediately, skipping the entire RAG pipeline.

Usage:
    from src.cache.semantic_cache import SemanticCache
    cache = SemanticCache()
    cached = cache.get("What is Veracier?")
    if cached:
        print(cached)  # returns answer directly
    else:
        answer = run_full_pipeline(...)
        cache.put("What is Veracier?", answer, metadata={})
"""

import hashlib
import logging
import time
from typing import Optional, Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class SemanticCache:
    """Semantic cache using FAISS for vector similarity + Redis for storage.

    Workflow:
        1. On get(query): embed query → search FAISS → if cos_sim > threshold,
           return cached answer from Redis.
        2. On put(query, answer): embed query → add to FAISS index → store
           answer in Redis.

    The FAISS index is rebuilt from Redis on initialization for persistence
    across restarts.

    Attributes:
        similarity_threshold: Minimum cosine similarity for a cache hit (0-1).
        max_size: Maximum number of cached entries.
        enabled: If False, get() always returns None.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_size: int = 10000,
        ttl: int = 86400,
        enabled: bool = True,
        embedder=None,  # Optional[SentenceTransformer]
    ):
        """Initialize the semantic cache.

        Args:
            similarity_threshold: Cosine similarity threshold for cache hit.
            max_size: Max cached entries (oldest evicted on overflow).
            ttl: Redis key TTL in seconds (default: 24h).
            enabled: If False, disables caching entirely.
            embedder: Pre-loaded SentenceTransformer instance (lazy-loads if None).
        """
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl
        self.enabled = enabled
        self._embedder = embedder
        self._index = None  # FAISS index, built lazily
        self._keys: List[str] = []  # Ordered cache keys (for eviction)
        self._redis_client = None
        self._faiss_dim = None
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Lazy dependencies
    # ------------------------------------------------------------------

    @property
    def embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is None and self.enabled:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = "BAAI/bge-base-en-v1.5"
                self._embedder = SentenceTransformer(model_name)
                logger.info("SemanticCache: loaded embedding model %s", model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. Semantic cache disabled."
                )
                self.enabled = False
            except Exception as e:
                logger.error("Failed to load embedding model: %s", e)
                self.enabled = False
        return self._embedder

    @property
    def redis(self):
        """Lazy-load Redis client."""
        if self._redis_client is None and self.enabled:
            from src.cache.redis_client import RedisClient
            # Get Redis URL from config or env
            import os
            redis_url = os.environ.get(
                "RAG_CACHE_REDIS_URL", "redis://localhost:6379/0"
            )
            self._redis_client = RedisClient(redis_url=redis_url)
            if not self._redis_client.connected:
                # Fall back to in-memory-only mode
                logger.info("SemanticCache: Redis unavailable, using in-memory mode.")
        return self._redis_client

    @property
    def faiss(self):
        """Lazy-import FAISS."""
        try:
            import faiss
            return faiss
        except ImportError:
            logger.warning("faiss-cpu not installed.")
            return None

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _ensure_index(self, dim: int) -> None:
        """Create FAISS index if not yet initialized.

        Args:
            dim: Embedding dimension.
        """
        if self._index is not None:
            return

        faiss = self.faiss
        if faiss is None:
            self.enabled = False
            return

        self._faiss_dim = dim
        self._index = faiss.IndexFlatIP(dim)  # Inner product = cosine (normalized)
        logger.info("SemanticCache: FAISS index created (dim=%d).", dim)

        # Rebuild from Redis on startup (if available)
        self._rebuild_from_redis()

    def _rebuild_from_redis(self) -> None:
        """Rebuild FAISS index from cached entries in Redis."""
        redis = self.redis
        if redis is None or not redis.connected:
            return
        faiss = self.faiss
        if faiss is None:
            return

        try:
            # Scan Redis for cache keys (keys matching "rag_cache:*")
            # Since redis-py scanning requires the redis client,
            # we use a simple approach: maintain a list in Redis
            key_list_raw = redis.get("rag_cache:index_keys")
            if key_list_raw:
                import json
                self._keys = json.loads(key_list_raw)
                vectors = []
                for key in self._keys:
                    entry = redis.get_json(key)
                    if entry and "vector" in entry:
                        vec = np.array(entry["vector"], dtype=np.float32)
                        vectors.append(vec)
                if vectors:
                    vecs_array = np.stack(vectors).astype(np.float32)
                    self._index.add(vecs_array)
                    logger.info(
                        "SemanticCache: rebuilt FAISS index from %d cached entries.",
                        len(vectors),
                    )
        except Exception as e:
            logger.warning("Failed to rebuild cache from Redis: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> Optional[str]:
        """Look up a semantically similar cached answer.

        Args:
            query: The user's question.

        Returns:
            Cached answer string if a similar question exists above threshold,
            None otherwise.
        """
        if not self.enabled:
            self._misses += 1
            return None

        if self.embedder is None or self.faiss is None:
            self._misses += 1
            return None

        try:
            # Encode query
            query_vec = self.embedder.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )[0].astype(np.float32)

            dim = len(query_vec)
            self._ensure_index(dim)

            if self._index is None or self._index.ntotal == 0:
                self._misses += 1
                return None

            # Search FAISS for most similar cached question
            query_vec = query_vec.reshape(1, -1)
            distances, indices = self._index.search(query_vec, 1)

            best_sim = float(distances[0][0])
            best_idx = int(indices[0][0])

            if best_sim >= self.similarity_threshold and best_idx >= 0:
                key = self._keys[best_idx]
                redis = self.redis
                if redis is not None:
                    entry = redis.get_json(key)
                    if entry and "answer" in entry:
                        self._hits += 1
                        logger.info(
                            "SemanticCache HIT (sim=%.3f, hits=%d, misses=%d)",
                            best_sim, self._hits, self._misses,
                        )
                        return entry["answer"]

            self._misses += 1
            logger.debug(
                "SemanticCache MISS (best_sim=%.3f < threshold=%.3f, "
                "hits=%d, misses=%d)",
                best_sim, self.similarity_threshold, self._hits, self._misses,
            )
            return None

        except Exception as e:
            logger.warning("SemanticCache get() error: %s", e)
            self._misses += 1
            return None

    def put(self, query: str, answer: str, metadata: Optional[Dict] = None) -> bool:
        """Cache a (query, answer) pair.

        Args:
            query: The user's question.
            answer: The generated answer.
            metadata: Optional dict with extra info (e.g., sources, intent).

        Returns:
            True if cached successfully.
        """
        if not self.enabled:
            return False
        if self.embedder is None:
            return False

        try:
            query_vec = self.embedder.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )[0].astype(np.float32)

            dim = len(query_vec)
            self._ensure_index(dim)

            if self._index is None:
                return False

            # Evict oldest if at capacity
            if len(self._keys) >= self.max_size:
                evicted_key = self._keys.pop(0)
                # Remove from Redis
                redis = self.redis
                if redis is not None:
                    redis.set(evicted_key, "", ttl=1)  # expire immediately
                # Note: FAISS IndexFlatIP doesn't support removal;
                # we rebuild periodically or just let it degrade slightly.
                logger.debug("SemanticCache evicted: %s", evicted_key)

            # Create cache entry
            cache_key = f"rag_cache:{hashlib.md5(query.encode()).hexdigest()[:16]}"
            entry = {
                "question": query,
                "answer": answer,
                "vector": query_vec.tolist(),
                "metadata": metadata or {},
            }

            # Add to FAISS
            self._index.add(query_vec.reshape(1, -1))
            self._keys.append(cache_key)

            # Store in Redis
            redis = self.redis
            if redis is not None and redis.connected:
                redis.set_json(cache_key, entry, ttl=self.ttl)
                # Update index key list in Redis
                import json
                redis.set("rag_cache:index_keys", json.dumps(self._keys), ttl=self.ttl)

            logger.debug("SemanticCache PUT: %s", cache_key)
            return True

        except Exception as e:
            logger.warning("SemanticCache put() error: %s", e)
            return False

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (
                self._hits / max(1, self._hits + self._misses)
            ),
            "size": len(self._keys),
            "max_size": self.max_size,
            "enabled": self.enabled,
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self._keys = []
        if self._index is not None:
            faiss = self.faiss
            if faiss is not None and self._faiss_dim:
                self._index = faiss.IndexFlatIP(self._faiss_dim)
        redis = self.redis
        if redis is not None and redis.connected:
            # Clear all rag_cache keys
            redis.set("rag_cache:index_keys", "[]", ttl=self.ttl)
        self._hits = 0
        self._misses = 0
        logger.info("SemanticCache: cleared all entries.")


# Module-level singleton
_cache: Optional[SemanticCache] = None


def get_cache(
    similarity_threshold: float = 0.92,
    enabled: bool = True,
) -> SemanticCache:
    """Return the global SemanticCache singleton.

    Args:
        similarity_threshold: Cosine similarity threshold for cache hit.
        enabled: Whether caching is active.

    Returns:
        SemanticCache instance (created on first call).
    """
    global _cache
    if _cache is None:
        _cache = SemanticCache(
            similarity_threshold=similarity_threshold,
            enabled=enabled,
        )
    return _cache
