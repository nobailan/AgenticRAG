"""
semantic_cache.py — 语义缓存模块

功能：把（问题, 答案）对做语义缓存。当用户提问时，先计算问题的 embedding 向量，
      在缓存中搜索语义相似的历史问题。如果相似度超过阈值，直接返回缓存的答案，
      跳过整个 RAG 流水线（意图分类 → 检索 → 精排 → 生成），响应时间 < 500ms。

原理：
    - 用 sentence-transformers（bge-base-en-v1.5）对问题做向量化
    - 用 FAISS IndexFlatIP 存储缓存问题的向量（内积 = 归一化后的余弦相似度）
    - 用 Redis 持久化存储 {cache_key: {question, answer, vector, metadata}}
    - 重启时从 Redis 重建 FAISS 索引，保证缓存可恢复

工作流：
    get(query) → embed(query) → FAISS.search → 相似度 > 阈值？
        → 是: 从 Redis 取答案返回（命中）
        → 否: 走完整流水线（未命中）
    put(query, answer) → embed(query) → 写入 FAISS + Redis

使用方式：
    from src.cache.semantic_cache import SemanticCache, get_cache
    cache = get_cache()
    cached = cache.get("Veracier 的注册地址是？")
    if cached:
        print(cached)  # 直接返回缓存答案
    else:
        answer = run_full_pipeline(...)
        cache.put("Veracier 的注册地址是？", answer)
"""

import hashlib
import json
import logging
import time
from typing import Optional, Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class SemanticCache:
    """语义缓存：FAISS 向量相似度 + Redis 持久化。

    特性：
        - 懒加载：首次调用 get/put 时才初始化 embedding 模型和 FAISS 索引
        - 降级友好：Redis 不可用时自动切换为纯内存模式（重启后缓存丢失）
        - 容量控制：超过 max_size 时淘汰最旧的条目（FIFO 策略）
        - 统计透明：hits / misses / hit_rate 可随时查询

    Attributes:
        similarity_threshold: 判定命中的余弦相似度阈值（默认 0.92，范围 0-1）
        max_size: 最大缓存条目数
        ttl: Redis 中每条缓存的过期时间（秒）
        enabled: 是否启用缓存；False 时 get() 永远返回 None
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_size: int = 10000,
        ttl: int = 86400,
        enabled: bool = True,
        embedder=None,
    ):
        """初始化语义缓存。

        Args:
            similarity_threshold: 余弦相似度阈值，超过此值判定为命中
            max_size: 最大缓存条目，超出时淘汰最旧条目
            ttl: 缓存过期时间（秒），默认 86400 = 24 小时
            enabled: 是否启用
            embedder: 可复用的 SentenceTransformer 实例（None 则自动创建）
        """
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl
        self.enabled = enabled
        self._embedder = embedder
        self._index = None          # FAISS 索引，首次使用时创建
        self._keys: List[str] = []  # 有序的缓存键列表（用于 FIFO 淘汰）
        self._redis_client = None
        self._faiss_dim = None
        self._hits = 0              # 命中次数
        self._misses = 0            # 未命中次数

    # ==================================================================
    # 懒加载依赖（启动时不初始化，首次调用时才加载）
    # ==================================================================

    @property
    def embedder(self):
        """懒加载 embedding 模型。

        使用 bge-base-en-v1.5 对问题进行向量化。若模型加载失败，自动禁用缓存。
        """
        if self._embedder is None and self.enabled:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = "BAAI/bge-base-en-v1.5"
                self._embedder = SentenceTransformer(model_name)
                logger.info("SemanticCache: 已加载 embedding 模型 %s", model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers 未安装，语义缓存不可用"
                )
                self.enabled = False
            except Exception as e:
                logger.error("embedding 模型加载失败: %s", e)
                self.enabled = False
        return self._embedder

    @property
    def redis(self):
        """懒加载 Redis 客户端。

        Redis 不可用时不会抛错，自动降级为纯内存缓存模式。
        """
        if self._redis_client is None and self.enabled:
            from src.cache.redis_client import RedisClient
            import os
            redis_url = os.environ.get(
                "RAG_CACHE_REDIS_URL", "redis://localhost:6379/0"
            )
            self._redis_client = RedisClient(redis_url=redis_url)
            if not self._redis_client.connected:
                logger.info("SemanticCache: Redis 不可用，使用纯内存模式")
        return self._redis_client

    @property
    def faiss(self):
        """懒导入 FAISS 库。"""
        try:
            import faiss
            return faiss
        except ImportError:
            logger.warning("faiss-cpu 未安装")
            return None

    # ==================================================================
    # FAISS 索引管理
    # ==================================================================

    def _ensure_index(self, dim: int) -> None:
        """确保 FAISS 索引已初始化。

        首次调用时创建 IndexFlatIP（内积索引），然后尝试从 Redis 恢复历史缓存。

        Args:
            dim: 向量维度
        """
        if self._index is not None:
            return

        faiss = self.faiss
        if faiss is None:
            self.enabled = False
            return

        self._faiss_dim = dim
        # IndexFlatIP：内积搜索。因为 embedding 已做 L2 归一化，
        # 内积等价于余弦相似度，值越大越相似
        self._index = faiss.IndexFlatIP(dim)
        logger.info("SemanticCache: FAISS 索引已创建 (dim=%d)", dim)

        # 从 Redis 恢复历史缓存数据
        self._rebuild_from_redis()

    def _rebuild_from_redis(self) -> None:
        """启动时从 Redis 重建 FAISS 索引。

        数据流程：
            Redis 中存储了一个 "rag_cache:index_keys" 键，值是所有缓存键的 JSON 列表。
            遍历每个键，取出 vector 字段，加入 FAISS 索引。
            如果 Redis 不可用或没有历史数据，跳过重建。
        """
        redis = self.redis
        if redis is None or not redis.connected:
            return
        faiss = self.faiss
        if faiss is None:
            return

        try:
            key_list_raw = redis.get("rag_cache:index_keys")
            if key_list_raw:
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
                        "SemanticCache: 从 Redis 恢复了 %d 条缓存",
                        len(vectors),
                    )
        except Exception as e:
            logger.warning("从 Redis 重建缓存失败: %s", e)

    # ==================================================================
    # 公开 API：查询和存储
    # ==================================================================

    def get(self, query: str) -> Optional[str]:
        """查询缓存：用语义相似度匹配历史问题。

        步骤：
            1. 对 query 做 embedding 得到向量
            2. 在 FAISS 中搜索最相似的缓存问题
            3. 若余弦相似度 ≥ threshold，返回缓存的答案（命中）
            4. 否则返回 None（未命中，需要走完整流水线）

        Args:
            query: 用户问题

        Returns:
            缓存的答案；未命中或缓存不可用时返回 None
        """
        if not self.enabled:
            self._misses += 1
            return None

        if self.embedder is None or self.faiss is None:
            self._misses += 1
            return None

        try:
            # 对问题做向量化（L2 归一化，使内积 = 余弦相似度）
            query_vec = self.embedder.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )[0].astype(np.float32)

            dim = len(query_vec)
            self._ensure_index(dim)

            # 索引为空（没有缓存数据），直接返回未命中
            if self._index is None or self._index.ntotal == 0:
                self._misses += 1
                return None

            # 在 FAISS 中搜索最相似的缓存条目
            query_vec = query_vec.reshape(1, -1)
            distances, indices = self._index.search(query_vec, 1)

            best_sim = float(distances[0][0])   # 最高相似度
            best_idx = int(indices[0][0])        # 最高相似度的索引

            # 相似度达到阈值 → 从 Redis 取完整答案
            if best_sim >= self.similarity_threshold and best_idx >= 0:
                key = self._keys[best_idx]
                redis = self.redis
                if redis is not None:
                    entry = redis.get_json(key)
                    if entry and "answer" in entry:
                        self._hits += 1
                        logger.info(
                            "SemanticCache 命中 (sim=%.3f, hits=%d, misses=%d)",
                            best_sim, self._hits, self._misses,
                        )
                        return entry["answer"]

            # 未命中
            self._misses += 1
            logger.debug(
                "SemanticCache 未命中 (best_sim=%.3f < threshold=%.3f)",
                best_sim, self.similarity_threshold,
            )
            return None

        except Exception as e:
            logger.warning("SemanticCache get() 异常: %s", e)
            self._misses += 1
            return None

    def put(self, query: str, answer: str, metadata: Optional[Dict] = None) -> bool:
        """将 (问题, 答案) 存入缓存。

        步骤：
            1. 对 query 做 embedding
            2. 若容量已满，淘汰最旧的条目（FIFO）
            3. 将向量加入 FAISS 索引
            4. 将 {question, answer, vector, metadata} 存入 Redis

        Args:
            query: 用户问题
            answer: 生成的答案
            metadata: 可选的附加信息（如 intent、sources）

        Returns:
            True 存入成功
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

            # FIFO 淘汰：超过 max_size 时删除最旧条目
            if len(self._keys) >= self.max_size:
                evicted_key = self._keys.pop(0)
                redis = self.redis
                if redis is not None:
                    redis.set(evicted_key, "", ttl=1)  # 立即过期
                logger.debug("SemanticCache 淘汰: %s", evicted_key)

            # 生成缓存键（问题文本的 MD5 前 16 位，避免 key 过长）
            cache_key = f"rag_cache:{hashlib.md5(query.encode()).hexdigest()[:16]}"
            entry = {
                "question": query,
                "answer": answer,
                "vector": query_vec.tolist(),
                "metadata": metadata or {},
            }

            # 加入 FAISS 索引
            self._index.add(query_vec.reshape(1, -1))
            self._keys.append(cache_key)

            # 持久化到 Redis
            redis = self.redis
            if redis is not None and redis.connected:
                redis.set_json(cache_key, entry, ttl=self.ttl)
                # 同时更新键列表，确保重启后可恢复
                redis.set(
                    "rag_cache:index_keys",
                    json.dumps(self._keys),
                    ttl=self.ttl,
                )

            logger.debug("SemanticCache 存入: %s", cache_key)
            return True

        except Exception as e:
            logger.warning("SemanticCache put() 异常: %s", e)
            return False

    # ==================================================================
    # 运维接口：统计与清理
    # ==================================================================

    def stats(self) -> Dict[str, Any]:
        """返回缓存运行统计，用于监控和调试。

        Returns:
            dict: {
                "hits": 命中次数,
                "misses": 未命中次数,
                "hit_rate": 命中率 (0-1),
                "size": 当前缓存条目数,
                "max_size": 最大容量,
                "enabled": 是否启用,
            }
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, self._hits + self._misses),
            "size": len(self._keys),
            "max_size": self.max_size,
            "enabled": self.enabled,
        }

    def clear(self) -> None:
        """清空全部缓存，重置 FAISS 索引和计数器。"""
        self._keys = []
        if self._index is not None:
            faiss = self.faiss
            if faiss is not None and self._faiss_dim:
                self._index = faiss.IndexFlatIP(self._faiss_dim)
        redis = self.redis
        if redis is not None and redis.connected:
            redis.set("rag_cache:index_keys", "[]", ttl=self.ttl)
        self._hits = 0
        self._misses = 0
        logger.info("SemanticCache: 已清空全部缓存")


# ---------------------------------------------------------------------------
# 模块级单例
# ---------------------------------------------------------------------------

_cache: Optional[SemanticCache] = None


def get_cache(
    similarity_threshold: float = 0.92,
    enabled: bool = True,
) -> SemanticCache:
    """获取全局 SemanticCache 单例。

    整个应用只初始化一次缓存模块（embedding 模型加载较慢），
    后续调用直接复用。

    Args:
        similarity_threshold: 余弦相似度阈值
        enabled: 是否启用

    Returns:
        SemanticCache 实例
    """
    global _cache
    if _cache is None:
        _cache = SemanticCache(
            similarity_threshold=similarity_threshold,
            enabled=enabled,
        )
    return _cache
