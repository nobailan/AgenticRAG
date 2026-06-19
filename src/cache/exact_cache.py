"""
exact_cache.py — 精确缓存模块

功能：对完全相同的问题做精确字符串匹配缓存，命中时响应时间 < 100ms。
      这是两级缓存体系的第一层（L1），优先级高于语义缓存。

原理：
    - Key = 用户问题文本的 MD5 哈希（去重 + 固定长度）
    - Value = 完整答案 JSON，包含 answer、intent、sources、cached_at 等字段
    - 存储 = Redis（可降级为进程内 dict）
    - TTL = 可配置，默认 7 天（604800 秒）

为什么需要精确缓存：
    - 语义缓存有误判风险（相似但不同的问题可能被错误命中）
    - 精确缓存零误判，适合高频重复问题（如"公司总部在哪里？"）
    - 两级配合：先查精确（快 + 准），未命中再查语义（稍慢但有泛化）

使用方式：
    from src.cache.exact_cache import ExactCache
    cache = ExactCache()
    cache.put("What is Veracier?", answer_json)
    cached = cache.get("What is Veracier?")  # → 返回 answer_json 或 None
"""

import hashlib
import json
import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ExactCache:
    """精确缓存：基于问题 MD5 的字符串匹配。

    L1 缓存层，无泛化能力，但零误判。适合缓存高频重复问题。

    Attributes:
        ttl: 过期时间（秒），默认 604800 = 7 天
        enabled: 是否启用
    """

    def __init__(
        self,
        ttl: int = 604800,
        enabled: bool = True,
    ):
        """初始化精确缓存。

        Args:
            ttl: 缓存过期时间（秒）
            enabled: 是否启用
        """
        self.ttl = ttl
        self.enabled = enabled
        self._redis = None
        self._fallback: Dict[str, Dict] = {}  # Redis 不可用时的进程内降级
        self._hits = 0
        self._misses = 0
        self._warned = False

    # ------------------------------------------------------------------
    # 懒加载 Redis
    # ------------------------------------------------------------------

    @property
    def redis(self):
        """懒加载 Redis 客户端，不可用时降级为内存 dict。"""
        if self._redis is None and self.enabled:
            try:
                from src.cache.redis_client import RedisClient
                import os
                redis_url = os.environ.get(
                    "RAG_CACHE_REDIS_URL", "redis://localhost:6379/0"
                )
                self._redis = RedisClient(redis_url=redis_url)
                if not self._redis.connected:
                    logger.info("ExactCache: Redis 不可用，降级为内存模式")
            except Exception as e:
                if not self._warned:
                    logger.warning("ExactCache: Redis 初始化失败: %s", e)
                    self._warned = True
        return self._redis

    # ------------------------------------------------------------------
    # 键生成
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(query: str) -> str:
        """把问题文本转成 Redis 键。

        格式: rag:exact:{md5}
        去重处理：先 strip + lowercase，减少大小写/空格导致的缓存穿透。

        Args:
            query: 用户问题

        Returns:
            Redis 键字符串
        """
        normalized = query.strip().lower()
        digest = hashlib.md5(normalized.encode()).hexdigest()
        return f"rag:exact:{digest}"

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """查询精确缓存。

        步骤：
            1. 规范化问题 → 生成 MD5 键
            2. 优先查 Redis
            3. Redis 不可用则查本地 dict
            4. 命中检查 TTL 是否过期

        Args:
            query: 用户问题

        Returns:
            缓存答案 dict（含 answer, intent, sources, cached_at 等），
            未命中返回 None
        """
        if not self.enabled:
            self._misses += 1
            return None

        key = self._make_key(query)

        try:
            redis = self.redis
            if redis is not None and redis.connected:
                raw = redis.get(key)
                if raw:
                    entry = json.loads(raw)
                    # 检查 TTL 是否过期
                    cached_at = entry.get("cached_at", 0)
                    if time.time() - cached_at < self.ttl:
                        self._hits += 1
                        logger.info(
                            "ExactCache 命中 (hits=%d, misses=%d)",
                            self._hits, self._misses,
                        )
                        return entry
                    else:
                        # 过期删除
                        redis.set(key, "", ttl=1)
            else:
                # Redis 不可用，查本地 dict
                if key in self._fallback:
                    entry = self._fallback[key]
                    if time.time() - entry.get("cached_at", 0) < self.ttl:
                        self._hits += 1
                        return entry
        except Exception as e:
            logger.debug("ExactCache get() 异常: %s", e)

        self._misses += 1
        return None

    def put(self, query: str, result: Dict[str, Any]) -> bool:
        """存入精确缓存。

        Args:
            query: 用户问题
            result: 答案 dict，必须包含 'answer' 键

        Returns:
            True 写入成功
        """
        if not self.enabled:
            return False

        key = self._make_key(query)
        entry = {
            "answer": result.get("answer", ""),
            "intent": result.get("intent", "unknown"),
            "sources": result.get("retrieved_sources", []),
            "cached_at": time.time(),
        }

        try:
            redis = self.redis
            if redis is not None and redis.connected:
                redis.set(key, json.dumps(entry, ensure_ascii=False), ttl=self.ttl)
            else:
                self._fallback[key] = entry
                # 内存模式下的简单容量限制（最多 1000 条）
                if len(self._fallback) > 1000:
                    oldest = min(self._fallback, key=lambda k: self._fallback[k].get("cached_at", 0))
                    del self._fallback[oldest]
            logger.debug("ExactCache 存入: %s", key)
            return True
        except Exception as e:
            logger.debug("ExactCache put() 异常: %s", e)
            return False

    def stats(self) -> Dict[str, Any]:
        """返回缓存统计。"""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, self._hits + self._misses),
            "enabled": self.enabled,
        }

    def clear(self) -> None:
        """清空所有精确缓存条目。"""
        self._fallback.clear()
        self._hits = 0
        self._misses = 0
        # Redis 端通过 TTL 自动过期，不做全量清理（避免 scan 开销）
        logger.info("ExactCache: 已清空内存缓存")
