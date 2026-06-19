"""
cache_manager.py — 两级缓存统一管理器

功能：将精确缓存（L1）和语义缓存（L2）整合为一个统一入口，
      按 L1 → L2 的顺序逐级查询，任意一级命中即返回。

查询流程：
    用户问题
      → L1 精确缓存（MD5 匹配，0 误判，< 100ms）
      → 命中？返回答案（cache_type="exact"）
      → 未命中
      → L2 语义缓存（向量相似度，有泛化，< 200ms）
      → 命中？返回答案（cache_type="semantic"）
      → 未命中 → 执行完整 RAG 流水线

写入流程：
    完整流水线生成答案
      → 同时写入 L1 精确缓存 + L2 语义缓存

使用方式：
    from src.cache.cache_manager import CacheManager, get_cache_manager
    cm = get_cache_manager()
    cached = cm.get("问题")
    if cached:
        print(cached["answer"], cached["cache_type"])
    else:
        answer = run_pipeline()
        cm.put("问题", answer, metadata)
"""

import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class CacheManager:
    """两级缓存管理器。

    特性：
        - 逐级查询：L1 精确 → L2 语义
        - 命中时标记 cache_type（"exact" / "semantic"）
        - L1 和 L2 可独立开关
        - 写入时同时更新两级缓存

    Attributes:
        exact_enabled: 是否启用 L1 精确缓存
        semantic_enabled: 是否启用 L2 语义缓存
    """

    def __init__(
        self,
        exact_enabled: bool = True,
        semantic_enabled: bool = True,
        exact_ttl: int = 604800,
        semantic_threshold: float = 0.92,
        semantic_max_size: int = 10000,
        semantic_ttl: int = 86400,
    ):
        """初始化缓存管理器。

        Args:
            exact_enabled: 是否启用精确缓存
            semantic_enabled: 是否启用语义缓存
            exact_ttl: L1 过期时间（秒），默认 7 天
            semantic_threshold: L2 相似度阈值
            semantic_max_size: L2 最大条目
            semantic_ttl: L2 过期时间（秒），默认 24 小时
        """
        self.exact_enabled = exact_enabled
        self.semantic_enabled = semantic_enabled
        self.exact_ttl = exact_ttl
        self.semantic_threshold = semantic_threshold
        self.semantic_max_size = semantic_max_size
        self.semantic_ttl = semantic_ttl

        self._exact_cache = None
        self._semantic_cache = None

    # ------------------------------------------------------------------
    # 懒加载子缓存
    # ------------------------------------------------------------------

    @property
    def exact(self):
        """懒加载 L1 精确缓存。"""
        if self._exact_cache is None and self.exact_enabled:
            from src.cache.exact_cache import ExactCache
            self._exact_cache = ExactCache(ttl=self.exact_ttl, enabled=self.exact_enabled)
        return self._exact_cache

    @property
    def semantic(self):
        """懒加载 L2 语义缓存。"""
        if self._semantic_cache is None and self.semantic_enabled:
            from src.cache.semantic_cache import SemanticCache
            self._semantic_cache = SemanticCache(
                similarity_threshold=self.semantic_threshold,
                max_size=self.semantic_max_size,
                ttl=self.semantic_ttl,
                enabled=self.semantic_enabled,
            )
        return self._semantic_cache

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """两级缓存查询：L1 精确 → L2 语义。

        逐级查询，任意一级命中即返回，不再查下一级。

        Args:
            query: 用户问题

        Returns:
            未命中返回 None。
            命中返回 dict:
                - "answer": 缓存答案
                - "cache_type": "exact" | "semantic"
                - "cache_hit": True
                - "intent": 意图分类
                - "sources": 检索来源列表
        """
        # ---- L1: 精确缓存 ----
        if self.exact_enabled and self.exact is not None:
            entry = self.exact.get(query)
            if entry:
                logger.info("CacheManager: L1 精确缓存命中")
                entry["cache_type"] = "exact"
                entry["cache_hit"] = True
                return entry

        # ---- L2: 语义缓存 ----
        if self.semantic_enabled and self.semantic is not None:
            answer = self.semantic.get(query)
            if answer:
                logger.info("CacheManager: L2 语义缓存命中")
                return {
                    "answer": answer,
                    "cache_type": "semantic",
                    "cache_hit": True,
                }

        logger.debug("CacheManager: 两级缓存均未命中")
        return None

    def put(
        self,
        query: str,
        result: Dict[str, Any],
        metadata: Optional[Dict] = None,
    ) -> bool:
        """同时写入两级缓存。

        L1 精确缓存：问题 → 答案的精确映射
        L2 语义缓存：问题向量 → 答案的相似度映射

        Args:
            query: 用户问题
            result: 完整结果 dict（必须含 "answer"）
            metadata: 可选的附加信息

        Returns:
            True 至少一级写入成功
        """
        ok = False

        # ---- L1: 精确缓存 ----
        if self.exact_enabled and self.exact is not None:
            if self.exact.put(query, result):
                ok = True

        # ---- L2: 语义缓存 ----
        if self.semantic_enabled and self.semantic is not None:
            answer = result.get("answer", "")
            if answer:
                if self.semantic.put(query, answer, metadata):
                    ok = True

        return ok

    def stats(self) -> Dict[str, Any]:
        """返回两级缓存的聚合统计。"""
        s = {"enabled": True}
        if self._exact_cache:
            s["exact"] = self._exact_cache.stats()
        if self._semantic_cache:
            s["semantic"] = self._semantic_cache.stats()
        return s

    def clear(self) -> None:
        """清空两级缓存。"""
        if self._exact_cache:
            self._exact_cache.clear()
        if self._semantic_cache:
            self._semantic_cache.clear()
        logger.info("CacheManager: 两级缓存已清空")


# ---------------------------------------------------------------------------
# 模块级单例
# ---------------------------------------------------------------------------

_manager: Optional[CacheManager] = None


def get_cache_manager(
    exact_enabled: bool = True,
    semantic_enabled: bool = True,
) -> CacheManager:
    """获取全局 CacheManager 单例。

    Args:
        exact_enabled: 是否启用精确缓存
        semantic_enabled: 是否启用语义缓存

    Returns:
        CacheManager 实例
    """
    global _manager
    if _manager is None:
        _manager = CacheManager(
            exact_enabled=exact_enabled,
            semantic_enabled=semantic_enabled,
        )
    return _manager
