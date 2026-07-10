"""
health.py — 健康检查与限流

功能：
    - /health 端点：检查各组件是否正常（LLM API、索引、Redis）
    - 简易限流器：基于时间窗口的 QPS 限制

用法：
    from src.security.health import HealthChecker, RateLimiter

    # 健康检查
    health = HealthChecker()
    status = health.check()

    # 限流
    limiter = RateLimiter(max_rpm=60)
    if limiter.acquire():
        process_query()
    else:
        return "请求过于频繁，请稍后再试。"
"""

import logging
import time
from collections import deque
from typing import Dict, Any

logger = logging.getLogger(__name__)


class HealthChecker:
    """组件健康检查器。

    检查项：
        - 索引是否加载
        - LLM API Key 是否配置
        - Redis 是否连通（如启用缓存）
    """

    def __init__(self):
        pass

    def check(self) -> Dict[str, Any]:
        """执行全量健康检查。

        Returns:
            {"status": "ok"|"degraded"|"down", "components": {...}}
        """
        components = {}

        # 1. 索引检查
        try:
            from src.retrieval.retriever import is_loaded, get_chunk_count
            if is_loaded():
                components["index"] = {"status": "ok", "chunks": get_chunk_count()}
            else:
                components["index"] = {"status": "down", "message": "索引未加载"}
        except Exception as e:
            components["index"] = {"status": "down", "message": str(e)}

        # 2. LLM API Key 检查
        import os
        providers = []
        for key in ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
            if os.environ.get(key):
                providers.append(key.replace("_API_KEY", "").lower())
        components["llm"] = {
            "status": "ok" if providers else "down",
            "available": providers or ["none"],
        }

        # 3. Redis 检查（如启用）
        from src.core.config import config
        if config.cache_enabled:
            try:
                from src.cache.redis_client import RedisClient
                redis = RedisClient(redis_url=config.cache_redis_url)
                if redis.connected:
                    components["redis"] = {"status": "ok"}
                else:
                    components["redis"] = {"status": "degraded", "message": "Redis 不可用，缓存降级"}
            except Exception as e:
                components["redis"] = {"status": "down", "message": str(e)}
        else:
            components["redis"] = {"status": "disabled"}

        # 综合状态
        statuses = [c.get("status") for c in components.values()]
        if "down" in statuses:
            overall = "degraded" if "ok" in statuses else "down"
        else:
            overall = "ok"

        return {"status": overall, "timestamp": time.time(), "components": components}

    def is_ready(self) -> bool:
        """快速检查核心组件是否就绪（用于 K8s readiness probe）。"""
        s = self.check()
        return s["components"].get("index", {}).get("status") == "ok"


class RateLimiter:
    """简易滑动窗口限流器。

    不做分布式协调（单进程适用），企业级需换 Redis token bucket。

    Attributes:
        max_rpm: 每分钟最大请求数
        window: 时间窗口（秒）
    """

    def __init__(self, max_rpm: int = 60, enabled: bool = True):
        self.max_rpm = max_rpm
        self.enabled = enabled
        self._window: deque = deque()

    def acquire(self) -> bool:
        """尝试获取一个请求许可。

        Returns:
            True 允许通过，False 触发限流
        """
        if not self.enabled:
            return True

        now = time.time()
        # 清理过期的时间戳
        while self._window and self._window[0] < now - 60:
            self._window.popleft()

        if len(self._window) < self.max_rpm:
            self._window.append(now)
            return True

        logger.warning("限流触发: %d RPM 已达上限", self.max_rpm)
        return False

    def remaining(self) -> int:
        """剩余可用请求数。"""
        now = time.time()
        while self._window and self._window[0] < now - 60:
            self._window.popleft()
        return max(0, self.max_rpm - len(self._window))

    def stats(self) -> Dict[str, Any]:
        return {
            "max_rpm": self.max_rpm,
            "current_rpm": len(self._window),
            "remaining": self.remaining(),
            "enabled": self.enabled,
        }


# 模块级单例
_health: HealthChecker = None
_limiter: RateLimiter = None


def get_health_checker() -> HealthChecker:
    global _health
    if _health is None:
        _health = HealthChecker()
    return _health


def get_rate_limiter(max_rpm: int = 60, enabled: bool = True) -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(max_rpm=max_rpm, enabled=enabled)
    return _limiter
