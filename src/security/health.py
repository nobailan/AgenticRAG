"""
health.py — 健康检查与分布式限流 (v0.7 → v0.7.1 Redis 升级)

功能：
    - /health 端点：各组件健康状态
    - RateLimiter：Redis token bucket 分布式限流（多副本安全）
    - 降级策略：Redis 不可用时自动切换本地内存限流

Redis 模式参考 E:\agentProject\harness_lab\backend\src\engine\artifact_store.py

用法：
    limiter = RateLimiter(max_rpm=60)
    if limiter.acquire("user_123"):
        process_query()
"""

import logging
import time
from collections import deque
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# ==========================================================================
# 健康检查
# ==========================================================================

class HealthChecker:
    """组件健康检查器。"""

    def check(self) -> Dict[str, Any]:
        components = {}

        # 索引
        try:
            from src.retrieval.retriever import is_loaded, get_chunk_count
            if is_loaded():
                components["index"] = {"status": "ok", "chunks": get_chunk_count()}
            else:
                components["index"] = {"status": "down", "message": "索引未加载"}
        except Exception as e:
            components["index"] = {"status": "down", "message": str(e)}

        # LLM
        import os
        providers = []
        for key in ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
            if os.environ.get(key):
                providers.append(key.replace("_API_KEY", "").lower())
        components["llm"] = {
            "status": "ok" if providers else "down",
            "available": providers or ["none"],
        }

        # Redis
        try:
            from src.core.config import config
            if config.cache_enabled:
                from src.cache.redis_client import RedisClient
                redis = RedisClient(redis_url=config.cache_redis_url)
                if redis.connected:
                    components["redis"] = {"status": "ok"}
                else:
                    components["redis"] = {"status": "degraded", "message": "不可用，缓存降级"}
            else:
                components["redis"] = {"status": "disabled"}
        except Exception as e:
            components["redis"] = {"status": "down", "message": str(e)}

        statuses = [c.get("status") for c in components.values()]
        overall = "degraded" if "down" in statuses else "ok"
        return {"status": overall, "timestamp": time.time(), "components": components}

    def is_ready(self) -> bool:
        s = self.check()
        return s["components"].get("index", {}).get("status") == "ok"


# ==========================================================================
# 分布式限流器（Redis token bucket）
# ==========================================================================

class RateLimiter:
    """Redis token bucket 限流器 + 本地内存降级。

    设计原则：
        - 优先用 Redis 做分布式限流（多副本共享计数）
        - Redis 不可用时自动降级为本地内存 deque
        - 降级时记录告警（不限流总比全拒好，视安全策略而定）

    算法：
        token bucket：每分钟补充 max_rpm 个 token，每个请求消耗 1 个。
        Redis Lua 脚本保证原子性。

    Attributes:
        max_rpm: 每分钟最大请求数
        enabled: 是否启用限流
        redis_url: Redis 连接地址（默认从 config 读取）
    """

    def __init__(
        self,
        max_rpm: int = 60,
        enabled: bool = True,
        redis_url: Optional[str] = None,
    ):
        self.max_rpm = max_rpm
        self.enabled = enabled
        self._redis_url = redis_url
        self._redis = None
        self._redis_ok: Optional[bool] = None
        # 本地降级
        self._local_window: deque = deque()

    # ------------------------------------------------------------------
    # Redis 连接
    # ------------------------------------------------------------------

    @property
    def redis(self):
        """懒连接 Redis（首次调用时连接）。"""
        if self._redis is None and self.enabled:
            try:
                import redis
                url = self._redis_url
                if not url:
                    from src.core.config import config
                    url = getattr(config, "cache_redis_url", "redis://localhost:6379/0")
                self._redis = redis.Redis.from_url(
                    url,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    decode_responses=True,
                )
                self._redis.ping()
                self._redis_ok = True
                logger.info("RateLimiter: Redis 已连接 (%s)", url)
            except ImportError:
                logger.warning("redis-py 未安装，限流降级为本地模式")
                self._redis_ok = False
            except Exception as e:
                logger.warning("Redis 不可用 (%s)，限流降级为本地模式", e)
                self._redis_ok = False
        return self._redis if self._redis_ok else None

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def acquire(self, user_id: str = "default") -> bool:
        """尝试获取一个请求许可（token bucket 算法）。

        Redis 模式下使用 EVAL 执行 Lua 脚本保证原子性。
        本地降级模式使用滑动窗口。

        Args:
            user_id: 用户标识（全局限流用同一个 key，每用户限流用不同 key）

        Returns:
            True 允许 / False 触发限流
        """
        if not self.enabled or self.max_rpm <= 0:
            return True

        redis = self.redis
        if redis:
            return self._acquire_redis(user_id)
        else:
            return self._acquire_local()

    def remaining(self, user_id: str = "default") -> int:
        """剩余可用请求数。"""
        if not self.enabled:
            return self.max_rpm

        redis = self.redis
        if redis:
            try:
                key = f"ratelimit:{user_id}"
                used = int(redis.get(key) or 0)
                return max(0, self.max_rpm - used)
            except Exception:
                pass
        return max(0, self.max_rpm - len(self._local_window))

    def stats(self) -> Dict[str, Any]:
        return {
            "max_rpm": self.max_rpm,
            "remaining": self.remaining(),
            "enabled": self.enabled,
            "backend": "redis" if self.redis else "local",
        }

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _acquire_redis(self, user_id: str) -> bool:
        """Redis token bucket（Lua 原子操作）。

        脚本逻辑：
            1. 获取当前 bucket 中的 token 数
            2. 检查上次填充时间，按时间比例补充 token
            3. 如果有 token，减 1 并返回 True
            4. 否则返回 False
        """
        lua_script = """
        local key = KEYS[1]
        local max_tokens = tonumber(ARGV[1])
        local now = tonumber(ARGV[2])
        local window = 60  -- 60 秒窗口

        local tokens = tonumber(redis.call('GET', key) or max_tokens)
        local last_refill = tonumber(redis.call('GET', key .. ':ts') or now)

        -- 按时间比例补充 token
        local elapsed = now - last_refill
        if elapsed > 0 then
            tokens = math.min(max_tokens, tokens + (elapsed / window) * max_tokens)
        end

        if tokens >= 1 then
            redis.call('SET', key, tokens - 1)
            redis.call('EXPIRE', key, window + 10)
            redis.call('SET', key .. ':ts', now)
            redis.call('EXPIRE', key .. ':ts', window + 10)
            return 1
        else
            return 0
        end
        """
        try:
            key = f"ratelimit:{user_id}"
            now = time.time()
            result = self.redis.eval(lua_script, 1, key, self.max_rpm, now)
            if not result:
                logger.warning("限流触发: %s (RPM=%d)", user_id, self.max_rpm)
            return bool(result)
        except Exception as e:
            logger.warning("Redis 限流失效: %s，降级放行", e)
            return True  # 故障开放（fail-open），不限流

    def _acquire_local(self) -> bool:
        """本地滑动窗口降级（单进程有效）。"""
        now = time.time()
        while self._local_window and self._local_window[0] < now - 60:
            self._local_window.popleft()

        if len(self._local_window) < self.max_rpm:
            self._local_window.append(now)
            return True

        logger.warning("本地限流触发: RPM=%d", self.max_rpm)
        return False


# ==========================================================================
# 模块级单例
# ==========================================================================

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
