"""
redis_client.py — Redis 连接客户端（轻量封装）

功能：对 redis-py 做薄封装，提供带容错降级的 get/set/exists 操作。
      Redis 不可用时（未安装 / 连接被拒 / 超时）自动降级，全部操作返回 None/False，
      不会导致应用崩溃。

设计原则：
    - 首次连接失败只在日志中告警一次，不重复刷屏
    - 所有方法都有 try/except 保护，Redis 宕机不影响主流程
    - 额外提供 get_json / set_json 方便存取字典数据

使用方式：
    from src.cache.redis_client import RedisClient
    client = RedisClient(redis_url="redis://localhost:6379/0")
    client.set("key", "value", ttl=3600)
    value = client.get("key")
"""

import json
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis 客户端封装，连接失败自动降级。

    所有操作在 Redis 不可用时会静默返回空值，不影响主业务逻辑。
    适合用作缓存层——缓存挂了，业务还能正常运行。

    Attributes:
        redis_url: Redis 连接地址（如 redis://localhost:6379/0）
        connected: 当前是否已连接
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """初始化 Redis 客户端（不会立即连接，首次使用时才连）。

        Args:
            redis_url: Redis 连接 URL 字符串
        """
        self.redis_url = redis_url
        self._client: Any = None
        self._connected: Optional[bool] = None  # None=未检查, True=通, False=断
        self._warned = False  # 连接失败只告警一次

    @property
    def connected(self) -> bool:
        """检查 Redis 是否可用（带缓存，避免每次操作都 ping）。"""
        if self._connected is None:
            self._connect()
        return self._connected or False

    def _connect(self) -> None:
        """尝试连接 Redis。

        连接失败不抛异常，设置 _connected=False 后静默降级。
        常见的失败原因：redis-py 未安装 / Redis 服务未启动 / 网络不通。
        """
        try:
            import redis
            self._client = redis.Redis.from_url(
                self.redis_url,
                socket_connect_timeout=2,  # 连接超时 2 秒
                socket_timeout=2,           # 读写超时 2 秒
                decode_responses=True,      # 返回 str 而非 bytes
            )
            self._client.ping()  # 验证连接
            self._connected = True
            logger.info("Redis 已连接: %s", self.redis_url)
        except ImportError:
            if not self._warned:
                logger.warning(
                    "redis-py 未安装，语义缓存不可用。安装: pip install redis"
                )
                self._warned = True
            self._connected = False
        except Exception as e:
            if not self._warned:
                logger.warning("Redis 连接失败 (%s)，缓存降级为内存模式", e)
                self._warned = True
            self._connected = False
            self._client = None

    def get(self, key: str) -> Optional[str]:
        """从 Redis 读取字符串值。

        Args:
            key: Redis 键名

        Returns:
            存储的值，键不存在或 Redis 不可用时返回 None
        """
        if not self.connected or self._client is None:
            return None
        try:
            return self._client.get(key)
        except Exception as e:
            logger.warning("Redis GET 失败: %s", e)
            return None

    def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        """写入键值对到 Redis，带过期时间。

        Args:
            key: 键名
            value: 字符串值
            ttl: 过期时间（秒），默认 1 小时

        Returns:
            True 写入成功，False 写入失败
        """
        if not self.connected or self._client is None:
            return False
        try:
            self._client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.warning("Redis SET 失败: %s", e)
            return False

    def exists(self, key: str) -> bool:
        """检查键是否存在。

        Args:
            key: Redis 键名

        Returns:
            True 键存在，False 不存在或 Redis 不可用
        """
        if not self.connected or self._client is None:
            return False
        try:
            return bool(self._client.exists(key))
        except Exception as e:
            logger.warning("Redis EXISTS 失败: %s", e)
            return False

    def get_json(self, key: str) -> Optional[dict]:
        """读取 JSON 值，自动反序列化为 dict。

        Args:
            key: Redis 键名

        Returns:
            反序列化后的 dict，读取失败返回 None
        """
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    def set_json(self, key: str, value: dict, ttl: int = 3600) -> bool:
        """将 dict 序列化后存入 Redis。

        Args:
            key: 键名
            value: 要存储的字典
            ttl: 过期时间（秒）

        Returns:
            True 成功
        """
        try:
            return self.set(key, json.dumps(value, ensure_ascii=False), ttl)
        except (TypeError, ValueError):
            return False

    def close(self) -> None:
        """关闭 Redis 连接。"""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._connected = False
