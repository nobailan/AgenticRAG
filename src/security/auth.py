"""
auth.py — 基于角色的访问控制 (RBAC) + Redis 持久化 (v0.7.1)

功能：文档级访问控制。用户→角色映射持久化到 Redis Hash，
      多副本共享、重启不丢失。Redis 不可用时降级为内存 dict。

角色定义：
    admin    — 全部文档
    legal    — 法务 + 合同 + 合规 + IP
    finance  — 财务 + 审计 + 转让定价
    hr       — HR + CSE 会议纪要 + 培训
    engineer — 技术 + 生产 + 质量 + 专利
    viewer   — 仅公开级文档

文档分类 → 角色的映射逻辑见 _classification_to_role()。

用法：
    auth = get_authorizer()
    auth.set_user("marie", "legal")           # 持久化到 Redis
    auth.filter_chunks(chunks, "marie")       # 过滤 Marie 无权看的 chunk
"""

import logging
from typing import Dict, List, Set, Optional

logger = logging.getLogger(__name__)

# 角色 → 可访问的分类集合
ROLE_PERMISSIONS: Dict[str, Set[str]] = {
    "admin": {"*"},
    "legal": {
        "PUBLIC", "LEGAL", "LITIGATION", "CONTRACTS", "COMPLIANCE_RISK",
        "SANCTIONS_RISK", "LICENSE_RISK", "IP_REGISTER", "IP_AGREEMENT",
        "NDA", "POLICY", "REGULATORY", "AUDIT", "GAP_ANALYSIS", "RELEVANT",
    },
    "finance": {
        "PUBLIC", "FINANCIAL", "IFRS15_FLAG", "TP_DOC", "ANALYSIS",
        "INVESTMENT", "KPI", "AUDIT", "RELEVANT",
    },
    "hr": {
        "PUBLIC", "HR_PLANNING", "MINUTES", "POLICY", "RELEVANT",
    },
    "engineer": {
        "PUBLIC", "TECHNICAL", "PATENT", "STANDARD", "PRODUCTION",
        "QUALITY", "NCR", "CAPA", "TRACEABILITY", "SUPPORT", "RELEVANT",
    },
    "viewer": {"PUBLIC"},
}

# 机密文档标记
_CLASSIFIED_TERMS = {"CLASSIFIED", "CONFIDENTIAL_DEFENSE", "SECRET"}

# Redis key 前缀
_REDIS_KEY = "rbac:users"


class RBACAuthorizer:
    """RBAC 授权器 + Redis 持久化。

    特性：
        - 用户→角色映射存储于 Redis Hash（rbac:users）
        - Redis 不可用时自动降级为内存 dict（重启丢失，但可用）
        - 权限判断：角色权限表 + 机密文档额外检查
    """

    def __init__(self, enabled: bool = True, redis_url: Optional[str] = None):
        self.enabled = enabled
        self._redis_url = redis_url
        self._redis = None
        self._redis_ok: Optional[bool] = None
        self._fallback: Dict[str, str] = {}  # Redis 不可用时的本地缓存

    # ------------------------------------------------------------------
    # Redis 连接
    # ------------------------------------------------------------------

    @property
    def redis(self):
        if self._redis is None and self.enabled:
            try:
                import redis
                url = self._redis_url
                if not url:
                    from src.core.config import config
                    url = getattr(config, "cache_redis_url", "redis://localhost:6379/0")
                self._redis = redis.Redis.from_url(
                    url, socket_connect_timeout=2, socket_timeout=2,
                    decode_responses=True,
                )
                self._redis.ping()
                self._redis_ok = True
                logger.info("RBAC: Redis 已连接 (%s)", url)
            except ImportError:
                logger.warning("RBAC: redis-py 未安装，降级为内存模式")
                self._redis_ok = False
            except Exception as e:
                logger.warning("RBAC: Redis 不可用 (%s)，降级为内存模式", e)
                self._redis_ok = False
        return self._redis if self._redis_ok else None

    # ------------------------------------------------------------------
    # 用户管理
    # ------------------------------------------------------------------

    def set_user(self, user_id: str, role: str) -> None:
        """设置用户角色（持久化到 Redis + 内存缓存）。

        Args:
            user_id: 用户标识（来自 SSO/API key）
            role: 角色名
        """
        if role not in ROLE_PERMISSIONS:
            logger.warning("未知角色 '%s'，降级为 viewer", role)
            role = "viewer"

        redis = self.redis
        if redis:
            try:
                redis.hset(_REDIS_KEY, user_id, role)
                logger.info("用户 '%s' 角色设为 '%s' (Redis)", user_id, role)
            except Exception as e:
                logger.warning("Redis 写入失败: %s，降级本地", e)
        self._fallback[user_id] = role

    def get_role(self, user_id: str = "default") -> str:
        """查询用户角色（优先 Redis，不可用时读本地）。"""
        redis = self.redis
        if redis:
            try:
                role = redis.hget(_REDIS_KEY, user_id)
                if role:
                    return role
            except Exception:
                pass
        return self._fallback.get(user_id, "viewer")

    def list_users(self) -> Dict[str, str]:
        """列出所有用户及角色。"""
        redis = self.redis
        if redis:
            try:
                return redis.hgetall(_REDIS_KEY) or {}
            except Exception:
                pass
        return dict(self._fallback)

    def remove_user(self, user_id: str) -> None:
        """删除用户。"""
        redis = self.redis
        if redis:
            try:
                redis.hdel(_REDIS_KEY, user_id)
            except Exception:
                pass
        self._fallback.pop(user_id, None)

    # ------------------------------------------------------------------
    # 权限判断
    # ------------------------------------------------------------------

    def get_permissions(self, user_id: str = "default") -> Set[str]:
        """获取用户可访问的文档分类集合。"""
        if not self.enabled:
            return {"*"}
        role = self.get_role(user_id)
        return ROLE_PERMISSIONS.get(role, {"PUBLIC"})

    def can_access(self, chunk_metadata: dict, user_id: str = "default") -> bool:
        """判断用户是否可以访问某条 chunk。

        Args:
            chunk_metadata: chunk.metadata 字典
            user_id: 用户标识

        Returns:
            True 可访问
        """
        if not self.enabled:
            return True

        permissions = self.get_permissions(user_id)
        if "*" in permissions:
            return True

        classification = str(chunk_metadata.get("classification", "PUBLIC")).upper()

        # 机密文档：仅 admin 可看
        for term in _CLASSIFIED_TERMS:
            if term in classification:
                return False

        if classification in permissions:
            return True
        if classification == "RELEVANT":
            return True

        return False

    def filter_chunks(self, chunks: list, user_id: str = "default") -> list:
        """从 chunk 列表中过滤无权访问的条目。

        Args:
            chunks: RetrievedChunk 对象列表
            user_id: 用户标识

        Returns:
            过滤后的 chunk 列表
        """
        if not self.enabled:
            return chunks

        before = len(chunks)
        filtered = [
            c for c in chunks
            if self.can_access(
                c.metadata if hasattr(c, "metadata") else c.get("metadata", {}),
                user_id,
            )
        ]
        after = len(filtered)
        if before != after:
            logger.info("RBAC 过滤: %d → %d chunks (用户=%s)", before, after, user_id)
        return filtered


# 模块级单例
_authorizer: Optional[RBACAuthorizer] = None


def get_authorizer(enabled: bool = True) -> RBACAuthorizer:
    global _authorizer
    if _authorizer is None:
        _authorizer = RBACAuthorizer(enabled=enabled)
    return _authorizer
