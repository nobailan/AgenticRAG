"""
auth.py — 基于角色的访问控制 (RBAC)

功能：提供文档级别的访问控制。每个用户/角色只能检索其授权范围内的文档。
      当前实现为轻量级内存方案，适合 Demo 和单机部署。

企业级升级方向：对接 LDAP/AD、JWT token、OPA 策略引擎。

角色定义（按企业知识库常见场景）：
    - admin:    全部文档
    - legal:    法务 + 合同 + 合规
    - finance:  财务 + 审计 + 转让定价
    - hr:       HR + CSE 会议纪要 + 培训
    - engineer: 技术 + 生产 + 质量 + 专利
    - viewer:   仅公开级文档

文档分类 → 角色映射（基于 metadata.classification）：
    - PUBLIC:              所有人
    - FINANCIAL/IFRS15_FLAG: finance, admin
    - LEGAL/LITIGATION:     legal, admin
    - HR/HR_PLANNING:       hr, admin
    - TECHNICAL/PATENT:     engineer, admin
    - CONFIDENTIAL:         admin only
"""

import logging
from typing import List, Set, Optional

logger = logging.getLogger(__name__)

# 角色 → 可访问的分类
ROLE_PERMISSIONS: dict = {
    "admin": {"*"},  # 全部
    "legal": {"PUBLIC", "LEGAL", "LITIGATION", "CONTRACTS", "COMPLIANCE_RISK",
              "SANCTIONS_RISK", "LICENSE_RISK", "IP_REGISTER", "IP_AGREEMENT",
              "NDA", "POLICY", "REGULATORY", "AUDIT", "GAP_ANALYSIS", "RELEVANT"},
    "finance": {"PUBLIC", "FINANCIAL", "IFRS15_FLAG", "TP_DOC", "ANALYSIS",
                "INVESTMENT", "KPI", "AUDIT", "RELEVANT"},
    "hr": {"PUBLIC", "HR_PLANNING", "MINUTES", "POLICY", "RELEVANT"},
    "engineer": {"PUBLIC", "TECHNICAL", "PATENT", "STANDARD", "PRODUCTION",
                 "QUALITY", "NCR", "CAPA", "TRACEABILITY", "SUPPORT", "RELEVANT"},
    "viewer": {"PUBLIC"},
}

# 任务敏感文档的元数据标记
_CLASSIFIED_TERMS = {"CLASSIFIED", "CONFIDENTIAL_DEFENSE", "SECRET"}


class RBACAuthorizer:
    """轻量 RBAC 授权器。

    用法：
        auth = RBACAuthorizer()
        auth.set_user("marie", "legal")
        allowed = auth.filter_chunks(chunks)  # 只返回 Marie 有权看的文档
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._user_roles: dict = {}  # user_id → role

    def set_user(self, user_id: str, role: str) -> None:
        """设置当前用户的角色。

        Args:
            user_id: 用户标识（来自 SSO 或 API key）
            role: 角色名（admin/legal/finance/hr/engineer/viewer）
        """
        if role not in ROLE_PERMISSIONS:
            logger.warning("未知角色 '%s'，降级为 viewer", role)
            role = "viewer"
        self._user_roles[user_id] = role
        logger.info("用户 '%s' 角色设为 '%s'", user_id, role)

    def get_permissions(self, user_id: str = "default") -> Set[str]:
        """获取用户可访问的文档分类集合。"""
        if not self.enabled:
            return {"*"}
        role = self._user_roles.get(user_id, "viewer")
        return ROLE_PERMISSIONS.get(role, {"PUBLIC"})

    def can_access(self, chunk_metadata: dict, user_id: str = "default") -> bool:
        """判断用户是否可以访问某条 chunk。

        Args:
            chunk_metadata: chunk.metadata 字典，含 classification 字段
            user_id: 用户标识

        Returns:
            True 可访问
        """
        if not self.enabled:
            return True

        permissions = self.get_permissions(user_id)
        if "*" in permissions:
            return True

        # 机密文档：仅 admin 可访问
        classification = str(chunk_metadata.get("classification", "PUBLIC")).upper()
        for term in _CLASSIFIED_TERMS:
            if term in classification:
                return False  # 非 admin 不能访问机密文档

        # 检查分类是否在授权范围内
        if classification in permissions:
            return True

        # "RELEVANT" 类型：对所有角色开放（通用业务文档）
        if classification == "RELEVANT":
            return True

        return False

    def filter_chunks(self, chunks: list, user_id: str = "default") -> list:
        """从 chunk 列表中过滤掉用户无权访问的条目。

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
