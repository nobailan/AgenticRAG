"""
audit.py — 查询审计日志

功能：记录每一次查询的完整上下文，满足合规审计要求。
      日志为 JSON Lines 格式，可按日期滚动。

记录内容：
    - 时间戳、用户标识、角色
    - 原始问题、意图分类
    - 检索到的文档数、最终答案长度
    - 缓存命中情况
    - 各节点耗时（可选）
    - 是否触发了 PII 告警

用法：
    from src.security.audit import AuditLogger
    audit = AuditLogger()
    audit.log_query(question="...", result={...})
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """查询审计日志器。

    每条日志为一行 JSON，写入 logs/audit_{date}.jsonl。
    设计原则：日志写入失败不影响主流程（try/except 包裹）。
    """

    def __init__(self, enabled: bool = True, log_dir: str = "logs"):
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self._ensure_dir()

    def _ensure_dir(self):
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _log_path(self) -> Path:
        date_str = time.strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.jsonl"

    def log_query(
        self,
        question: str,
        result: Dict[str, Any],
        user_id: str = "anonymous",
        role: str = "viewer",
        extra: Optional[Dict] = None,
    ) -> None:
        """记录一次查询。

        Args:
            question: 用户问题
            result: run_rag() 返回的完整结果 dict
            user_id: 用户标识
            role: 用户角色
            extra: 额外信息（如节点耗时、PII 命中）
        """
        if not self.enabled:
            return

        try:
            entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "user_id": user_id,
                "role": role,
                "question": question[:500],  # 截断过长问题
                "question_hash": _hash_str(question),  # 用于去重分析
                "intent": result.get("intent", "unknown"),
                "answer_length": len(result.get("answer", "")),
                "sources_count": len(result.get("retrieved_sources", [])),
                "from_cache": result.get("from_cache", False),
                "cache_type": result.get("cache_type", "none"),
                "trace_steps": len(result.get("reasoning_trace", [])),
            }
            if extra:
                entry.update(extra)

            with open(self._log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        except Exception as e:
            logger.debug("审计日志写入失败: %s", e)

    def recent_queries(self, n: int = 20) -> list:
        """读取最近 n 条审计日志（调试用）。"""
        path = self._log_path()
        if not path.exists():
            return []
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries[-n:]

    def stats(self) -> Dict[str, Any]:
        """统计今日查询概况。"""
        entries = self.recent_queries(1000)
        today = time.strftime("%Y-%m-%d")
        today_entries = [e for e in entries if e.get("timestamp", "").startswith(today)]
        return {
            "today_queries": len(today_entries),
            "avg_answer_len": (
                sum(e.get("answer_length", 0) for e in today_entries) / max(1, len(today_entries))
            ),
            "cache_hit_rate": (
                sum(1 for e in today_entries if e.get("from_cache"))
                / max(1, len(today_entries))
            ),
        }


def _hash_str(s: str) -> str:
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()[:12]


# 模块级单例
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(enabled: bool = True) -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(enabled=enabled)
    return _audit_logger
