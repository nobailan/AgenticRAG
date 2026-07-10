"""
audit.py — 查询审计日志 (v0.7.1 Redis 升级)

功能：记录每次查询的完整上下文，支持双写：
      ① 本地 JSONL 文件（始终可用，无需外部依赖）
      ② Redis List（可选，用于集中收集、实时监控仪表盘）

记录内容：时间戳、用户、角色、问题、意图、答案长度、来源数、缓存情况

用法：
    audit = get_audit_logger()
    audit.log_query(question, result, user_id="marie", role="legal")
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

_AUDIT_REDIS_KEY = "audit:queries"


class AuditLogger:
    """查询审计日志器（JSONL + Redis 双写）。

    特性：
        - JSONL 文件按日滚动，防篡改（append-only）
        - Redis List 写入最近 N 条（便于仪表盘实时读取）
        - 写入失败不影响主流程（try/except + debug log）
    """

    def __init__(
        self,
        enabled: bool = True,
        log_dir: str = "logs",
        redis_url: Optional[str] = None,
    ):
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self._redis_url = redis_url
        self._redis = None
        self._redis_ok: Optional[bool] = None
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

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
            except Exception:
                self._redis_ok = False
        return self._redis if self._redis_ok else None

    # ------------------------------------------------------------------
    # 日志写入
    # ------------------------------------------------------------------

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
            result: run_rag() 返回的完整结果
            user_id: 用户标识
            role: 用户角色
            extra: 额外字段（节点耗时、PII 命中数等）
        """
        if not self.enabled:
            return

        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "user_id": user_id,
            "role": role,
            "question": question[:500],
            "question_hash": _hash_str(question),
            "intent": result.get("intent", "unknown"),
            "answer_length": len(result.get("answer", "")),
            "sources_count": len(result.get("retrieved_sources", [])),
            "from_cache": result.get("from_cache", False),
            "cache_type": result.get("cache_type", "none"),
            "trace_steps": len(result.get("reasoning_trace", [])),
        }
        if extra:
            entry.update(extra)

        # ① 写入本地 JSONL
        try:
            with open(self._log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("审计日志 JSONL 写入失败: %s", e)

        # ② 推送 Redis List（保留最近 1000 条）
        redis = self.redis
        if redis:
            try:
                redis.lpush(_AUDIT_REDIS_KEY, json.dumps(entry, ensure_ascii=False))
                redis.ltrim(_AUDIT_REDIS_KEY, 0, 999)  # 保留最近 1000 条
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def recent_queries(self, n: int = 20, source: str = "file") -> List[Dict]:
        """读取最近 n 条审计日志。

        Args:
            n: 条数
            source: "file"（本地 JSONL）或 "redis"（Redis List）

        Returns:
            日志条目列表（最新在前）
        """
        if source == "redis":
            redis = self.redis
            if redis:
                try:
                    raw = redis.lrange(_AUDIT_REDIS_KEY, 0, n - 1)
                    return [json.loads(r) for r in raw]
                except Exception:
                    pass

        # 文件读取
        path = self._log_path()
        if not path.exists():
            return []
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return list(reversed(entries[-n:]))

    def stats(self) -> Dict[str, Any]:
        """统计今日查询概况。"""
        entries = self.recent_queries(1000)
        today = time.strftime("%Y-%m-%d")
        today_entries = [e for e in entries if e.get("timestamp", "").startswith(today)]
        return {
            "today_queries": len(today_entries),
            "avg_answer_len": (
                sum(e.get("answer_length", 0) for e in today_entries)
                / max(1, len(today_entries))
            ) if today_entries else 0,
            "cache_hit_rate": (
                sum(1 for e in today_entries if e.get("from_cache"))
                / max(1, len(today_entries))
            ) if today_entries else 0,
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
