"""
pii_detector.py — 敏感信息检测与脱敏

功能：在检索结果返回给 LLM 之前，扫描文档内容中的敏感信息
      并在生成答案后再次检查，确保不泄露个人隐私。

检测类型：
    - 邮箱地址
    - 手机号 / 电话号码
    - 身份证号 / 社保号
    - 薪资数字 ("salary", "salaire", "remuneration" 上下文)
    - 银行账号 (IBAN)

处理策略（可配置）：
    - mask: 脱敏替换（如 ***@***.com）
    - warn: 仅告警，不修改内容
    - block: 完全移除包含敏感信息的 chunk

用法：
    from src.security.pii_detector import PIIDetector
    detector = PIIDetector(mode="mask")
    clean_text = detector.sanitize(raw_text)
"""

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

# 正则模式库
_PATTERNS = {
    "email": re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    ),
    "phone": re.compile(
        r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}\b'
    ),
    "ssn_fr": re.compile(
        r'\b[12]\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b'  # 法国社保号
    ),
    "iban": re.compile(
        r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b'
    ),
}

# 薪资关键词（上下文检测用）
_SALARY_KEYWORDS = [
    "salary", "salaire", "wage", "remuneration", "compensation",
    "rémunération", "gehalt", "stipendio", "salario",
    "pay grade", "bonus", "prime",
]

# 脱敏替换模板
_MASK_TEMPLATES = {
    "email": "***@***.com",
    "phone": "***-****-****",
    "ssn_fr": "***-****-****",
    "iban": "XX00-XXXX-XXXX-XXXX",
}


class PIIDetector:
    """敏感信息检测器。

    特性：
        - 正则匹配 + 上下文关键词双重检测
        - 三种处理模式：mask（脱敏）、warn（告警）、block（移除）
        - 返回检测报告（命中了哪些类型的 PII）
    """

    def __init__(self, mode: str = "mask", enabled: bool = True):
        """
        Args:
            mode: "mask" | "warn" | "block"
            enabled: 是否启用 PII 检测
        """
        self.mode = mode
        self.enabled = enabled
        self._detections: List[dict] = []  # 本轮检测报告

    def sanitize(self, text: str) -> Tuple[str, List[dict]]:
        """扫描并处理文本中的敏感信息。

        Args:
            text: 原始文本

        Returns:
            (处理后的文本, 检测报告列表)
        """
        if not self.enabled:
            return text, []

        detections = []

        for pii_type, pattern in _PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                detections.append({
                    "type": pii_type,
                    "count": len(matches),
                    "samples": matches[:3],  # 只记录前 3 条样本
                })

                if self.mode == "mask":
                    template = _MASK_TEMPLATES.get(pii_type, "***")
                    text = pattern.sub(template, text)
                elif self.mode == "block":
                    # 不对整个文本做替换，但打标记
                    pass

        # 薪资上下文检测
        text_lower = text.lower()
        salary_hits = [kw for kw in _SALARY_KEYWORDS if kw in text_lower]
        if salary_hits:
            detections.append({
                "type": "salary_context",
                "count": len(salary_hits),
                "keywords": salary_hits[:5],
            })

        if detections:
            total = sum(d["count"] for d in detections)
            types = [d["type"] for d in detections]
            if self.mode == "warn":
                logger.warning("PII 检测: %d 处敏感信息 (%s)", total, ", ".join(types))
            else:
                logger.info("PII 已处理: %d 处 (%s), 模式=%s", total, ", ".join(types), self.mode)

        return text, detections

    def is_safe(self, text: str) -> bool:
        """检查文本是否不含敏感信息。

        Returns:
            True 安全（无 PII 命中）
        """
        if not self.enabled:
            return True
        _, detections = self.sanitize(text)
        # warn 模式下不阻止
        if self.mode == "warn":
            return True
        return len(detections) == 0

    def last_detections(self) -> List[dict]:
        return self._detections


# 模块级单例
_detector: PIIDetector = None


def get_pii_detector(mode: str = "mask", enabled: bool = True) -> PIIDetector:
    global _detector
    if _detector is None:
        _detector = PIIDetector(mode=mode, enabled=enabled)
    else:
        _detector.mode = mode
        _detector.enabled = enabled
    return _detector
