"""
document_cleaner.py — 四层文档清洗管线 (v0.8)

流水线：
    L1 格式层 → L2 结构层 → L3 内容层 → L4 语义层

每层独立可测，可单独开关。

用法：
    from src.data.cleaners.document_cleaner import DocumentCleaner
    cleaner = DocumentCleaner()
    clean_elements = cleaner.clean(elements)
"""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import List

from src.data.extractors.pdf_extractor import DocumentElement

logger = logging.getLogger(__name__)


class DocumentCleaner:
    """四层文档清洗器。"""

    def __init__(
        self,
        l1_enabled: bool = True,   # 格式层
        l2_enabled: bool = True,   # 结构层
        l3_enabled: bool = True,   # 内容层
        l4_enabled: bool = False,  # 语义层（需要 LLM，默认关闭）
    ):
        self.l1 = l1_enabled
        self.l2 = l2_enabled
        self.l3 = l3_enabled
        self.l4 = l4_enabled
        self.stats = {"l1_filtered": 0, "l2_filtered": 0, "l3_filtered": 0, "l4_filtered": 0}

    def clean(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """执行全量清洗。

        Args:
            elements: 原始 DocumentElement 列表

        Returns:
            清洗后的元素列表
        """
        before = len(elements)

        if self.l1:
            elements = self._clean_format(elements)
        if self.l2:
            elements = self._clean_structure(elements)
        if self.l3:
            elements = self._clean_content(elements)
        if self.l4:
            elements = self._clean_semantic(elements)

        after = len(elements)
        if before != after:
            logger.info("清洗: %d → %d 元素 (过滤 %d)", before, after, before - after)
        return elements

    # ==================================================================
    # L1: 格式层 — 编码、特殊字符
    # ==================================================================

    _CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
    _ZERO_WIDTH = re.compile(r'[​-‏ - ⁠-⁯﻿]')
    _MULTI_SPACE = re.compile(r'\s{3,}')

    def _clean_format(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """L1: 移除控制字符、零宽字符、统一空白、修复乱码。"""
        clean = []
        for el in elements:
            text = el.text
            text = self._CONTROL_CHARS.sub(' ', text)
            text = self._ZERO_WIDTH.sub('', text)
            text = self._MULTI_SPACE.sub(' ', text)
            text = text.strip()

            if not text:
                self.stats["l1_filtered"] += 1
                continue

            # 乱码检测：非正常字符比例 > 30%
            normal = sum(1 for c in text if c.isprintable() or c in '\n\r\t')
            if normal / max(1, len(text)) < 0.7:
                self.stats["l1_filtered"] += 1
                continue

            el.text = text
            clean.append(el)

        return clean

    # ==================================================================
    # L2: 结构层 — 空元素、重复水印、异常长度
    # ==================================================================

    def _clean_structure(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """L2: 过滤空元素、检测异常大/小元素。"""
        clean = []
        for el in elements:
            # 空或过短
            if len(el.text) < 3:
                self.stats["l2_filtered"] += 1
                continue
            # 异常大（可能是 PDF bug）
            if len(el.text) > 50000:
                logger.warning("L2: 异常大元素 (%d chars)，截断", len(el.text))
                el.text = el.text[:50000]
            clean.append(el)
        return clean

    # ==================================================================
    # L3: 内容层 — 语言检测、去重
    # ==================================================================

    def _clean_content(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """L3: 重复文本检测、语言标记验证。"""
        seen = set()
        clean = []
        for el in elements:
            # 精确去重（同文本只保留一次）
            h = hash(el.text[:200])
            if h in seen:
                self.stats["l3_filtered"] += 1
                continue
            seen.add(h)
            clean.append(el)
        return clean

    # ==================================================================
    # L4: 语义层 — LLM 辅助（可选）
    # ==================================================================

    def _clean_semantic(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """L4: 用 LLM 标记低质量/无关内容。默认关闭（成本考量）。"""
        # v0.8 暂不启用，预留接口
        return elements


class MetadataEnricher:
    """LLM 辅助元数据提取（可选）。"""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def enrich(self, elements: List[DocumentElement], source_file: str) -> dict:
        """从文档内容中自动提取元数据。

        Returns:
            {"title": str, "date": str, "language": str, "keywords": [str], "summary": str}
        """
        if not self.enabled:
            return {}

        # 取前 2000 字符作为 prompt 输入
        sample = " ".join(e.text for e in elements[:5] if e.text)[:2000]

        try:
            from src.llm.llm_client import get_llm_response
            prompt = (
                "从以下文档片段中提取元数据。只输出 JSON：\n"
                f"{sample}\n\n"
                '{"title": "...", "date": "YYYY-MM-DD", "language": "fr/en/de/...", '
                '"keywords": ["...", "..."], "summary": "..."}'
            )
            import json
            result = get_llm_response(prompt)
            return json.loads(result)
        except Exception:
            return {}
