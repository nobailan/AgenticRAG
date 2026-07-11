"""
adaptive_chunker.py — 自适应分块器 + 小2大 Sentence Window (v0.8)

核心改进：
    1. 按文档类型选择分块策略（不再一刀切）
    2. Sentence Window：小 chunk 检索 + 大 chunk 生成

分块策略表：
    合同/法律  → 句子级，512 tokens
    财报/表格  → 表格独立 chunk + 周边文字 256 tokens
    技术文档   → 语义分块（实验），384 tokens
    通用文档   → 句子级，512 tokens（默认）

用法：
    from src.data.chunkers.adaptive_chunker import AdaptiveChunker
    chunker = AdaptiveChunker()
    chunks = chunker.chunk(elements, doc_type="contract")
"""

import logging
from typing import List, Dict, Any, Optional

from src.data.extractors.pdf_extractor import DocumentElement

logger = logging.getLogger(__name__)


# 按文档类型的分块配置
CHUNK_CONFIG = {
    "contract":   {"size": 512, "overlap": 128, "strategy": "sentence"},
    "financial":  {"size": 256, "overlap": 64,  "strategy": "table_aware"},
    "technical":  {"size": 384, "overlap": 96,  "strategy": "semantic"},
    "email":      {"size": 256, "overlap": 64,  "strategy": "by_thread"},
    "policy":     {"size": 768, "overlap": 192, "strategy": "sentence"},
    "generic":    {"size": 512, "overlap": 128, "strategy": "sentence"},
}


class AdaptiveChunker:
    """按文档类型自适应分块 + Sentence Window。

    两种输出模式：
        - standard: 每个 chunk 独立（兼容旧版）
        - sentence_window: 每个 chunk 附带 expanded_text（推荐的 v0.8 模式）
    """

    def __init__(
        self,
        doc_type: str = "generic",
        mode: str = "sentence_window",
        sentence_window_size: int = 1,  # 小 chunk 前后各扩展 N 句
        tokenizer_name: str = "cl100k_base",
    ):
        self.doc_type = doc_type
        self.mode = mode
        self.sentence_window_size = sentence_window_size
        self._token_counter = self._build_token_counter(tokenizer_name)

    @staticmethod
    def _build_token_counter(name: str):
        try:
            import tiktoken
            enc = tiktoken.get_encoding(name)
            return lambda t: len(enc.encode(t))
        except ImportError:
            return lambda t: len(t) // 4

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def chunk(
        self,
        elements: List[DocumentElement],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """对文档元素列表做自适应分块。

        Args:
            elements: 清洗后的 DocumentElement 列表
            metadata: 文档级元数据（会附加到每个 chunk）

        Returns:
            chunk dict 列表，每个 dict: {chunk_id, text, expanded_text?, metadata}
        """
        config = CHUNK_CONFIG.get(self.doc_type, CHUNK_CONFIG["generic"])
        strategy = config["strategy"]
        chunk_size = config["size"]
        chunk_overlap = config["overlap"]

        logger.info("分块: strategy=%s, size=%d, overlap=%d, mode=%s",
                     strategy, chunk_size, chunk_overlap, self.mode)

        if strategy == "table_aware":
            chunks = self._chunk_table_aware(elements, chunk_size, chunk_overlap, metadata)
        elif strategy == "sentence":
            chunks = self._chunk_sentence_level(elements, chunk_size, chunk_overlap, metadata)
        elif strategy == "by_thread":
            chunks = self._chunk_by_thread(elements, chunk_size, metadata)
        else:
            chunks = self._chunk_sentence_level(elements, chunk_size, chunk_overlap, metadata)

        logger.info("分块完成: %d chunks", len(chunks))
        return chunks

    # ------------------------------------------------------------------
    # 表格感知分块
    # ------------------------------------------------------------------

    def _chunk_table_aware(
        self, elements, chunk_size, chunk_overlap, metadata
    ) -> List[Dict]:
        """财报/表格类文档：表格元素独立保留，文字元素正常分块。"""
        chunks = []
        table_buffer = []
        text_buffer = []

        for el in elements:
            if el.category == "Table":
                # 表格独立成 chunk（不参与分词）
                chunks.append(self._make_chunk(
                    text=f"[TABLE]\n{el.text}",
                    metadata={**(metadata or {}), "element_type": "table",
                              "page": el.page_number},
                ))
            else:
                text_buffer.append(el.text)

        # 文字部分按句子级分块
        if text_buffer:
            chunks.extend(self._chunk_sentence_level(
                [DocumentElement(category="NarrativeText", text=" ".join(text_buffer),
                                 page_number=1, metadata={})],
                chunk_size, chunk_overlap, metadata,
            ))

        return chunks

    # ------------------------------------------------------------------
    # 句子级分块（默认策略）+ Sentence Window
    # ------------------------------------------------------------------

    def _chunk_sentence_level(
        self, elements, chunk_size, chunk_overlap, metadata
    ) -> List[Dict]:
        """句子级分块 + 可选 Sentence Window 扩展。"""
        # 收集所有句子
        sentences = []
        for el in elements:
            if el.category in ("Title", "NarrativeText", "ListItem"):
                sents = self._split_sentences(el.text)
                for s in sents:
                    sentences.append({"text": s, "page": el.page_number,
                                      "category": el.category})

        if not sentences:
            return []

        # 预计算 token 数
        sent_tokens = [self._token_counter(s["text"]) for s in sentences]

        # 贪婪分组
        chunks = []
        current_sents = []
        current_tokens = 0
        chunk_idx = 0

        for i, sent in enumerate(sentences):
            st = sent_tokens[i]
            if current_tokens + st > chunk_size and current_sents:
                # 当前 chunk 满了，产出
                chunks.append(self._make_sentence_window_chunk(
                    current_sents, sentences, chunk_idx, chunk_size, metadata
                ))
                chunk_idx += 1
                # Sentence overlap
                overlap_sents = []
                overlap_tokens = 0
                for s in reversed(current_sents):
                    t = self._token_counter(s["text"])
                    if overlap_tokens + t <= chunk_overlap:
                        overlap_sents.insert(0, s)
                        overlap_tokens += t
                    else:
                        break
                current_sents = overlap_sents
                current_tokens = overlap_tokens

            current_sents.append(sent)
            current_tokens += st

        # 尾部
        if current_sents:
            chunks.append(self._make_sentence_window_chunk(
                current_sents, sentences, chunk_idx, chunk_size, metadata
            ))

        return chunks

    def _make_sentence_window_chunk(
        self, current_sents, all_sents, chunk_idx, chunk_size, metadata
    ) -> Dict:
        """创建 Sentence Window chunk。

        小 chunk（检索用）= 当前句子组
        大 chunk（生成用）= 向前后各扩展 sentence_window_size 句
        """
        small_text = " ".join(s["text"] for s in current_sents).strip()

        if self.mode == "sentence_window":
            # 找到 current_sents 在 all_sents 中的位置
            first_text = current_sents[0]["text"] if current_sents else ""
            idx = 0
            for i, s in enumerate(all_sents):
                if s["text"] == first_text:
                    idx = i
                    break

            start = max(0, idx - self.sentence_window_size)
            end = min(len(all_sents), idx + len(current_sents) + self.sentence_window_size)
            expanded_text = " ".join(s["text"] for s in all_sents[start:end]).strip()
        else:
            expanded_text = small_text

        return {
            "chunk_id": f"doc_{chunk_idx}",
            "text": small_text,
            "expanded_text": expanded_text,
            "metadata": {**(metadata or {}),
                         "chunk_index": chunk_idx,
                         "chunk_size_tokens": self._token_counter(small_text),
                         "expanded_size_tokens": self._token_counter(expanded_text)},
        }

    # ------------------------------------------------------------------
    # 邮件线程分块
    # ------------------------------------------------------------------

    def _chunk_by_thread(self, elements, chunk_size, metadata) -> List[Dict]:
        """邮件类文档：按 From/To/Subject 边界切分。"""
        chunks = []
        for i, el in enumerate(elements):
            chunks.append(self._make_chunk(el.text, {
                **(metadata or {}), "element_type": "email_thread", "index": i,
            }))
        return chunks

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _make_chunk(self, text: str, metadata: dict) -> Dict:
        return {"chunk_id": "", "text": text, "metadata": metadata}

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """简单句子切分。"""
        import re
        # 在句末标点后切分
        parts = re.split(r'(?<=[.!?。！？])\s+', text)
        return [p.strip() for p in parts if p.strip()]
