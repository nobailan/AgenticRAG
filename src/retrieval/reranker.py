"""
reranker.py — Cross-encoder 精排模块

功能：在混合检索（FAISS + BM25 + RRF）之后，用 cross-encoder 模型对候选文档
      做二次精排，把最相关的文档排在前面，砍掉相关性低的文档。

原理：bi-encoder（bge-base-en-v1.5）速度快但精度有限；cross-encoder 把
      (query, document) 拼成一对送进模型打分，精度高但速度慢。
      所以先用 bi-encoder 粗筛 20 篇，再用 cross-encoder 精排保留 top-5。

在工作流中的位置：retrieve → rerank → check_sufficiency

使用方式：
    from src.retrieval.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    top_docs = reranker.rerank(query, documents)
"""

import logging
import time
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder 精排器。

    加载 sentence-transformers 的 cross-encoder 模型（默认 BAAI/bge-reranker-base），
    对候选文档列表逐条计算与查询的相关性分数，按分数降序排列后截取 top-k 返回。

    设计要点：
        - 模型采用懒加载：首次调用 rerank() 时才加载模型，避免拖慢启动
        - 文档文本截断到 max_length（默认 512 字符），防止超长输入拖慢推理
        - 通过 enabled 开关控制是否启用精排，关闭时退化为按原始分排序
        - 重排耗时超过 500ms 会打印警告日志

    Attributes:
        model_name: HuggingFace 模型名或本地路径
        top_k: 精排后保留的文档数
        max_length: 送入模型的单文档最大字符数
        enabled: 是否启用精排；False 时退化为原始排序
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        top_k: int = 5,
        max_length: int = 512,
        enabled: bool = True,
    ):
        """初始化精排器。

        Args:
            model_name: cross-encoder 模型名（HuggingFace ID 或本地路径）
            top_k: 精排后保留的文档数量
            max_length: 单篇文档送入模型前的最大字符数（截断）
            enabled: 是否启用精排
        """
        self.model_name = model_name
        self.top_k = top_k
        self.max_length = max_length
        self.enabled = enabled
        self._model = None  # 懒加载，首次使用时才初始化

    @property
    def model(self):
        """懒加载 cross-encoder 模型。

        只有在 self.enabled=True 且首次调用时才初始化模型，启动时不会加载。
        如果 sentence-transformers 未安装或模型加载失败，自动降级为禁用状态。
        """
        if self._model is None and self.enabled:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("正在加载 cross-encoder 模型: %s", self.model_name)
                self._model = CrossEncoder(self.model_name)
                logger.info("Cross-encoder 模型加载完成")
            except ImportError:
                logger.error(
                    "sentence-transformers 未安装，无法使用精排功能。"
                    "安装命令: pip install sentence-transformers"
                )
                self.enabled = False
            except Exception as e:
                logger.error("Cross-encoder 模型加载失败: %s", e)
                self.enabled = False
        return self._model

    def rerank(
        self, query: str, documents: List[Dict], return_all_on_fallback: bool = True
    ) -> List[Dict]:
        """对候选文档列表做精排，返回 top-k 篇。

        流程：
            1. 检查 enabled 开关，若关闭则按原始检索分排序返回
            2. 对每篇文档截断到 max_length，与查询拼成 (query, doc) 对
            3. 用 cross-encoder 逐对打分
            4. 按分数降序排列，截取 top_k 篇
            5. 每篇文档的 dict 中新增 'rerank_score' 字段
            6. 若重排过程出错，降级为按原始分排序

        Args:
            query: 用户查询字符串
            documents: 候选文档列表，每个 dict 至少包含 'text' 和 'score' 键
            return_all_on_fallback: 降级时返回前 top_k 篇(True)还是全部(False)

        Returns:
            精排后的文档列表，长度 ≤ top_k，每篇含 'rerank_score' 字段
        """
        if not documents:
            return []

        # 精排已禁用 → 按原始检索分排序，直接截取 top_k
        if not self.enabled:
            docs = sorted(documents, key=lambda d: d.get("score", 0), reverse=True)
            return docs[: self.top_k]

        start_time = time.time()

        try:
            # 构建 (query, doc_text) 对，同时截断过长文本
            pairs = []
            for doc in documents:
                text = doc.get("text", "")
                if len(text) > self.max_length:
                    text = text[: self.max_length]
                pairs.append([query, text])

            # 用 cross-encoder 一次性对所有 pair 打分
            scores = self.model.predict(pairs, show_progress_bar=False)

            # 把分数挂到每个文档上
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)

            # 按精排分降序排列，截取 top_k
            ranked = sorted(
                documents, key=lambda d: d.get("rerank_score", 0), reverse=True
            )
            result = ranked[: self.top_k]

            # 耗时统计
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                "精排完成: %d 篇 → top-%d, 耗时 %.0f ms (模型: %s)",
                len(documents),
                len(result),
                elapsed_ms,
                self.model_name,
            )

            # 超过阈值时告警（目标 < 500ms）
            if elapsed_ms > 500:
                logger.warning(
                    "精排耗时 %.0f ms 超过 500ms 阈值，"
                    "建议换用小模型或减小 max_length",
                    elapsed_ms,
                )

            return result

        except Exception as e:
            logger.error("精排失败: %s，降级为原始排序", e)
            docs = sorted(documents, key=lambda d: d.get("score", 0), reverse=True)
            return docs if not return_all_on_fallback else docs[: self.top_k]


# ---------------------------------------------------------------------------
# 模块级单例
# ---------------------------------------------------------------------------

_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(
    model_name: str = "BAAI/bge-reranker-base",
    top_k: int = 5,
    enabled: bool = True,
) -> CrossEncoderReranker:
    """获取全局 CrossEncoderReranker 单例。

    整个应用只初始化一次 cross-encoder 模型（加载模型比较耗时），
    后续调用直接复用已加载的实例。

    Args:
        model_name: 模型名
        top_k: 精排保留数
        enabled: 是否启用

    Returns:
        CrossEncoderReranker 实例（首次调用时创建）
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(
            model_name=model_name, top_k=top_k, enabled=enabled
        )
    return _reranker
