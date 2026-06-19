"""
retriever_worker.py — 检索 Worker

功能：被 Supervisor 调度，执行文档检索任务。封装完整的检索链路（粗排 + 精排）。

使用方式：
    from src.agents.workers.retriever_worker import retriever_worker
    state = retriever_worker(state)
"""

import logging
from src.core.models import RAGState
from src.core.config import config

logger = logging.getLogger(__name__)


def retriever_worker(state: RAGState) -> RAGState:
    """检索 Worker：执行混合检索 + 可选精排。

    1. 确定检索查询（优先 active_query，否则 question）
    2. 调 HybridRetriever 做混合检索
    3. 去重合并到 accumulated_chunks
    4. 结果写入 worker_results["retriever"]

    Args:
        state: 当前状态

    Returns:
        更新后的 state
    """
    query = state.active_query or state.question
    logger.info("RetrieverWorker: 正在检索 '%s'...", query[:80])

    try:
        from src.retrieval.retriever import HybridRetriever
        retriever = HybridRetriever()
        chunks = retriever.search(query, top_k=config.top_k)

        # 去重合并
        existing_ids = {c.chunk_id for c in state.accumulated_chunks}
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        state.accumulated_chunks.extend(new_chunks)

        state.worker_results["retriever"] = (
            f"检索到 {len(chunks)} 篇文档（新增 {len(new_chunks)} 篇）"
        )
        state.reasoning_trace.append(
            f"[retriever_worker] '{query[:60]}' → {len(chunks)} 篇"
        )

    except Exception as e:
        logger.error("RetrieverWorker 失败: %s", e)
        state.worker_results["retriever"] = f"检索失败: {e}"
        state.reasoning_trace.append(f"[retriever_worker] 失败: {e}")

    return state
