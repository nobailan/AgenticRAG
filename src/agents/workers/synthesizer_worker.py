"""
synthesizer_worker.py — 综合生成 Worker

功能：被 Supervisor 调度，基于检索结果和批评意见生成最终答案。
      这是多 Agent 流程的最后一个 Worker，负责汇总所有信息。

职责：
    - 综合 retriever_worker 的检索结果
    - 参考 critic_worker 的评审意见
    - 生成带 [doc_N] 引用的最终答案

使用方式：
    from src.agents.workers.synthesizer_worker import synthesizer_worker
    state = synthesizer_worker(state)
"""

import logging
from src.core.models import RAGState
from src.llm.llm_client import get_llm_response

logger = logging.getLogger(__name__)

SYNTHESIZER_PROMPT = """你是一个企业知识库助手。基于以下文档片段回答用户问题。
不要编造任何信息。如果文档中没有答案，明确说"没有找到相关信息"。
在每个引用句后附上来源 ID（如 [doc_5]）。

文档片段:
{chunks}

用户问题: {question}

评审意见: {critic_feedback}

答案:"""


def synthesizer_worker(state: RAGState) -> RAGState:
    """综合生成 Worker：基于检索结果和评审意见生成最终答案。

    步骤：
        1. 收集检索文档和评审意见
        2. 调用 LLM 生成答案
        3. 写入 state.final_answer 和 state.worker_results["synthesizer"]

    Args:
        state: 当前状态

    Returns:
        更新后的 state
    """
    question = state.question

    if state.accumulated_chunks:
        chunks_text = "\n\n".join(
            f"[{c.chunk_id}] (来源: {c.metadata.get('source_file', 'unknown')})\n{c.text}"
            for c in state.accumulated_chunks
        )
    else:
        chunks_text = "(未检索到任何文档)"

    critic_feedback = state.worker_results.get("critic", "无评审意见")

    try:
        prompt = SYNTHESIZER_PROMPT.format(
            chunks=chunks_text,
            question=question,
            critic_feedback=critic_feedback,
        )
        answer = get_llm_response(prompt)
        state.final_answer = answer
        state.worker_results["synthesizer"] = f"生成答案 ({len(answer)} 字符)"
        state.reasoning_trace.append(
            f"[synthesizer_worker] 答案已生成 ({len(answer)} 字符)"
        )

    except Exception as e:
        logger.error("SynthesizerWorker 失败: %s", e)
        state.final_answer = f"答案生成失败: {e}"
        state.worker_results["synthesizer"] = f"生成失败: {e}"
        state.reasoning_trace.append(f"[synthesizer_worker] 失败: {e}")

    return state
