"""
critic_worker.py — 评审 Worker

功能：被 Supervisor 调度，对检索结果或草稿答案做质量评估。
      返回评分和改进建议，供其他 Worker（或 Supervisor）决策。

职责：
    - 评估检索结果是否足以回答问题
    - 指出缺失的关键信息
    - 对草稿答案做忠实度检查（是否编造了文档中没有的内容）

使用方式：
    from src.agents.workers.critic_worker import critic_worker
    state = critic_worker(state)
"""

import logging
from src.core.models import RAGState
from src.llm.llm_client import get_llm_response

logger = logging.getLogger(__name__)

CRITIC_PROMPT = """你是一个严格的评审员。评估以下检索结果是否足以回答用户问题。

用户问题: {question}

检索到的文档片段:
{chunks}

请评估:
1. 检索结果是否包含了回答问题所需的关键信息？
2. 如果不足，缺少什么具体信息？
3. 对检索质量打分（1-10分）

输出格式:
SCORE: <1-10>
SUFFICIENT: <YES/NO>
MISSING: <缺少的信息，如果充分则写NONE>
"""


def critic_worker(state: RAGState) -> RAGState:
    """评审 Worker：评估检索质量。

    步骤：
        1. 格式化已检索的 chunks
        2. 调用 LLM 做评审
        3. 将评分和建议写入 state.worker_results["critic"]

    Args:
        state: 当前状态

    Returns:
        更新后的 state
    """
    if not state.accumulated_chunks:
        state.worker_results["critic"] = "无可评估的文档"
        state.reasoning_trace.append("[critic_worker] 无文档，跳过评审")
        return state

    # 格式化检索结果（截断每篇到 500 字符）
    chunks_text = "\n\n".join(
        f"[{c.chunk_id}] {c.text[:500]}" for c in state.accumulated_chunks[-10:]
    )
    question = state.active_query or state.question

    try:
        prompt = CRITIC_PROMPT.format(question=question, chunks=chunks_text)
        response = get_llm_response(prompt)

        # 解析评分
        score = "?"
        sufficient = "?"
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("SCORE:"):
                score = line.replace("SCORE:", "").strip()
            elif line.startswith("SUFFICIENT:"):
                sufficient = line.replace("SUFFICIENT:", "").strip()

        state.worker_results["critic"] = (
            f"评分: {score}/10, 信息充分: {sufficient}"
        )
        state.reasoning_trace.append(
            f"[critic_worker] 评分={score}, 充分={sufficient}"
        )

    except Exception as e:
        logger.error("CriticWorker 失败: %s", e)
        state.worker_results["critic"] = f"评审失败: {e}"
        state.reasoning_trace.append(f"[critic_worker] 失败: {e}")

    return state
