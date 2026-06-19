"""
supervisor.py — 多 Agent 架构的 Supervisor（调度器）节点

功能：分析用户问题，将其拆解为子任务，分配给对应的 Worker 执行。
      充当整个多 Agent 系统的"大脑"和协调者。

决策流程：
    1. 分析问题类型（简单 / 复杂 / 多跳）
    2. 生成子任务计划（每个子任务指定 worker_type）
    3. 按顺序调度 Worker 执行
    4. 汇总结果

Worker 类型：
    - retriever:    检索相关文档
    - critic:       对检索结果/草稿答案做质量评估
    - synthesizer:  综合所有信息生成最终答案

使用方式（在 workflow 中）：
    from src.agents.supervisor import supervisor_node
    state = supervisor_node(state)
"""

import logging
from typing import List, Dict

from src.core.config import config
from src.core.models import RAGState
from src.llm.llm_client import get_llm_response

logger = logging.getLogger(__name__)

# Supervisor 的 system prompt
SUPERVISOR_SYSTEM_PROMPT = """你是一个 RAG 系统的 Supervisor Agent。

你的职责是分析用户问题，决定需要哪些 Worker 来协作完成任务。

可用的 Worker：
- retriever: 从企业文档库检索相关文档片段
- critic: 评估检索结果的质量，判断是否足以回答问题
- synthesizer: 综合检索结果生成最终答案

对于简单问题（直接可以从文档中查找答案），使用：retriever → synthesizer
对于复杂/多跳问题，使用：retriever → critic → (可能再次retriever) → synthesizer

输出格式（严格遵守）：
PLAN: <用一句话描述整体计划>
WORKERS:
- retriever: <检索什么内容>
- critic: <评估什么方面>
- synthesizer: <如何综合信息>
"""


def supervisor_node(state: RAGState) -> RAGState:
    """Supervisor 节点：分析问题，生成任务计划，分配给 Worker。

    这个节点在意图分类之后、Worker 执行之前运行。
    它不直接执行检索或生成，只做规划和调度。

    Args:
        state: 当前 RAGState，至少需包含 question 和 intent

    Returns:
        更新后的 state，填充了 supervisor_plan 和 subtasks
    """
    question = state.question
    intent = state.intent

    # 简单问题：跳过复杂的多 Agent 流程
    if intent == "simple":
        state.supervisor_plan = "简单查询，直接检索+生成"
        state.subtasks = [
            {"worker_type": "retriever", "description": f"检索与'{question}'相关的文档"},
            {"worker_type": "synthesizer", "description": "基于检索结果生成答案"},
        ]
        state.reasoning_trace.append(
            f"[supervisor] 简单问题 → retriever + synthesizer"
        )
        return state

    # 复杂/多跳问题：用 LLM 生成详细计划
    try:
        prompt = (
            f"用户问题: {question}\n"
            f"意图类型: {intent}\n\n"
            f"请分析这个问题，生成执行计划和 Worker 分配。"
        )
        plan = get_llm_response(prompt, system_prompt=SUPERVISOR_SYSTEM_PROMPT)
        state.supervisor_plan = plan
        state.reasoning_trace.append(f"[supervisor] 计划: {plan[:100]}...")

        # 解析 Worker 分配
        subtasks = _parse_worker_plan(plan)
        if not subtasks:
            # 解析失败时提供默认分配
            subtasks = [
                {"worker_type": "retriever", "description": f"检索'{question}'相关文档"},
                {"worker_type": "synthesizer", "description": "基于检索结果生成答案"},
            ]
        state.subtasks = subtasks

    except Exception as e:
        logger.warning("Supervisor 规划失败: %s，使用默认分配", e)
        state.supervisor_plan = "规划失败，使用默认流程"
        state.subtasks = [
            {"worker_type": "retriever", "description": f"检索'{question}'"},
            {"worker_type": "synthesizer", "description": "生成答案"},
        ]
        state.reasoning_trace.append(f"[supervisor] 规划失败: {e}")

    return state


def _parse_worker_plan(plan: str) -> List[Dict[str, str]]:
    """从 Supervisor 的 LLM 输出中解析 Worker 分配。

    支持的格式：
        - retriever: <描述>
        - critic: <描述>
        - synthesizer: <描述>

    Args:
        plan: LLM 生成的计划文本

    Returns:
        子任务列表
    """
    subtasks = []
    lines = plan.split("\n")
    for line in lines:
        line = line.strip().lstrip("- ")
        for worker_type in ["retriever", "critic", "synthesizer"]:
            if line.lower().startswith(f"{worker_type}:"):
                desc = line[len(worker_type) + 1:].strip()
                subtasks.append({"worker_type": worker_type, "description": desc})
    return subtasks
