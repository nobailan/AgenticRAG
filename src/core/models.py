"""
models.py — 共享数据模型

定义 RAG 系统中跨模块使用的核心数据结构。提取到独立文件是为了避免
retriever 和 workflow 之间的循环导入问题。

包含：
    - RetrievedChunk: 单条检索结果（chunk_id, text, score, metadata）
    - RAGState:     LangGraph 工作流全程流转的状态对象

使用方式：
    from src.core.models import RetrievedChunk, RAGState
"""

from typing import List, Optional, Dict, Literal, Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """单条检索结果。

    这是 retriever 和 workflow 之间传递文档片段的统一数据格式。
    FAISS + BM25 混合检索后，每条结果都会封装为这个对象。

    Attributes:
        chunk_id: 唯一标识，格式 "doc_0", "doc_1" ...
        text: 文档片段的完整文本
        score: RRF 融合后的检索得分（越高越相关）
        metadata: 来源元数据（source_file, page, entity, language 等）
    """

    chunk_id: str
    """唯一标识，格式 'doc_N'，N 是从 0 开始的整数"""

    text: str
    """文档片段的完整文本内容"""

    score: float
    """RRF 融合后的检索得分，越高越相关"""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """来源元数据。常见字段：source_file, page, entity, language, format, classification"""


class RAGState(BaseModel):
    """LangGraph 工作流全程状态。

    这个对象在 8 个节点之间流转，每个节点从中读取输入、写入输出。
    所有字段都有合理的默认值，初始化时只需传入 question。

    工作流节点与字段对应关系：
        classify_intent    → intent
        ask_clarification  → clarification_question
        plan_sub_questions → sub_questions, current_sub_idx
        retrieve           → accumulated_chunks
        rerank             → accumulated_chunks（替换为精排后的 top-k）
        check_sufficiency  → information_sufficient, missing_information
        refine_query       → active_query
        generate_answer    → final_answer
    """

    # ---- 输入 ----
    question: str = ""
    """用户原始问题"""

    # ---- classify_intent 输出 ----
    intent: Literal["simple", "multi_hop", "unclear"] = "simple"
    """意图分类结果：简单查询 / 多跳推理 / 语义模糊"""

    # ---- plan_sub_questions 输出 ----
    sub_questions: List[str] = Field(default_factory=list)
    """多跳推理时拆解出的子问题列表，长度 ≥ 2"""

    current_sub_idx: int = 0
    """当前正在处理的子问题索引（从 0 开始）"""

    active_query: str = ""
    """当前检索用的查询字符串。
    simple 路径 = 原问题；multi_hop 路径 = 当前子问题；retry 路径 = 改写后的查询"""

    # ---- retrieve + rerank 输出 ----
    accumulated_chunks: List[RetrievedChunk] = Field(default_factory=list)
    """累积检索结果，按 chunk_id 去重，按得分降序"""

    # ---- check_sufficiency 输出 ----
    information_sufficient: bool = False
    """当前检索结果是否足以回答问题"""

    missing_information: Optional[str] = None
    """信息不足时，描述缺少什么（如"缺少 2022 年的研发支出数字"）"""

    # ---- generate_answer 输出 ----
    final_answer: Optional[str] = None
    """最终生成的答案，含 [doc_N] 引用"""

    # ---- ask_clarification 输出 ----
    clarification_question: Optional[str] = None
    """意图为 unclear 时，生成的反问句"""

    # ---- 控制字段 ----
    retry_count: int = 0
    """当前子问题的检索-改写重试计数，上限由 config.max_retries 控制"""

    # ---- 审计追溯 ----
    reasoning_trace: List[str] = Field(default_factory=list)
    """每一步决策的日志。示例: '[classify_intent] intent=multi_hop'"""
