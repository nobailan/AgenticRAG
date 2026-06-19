"""
ragas_metrics.py — RAGAS 评测指标封装

功能：封装 RAGAS 库的三个核心指标，并提供纯 Python 的简易替代实现
      （用于 ragas 未安装时的 fallback）。

三个指标：
    - Faithfulness（忠实度）：生成的答案中有多少信息确实来自检索的文档
    - AnswerRelevancy（答案相关性）：答案与问题的相关程度
    - ContextRelevancy（上下文相关性）：检索到的文档与问题的相关程度

使用方式：
    from evaluation.ragas_metrics import compute_ragas_metrics_simple
    scores = compute_ragas_metrics_simple(question, answer, contexts, ground_truth)
"""

import logging
from typing import Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RAGASScore:
    """RAGAS 评测得分容器。

    Attributes:
        faithfulness: 忠实度，答案信息是否来自文档（0-1，越高越好）
        answer_relevancy: 答案相关性，答案是否切题（0-1）
        context_relevancy: 上下文相关性，检索文档是否相关（0-1）
    """
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_relevancy: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """转为字典，保留 4 位小数。"""
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_relevancy": round(self.context_relevancy, 4),
        }

    @property
    def average(self) -> float:
        """三项指标的算术平均，作为综合得分。"""
        return (self.faithfulness + self.answer_relevancy + self.context_relevancy) / 3


def _check_ragas() -> bool:
    """检测 ragas 库是否已安装。"""
    try:
        import ragas  # noqa: F401
        return True
    except ImportError:
        return False


def compute_ragas_metrics(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    llm_model: str = "gpt-4o-mini",
) -> RAGASScore:
    """用 RAGAS 库计算评测指标（需要安装 ragas 和 langchain-openai）。

    这是完整的基于 LLM 的评分方式，精度高但依赖外部 API。
    如果 ragas 未安装，会降级到简易计算方式。

    Args:
        question: 用户问题
        answer: 系统生成的答案
        contexts: 检索返回的文档文本列表
        ground_truth: 参考答案
        llm_model: 用于评分的 LLM（默认 gpt-4o-mini，成本低）

    Returns:
        RAGASScore 对象
    """
    if not _check_ragas():
        logger.warning("ragas 未安装，降级为简易评分。安装: pip install ragas")
        return compute_ragas_metrics_simple(question, answer, contexts, ground_truth)

    try:
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRelevancy
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI

        # 用便宜的模型做评分（不参与生成，只做评判）
        eval_llm = LangchainLLMWrapper(ChatOpenAI(
            model=llm_model,
            temperature=0,
        ))

        from ragas import SingleTurnSample

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )

        faithfulness = Faithfulness(llm=eval_llm)
        answer_rel = AnswerRelevancy(llm=eval_llm)
        context_rel = ContextRelevancy(llm=eval_llm)

        return RAGASScore(
            faithfulness=float(faithfulness.score(sample)),
            answer_relevancy=float(answer_rel.score(sample)),
            context_relevancy=float(context_rel.score(sample)),
        )

    except Exception as e:
        logger.error("RAGAS 计算失败: %s，降级为简易评分", e)
        return compute_ragas_metrics_simple(question, answer, contexts, ground_truth)


def compute_ragas_metrics_simple(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
) -> RAGASScore:
    """简易 RAGAS 评分——不依赖外部 LLM，纯基于词重叠计算。

    这是 ragas 不可用时的降级方案。评分逻辑简单粗暴，不如 LLM-based 评分精确，
    但可以提供大致的质量参考。

    计算逻辑：
        - Faithfulness: 答案中的词有多少出现在检索文档中
        - AnswerRelevancy: 答案长度合理性 + 答案与问题的词重叠
        - ContextRelevancy: 检索文档与问题的词重叠

    Args:
        question: 用户问题
        answer: 系统生成的答案
        contexts: 检索文档文本列表
        ground_truth: 参考答案（简易模式下暂未使用，预留参数）

    Returns:
        RAGASScore 对象
    """
    score = RAGASScore()

    # ContextRelevancy：检索文档与问题的关键词重叠率
    if contexts and question:
        q_words = set(question.lower().split())
        ctx_text = " ".join(contexts).lower()
        context_words = set(ctx_text.split())
        overlap = q_words & context_words
        score.context_relevancy = min(1.0, len(overlap) / max(1, len(q_words)))

    # AnswerRelevancy：答案长度得分 + 答案与问题的词重叠
    if answer and question:
        a_words = set(answer.lower().split())
        q_words = set(question.lower().split())
        overlap = a_words & q_words
        # 长度得分：太短(< 30 词)的答案扣分
        len_score = min(1.0, len(answer.split()) / 30)
        # 重叠得分
        overlap_score = min(1.0, len(overlap) / max(1, len(q_words)))
        score.answer_relevancy = (len_score + overlap_score) / 2

    # Faithfulness：答案中的词有多少来自检索文档
    if answer and contexts:
        C = set(" ".join(contexts).lower().split())
        A = set(answer.lower().split())
        if A:
            overlap = A & C
            score.faithfulness = len(overlap) / len(A)

    return score
