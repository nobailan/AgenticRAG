"""
ragas_metrics.py -- RAGAS evaluation metrics wrapper.

Encapsulates Faithfulness, AnswerRelevancy, and ContextRelevancy metrics
from the RAGAS library. Falls back gracefully if ragas is not installed.

Usage:
    from evaluation.ragas_metrics import compute_ragas_metrics
    scores = compute_ragas_metrics(question, answer, contexts, ground_truth)
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RAGASScore:
    """Container for RAGAS evaluation scores."""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_relevancy: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_relevancy": round(self.context_relevancy, 4),
        }

    @property
    def average(self) -> float:
        return (self.faithfulness + self.answer_relevancy + self.context_relevancy) / 3


def _check_ragas() -> bool:
    """Check if ragas library is installed."""
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
    """Compute RAGAS evaluation metrics for a single Q&A pair.

    Args:
        question: The user question.
        answer: The generated answer.
        contexts: List of retrieved context strings.
        ground_truth: The reference/expected answer.
        llm_model: LLM model name for RAGAS scoring (e.g., 'gpt-4o-mini').

    Returns:
        RAGASScore with faithfulness, answer_relevancy, context_relevancy.
    """
    if not _check_ragas():
        logger.warning("ragas not installed. Run: pip install ragas")
        return RAGASScore()

    try:
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            ContextRelevancy,
        )
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI

        # Set up LLM for RAGAS scoring
        eval_llm = LangchainLLMWrapper(ChatOpenAI(
            model=llm_model,
            temperature=0,
        ))

        # Prepare dataset entry
        from ragas import SingleTurnSample

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )

        # Compute metrics
        faithfulness = Faithfulness(llm=eval_llm)
        answer_rel = AnswerRelevancy(llm=eval_llm)
        context_rel = ContextRelevancy(llm=eval_llm)

        return RAGASScore(
            faithfulness=float(faithfulness.score(sample)),
            answer_relevancy=float(answer_rel.score(sample)),
            context_relevancy=float(context_rel.score(sample)),
        )

    except Exception as e:
        logger.error("RAGAS computation failed: %s", e)
        return RAGASScore()


def compute_ragas_metrics_simple(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
) -> RAGASScore:
    """Simplified RAGAS computation without external LLM dependency.

    Uses heuristic-based scoring when the full RAGAS library is unavailable.
    This is NOT as accurate as the LLM-based version but provides a baseline.

    Args:
        question: The user question.
        answer: The generated answer.
        contexts: List of retrieved context strings.
        ground_truth: The reference/expected answer.

    Returns:
        RAGASScore with approximate scores.
    """
    score = RAGASScore()

    # Context Relevancy: check if contexts contain question keywords
    if contexts and question:
        q_words = set(question.lower().split())
        context_words = set(" ".join(contexts).lower().split())
        overlap = q_words & context_words
        score.context_relevancy = min(1.0, len(overlap) / max(1, len(q_words)))

    # Answer Relevancy: check answer length and content
    if answer and question:
        a_words = set(answer.lower().split())
        q_words = set(question.lower().split())
        overlap = a_words & q_words
        # Penalize very short or off-topic answers
        len_score = min(1.0, len(answer.split()) / 30)
        overlap_score = min(1.0, len(overlap) / max(1, len(q_words)))
        score.answer_relevancy = (len_score + overlap_score) / 2

    # Faithfulness: check if answer content appears in contexts
    if answer and contexts:
        C = set(" ".join(contexts).lower().split())
        A = set(answer.lower().split())
        if A:
            overlap = A & C
            score.faithfulness = len(overlap) / len(A)

    return score
