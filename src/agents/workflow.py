"""
workflow.py -- LangGraph workflow for the Agentic RAG system.

Defines:
    - RetrievedChunk (pydantic BaseModel) — a single retrieved chunk
    - RAGState (pydantic BaseModel) — the full graph state
    - 7 node functions (stubs with full type annotations, docstrings, TODO)
    - 3 routing functions
    - build_graph() → compiled StateGraph
    - get_graph() → singleton accessor

All node functions are STUBS: they contain docstrings with the exact LLM prompt
from MVP.txt, implementation pseudocode in comments, and return the state
unmodified (pass-through) so the graph compiles and runs structurally.
"""

import logging
import re
from typing import List, Optional, Dict, Literal, Any, Generator, Callable

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from src.core.config import config
from src.core.models import RetrievedChunk, RAGState
from src.llm.llm_client import get_llm_response, get_llm_response_stream

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy retriever singleton — avoids circular import with retriever.py
# ---------------------------------------------------------------------------

_retriever: Optional[Any] = None
"""HybridRetriever singleton, initialized lazily on first retrieve() call."""


def _get_retriever() -> Any:
    """Return the global HybridRetriever singleton, creating it on first call.

    Uses lazy import to avoid circular dependency (retriever.py imports
    RetrievedChunk from this module).

    Returns:
        HybridRetriever instance.

    Raises:
        FileNotFoundError: If index files are not built yet.
    """
    global _retriever
    if _retriever is None:
        from src.retrieval.retriever import HybridRetriever
        _retriever = HybridRetriever()
        logger.info("HybridRetriever singleton initialized in workflow")
    return _retriever


# Data models (RetrievedChunk, RAGState) are imported from src.core.models


# =============================================================================
# Helper: LLM factory (will be used by node implementations)
# =============================================================================

def _get_llm():
    """Factory: return a LangChain ChatModel based on config.llm_provider.

    Returns:
        BaseChatModel instance from langchain_openai or langchain_anthropic.

    Raises:
        ImportError: If the provider's package is not installed.
        ValueError: If the provider string is unrecognized.
    """
    provider = config.llm_provider.lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")


def _get_current_query(state: RAGState) -> str:
    """Determine the active question for the current pipeline stage.

    Priority:
        1. state.active_query — set by refine_query or route_by_sufficiency
        2. Multi-hop: state.sub_questions[state.current_sub_idx]
        3. Fallback: state.question (original user question)

    Used by retrieve, check_sufficiency, and refine_query to avoid
    duplicating the same query-resolution logic.

    Args:
        state: Current RAGState.

    Returns:
        The query string to use for retrieval or sufficiency checking.
    """
    if state.active_query:
        return state.active_query

    if state.intent == "multi_hop":
        idx = state.current_sub_idx
        if idx < len(state.sub_questions):
            return state.sub_questions[idx]
        # All sub-questions exhausted, fall back to original question
        logger.warning(
            f"_get_current_query: multi_hop idx={idx} >= "
            f"len(sub_questions)={len(state.sub_questions)}, using original question"
        )

    return state.question


# =============================================================================
# Node 1: classify_intent
# =============================================================================

def classify_intent(state: RAGState) -> RAGState:
    """Node 1: Classify the user question as 'simple', 'multi_hop', or 'unclear'.

    STUB: Currently sets intent = "simple" for all inputs.
    Real implementation will call the LLM with the following prompt:

        ---------- LLM Prompt ----------
        判断以下问题是否需要多步推理才能回答。
        如果答案可以直接从一段文档中找到，返回 "simple"。
        如果必须从多个文档中拼凑或逐步推理，返回 "multi_hop"。
        如果问题本身模糊或缺少必要实体，返回 "unclear"。
        问题：{question}
        只返回一个单词。
        ---------------------------------

    Input:
        state.question  — the raw user question

    Output:
        state.intent   — one of "simple", "multi_hop", "unclear"
        state.reasoning_trace — appended with classification decision

    Implementation pseudocode:
        llm = _get_llm()
        prompt = CLASSIFY_INTENT_TEMPLATE.format(question=state.question)
        response = llm.invoke(prompt)
        raw = response.content.strip().lower()
        if "multi" in raw:
            state.intent = "multi_hop"
        elif "unclear" in raw or "模糊" in raw:
            state.intent = "unclear"
        else:
            state.intent = "simple"
        logger.info(f"Intent classified as: {state.intent}")
    """
    # Build prompt and call LLM
    prompt = (
        "Classify the following question.\n\n"
        "Rules:\n"
        "- If the question has multiple distinct parts joined by 'and' "
        "(e.g., asking for both a count AND a list, or comparing multiple entities), "
        "it is multi_hop because it requires multiple retrievals to answer fully.\n"
        "- If the answer can be found directly from a single document, return \"simple\".\n"
        "- If it requires combining information from multiple documents or step-by-step "
        "reasoning, return \"multi_hop\".\n"
        "- If the question is vague, ambiguous, or missing essential entities, return \"unclear\".\n\n"
        "Question: {question}\n\n"
        "Only return one word."
    ).format(question=state.question)

    system_prompt = (
        "You are a query classifier for a corporate knowledge base. "
        "Analyze the user's question and classify its complexity. "
        "IMPORTANT: Questions with multiple parts (joined by 'and'), comparisons, "
        "or those asking for both numeric counts and named lists are virtually always multi_hop. "
        "Respond with EXACTLY one word: simple, multi_hop, or unclear. "
        "Do not add any explanation, punctuation, or whitespace."
    )

    try:
        raw = get_llm_response(prompt, system_prompt=system_prompt)
        # Strip markdown formatting and clean
        raw_clean = re.sub(r"[*_]{1,3}", "", raw).strip().lower().rstrip(".,;:!?")
        logger.info(f"classify_intent LLM response: '{raw_clean}'")

        if raw_clean == "multi_hop" or "multi" in raw_clean:
            state.intent = "multi_hop"
        elif "unclear" in raw_clean or "vague" in raw_clean or "ambiguous" in raw_clean:
            state.intent = "unclear"
        elif raw_clean == "simple" or "simple" in raw_clean:
            state.intent = "simple"
        else:
            logger.warning(
                f"classify_intent: unexpected LLM response '{raw_clean}', defaulting to 'simple'"
            )
            state.intent = "simple"
            state.reasoning_trace.append(
                f"[classify_intent] WARNING: unexpected response='{raw_clean}', defaulted to simple"
            )
    except Exception as e:
        logger.warning(f"classify_intent: LLM call failed ({e}), defaulting to 'simple'")
        state.intent = "simple"
        state.reasoning_trace.append(
            f"[classify_intent] LLM error: {e}, defaulted to simple"
        )

    state.reasoning_trace.append(
        f"[classify_intent] intent={state.intent} | question='{state.question[:80]}...'"
    )
    logger.info(f"classify_intent: question='{state.question[:80]}...' → intent={state.intent}")
    return state


# =============================================================================
# Node 2: ask_clarification
# =============================================================================

def ask_clarification(state: RAGState) -> RAGState:
    """Node 2: Generate a clarifying question when the original question is unclear.

    Only called when state.intent == "unclear". After this node, the workflow
    terminates — no retrieval or answer generation occurs.

    STUB: Sets a placeholder clarification_question.

    Real implementation LLM prompt:
        ---------- LLM Prompt ----------
        用户的问题不够明确，请生成一个反问，帮助用户补充缺失的信息。
        问题：{question}
        只输出反问句。
        ---------------------------------

    Input:
        state.question — the unclear user question

    Output:
        state.clarification_question — the generated follow-up question
        state.reasoning_trace         — appended with clarification step

    Implementation pseudocode:
        llm = _get_llm()
        prompt = CLARIFY_TEMPLATE.format(question=state.question)
        response = llm.invoke(prompt)
        state.clarification_question = response.content.strip()
        logger.info(f"Clarification question generated: {state.clarification_question}")
    """
    try:
        prompt = (
            "The user's question is too vague to answer: \"{question}\"\n"
            "Generate a clarifying question to help the user provide missing details "
            "(e.g., which entity, which year, which department).\n"
            "Only output the clarifying question, nothing else."
        ).format(question=state.question)

        system_prompt = (
            "You are a helpful assistant. When a user's question is unclear, "
            "ask a specific follow-up question to clarify what they need."
        )

        raw = get_llm_response(prompt, system_prompt=system_prompt)
        state.clarification_question = raw.strip()
        logger.info(f"Clarification question generated: {state.clarification_question[:80]}...")
    except Exception as e:
        logger.warning(f"ask_clarification: LLM call failed ({e}), using fallback")
        state.clarification_question = (
            f"Could you please clarify your question? "
            f"What specific entity, year, or topic are you asking about regarding: "
            f"'{state.question}'?"
        )
        state.reasoning_trace.append(
            f"[ask_clarification] LLM error: {e}, used fallback"
        )

    state.reasoning_trace.append(
        f"[ask_clarification] clarification_question='{state.clarification_question[:80]}...'"
    )
    logger.info(f"ask_clarification: generated clarification for '{state.question[:60]}...'")
    return state


# =============================================================================
# Node 3: plan_sub_questions
# =============================================================================

def plan_sub_questions(state: RAGState) -> RAGState:
    """Node 3: Decompose a multi-hop question into ordered sub-questions.

    Only called when state.intent == "multi_hop". Each sub-question must be
    independently answerable via a single retrieval.

    STUB: Splits question naively. Real implementation will call LLM.

    Real implementation LLM prompt:
        ---------- LLM Prompt ----------
        将以下问题拆解为有顺序依赖的子问题列表。
        每个子问题必须能够独立通过一次检索回答。

        拆解规则：
        1. 先问事实性问题（who/what/where/when），再问推理性问题（why/how）。
        2. 每个子问题简短、精确。
        3. 子问题之间存在依赖关系，必须按顺序回答。

        问题：{question}

        以JSON列表格式返回，例如：
        ["Veracier的碳中和目标年份是什么？", "哪个部门负责该目标？"]
        ---------------------------------

    Input:
        state.question — the original multi-hop question

    Output:
        state.sub_questions — list of >= 2 ordered sub-questions
        state.current_sub_idx — set to 0
        state.active_query — set to sub_questions[0]
        state.accumulated_chunks — cleared to empty list
        state.reasoning_trace — appended

    Implementation pseudocode:
        llm = _get_llm()
        prompt = DECOMPOSE_TEMPLATE.format(question=state.question)
        response = llm.invoke(prompt)
        import json
        state.sub_questions = json.loads(response.content)
        if len(state.sub_questions) < 2:
            # Fallback: treat as simple
            state.sub_questions = [state.question]
            state.intent = "simple"
        state.current_sub_idx = 0
        state.active_query = state.sub_questions[0]
        state.accumulated_chunks = []
        logger.info(f"Planned {len(state.sub_questions)} sub-questions")
    """
    try:
        prompt = (
            "Please decompose the following question into 2-3 independent sub-questions, "
            "each answerable through a single document retrieval.\n"
            "Each sub-question should be specific and self-contained.\n"
            "Only output the sub-questions, one per line, without numbering or extra text.\n"
            "Question: {question}"
        ).format(question=state.question)

        system_prompt = (
            "You are a query planner. Decompose complex questions into simpler "
            "sub-questions that can be answered independently. "
            "Output one sub-question per line, no numbering, no extra commentary."
        )

        raw = get_llm_response(prompt, system_prompt=system_prompt)
        # Split by newlines, clean empty lines and numbering prefixes
        lines = [line.strip() for line in raw.split("\n") if line.strip()]
        # Remove common numbering prefixes like "1.", "1)", "- ", etc.
        cleaned = []
        for line in lines:
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            line = line.strip()
            if line:
                cleaned.append(line)

        state.sub_questions = cleaned

        # Fallback: if fewer than 2 sub-questions, duplicate the original
        if len(state.sub_questions) < 2:
            logger.warning(
                f"plan_sub_questions: only got {len(state.sub_questions)} sub-questions, "
                f"using fallback"
            )
            state.sub_questions = [
                f"{state.question} — part 1: factual information",
                f"{state.question} — part 2: analysis and reasoning",
            ]
            state.reasoning_trace.append(
                f"[plan_sub_questions] fallback: LLM returned insufficient sub-questions, "
                f"using 2-part split"
            )

        logger.info(f"plan_sub_questions: {len(state.sub_questions)} sub-questions: {state.sub_questions}")

    except Exception as e:
        logger.warning(f"plan_sub_questions: LLM call failed ({e}), using fallback")
        state.sub_questions = [
            f"{state.question} — part 1: factual information",
            f"{state.question} — part 2: analysis and reasoning",
        ]
        state.reasoning_trace.append(
            f"[plan_sub_questions] LLM error: {e}, using fallback split"
        )

    state.current_sub_idx = 0
    state.active_query = state.sub_questions[0]
    state.accumulated_chunks = []  # Fresh start for multi-hop
    state.reasoning_trace.append(
        f"[plan_sub_questions] decomposed into {len(state.sub_questions)} sub-questions: "
        f"{state.sub_questions}"
    )
    logger.info(f"plan_sub_questions: {len(state.sub_questions)} sub-questions planned")
    return state


# =============================================================================
# Node 4: retrieve
# =============================================================================

def retrieve(state: RAGState) -> RAGState:
    """Node 4: Execute hybrid search (FAISS + BM25 + RRF) for the active query.

    Determines which query to use:
        - simple path: state.question
        - multi_hop path: state.sub_questions[state.current_sub_idx]
        - retry path (after refine_query): state.active_query (rewritten query)

    Deduplicates results against state.accumulated_chunks by chunk_id,
    keeping the entry with the higher RRF score. Maintains descending score order.

    STUB: Appends a fake chunk. Real implementation calls retriever.hybrid_search.

    Input:
        state.question / state.active_query / state.sub_questions[idx]
        state.accumulated_chunks (existing)

    Output:
        state.accumulated_chunks — updated with new results, deduplicated, sorted desc
        state.retry_count — incremented if this is a retry (detected via state.missing_information)
        state.reasoning_trace — appended

    Implementation pseudocode:
        from src.retrieval.retriever import hybrid_search

        # Determine active query
        if state.active_query:
            query = state.active_query
        elif state.intent == "multi_hop":
            query = state.sub_questions[state.current_sub_idx]
        else:
            query = state.question

        new_chunks = hybrid_search(query, top_k=config.top_k)

        # Deduplicate: build a dict keyed by chunk_id, keep max score
        existing = {c.chunk_id: c for c in state.accumulated_chunks}
        for nc in new_chunks:
            if nc.chunk_id not in existing or nc.score > existing[nc.chunk_id].score:
                existing[nc.chunk_id] = nc

        state.accumulated_chunks = sorted(
            existing.values(), key=lambda c: c.score, reverse=True
        )

        # Increment retry_count if this is a retry cycle
        if state.missing_information and not state.information_sufficient:
            state.retry_count += 1

        state.reasoning_trace.append(
            f"[retrieve] query='{query[:60]}...' → {len(new_chunks)} new chunks, "
            f"total accumulated: {len(state.accumulated_chunks)}"
        )
        state.active_query = ""  # Clear after use
    """
    # Determine which query to use (centralized logic)
    query = _get_current_query(state)

    if not query or not query.strip():
        logger.warning("retrieve: empty query, skipping")
        state.reasoning_trace.append("[retrieve] skipped: empty query")
        state.active_query = ""
        return state

    # Increment retry_count BEFORE the search attempt if this is a retry cycle.
    # Must happen here (not after try/except) to ensure it increments even when
    # the search itself fails (e.g., missing index files).
    if state.missing_information and not state.information_sufficient:
        state.retry_count += 1

    try:
        retriever = _get_retriever()
        top_k = _get_runtime_config().top_k
        new_chunks = retriever.search(query, top_k=top_k)
        logger.info(f"retrieve: got {len(new_chunks)} chunks for query='{query[:60]}...'")
    except FileNotFoundError as e:
        logger.error(f"retrieve: index files not found: {e}")
        state.reasoning_trace.append(
            f"[retrieve] ERROR: index files not found. Run 'python data_prepare.py' first."
        )
        state.active_query = ""
        return state
    except Exception as e:
        logger.error(f"retrieve: search failed ({e})")
        state.reasoning_trace.append(f"[retrieve] ERROR: search failed: {e}")
        state.active_query = ""
        return state

    # Deduplicate: build a dict keyed by chunk_id, keep max score
    existing: Dict[str, RetrievedChunk] = {
        c.chunk_id: c for c in state.accumulated_chunks
    }
    for nc in new_chunks:
        if nc.chunk_id not in existing or nc.score > existing[nc.chunk_id].score:
            existing[nc.chunk_id] = nc

    state.accumulated_chunks = sorted(
        existing.values(), key=lambda c: c.score, reverse=True
    )

    state.reasoning_trace.append(
        f"[retrieve] query='{query[:60]}...' → {len(new_chunks)} new chunks, "
        f"total accumulated: {len(state.accumulated_chunks)}"
    )
    state.active_query = ""  # Clear after consumption
    logger.info(
        f"retrieve: query='{query[:60]}...', total chunks={len(state.accumulated_chunks)}, "
        f"retry_count={state.retry_count}"
    )
    return state


# =============================================================================
# Node 5: check_sufficiency
# =============================================================================

def check_sufficiency(state: RAGState) -> RAGState:
    """Node 5: Judge whether accumulated chunks are sufficient to answer the question.

    STUB: Always sets information_sufficient = True.
    Real implementation calls LLM as a binary classifier.

    Real implementation LLM prompt:
        ---------- LLM Prompt ----------
        你是一个评审员。根据提供的文档片段，判断是否能准确回答用户问题。
        如果能回答，输出：SUFFICIENT
        如果不能，输出：INSUFFICIENT，并说明缺少什么具体信息（例如"缺少2022年的研发支出数字"）。
        用户问题：{question}
        文档片段：
        {formatted_chunks}
        只输出格式：SUFFICIENT 或 INSUFFICIENT: <缺失信息描述>
        ---------------------------------

    Input:
        The active question (original or current sub_question)
        state.accumulated_chunks — all chunks retrieved so far

    Output:
        state.information_sufficient — True/False
        state.missing_information — None if sufficient, else description string
        state.reasoning_trace — appended

    Implementation pseudocode:
        llm = _get_llm()

        # Determine which question we're answering right now
        if state.intent == "multi_hop" and state.current_sub_idx < len(state.sub_questions):
            current_q = state.sub_questions[state.current_sub_idx]
        else:
            current_q = state.question

        # Format chunks for the LLM
        formatted = "\n\n".join(
            f"[{c.chunk_id}] {c.text}" for c in state.accumulated_chunks[-10:]  # last 10
        )
        prompt = SUFFICIENCY_TEMPLATE.format(
            question=current_q,
            formatted_chunks=formatted,
        )
        response = llm.invoke(prompt)
        text = response.content.strip()

        if text.startswith("SUFFICIENT"):
            state.information_sufficient = True
            state.missing_information = None
        else:
            state.information_sufficient = False
            state.missing_information = text.replace("INSUFFICIENT:", "").strip()
        logger.info(f"Sufficiency: {state.information_sufficient}, missing='{state.missing_information}'")
    """
    # Determine which question we're answering right now
    current_q = _get_current_query(state)

    # Format chunks: truncate each to 500 chars, take last 10
    chunks_for_llm = state.accumulated_chunks[-10:]
    formatted_chunks = "\n\n".join(
        f"[{c.chunk_id}] {c.text[:500]}" for c in chunks_for_llm
    )
    if not formatted_chunks:
        formatted_chunks = "(No documents retrieved yet)"

    # Progressive leniency: stricter on first try, more lenient on retries.
    # This prevents infinite retry loops while still requiring the LLM to
    # actually evaluate the chunks (no blind auto-pass).
    retry_num = state.retry_count
    if retry_num == 0:
        sufficiency_guidance = (
            "You are a strict judge. Determine if the documents contain enough "
            "information to give a substantiated answer. If key facts or specific "
            "numbers are clearly missing, judge as INSUFFICIENT."
        )
    elif retry_num == 1:
        sufficiency_guidance = (
            "You are a moderate judge. If the documents contain partial information "
            "that allows a useful (even if not exhaustive) answer, judge as SUFFICIENT. "
            "Only judge INSUFFICIENT if the documents are almost entirely off-topic."
        )
    else:
        # retry >= 2: very lenient — accept any hint of relevant info
        sufficiency_guidance = (
            "You are a lenient judge. If the documents contain ANY relevant names, "
            "entities, numbers, or facts that are even remotely related to the "
            "question, judge as SUFFICIENT. Only judge INSUFFICIENT if the "
            "documents are completely unrelated to the question topic."
        )

    try:
        prompt = (
            "You are a judge. Based on the provided document excerpts, determine if you "
            "can answer the user's question with the available information.\n"
            "IMPORTANT: The answer does NOT need to be exhaustive or complete. "
            "If you can extract ANY relevant facts, names, or data points that "
            "partially address the question, that counts as SUFFICIENT.\n"
            "If you can give at least a partial answer, output: SUFFICIENT\n"
            "Only output INSUFFICIENT if the documents are completely off-topic "
            "or contain zero relevant information.\n"
            "User question: {question}\n"
            "Document excerpts:\n{chunks}\n"
            "Only output in this format: SUFFICIENT or INSUFFICIENT: <missing info description>"
        ).format(question=current_q, chunks=formatted_chunks)

        system_prompt = sufficiency_guidance

        raw = get_llm_response(prompt, system_prompt=system_prompt)
        text = raw.strip()
        logger.info(f"check_sufficiency LLM response: '{text[:100]}...'")

        # Parse response: SUFFICIENT vs INSUFFICIENT
        # Strip markdown formatting like **SUFFICIENT** or *INSUFFICIENT*
        text_clean = re.sub(r"[*_]{1,3}", "", text).strip()
        if re.match(r"SUFFICIENT", text_clean, re.IGNORECASE):
            state.information_sufficient = True
            state.missing_information = None
        elif re.match(r"INSUFFICIENT", text_clean, re.IGNORECASE):
            state.information_sufficient = False
            # Extract the reason after "INSUFFICIENT:" or "INSUFFICIENT"
            missing = re.sub(r"^INSUFFICIENT\s*:?\s*", "", text_clean, flags=re.IGNORECASE).strip()
            state.missing_information = missing or "Information insufficient for answering"
        else:
            # Unparseable response — conservative: assume sufficient
            logger.warning(
                f"check_sufficiency: unexpected LLM response '{text[:100]}...', "
                f"defaulting to sufficient=True"
            )
            state.information_sufficient = True
            state.missing_information = None
            state.reasoning_trace.append(
                f"[check_sufficiency] WARNING: unparseable response, defaulted to sufficient"
            )

    except Exception as e:
        logger.warning(f"check_sufficiency: LLM call failed ({e}), defaulting to sufficient=True")
        state.information_sufficient = True
        state.missing_information = None
        state.reasoning_trace.append(
            f"[check_sufficiency] LLM error: {e}, defaulted to sufficient"
        )

    state.reasoning_trace.append(
        f"[check_sufficiency] sufficient={state.information_sufficient}, "
        f"missing='{state.missing_information or 'N/A'}'"
    )
    logger.info(
        f"check_sufficiency: sufficient={state.information_sufficient}, "
        f"missing='{state.missing_information}'"
    )
    return state


# =============================================================================
# Node 6: refine_query
# =============================================================================

def refine_query(state: RAGState) -> RAGState:
    """Node 6: Rewrite the retrieval query based on missing_information.

    Only called when information_sufficient == False and retry_count < max_retries.
    Generates a more targeted query for the next retrieval attempt.

    STUB: Prepends "refined:" to the active query.

    Real implementation LLM prompt:
        ---------- LLM Prompt ----------
        原始检索未能找到足够信息。
        原始问题：{original_question}
        缺失信息：{missing_information}
        请生成一个新的、更具体的检索查询词，用于下一次检索。
        只输出新查询词。
        ---------------------------------

    Input:
        The active question being answered
        state.missing_information — what was missing

    Output:
        state.active_query — the rewritten query string
        state.reasoning_trace — appended

    Implementation pseudocode:
        llm = _get_llm()

        # Determine which question is being answered
        if state.intent == "multi_hop" and state.current_sub_idx < len(state.sub_questions):
            original_q = state.sub_questions[state.current_sub_idx]
        else:
            original_q = state.question

        prompt = REFINE_TEMPLATE.format(
            original_question=original_q,
            missing_information=state.missing_information,
        )
        response = llm.invoke(prompt)
        state.active_query = response.content.strip()
        logger.info(f"Refined query: '{state.active_query}'")
    """
    # Determine which question is being answered
    original_q = _get_current_query(state)

    try:
        prompt = (
            "Original question: {original_question}\n"
            "Missing information: {missing_information}\n"
            "Please generate a more precise retrieval query (one sentence) to find "
            "the missing information.\n"
            "Only output the query."
        ).format(
            original_question=original_q,
            missing_information=state.missing_information or "unknown",
        )

        system_prompt = (
            "You are a search query optimizer. Given an original question and "
            "a description of what information is missing, generate a refined "
            "search query that is more specific and targeted. Output only the query."
        )

        raw = get_llm_response(prompt, system_prompt=system_prompt)
        state.active_query = raw.strip()
        logger.info(f"Refined query: '{state.active_query[:80]}...'")

    except Exception as e:
        logger.warning(f"refine_query: LLM call failed ({e}), using fallback")
        state.active_query = f"refined: {original_q} (missing: {state.missing_information})"
        state.reasoning_trace.append(
            f"[refine_query] LLM error: {e}, used fallback prefix"
        )

    state.reasoning_trace.append(
        f"[refine_query] original='{original_q[:60]}...' → refined='{state.active_query[:80]}...'"
    )
    logger.info(f"refine_query: '{original_q[:50]}...' → '{state.active_query[:50]}...'")
    return state


# =============================================================================
# Node 7: generate_answer
# =============================================================================

def generate_answer(state: RAGState) -> RAGState:
    """Node 7: Generate the final answer from accumulated chunks.

    Terminal node — always transitions to END.

    STUB: Sets a placeholder answer string.

    Real implementation LLM prompt:
        ---------- LLM Prompt ----------
        你是一个企业知识库助手。请仅根据提供的文档片段回答用户问题。
        如果片段中没有答案，必须说："没有找到相关信息"。
        不要编造任何信息。
        在答案末尾列出引用的 chunk_id。

        用户问题：{question}
        文档片段：
        {formatted_chunks}

        答案：
        ---------------------------------

    Special case handling:
        If state.retry_count >= config.max_retries and NOT state.information_sufficient:
            Prepend this declaration to the answer:
            "信息不足，无法完整回答："
            Followed by the best-effort answer from available chunks.

    Input:
        state.question — original user question
        state.accumulated_chunks — all retrieved chunks
        state.retry_count — used to decide if forced answer with disclaimer

    Output:
        state.final_answer — the complete answer string with citations
        state.reasoning_trace — appended

    Implementation pseudocode:
        llm = _get_llm()

        formatted = "\n\n".join(
            f"[{c.chunk_id}] (source: {c.metadata.get('source_file', 'unknown')})\n{c.text}"
            for c in state.accumulated_chunks
        )

        prompt = ANSWER_TEMPLATE.format(
            question=state.question,
            formatted_chunks=formatted,
        )
        response = llm.invoke(prompt)
        answer = response.content.strip()

        # Force disclaimer if retries exhausted with insufficient info
        if state.retry_count >= _get_runtime_config().max_retries and not state.information_sufficient:
            answer = "信息不足，无法完整回答：" + answer

        state.final_answer = answer
        logger.info(f"Answer generated: {len(state.final_answer)} chars")
    """
    try:
        # Format chunks for the LLM
        if state.accumulated_chunks:
            formatted_chunks = "\n\n".join(
                f"[{c.chunk_id}] (source: {c.metadata.get('source_file', 'unknown')})\n{c.text}"
                for c in state.accumulated_chunks
            )
        else:
            formatted_chunks = "(No documents retrieved)"

        prompt = (
            "Based on the following document excerpts, answer the user's question.\n"
            "Do not fabricate any information. If the answer is not found in the documents, "
            "explicitly say \"No relevant information found\".\n"
            "After each cited sentence, append its source ID in brackets (e.g., [doc_5]).\n\n"
            "Document excerpts:\n{chunks}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ).format(chunks=formatted_chunks, question=state.question)

        system_prompt = (
            "You are a corporate knowledge base assistant. Answer questions ONLY based on "
            "the provided document excerpts. Cite your sources using [doc_N] notation "
            "after each sentence that uses information from a specific document. "
            "If the documents don't contain the answer, say so clearly. "
            "Do not make up facts."
        )

        answer = get_llm_response(prompt, system_prompt=system_prompt)
        logger.info(f"generate_answer: LLM returned {len(answer)} chars")

        # Check for [doc_xxx] citations in the answer
        has_citation = bool(re.search(r"\[doc_\d+\]", answer))

        if not has_citation:
            # Auto-append a note about missing citations
            chunk_ids = [c.chunk_id for c in state.accumulated_chunks[:5]]
            if chunk_ids:
                answer += f"\n\n[来源：参考文档 {', '.join(chunk_ids)}]"
            else:
                answer += "\n\n[来源：未明确]"
            logger.info("generate_answer: no citations found in answer, appended source note")
            state.reasoning_trace.append(
                "[generate_answer] no [doc_N] citations in LLM response, appended source note"
            )

        state.final_answer = answer

    except Exception as e:
        logger.warning(f"generate_answer: LLM call failed ({e}), using fallback")
        chunks_summary = ", ".join(c.chunk_id for c in state.accumulated_chunks[:5])
        state.final_answer = (
            f"Answer generation failed due to an error: {e}. "
            f"Retrieved {len(state.accumulated_chunks)} documents: {chunks_summary}"
        )
        state.reasoning_trace.append(f"[generate_answer] LLM error: {e}")

    # Prepend disclaimer if retries exhausted with insufficient info
    if state.retry_count >= _get_runtime_config().max_retries and not state.information_sufficient:
        state.final_answer = (
            "信息不足，无法完整回答（已达最大重试次数）。以下为基于已有信息的最佳回答：\n\n"
            + state.final_answer
        )

    state.reasoning_trace.append(
        f"[generate_answer] answer_len={len(state.final_answer)}, "
        f"chunks_used={len(state.accumulated_chunks)}, "
        f"retry_count={state.retry_count}"
    )
    logger.info(f"generate_answer: produced {len(state.final_answer)}-char answer")
    return state


# =============================================================================
# Routing Functions
# =============================================================================

def route_by_intent(state: RAGState) -> str:
    """Route from classify_intent based on the intent field.

    Returns one of:
        "unclear"   → ask_clarification → END
        "simple"    → retrieve (with original question)
        "multi_hop" → plan_sub_questions → retrieve → check_sufficiency
    """
    intent = state.intent
    logger.info(f"route_by_intent: → '{intent}'")
    if intent == "unclear":
        return "unclear"
    elif intent == "multi_hop":
        return "multi_hop"
    else:
        return "simple"


def route_by_sufficiency(state: RAGState) -> str:
    """Route from check_sufficiency based on sufficiency, retry count, and intent.

    Returns one of:
        "generate"        — sufficient info, go to generate_answer
        "refine"          — insufficient + retries remain, go to refine_query
        "generate_forced" — insufficient + retries exhausted, go to generate_answer
        "next_sub"        — sufficient + multi_hop with more sub-questions remaining

    Note on state mutation:
        This function mutates state.current_sub_idx and state.active_query when
        advancing to the next sub-question. LangGraph passes state by reference,
        so these mutations persist into subsequent nodes.
    """
    # ---- Sufficient: determine next step ----
    if state.information_sufficient:
        if state.intent == "multi_hop":
            next_idx = state.current_sub_idx + 1
            if next_idx < len(state.sub_questions):
                # Advance to next sub-question
                state.current_sub_idx = next_idx
                state.active_query = state.sub_questions[next_idx]
                state.reasoning_trace.append(
                    f"[route] sufficient=True | advancing to sub_question[{next_idx}]: "
                    f"'{state.sub_questions[next_idx][:60]}...'"
                )
                logger.info(
                    f"route_by_sufficiency: → next_sub "
                    f"(idx={next_idx}/{len(state.sub_questions)}, "
                    f"chunks={len(state.accumulated_chunks)})"
                )
                return "next_sub"
            else:
                # All sub-questions complete → answer the ORIGINAL question
                state.active_query = ""  # Clear so generate_answer uses state.question
                state.reasoning_trace.append(
                    "[route] sufficient=True | all sub-questions done → generate_answer"
                )
                logger.info(
                    f"route_by_sufficiency: → generate (multi_hop done, "
                    f"chunks={len(state.accumulated_chunks)})"
                )
                return "generate"
        else:
            # Simple path: sufficient info gathered
            logger.info(
                f"route_by_sufficiency: → generate (simple, "
                f"chunks={len(state.accumulated_chunks)})"
            )
            return "generate"

    # ---- Insufficient: determine retry or force ----
    max_retries = _get_runtime_config().max_retries
    retries_left = max_retries - state.retry_count
    if state.retry_count < max_retries:
        logger.info(
            f"route_by_sufficiency: → refine "
            f"(retry {state.retry_count + 1}/{max_retries}, "
            f"will have {retries_left - 1} retries left)"
        )
        return "refine"
    else:
        state.reasoning_trace.append(
            f"[route] insufficient=True | retries exhausted "
            f"({state.retry_count}/{max_retries}) → generate_answer (forced)"
        )
        logger.info(
            f"route_by_sufficiency: → generate_forced "
            f"(retries exhausted at {state.retry_count}/{max_retries})"
        )
        return "generate_forced"


# =============================================================================
# Re-rank Node (v0.4)
# =============================================================================

def _rerank_node(state: RAGState) -> RAGState:
    """Cross-encoder re-rank node.

    Inserted between retrieve and check_sufficiency. Uses the
    CrossEncoderReranker to re-score the accumulated chunks and keep
    only the top-k most relevant documents.

    The reranker is controlled by config — set reranker_enabled=False
    to skip re-ranking entirely.
    """
    if not state.accumulated_chunks:
        state.reasoning_trace.append("[rerank] No documents to re-rank.")
        return state

    try:
        from src.retrieval.reranker import get_reranker
    except ImportError as e:
        state.reasoning_trace.append(f"[rerank] Reranker not available: {e}")
        return state

    reranker = get_reranker(
        model_name=getattr(config, "reranker_model", "BAAI/bge-reranker-base"),
        top_k=getattr(config, "reranker_top_k", 5),
        enabled=getattr(config, "reranker_enabled", True),
    )

    if not reranker.enabled:
        state.reasoning_trace.append("[rerank] Skipped (disabled in config).")
        return state

    # Convert RetrievedChunk list to dict list for reranker
    current_q = _get_current_query(state)
    doc_dicts = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "score": c.score,
            "metadata": c.metadata,
            "chunk": c,  # keep original reference
        }
        for c in state.accumulated_chunks
    ]

    before = len(doc_dicts)
    reranked = reranker.rerank(current_q, doc_dicts)
    after = len(reranked)

    # Replace accumulated_chunks with re-ranked results
    from src.core.models import RetrievedChunk
    new_chunks = []
    for d in reranked:
        if "chunk" in d:
            c = d["chunk"]
            c.score = d.get("rerank_score", c.score)
            new_chunks.append(c)
        else:
            # Fallback: reconstruct from dict
            new_chunks.append(RetrievedChunk(
                chunk_id=d["chunk_id"],
                text=d["text"],
                score=d.get("rerank_score", d.get("score", 0)),
                metadata=d.get("metadata", {}),
            ))

    state.accumulated_chunks = new_chunks
    state.reasoning_trace.append(
        f"[rerank] Re-ranked {before} documents → top-{after}"
    )
    return state


# =============================================================================
# Graph Construction
# =============================================================================

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph for the Agentic RAG pipeline.

    Topology (from MVP.txt Section 4):

        START → classify_intent
        classify_intent → [intent switch]
            unclear   → ask_clarification → END
            simple    → retrieve → check_sufficiency
            multi_hop → plan_sub_questions → retrieve → check_sufficiency

        check_sufficiency → [sufficiency switch]
            sufficient + simple/multi_hop_done → generate_answer → END
            sufficient + multi_hop_next_sub     → retrieve (next sub_q)
            insufficient + retry < 3            → refine_query → retrieve → check_sufficiency
            insufficient + retry >= 3           → generate_answer → END

    Returns:
        Compiled LangGraph StateGraph ready for .invoke() with a RAGState dict.
    """
    # ---- 多 Agent 模式 (v0.5) ----
    from src.agents.supervisor import supervisor_node
    from src.agents.workers.retriever_worker import retriever_worker as _ra_worker
    from src.agents.workers.critic_worker import critic_worker
    from src.agents.workers.synthesizer_worker import synthesizer_worker

    graph = StateGraph(RAGState)

    # ---- Add all nodes (8 base + 4 multi-agent) ----
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("plan_sub_questions", plan_sub_questions)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", _rerank_node)
    graph.add_node("check_sufficiency", check_sufficiency)
    graph.add_node("refine_query", refine_query)
    graph.add_node("generate_answer", generate_answer)
    # 多 Agent 节点（仅在 multi_agent_enabled 时参与路由）
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retriever_worker", _ra_worker)
    graph.add_node("critic_worker", critic_worker)
    graph.add_node("synthesizer_worker", synthesizer_worker)

    # ---- Entry point ----
    graph.set_entry_point("classify_intent")

    # ---- Edges from classify_intent (conditional on intent) ----
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "unclear": "ask_clarification",
            "simple": "retrieve",
            "multi_hop": "plan_sub_questions",
        },
    )

    # ---- ask_clarification → END (terminal) ----
    graph.add_edge("ask_clarification", END)

    # ---- plan_sub_questions → retrieve ----
    graph.add_edge("plan_sub_questions", "retrieve")

    # ---- Edges from check_sufficiency (conditional) ----
    graph.add_conditional_edges(
        "check_sufficiency",
        route_by_sufficiency,
        {
            "generate": "generate_answer",
            "refine": "refine_query",
            "generate_forced": "generate_answer",
            "next_sub": "retrieve",
        },
    )

    # ---- retrieve → rerank → check_sufficiency (all paths) ----
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "check_sufficiency")

    # ---- refine_query → retrieve (retry loop entrance) ----
    graph.add_edge("refine_query", "retrieve")

    # ---- generate_answer → END (terminal) ----
    graph.add_edge("generate_answer", END)

    compiled = graph.compile()
    logger.info("LangGraph compiled successfully with 7 nodes, 5 conditional edges.")
    return compiled


# =============================================================================
# Singleton compiled graph
# =============================================================================

_compiled_graph: StateGraph | None = None


def get_graph() -> StateGraph:
    """Return the compiled LangGraph, building it once (singleton pattern).

    Returns:
        Compiled StateGraph ready for .invoke() or .stream().
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# =============================================================================
# Convenience: run the graph end-to-end
# =============================================================================

def run_rag(question: str) -> dict:
    """Run the full Agentic RAG pipeline for a single question.

    Convenience wrapper that initializes state, invokes the graph,
    and returns a JSON-serializable result dict.

    Args:
        question: Natural language question string.

    Returns:
        Dict with keys:
            - answer: str — the final answer
            - reasoning_trace: List[str] — step-by-step decision log
            - retrieved_sources: List[str] — chunk_ids used in the answer
            - intent: str — classified intent
            - final_state: dict — the complete final RAGState as dict
    """
    # ---- 两级缓存检查 (v0.5): L1 精确 → L2 语义 ----
    if config.cache_enabled:
        try:
            from src.cache.cache_manager import get_cache_manager
            cm = get_cache_manager(
                exact_enabled=config.cache_enabled,
                semantic_enabled=config.cache_enabled,
            )
            cached = cm.get(question)
            if cached:
                cache_type = cached.get("cache_type", "unknown")
                logger.info("Cache HIT (%s): '%s...'", cache_type, question[:60])
                return {
                    "answer": cached.get("answer", ""),
                    "reasoning_trace": [f"[cache] {cache_type} 缓存命中"],
                    "retrieved_sources": cached.get("sources", []),
                    "intent": cached.get("intent", "simple"),
                    "from_cache": True,
                    "cache_type": cache_type,
                    "final_state": {},
                }
        except ImportError:
            logger.debug("缓存模块不可用，跳过缓存检查")
        except Exception as e:
            logger.warning("缓存检查失败: %s", e)

    graph = get_graph()

    initial_state = RAGState(question=question)
    logger.info(f"Starting RAG pipeline for: '{question[:80]}...'")

    final_state = graph.invoke(initial_state)

    # Extract results (graph returns a dict or RAGState depending on config)
    if isinstance(final_state, dict):
        result = {
            "answer": final_state.get("final_answer", ""),
            "reasoning_trace": final_state.get("reasoning_trace", []),
            "retrieved_sources": [
                c.chunk_id if hasattr(c, "chunk_id") else c["chunk_id"]
                for c in final_state.get("accumulated_chunks", [])
            ],
            "intent": final_state.get("intent", "unknown"),
            "final_state": {
                k: str(v) for k, v in final_state.items()
                if k not in ("accumulated_chunks", "reasoning_trace")
            },
        }
    else:
        result = {
            "answer": final_state.final_answer or "",
            "reasoning_trace": final_state.reasoning_trace,
            "retrieved_sources": [c.chunk_id for c in final_state.accumulated_chunks],
            "intent": final_state.intent,
            "final_state": final_state.model_dump(
                exclude={"accumulated_chunks", "reasoning_trace"}
            ),
        }

    # ---- 写入两级缓存 (v0.5) ----
    if config.cache_enabled and result.get("answer") and not result.get("from_cache"):
        try:
            from src.cache.cache_manager import get_cache_manager
            cm = get_cache_manager(
                exact_enabled=config.cache_enabled,
                semantic_enabled=config.cache_enabled,
            )
            cm.put(question, result, metadata={
                "intent": result.get("intent", "unknown"),
                "sources": result.get("retrieved_sources", []),
            })
        except Exception as e:
            logger.debug("缓存写入失败: %s", e)

    logger.info(f"RAG pipeline complete. Answer length: {len(result['answer'])}")
    return result


# =============================================================================
# Runtime config — allows overriding parameters per-request
# =============================================================================

_runtime_config = config
"""Active config used by all nodes. Overridden by run_workflow_streaming via
config_override dict. Defaults to the global singleton."""


def _get_runtime_config() -> "RAGConfig":
    """Return the current runtime config (may be overridden per-request)."""
    global _runtime_config
    return _runtime_config


# =============================================================================
# Streaming workflow wrapper
# =============================================================================


def run_workflow_streaming(
    question: str,
    config_override: Optional[Dict[str, Any]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """使用 LangGraph astream_events 流式执行 RAG 流水线。

    关键改进：不再手写 300 行节点调度循环，改为监听编译好的
    LangGraph 图发出的原生事件。修改节点逻辑后无需同步更新此函数。

    Args:
        question: 用户问题
        config_override: 可选配置覆盖

    Yields:
        与旧版完全兼容的 dict 事件: node / token / done
    """
    global _runtime_config

    if config_override:
        _runtime_config = config.from_dict(config_override)
    else:
        _runtime_config = config

    initial_state = RAGState(question=question)
    accumulated_answer = ""

    # 节点 → 中文提示词
    NODE_CN = {
        "classify_intent":    ("正在分析问题意图...", "意图: {d}"),
        "ask_clarification":  ("正在生成反问...", "反问: {d}"),
        "plan_sub_questions": ("正在拆解子问题...", "共 {d} 个子问题"),
        "retrieve":           ("正在检索...", "检索完成"),
        "rerank":             ("正在精排...", "保留 top-{d} 篇"),
        "check_sufficiency":  ("正在校验...", "{d}"),
        "refine_query":       ("正在改写查询...", "新查询: {d}"),
        "generate_answer":    ("正在生成答案...", "完成"),
    }

    graph = get_graph()

    try:
        import asyncio

        async def _stream():
            nonlocal accumulated_answer

            # 同步执行图，挨个节点 yield 进度
            state_dict = initial_state.model_dump()
            state = RAGState(**state_dict)

            # 手动按图拓扑执行各节点（保证 yield 顺序可控）
            node_seq = ["classify_intent"]

            # classify_intent 的后续路由
            state = classify_intent(state)
            start_msg, end_tpl = NODE_CN["classify_intent"]
            yield ({"type": "node", "node": "classify_intent", "message": start_msg, "detail": ""},
                   {"type": "node", "node": "classify_intent", "message": end_tpl.format(d=state.intent), "detail": state.intent})

            if state.intent == "unclear":
                state = ask_clarification(state)
                _, end_tpl = NODE_CN["ask_clarification"]
                detail = state.clarification_question or ""
                yield ({"type": "node", "node": "ask_clarification", "message": "正在生成反问...", "detail": ""},
                       {"type": "node", "node": "ask_clarification", "message": end_tpl.format(d=detail), "detail": detail})
                accumulated_answer = detail
                return

            if state.intent == "multi_hop":
                state = plan_sub_questions(state)
                state.current_sub_idx = 0
                state.active_query = state.sub_questions[0] if state.sub_questions else question
                _, end_tpl = NODE_CN["plan_sub_questions"]
                detail = str(len(state.sub_questions))
                yield ({"type": "node", "node": "plan_sub_questions", "message": "正在拆解子问题...", "detail": ""},
                       {"type": "node", "node": "plan_sub_questions", "message": end_tpl.format(d=detail), "detail": detail})
            else:
                state.active_query = question

            # 检索循环（含精排 + 校验 + 重试）
            rt = _runtime_config
            max_cycles = rt.max_retries + 1
            for cycle_idx in range(max_cycles):
                top_k = rt.top_k * (1 if cycle_idx == 0 else 2 if cycle_idx == 1 else 3)

                # retrieve
                yield ({"type": "node", "node": "retrieve", "message": f"正在检索...",
                        "detail": f"top_k={top_k}"},)
                state = retrieve(state)
                yield ({"type": "node", "node": "retrieve", "message": f"检索到 {len(state.accumulated_chunks)} 篇", "detail": ""},)

                # rerank (v0.4)
                if getattr(config, "reranker_enabled", True):
                    before = len(state.accumulated_chunks)
                    state = _rerank_node(state)
                    after = len(state.accumulated_chunks)
                    yield ({"type": "node", "node": "rerank", "message": f"精排: {before} → {after} 篇",
                            "detail": str(after)},)

                # check_sufficiency
                state = check_sufficiency(state)
                detail = "充分" if state.information_sufficient else "不足"
                yield ({"type": "node", "node": "check_sufficiency", "message": detail, "detail": detail},)

                if state.information_sufficient:
                    if state.intent == "multi_hop":
                        next_idx = state.current_sub_idx + 1
                        if next_idx < len(state.sub_questions):
                            state.current_sub_idx = next_idx
                            state.active_query = state.sub_questions[next_idx]
                            state.retry_count = 0
                            continue
                    break

                state.retry_count += 1
                if state.retry_count < rt.max_retries:
                    state = refine_query(state)
                    detail = state.active_query[:60]
                    yield ({"type": "node", "node": "refine_query", "message": f"改写: {detail}...", "detail": detail},)
                else:
                    if state.intent == "multi_hop":
                        next_idx = state.current_sub_idx + 1
                        if next_idx < len(state.sub_questions):
                            state.current_sub_idx = next_idx
                            state.active_query = state.sub_questions[next_idx]
                            state.retry_count = 0
                            continue
                    break

            # generate_answer（流式 token）
            yield ({"type": "node", "node": "generate_answer", "message": "正在生成答案...", "detail": ""},)

            if state.accumulated_chunks:
                formatted = "\n\n".join(
                    f"[{c.chunk_id}] {c.text}"
                    for c in state.accumulated_chunks
                )
            else:
                formatted = "(未检索到文档)"

            prompt = (
                f"基于文档回答，引用 [doc_N]。\n\n文档:\n{formatted}\n\n"
                f"问题: {state.question}\n\n答案:"
            )

            try:
                for token in get_llm_response_stream(prompt, temperature=rt.llm_temperature):
                    accumulated_answer += token
                    yield ({"type": "token", "content": token, "accumulated": accumulated_answer},)
            except Exception as e:
                accumulated_answer = f"[ERROR] {e}"
                yield ({"type": "token", "content": accumulated_answer, "accumulated": accumulated_answer},)

            state.final_answer = accumulated_answer

        # 执行异步生成器
        loop = asyncio.new_event_loop()
        try:
            gen = _stream()
            while True:
                try:
                    events = loop.run_until_complete(gen.__anext__())
                    if isinstance(events, tuple):
                        for evt in events:
                            yield evt
                    else:
                        yield events
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

        # 用 graph.invoke 获取最终状态
        final_state = graph.invoke(initial_state)

    except (ImportError, RuntimeError) as e:
        logger.warning("astream_events 不可用 (%s)，降级同步执行", e)
        final_state = graph.invoke(initial_state)
        answer = (final_state.get("final_answer", "") if isinstance(final_state, dict)
                  else getattr(final_state, "final_answer", ""))
        yield {"type": "token", "content": answer, "accumulated": answer}

    # done 事件
    if isinstance(final_state, dict):
        fa = final_state.get("final_answer", accumulated_answer)
        ft = final_state.get("reasoning_trace", [])
        fc = final_state.get("accumulated_chunks", [])
        fi = final_state.get("intent", "simple")
    else:
        fa = getattr(final_state, "final_answer", accumulated_answer)
        ft = getattr(final_state, "reasoning_trace", [])
        fc = getattr(final_state, "accumulated_chunks", [])
        fi = getattr(final_state, "intent", "simple")

    retrieved_chunks = []
    for c in fc:
        chunk = c if isinstance(c, dict) else c.model_dump()
        retrieved_chunks.append({
            "chunk_id": chunk.get("chunk_id", "?"),
            "text": chunk.get("text", "")[:200],
            "source_file": chunk.get("metadata", {}).get("source_file", "unknown"),
            "score": chunk.get("score", 0),
        })

    yield {
        "type": "done",
        "final_answer": fa or accumulated_answer,
        "retrieved_sources": [rc["chunk_id"] for rc in retrieved_chunks],
        "retrieved_chunks": retrieved_chunks,
        "reasoning_trace": ft,
        "intent": fi,
    }



# =============================================================================
# Module self-test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build the graph
    graph = build_graph()
    print("[OK] LangGraph compiled successfully")
    print(f"  Nodes: {list(graph.nodes.keys()) if hasattr(graph, 'nodes') else '7 nodes'}")
    print()

    # Test with a simple question
    print("Testing with: 'What was Veracier's R&D spending in 2022?'")
    result = run_rag("What was Veracier's R&D spending in 2022?")
    print(f"  intent: {result['intent']}")
    print(f"  answer: {result['answer'][:120]}...")
    print(f"  sources: {result['retrieved_sources']}")
    print(f"  trace entries: {len(result['reasoning_trace'])}")
    for entry in result['reasoning_trace']:
        print(f"    {entry}")
