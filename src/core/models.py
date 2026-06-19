"""
models.py -- Shared data models for the Agentic RAG system.

Defines:
    - RetrievedChunk (pydantic BaseModel) — a single retrieved chunk
    - RAGState (pydantic BaseModel) — the full graph state

These are extracted to a shared location to avoid circular imports
between retriever and workflow modules.
"""

from typing import List, Optional, Dict, Literal, Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single retrieved document chunk with its score and source metadata."""

    chunk_id: str
    """Unique chunk identifier, format 'doc_N' where N is a zero-based integer."""

    text: str
    """The chunk's full text content."""

    score: float
    """RRF-fused retrieval score (higher = more relevant)."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Source metadata.
    Typical keys: source_file, page, entity, language, format, classification.
    """


class RAGState(BaseModel):
    """The complete state object flowing through the LangGraph.

    All fields have sensible defaults so the graph can be initialized
    with only a question string and nodes add information incrementally.
    """

    # ---- Input ----
    question: str = ""
    """The user's natural language question."""

    # ---- Intent classification (Node 1) ----
    intent: Literal["simple", "multi_hop", "unclear"] = "simple"
    """Classified question type. Set by classify_intent node."""

    # ---- Multi-hop planning (Node 3) ----
    sub_questions: List[str] = Field(default_factory=list)
    """Ordered list of sub-questions for multi-hop reasoning. Length >= 2."""

    current_sub_idx: int = 0
    """Index of the currently active sub_question (0-based)."""

    active_query: str = ""
    """The query string that retrieve() should use.
    Set by: simple path uses question, multi_hop uses sub_questions[idx],
    refine_query sets a rewritten query."""

    # ---- Accumulated retrieval results (Node 4) ----
    accumulated_chunks: List[RetrievedChunk] = Field(default_factory=list)
    """All retrieved chunks, deduplicated by chunk_id, sorted by score descending."""

    # ---- Sufficiency check (Node 5) ----
    information_sufficient: bool = False
    """Whether accumulated_chunks contain enough info to answer the question."""

    missing_information: Optional[str] = None
    """When insufficient, describes what specific information is missing.
    Example: 'missing specific year for R&D spending'."""

    # ---- Final answer (Node 7) ----
    final_answer: Optional[str] = None
    """The generated answer string, or None if not yet generated."""

    # ---- Clarification (Node 2) ----
    clarification_question: Optional[str] = None
    """When intent is 'unclear', a follow-up question asking the user to clarify."""

    # ---- Control ----
    retry_count: int = 0
    """Number of retrieval-refine retries attempted. Capped at config.max_retries."""

    # ---- Audit trail ----
    reasoning_trace: List[str] = Field(default_factory=list)
    """Step-by-step decision log. Each node appends a timestamped entry.
    Example: '[classify_intent] intent=multi_hop | confidence=high'"""
