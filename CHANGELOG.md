# Changelog

## [v0.4.0] — 2026-06-19

### Added
- **Cross-encoder re-ranking** (`src/retrieval/reranker.py`): BAAI/bge-reranker-base model inserted between retrieve and check_sufficiency nodes for improved retrieval precision. Controllable via `RAG_RERANKER_ENABLED` env var.
- **Semantic cache** (`src/cache/`): FAISS vector similarity + Redis persistent storage. Cached answers returned <500ms. Configurable threshold, TTL, and max size.
- **Multilingual E5 support** (`config.py`): Added `embedding_query_prefix` / `embedding_passage_prefix` for E5 models (e.g., `intfloat/multilingual-e5-base`). Query and passage prefixes applied automatically in retriever and data preparation.
- **RAGAS evaluation framework** (`evaluation/`): 10-question testset, RAGAS metrics (Faithfulness, AnswerRelevancy, ContextRelevancy), auto-generated comparison report in Markdown.
- **Docker support**: `Dockerfile` (Python 3.10-slim), `docker-compose.yml` (app + Redis), `.dockerignore`. One-command deployment with `docker-compose up --build`.

### Changed
- **Workflow graph**: Inserted `rerank` node between `retrieve` and `check_sufficiency` (8 nodes total).
- **Config expanded**: 11 new configuration variables for re-ranker, semantic cache, and E5 prefixes.
- **`run_rag()`**: Added cache check before graph invocation and cache write after answer generation.
- **Retriever encoding**: Query encoding now respects `embedding_query_prefix` from config.

### Fixed
- (No bugfixes in this release — feature-focused iteration)

---

## [v0.3.1] — 2026-06-17

### Changed
- Modular project structure (`src/core/`, `src/llm/`, `src/retrieval/`, `src/agents/`, `src/data/`, `src/web/`)
- Data models extracted to `src/core/models.py` to eliminate circular imports
- Root-level entry points updated to `from src.xxx import yyy` import style
- Git remote switched to SSH (HTTPS timeout behind firewall)
- `document/` and `veracier-industries/` excluded from Git tracking

### Fixed
- Gradio input box not clearing after send (generator now yields 4 values)
- Port 7860 conflict detection and stale process cleanup
- Reasoning GIF accelerated ~14x (68s → 4.8s, 800px)

---

## [v0.3.0] — 2026-06-16

### Added
- Gradio Web UI with streaming answers and reasoning trace panel
- LLM streaming support (`get_llm_response_stream`)
- Runtime config override (Top-K / Temperature sliders)
- `run_workflow_streaming()` generator for node-level event streaming

---

## [v0.2.0] — 2026-06-16

### Added
- Local embedding model deployment (bge-base-en-v1.5)
- FAISS + BM25 indexes for 200 PDFs (558 chunks)
- End-to-end validation 5/5 passed

---

## [v0.1.0] — 2026-06-16

### Added
- Real LLM integration replacing stub nodes
- Hybrid retriever (FAISS + BM25 + RRF)
- CLI REPL mode, single-question, and batch processing
- End-to-end integration tests
