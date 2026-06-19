# Changelog

## [v0.6.0] — 2026-06-20

### Added
- **多轮对话上下文**: `RAGState` 新增 `conversation_history` 字段，`generate_answer` 自动注入最近 3 轮对话到 prompt，支持追问、指代消解
- **答案来源可追溯**: 答案中 `[doc_xxx]` 自动替换为 `[来源N]`，对应检索来源面板的编号列表，一眼定位每句话的出处
- **UI 来源面板美化**: 来源条目改为盒装格式（得分 + 文件路径 + 内容预览）

### Changed
- `run_rag()` 和 `run_workflow_streaming()` 新增 `conversation_history` 参数
- Gradio `handle_user_message` 自动从 Chatbot 历史中提取上下文传给流水线

---

## [v0.5.0] — 2026-06-19

### Added
- **两级缓存体系** (`src/cache/exact_cache.py` + `cache_manager.py`): L1 精确缓存（MD5 匹配，<100ms）+ L2 语义缓存（向量相似度，<200ms），统一入口逐级查询
- **自动化回归测试** (`evaluation/regression_test.py`): 30 题黄金测试集 + 基线对比 + 退化检测（5% 阈值）
- **CI 流水线** (`.github/workflows/ci.yml`): GitHub Actions 自动运行回归测试，PR 时自动检测性能退化
- **多 Agent 架构基础版** (`src/agents/supervisor.py` + `workers/`): Supervisor + retriever/critic/synthesizer 三个 Worker，支持复杂问题的协作式处理
- **缓存状态展示**: Gradio 界面答案区下方显示命中类型（✅ 精确缓存 / 🔍 语义缓存）

### Changed
- `workflow.py` 的缓存集成从单级语义缓存升级为 cache_manager 两级查询
- `app_gradio.py` 的 done 事件处理增加 cache_type 显示
- `RAGState` 增加 `supervisor_plan`、`subtasks`、`worker_results` 字段
- `build_graph()` 注册了 4 个多 Agent 节点（supervisor, retriever_worker, critic_worker, synthesizer_worker）

---

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
