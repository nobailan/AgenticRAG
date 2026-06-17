# Agentic RAG 项目改进说明 — 2026年6月17日晚

## 项目现状回顾

本项目是一个基于 LangGraph 的 **Agentic RAG 企业知识库问答系统**，服务于 Veracier Industries 的约 1100+ 份企业文档（含法/英/德/意/西五种语言）。

### 当前架构
- **检索**: FAISS (dense) + BM25 (sparse) + RRF 融合 → 混合检索
- **推理流程**: LangGraph 7节点工作流（意图分类→澄清/子问题拆解→检索→充分性检查→查询优化→答案生成）
- **LLM**: DeepSeek v4-pro（通过 OpenAI 兼容 API）
- **Embedding**: BAAI/bge-base-en-v1.5（本地加载）
- **前端**: Gradio Web UI（流式输出 + 推理轨迹面板）
- **数据**: chunks.jsonl (4750 chunks, ~17MB) + faiss.index (~26MB) + bm25_index.pkl (~17MB)

---

## ✅ 已解决: multi-hop retry 耗尽后跳过后续子问题

**根因**: `run_workflow_streaming()` 中 `for cycle_idx in range(max_cycles)` 的 `max_cycles` 是所有子问题共享的硬上限（=4）。第一个子问题用完 4 个 cycle 后循环直接退出，第二个子问题完全未被检索。

**修复**: 将外层 `for` 循环改为 `while True`，在切换到下一个子问题时重置 `state.retry_count = 0`，使每个子问题独立获得完整的 retry 配额。

**验证**: ✅ 已通过测试 — "Compare Veracier's revenue growth between 2021 and 2022 across its main business segments" 现在两个子问题均能独立检索。

---

## 今晚待改进项

### 🔴 P0 — 稳定性修复

#### 1. `bash.exe.stackdump` 崩溃问题
- **文件**: `E:\agentProject\companyrag\bash.exe.stackdump`
- **现象**: 执行某些 Shell 命令时 bash.exe 崩溃产生 stackdump
- **排查方向**:
  - 检查是否是 Git Bash 与 conda Python 路径交互问题
  - 可能是 `data_prepare.py` 中 `Path.home()` 或 Windows 路径处理触发
  - 建议: 所有长路径操作统一使用 `pathlib.Path`，避免 shell 拼接

#### 2. Tesseract OCR 路径健壮性
- **问题**: `data_prepare.py` 中 poppler/tesseract 路径搜索依赖 conda env 结构假设
- **当前**: 硬编码猜测 `Library/bin` / `share/tessdata` 位置
- **改进**:
  - 先检查系统 PATH 中是否已有 tesseract/pdftoppm
  - 使用 `shutil.which()` 查找可执行文件
  - 添加 `--tesseract-path` 和 `--poppler-path` CLI 参数作为手动覆盖

### 🟡 P1 — 架构优化

#### 3. `run_workflow_streaming()` 与 `build_graph()` 逻辑重复
- **问题**: `run_workflow_streaming()` (workflow.py:1254-1563) 手动实现了完整的 300 行工作流逻辑，与 `build_graph()` 编译的 LangGraph 几乎重复
- **后果**: 修改节点逻辑时必须同时改两处，极易遗漏
- **建议**:
  - 使用 `graph.stream()` 或 `graph.astream_events()` 替代手写循环
  - 将 streaming 事件通过 LangGraph 的 `astream_events` 回调发出
  - 保留 `run_workflow_streaming()` 作为事件翻译层，不重复节点逻辑

#### 4. 子问题拆解质量不稳定
- **问题**: `plan_sub_questions` 的 fallback 逻辑（line 449-464）在 LLM 返回格式异常时简单拆成 "part 1: factual / part 2: analysis"
- **改进**:
  - 添加 JSON 格式解析（尝试 `json.loads` 匹配 `[...]` 数组）
  - 对 `deepseek-v4-pro` 模型调优 system prompt（当前 prompt 是通用英文，未针对 DeepSeek 优化）
  - 添加子问题数量上限（防止拆出 10+ 个子问题）

#### 5. 检索结果缓存缺失
- **问题**: 相同或相似的查询每次都要走完整的 FAISS + BM25 + RRF 流程
- **建议**:
  - 对原始用户问题做 embedding → 存入简单的 in-memory LRU cache（128 条）
  - 缓存 key = 问题 embedding 的 hash，value = top-K chunk_ids
  - 可大幅减少 embedding 调用（尤其是 multi-hop 中的重复查询）

### 🟢 P2 — 用户体验提升

#### 6. Gradio 对话历史管理
- **问题**: 当前 `clear_chat()` 只是清空 UI，没有保留对话上下文
- **改进**:
  - 添加 "导出对话" 按钮（导出为 Markdown/JSON）
  - 每次问答后将 Q&A pair 存入 `state.accumulated_chunks` 旁的对话历史
  - 支持多轮对话上下文（目前每次提问完全独立）

#### 7. Top-K 动态扩展策略文档化
- **当前**: `run_workflow_streaming` 中 Top-K 在 retry 时自动翻倍（5→10→15），但用户不可见
- **改进**: 在 Gradio trace panel 中显示 "Top-K 已从 5 提升到 10（第 1 次重试）"

#### 8. 答案来源可追溯性
- **当前**: 答案末尾附加 `[来源：参考文档 doc_0, doc_1, ...]`，但用户无法直接跳转查看原文
- **改进**:
  - Gradio sources panel 中每个 chunk 添加可点击展开的全文预览
  - 答案中的 `[doc_N]` 标记改为高亮可点击，点击后定位到 sources panel 对应条目

### ⚪ P3 — 工程完善

#### 9. 日志系统升级
- **当前**: 仅 `logging.basicConfig` 输出到控制台
- **改进**:
  - 添加 `RotatingFileHandler` 写入 `logs/rag_{date}.log`
  - 记录每次问答的完整 trace（question、intent、retrieved_chunks、answer、latency）
  - 便于后续分析检索质量和延迟瓶颈

#### 10. 测试覆盖
- **当前**: `test_e2e.py` 和 `test_mvp.py` 存在但依赖真实索引和 API
- **建议**:
  - 添加 mock LLM 和 mock retriever 的单元测试
  - 测试 `route_by_sufficiency` 的状态转换逻辑（当前未覆盖边界条件）
  - 测试 `_rrf_fusion` 的数学正确性（用已知输入验证输出）

#### 11. 配置文件拆分
- **问题**: `config.py` 既包含配置定义又包含设备检测逻辑
- **建议**:
  - 将 `_detect_best_device()` 移到 `env_loader.py`
  - `config.py` 只保留纯 dataclass 定义
  - 添加 `config.validate()` 方法，启动时检查路径是否存在、模型文件是否可访问

---

## 今晚建议执行顺序

| 优先级 | 项目 | 预计耗时 | 影响范围 |
|--------|------|---------|----------|
| 1 | bash.exe.stackdump 排查 | 20min | 稳定性 |
| 2 | Tesseract 路径修复 | 15min | data_prepare.py |
| 3 | streaming 去重（用 graph.stream） | 45min | workflow.py 重构 |
| 4 | 检索缓存 | 30min | retriever.py |
| 5 | Gradio trace 显示 Top-K 变化 | 10min | app_gradio.py |
| 6 | 日志写入文件 | 15min | 全局 |

---

## 项目关键路径速查

| 文件 | 大小 | 职责 |
|------|------|------|
| `config.py` | 9KB | 全局配置 dataclass + 设备检测 |
| `env_loader.py` | 5KB | 环境变量注入 + .env 加载 |
| `data_prepare.py` | 37KB | PDF→文本→分块→FAISS+BM25 索引构建 |
| `retriever.py` | 17KB | 混合检索（FAISS+BM25+RRF） |
| `workflow.py` | 63KB | LangGraph 工作流 + streaming 包装 |
| `llm_client.py` | 15KB | LLM API 调用（OpenAI/DeepSeek/Anthropic） |
| `app_gradio.py` | 13KB | Gradio Web UI |
| `main.py` | 9KB | CLI 入口（REPL + 批处理） |

---

*注：由于 Claude Code 闪退，本次会话的历史对话已丢失。以上改进建议基于对当前代码库完整阅读后的分析。如需回顾特定上下文，请告知。*
