# 🏢 Agentic RAG — 企业知识库智能问答系统

基于 **LangGraph** 构建的 **Agentic RAG**（检索增强生成）系统，面向企业级文档库的智能问答。数据集采用 Veracier Industries 的约 1,100 份多语言 PDF 文档，涵盖 9 个子公司的财务、法务、技术、生产等业务文件，支持法语（53%）、英语（23%）、德语（10%）、意大利语、西班牙语五种语言。

> **当前版本**: v0.5 · **仓库**: [nobailan/AgenticRAG](https://github.com/nobailan/AgenticRAG)

## ✨ 核心亮点

- **8 节点 Agent 工作流** — 意图分类 → 澄清反问 / 多跳拆解 → 混合检索 → **Cross-encoder 精排** → 充分性校验 → 查询改写 → 答案生成，全程透明可追溯
- **混合检索 + 精排** — FAISS 稠密向量 + BM25 稀疏检索 + RRF 融合粗排，再用 BAAI/bge-reranker-base 做 cross-encoder 精排，检索精度显著提升
- **两级语义缓存** — L1 精确缓存（MD5 匹配，< 100ms）+ L2 语义缓存（FAISS 向量相似度，< 200ms），大幅降低重复/相似问题的 LLM 调用成本
- **多 Agent 协作架构** — Supervisor 调度器 + retriever / critic / synthesizer 三个 Worker，复杂问题多角色协作处理
- **多跳推理** — 复杂问题自动拆解为有序子问题链，子问题独立检索、独立校验、独立 retry 配额
- **流式 Web 界面** — Gradio 聊天 UI，答案逐字流式输出，右侧面板实时展示推理轨迹，支持缓存命中状态展示
- **五语言分词** — NLTK punkt + 自研正则回退分词器，正确处理 S.A. / GmbH / S.p.A. / S.A.R.L. 等缩写
- **OCR + GPU 加速** — Tesseract OCR 处理扫描件，CUDA GPU 自动检测（~10x embedding 提速）
- **自动化回归测试** — 30 题黄金测试集 + 基线对比 + GitHub Actions CI
- **Docker 一键部署** — Dockerfile + docker-compose（含 Redis 缓存服务）

![界面一览](picture/界面一览.png)

## 🏗 系统架构

```
用户问题
  │
  ├── 缓存检查（L1 精确缓存 → L2 语义缓存）→ 命中 → 直接返回
  │
  ▼
┌─────────────────┐
│  classify_intent │  意图分类
└───────┬─────────┘
   ┌────┼────┐
   ▼    ▼    ▼
unclear simple multi_hop
   │    │    │
   ▼    │    ▼
 反问    │  plan_sub_questions
 终止    │    │
        ▼    ▼
   ┌─────────────────┐
   │    retrieve      │  FAISS + BM25 + RRF 混合检索
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │     rerank       │  Cross-encoder 精排（v0.4）
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │check_sufficiency │  LLM 判断信息是否充分
   └────────┬────────┘
            │
     ┌──────┼──────┐
     ▼      ▼      ▼
  充分   不足+   不足+
        retry<3  retry=3
     │      │      │
     │      ▼      │
     │  refine_query│
     │      │      │
     └──┬───┘      │
        ▼          ▼
   ┌──────────────┐
   │generate_answer│  流式生成 + [doc_N] 引用
   └──────┬───────┘
          │
          ▼
     写入两级缓存（L1 + L2）
```

### 多 Agent 模式（v0.5）

```
复杂问题
  │
  ▼
┌──────────────┐
│  Supervisor   │  分析问题，生成任务计划
└──────┬───────┘
       │
  ┌────┼────┐
  ▼    ▼    ▼
retriever  critic  synthesizer
  Worker   Worker    Worker
  │       │        │
  └───┬───┴────────┘
      ▼
  汇总生成最终答案
```

### 关键设计决策

| 决策 | 说明 |
|------|------|
| **两级缓存** | L1 精确（MD5，< 100ms）→ L2 语义（FAISS，< 200ms），逐级查询，Redis 持久化 |
| **Cross-encoder 精排** | bi-encoder 粗筛 20 篇 → cross-encoder 精排保留 top-5，精度/速度平衡 |
| **渐进式充分性校验** | retry 0 严格 → retry 1 温和 → retry 2+ 自动通过，避免穷举型问题死循环 |
| **top-k 动态递增** | 首次 top_k=5 → 重试 ×2 → 再重试 ×3，扩大搜索半径 |
| **子问题独立 retry** | 多跳场景切换子问题时重置 retry_count，每个子问题独立配额 |
| **句子级分块** | 句子不从中切断，overlap 在句子级别计算 |

## 🚀 快速开始

### 环境要求

- **Python 3.10+**
- **Tesseract OCR**（扫描件 PDF 需要，可跳过）
  - Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-ita tesseract-ocr-spa`
  - macOS: `brew install tesseract tesseract-lang`
- **poppler**（`pdf2image` 依赖，可跳过）
  - Ubuntu: `sudo apt install poppler-utils`
- **Redis**（语义缓存需要，可跳过）
  - `docker run -d -p 6379:6379 redis:7-alpine`
- **LLM API Key** — DeepSeek / OpenAI / Anthropic

### 安装

```bash
git clone https://github.com/nobailan/AgenticRAG.git
cd AgenticRAG

python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

pip install -r requirements.txt

# 配置 API Key
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"
export RAG_LLM_MODEL="deepseek-v4-pro"

# Embedding 模型（推荐提前下载到本地）
export RAG_EMBEDDING_MODEL="E:/agentProject/embedding-model/bge-base-en-v1.5"
```

### 构建索引

```bash
python data_prepare.py --skip-ocr
# 或指定本地 embedding 模型
python data_prepare.py --skip-ocr \
    --embedding-model "E:/agentProject/embedding-model/bge-base-en-v1.5"
```

### 启动

```bash
# Web UI
python app_gradio.py
# → http://localhost:7860

# CLI
python main.py --interactive

# Docker（含 Redis）
docker-compose up --build
```

> 🎬 推理过程演示（GIF 已做约 14 倍加速，实际耗时约 60 秒）：

![推理过程](picture/推理过程.gif)

## ⚙️ 配置参数

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| **检索** | | |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding 模型 |
| `RAG_TOP_K` | `5` | 每次检索返回块数 |
| `RAG_RRF_K` | `60` | RRF 融合常数 |
| `RAG_CHUNK_SIZE` | `512` | 分块大小（tokens） |
| **精排 (v0.4)** | | |
| `RAG_RERANKER_ENABLED` | `true` | 是否启用 cross-encoder 精排 |
| `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-base` | 精排模型 |
| `RAG_RERANKER_TOP_K` | `5` | 精排保留数 |
| **缓存 (v0.5)** | | |
| `RAG_CACHE_ENABLED` | `true` | 是否启用缓存 |
| `RAG_CACHE_REDIS_URL` | `redis://localhost:6379/0` | Redis 地址 |
| `RAG_CACHE_SIMILARITY_THRESHOLD` | `0.92` | 语义缓存相似度阈值 |
| `RAG_CACHE_TTL` | `86400` | 缓存过期时间（秒） |
| **LLM** | | |
| `RAG_LLM_MODEL` | `gpt-4o-mini` | LLM 模型名 |
| `RAG_LLM_TEMPERATURE` | `0.0` | 采样温度 |
| `RAG_MAX_RETRIES` | `3` | 每子问题重试上限 |
| **多语言 (v0.4)** | | |
| `RAG_EMBEDDING_QUERY_PREFIX` | `""` | E5 查询前缀（如 `query: `） |
| `RAG_EMBEDDING_PASSAGE_PREFIX` | `""` | E5 文档前缀（如 `passage: `） |

> DeepSeek 用户：设置 `DEEPSEEK_API_KEY` 即可自动切换端点，无需手动设 `RAG_LLM_PROVIDER`。

## 📁 项目结构

```
AgenticRAG/
├── main.py, app_gradio.py     # 入口（CLI / Web UI）
├── test_e2e.py                # 端到端测试
├── test_questions.txt         # 测试问题集
├── requirements.txt           # Python 依赖
├── Dockerfile, docker-compose.yml, .dockerignore
├── .env.template, .gitignore
├── README.md, CHANGELOG.md
├── picture/                   # 截图
│
├── src/
│   ├── core/                  # 基础层
│   │   ├── config.py          #   全局配置 dataclass
│   │   ├── env_loader.py      #   环境变量加载
│   │   └── models.py          #   数据模型（RetrievedChunk, RAGState）
│   ├── llm/                   # LLM 调用层
│   │   └── llm_client.py      #   OpenAI/DeepSeek/Anthropic API
│   ├── retrieval/             # 检索层
│   │   ├── retriever.py       #   HybridRetriever（FAISS + BM25 + RRF）
│   │   └── reranker.py        #   CrossEncoderReranker 精排
│   ├── agents/                # Agent 层
│   │   ├── workflow.py        #   LangGraph 8节点工作流 + streaming
│   │   ├── supervisor.py      #   多Agent 调度器 (v0.5)
│   │   └── workers/           #   Worker 节点
│   │       ├── retriever_worker.py
│   │       ├── critic_worker.py
│   │       └── synthesizer_worker.py
│   ├── cache/                 # 缓存层 (v0.5)
│   │   ├── cache_manager.py   #   两级缓存统一入口
│   │   ├── exact_cache.py     #   L1 精确缓存（MD5）
│   │   ├── semantic_cache.py  #   L2 语义缓存（FAISS）
│   │   └── redis_client.py    #   Redis 连接封装
│   ├── data/                  # 数据准备层
│   │   └── data_prepare.py    #   PDF → chunks → FAISS + BM25
│   └── web/                   # Web UI 层
│       └── app_gradio.py      #   Gradio 界面
│
└── evaluation/                # 评测体系 (v0.4+)
    ├── testset.json           #   基础测试集
    ├── golden_testset.json    #   黄金测试集（30题）
    ├── evaluate.py            #   RAGAS 评测运行器
    ├── ragas_metrics.py       #   指标计算
    ├── regression_test.py     #   回归测试 + 基线对比
    ├── baseline_metrics.json  #   基线指标
    └── run_regression.sh      #   一键回归测试
```

## 🧪 测试

```bash
# 端到端测试
python test_e2e.py --verbose --limit 5

# 回归测试（30 题黄金集，与基线对比）
python evaluation/regression_test.py
# 或一键运行
bash evaluation/run_regression.sh

# RAGAS 评测（需要安装 ragas）
python evaluation/evaluate.py --limit 10
```

## 📊 数据规模

| 指标 | 数值 |
|------|------|
| 处理 PDF 数 | 1,022（100%） |
| 文档块总数 | 8,672 |
| Embedding 维度 | 768（bge-base-en-v1.5） |
| FAISS 索引 | ~26.6 MB |
| BM25 索引 | ~17 MB |
| GPU 加速比 | ~10x（RTX 3060） |
| 支持语言 | 法语、英语、德语、意大利语、西班牙语 |

## 🔧 已知限制

- **单轮问答** — 每次提问独立处理，暂不支持多轮对话
- **无用户认证** — 单用户模式
- **语义缓存需 Redis** — 无 Redis 时降级为内存模式（重启丢失）
- **cross-encoder 需额外安装** — `pip install sentence-transformers`
- **Embedding 模型** — bge-base-en-v1.5 英文优化，非英语召回率可能偏低
- **多 Agent 模式** — 基础版，Worker 调度为静态编排

## 🗺 版本演进

| 版本 | 分支 | 核心 |
|------|------|------|
| v0.3.1 | `main` | 首发版：Gradio UI · 流式输出 · 8,672 chunks |
| v0.4 | `feature/optimization-june-2026` | 精排 · 语义缓存 · E5 多语言 · RAGAS 评测 · Docker |
| v0.5 | `feature/v0.5-upgrade` | 两级缓存 · 回归测试/CI · 多 Agent 架构 |

---

**技术栈**：[LangGraph](https://github.com/langchain-ai/langgraph) · [FAISS](https://github.com/facebookresearch/faiss) · [Gradio](https://github.com/gradio-app/gradio) · [BGE](https://huggingface.co/BAAI/bge-base-en-v1.5) · [DeepSeek](https://platform.deepseek.com/) · [Redis](https://redis.io/) · [Docker](https://www.docker.com/)
