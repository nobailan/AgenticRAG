# AgenticRAG v0.8 — 一键启动脚本
# 用法: 右键此文件 → "使用 PowerShell 运行" 或在终端输入: .\启动WebUI.ps1
# 首次运行若报"无法加载文件"，先执行: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$ErrorActionPreference = "Stop"

# ====== 配置区域 ======
$CondaPath = "E:\anaconda3"
$EnvName = "graph"
$ProjectDir = "E:\agentProject\companyrag"

# LLM 配置（如已设系统环境变量可删掉这几行）
#$env:DEEPSEEK_API_KEY = $env:DEEPSEEK_API_KEY ?? ""
$env:RAG_LLM_MODEL = "deepseek-v4-pro"

# Embedding 模型（bge-m3 下载后改这里即可切换多语言）
$env:RAG_EMBEDDING_MODEL = "E:/agentProject/embedding-model/bge-base-en-v1.5"
$env:RAG_EMBEDDING_DEVICE = "cuda"

# 索引文件路径（v0.8 新索引, 60,818 chunks）
$env:RAG_CHUNK_JSONL = "data/indices/rag_bench/chunks.jsonl"
$env:RAG_FAISS_INDEX = "data/indices/rag_bench/faiss.index"
$env:RAG_BM25_INDEX  = "data/indices/rag_bench/bm25_index.pkl"

# ====== 激活 Conda ======
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Agentic RAG v0.8 — 启动 Web UI" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if (-not (Test-Path "$CondaPath\Scripts\conda.exe")) {
    Write-Host "[ERROR] Conda 未找到: $CondaPath" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "[1/3] 激活 Conda 环境: $EnvName" -ForegroundColor Yellow
& "$CondaPath\Scripts\conda.exe" activate $EnvName 2>$null
# conda activate 在脚本中需要特殊处理
$env:PATH = "$CondaPath\envs\$EnvName;$CondaPath\envs\$EnvName\Scripts;$CondaPath\envs\$EnvName\Library\bin;$env:PATH"
$env:CONDA_PREFIX = "$CondaPath\envs\$EnvName"

Write-Host "[2/3] 初始化 Python 环境..." -ForegroundColor Yellow
$PythonExe = "$CondaPath\envs\$EnvName\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "[ERROR] Python 未找到: $PythonExe" -ForegroundColor Red
    pause
    exit 1
}

# 检查关键依赖
$check = & $PythonExe -c "import gradio, sentence_transformers, faiss; print('OK')" 2>&1
if ($check -notmatch "OK") {
    Write-Host "[WARN] 部分依赖缺失: $check" -ForegroundColor Yellow
}

Write-Host "[3/3] 启动 Gradio Web UI..." -ForegroundColor Yellow
Write-Host "  LLM: $env:RAG_LLM_MODEL" -ForegroundColor Gray
Write-Host "  Embedding: $env:RAG_EMBEDDING_MODEL" -ForegroundColor Gray
Write-Host ""

Set-Location $ProjectDir
& $PythonExe app_gradio.py

Write-Host ""
Write-Host "服务已停止。" -ForegroundColor Gray
pause
