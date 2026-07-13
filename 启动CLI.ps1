# AgenticRAG v0.8 — CLI 交互模式一键启动
$CondaPath = "E:\anaconda3"
$EnvName = "graph"
$ProjectDir = "E:\agentProject\companyrag"
$env:RAG_LLM_MODEL = "deepseek-v4-pro"
$env:RAG_EMBEDDING_MODEL = "E:/agentProject/embedding-model/bge-base-en-v1.5"
$env:RAG_EMBEDDING_DEVICE = "cuda"
$env:RAG_CHUNK_JSONL = "data/indices/rag_bench/chunks.jsonl"
$env:RAG_FAISS_INDEX = "data/indices/rag_bench/faiss.index"
$env:RAG_BM25_INDEX  = "data/indices/rag_bench/bm25_index.pkl"
$env:PATH = "$CondaPath\envs\$EnvName;$CondaPath\envs\$EnvName\Scripts;$CondaPath\envs\$EnvName\Library\bin;$env:PATH"

Write-Host "AgenticRAG v0.8 — CLI 交互模式" -ForegroundColor Cyan
Set-Location $ProjectDir
& "$CondaPath\envs\$EnvName\python.exe" main.py --interactive
pause
