<#
.SYNOPSIS
    一键启动 Agentic RAG 企业知识库 (Veracier Industries)。

.DESCRIPTION
    - 激活 conda 虚拟环境 (默认 E:\anaconda3\envs\graph)，正确注入 DLL 路径
      (Library\bin 等)，保证 torch / faiss / CUDA 依赖库可加载。
    - 加载 .env、校验索引文件与 API Key，给出友好提示。
    - 默认启动 Gradio Web UI (app_gradio.py)，并自动打开浏览器。

.PARAMETER Mode
    web  (默认) 启动 Gradio Web UI，浏览器访问 http://localhost:7860
    cli  启动交互式命令行 REPL (main.py --interactive)

.PARAMETER NoBrowser
    web 模式下不自动打开浏览器。

.PARAMETER EnvPath
    conda 虚拟环境根目录，默认 E:\anaconda3\envs\graph。

.EXAMPLE
    .\start.ps1
    .\start.ps1 -Mode cli
    .\start.ps1 -NoBrowser
#>
param(
    [ValidateSet('web', 'cli')]
    [string]$Mode = 'web',

    [switch]$NoBrowser,

    [string]$EnvPath = 'E:\anaconda3\envs\graph'
)

$ErrorActionPreference = 'Stop'

# 控制台使用 UTF-8，避免中文乱码
try { chcp 65001 > $null } catch { }
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 项目根目录 = 本脚本所在目录
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

function Write-Section($text) {
    Write-Host ''
    Write-Host ('=' * 60) -ForegroundColor DarkCyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host ('=' * 60) -ForegroundColor DarkCyan
}

Write-Section 'Agentic RAG — Veracier Industries 一键启动'

# ---------------------------------------------------------------------------
# 1. 校验并激活 conda 虚拟环境
# ---------------------------------------------------------------------------
$PythonExe = Join-Path $EnvPath 'python.exe'
if (-not (Test-Path $PythonExe)) {
    Write-Host "[ERROR] 未找到 Python 解释器: $PythonExe" -ForegroundColor Red
    Write-Host "        请用 -EnvPath 指定正确的 conda 环境目录。" -ForegroundColor Red
    exit 1
}

# 手动注入 conda 环境路径（等价于 conda activate），确保原生依赖库 DLL 可被加载
$env:CONDA_PREFIX = $EnvPath
$env:CONDA_DEFAULT_ENV = Split-Path $EnvPath -Leaf
$condaPaths = @(
    $EnvPath,
    (Join-Path $EnvPath 'Library\mingw-w64\bin'),
    (Join-Path $EnvPath 'Library\usr\bin'),
    (Join-Path $EnvPath 'Library\bin'),
    (Join-Path $EnvPath 'Scripts'),
    (Join-Path $EnvPath 'bin')
) | Where-Object { Test-Path $_ }
$env:PATH = ($condaPaths -join ';') + ';' + $env:PATH

$pyVer = (& $PythonExe --version 2>&1)
Write-Host "[OK] 虚拟环境: $EnvPath" -ForegroundColor Green
Write-Host "[OK] $pyVer" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 2. 预检查：索引文件 & API Key（仅提示，不阻塞启动）
# ---------------------------------------------------------------------------
$indexFiles = @('faiss.index', 'chunks.jsonl', 'bm25_index.pkl')
$missing = $indexFiles | Where-Object { -not (Test-Path (Join-Path $ProjectRoot $_)) }
if ($missing) {
    Write-Host "[WARN] 缺少索引文件: $($missing -join ', ')" -ForegroundColor Yellow
    Write-Host "       请先运行数据准备脚本生成索引 (python -m src.data.data_prepare ...)。" -ForegroundColor Yellow
} else {
    Write-Host "[OK] 索引文件齐全 (faiss.index / chunks.jsonl / bm25_index.pkl)" -ForegroundColor Green
}

$hasApiKey = $env:DEEPSEEK_API_KEY -or $env:OPENAI_API_KEY -or $env:ANTHROPIC_API_KEY
if (-not $hasApiKey) {
    Write-Host "[WARN] 未检测到 API Key (DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)。" -ForegroundColor Yellow
    Write-Host "       请设置系统环境变量后再问答，否则 LLM 调用会失败。" -ForegroundColor Yellow
} else {
    Write-Host "[OK] 已检测到 API Key" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 3. 启动
# ---------------------------------------------------------------------------
if ($Mode -eq 'cli') {
    Write-Section '启动 CLI 交互模式 (main.py --interactive)'
    & $PythonExe main.py --interactive
    exit $LASTEXITCODE
}

# web 模式
Write-Section '启动 Gradio Web UI (app_gradio.py)'
Write-Host '  本地地址: http://localhost:7860' -ForegroundColor Green
Write-Host '  健康检查: http://localhost:7860/health' -ForegroundColor DarkGray
Write-Host '  按 Ctrl+C 停止服务。' -ForegroundColor DarkGray
Write-Host ''

$browserJob = $null
if (-not $NoBrowser) {
    # 后台任务：等端口就绪后自动打开浏览器
    $browserJob = Start-Job -ScriptBlock {
        $url = 'http://localhost:7860'
        for ($i = 0; $i -lt 60; $i++) {
            Start-Sleep -Seconds 1
            try {
                $conn = New-Object System.Net.Sockets.TcpClient
                $conn.Connect('127.0.0.1', 7860)
                $conn.Close()
                Start-Process $url
                break
            } catch { }
        }
    }
}

try {
    & $PythonExe app_gradio.py
} finally {
    if ($browserJob) { Stop-Job $browserJob -ErrorAction SilentlyContinue; Remove-Job $browserJob -ErrorAction SilentlyContinue }
}
