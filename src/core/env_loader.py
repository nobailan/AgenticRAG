"""
env_loader.py — 运行时环境变量注入模块

在启动应用前调用 load_env()，自动设置 RAG_EMBEDDING_MODEL、RAG_CHUNK_JSONL 等路径。
API Key 必须通过系统环境变量（如 DEEPSEEK_API_KEY）设置，不要放在 .env 文件中。
"""

import os
import warnings
from pathlib import Path
from typing import Optional


def load_env(env_file: Optional[Path] = None) -> None:
    """
    加载环境变量，优先从系统环境变量读取，未设置的再使用默认值或 .env 文件。

    优先级：系统环境变量 > .env 文件 > 代码内默认值
    注意：API Key（如 DEEPSEEK_API_KEY）建议直接设置在系统环境变量中，以保安全。

    Args:
        env_file: 可选，指定 .env 文件路径，默认为项目根目录下的 .env
    """
    # env_loader is at src/core/env_loader.py → root is 3 levels up
    project_root = Path(__file__).resolve().parent.parent.parent

    if env_file is None:
        env_file = project_root / ".env"

    # 1. 从 .env 文件加载（但不会覆盖已存在的系统环境变量）
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    key = key.strip()
                    # 如果系统环境变量已有该 key，则跳过（不覆盖）
                    if key not in os.environ:
                        os.environ[key] = value.strip()
        print(f"[env_loader] Loaded non-sensitive variables from {env_file}")
    else:
        print(f"[env_loader] No .env file found at {env_file}, using defaults and system env")

    # 2. 设置默认值（仅当系统环境变量和 .env 都未设置时）
    # 这些默认值指向项目根目录下的常见路径
    defaults = {
        "RAG_EMBEDDING_MODEL": str(project_root / "embedding-model" / "bge-base-en-v1.5"),
        "RAG_CHUNK_JSONL": str(project_root / "chunks.jsonl"),
        "RAG_FAISS_INDEX": str(project_root / "faiss.index"),
        "RAG_BM25_INDEX": str(project_root / "bm25_index.pkl"),
        "RAG_LLM_MODEL": "deepseek-v4-pro",
        "RAG_TOP_K": "5",
        "RAG_MAX_RETRIES": "3",
        "RAG_LLM_TEMPERATURE": "0.0",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    # Resolve any relative paths to absolute (handles .env values like ./chunks.jsonl)
    path_keys = ["RAG_EMBEDDING_MODEL", "RAG_CHUNK_JSONL", "RAG_FAISS_INDEX", "RAG_BM25_INDEX"]
    for key in path_keys:
        raw = os.environ.get(key, "")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                os.environ[key] = str((project_root / p).resolve())
    # Also resolve RAG_DATA_DIR and RAG_MASTER_INDEX if present
    for key in ["RAG_DATA_DIR", "RAG_MASTER_INDEX", "RAG_ANSWER_KEY"]:
        raw = os.environ.get(key, "")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                os.environ[key] = str((project_root / p).resolve())

    # 3. 检查必要的 API Key 是否已设置（系统环境变量）
    api_keys = ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    has_key = any(os.environ.get(k) for k in api_keys)
    if not has_key:
        warnings.warn(
            "No API Key found in system environment variables. "
            "Please set DEEPSEEK_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY "
            "before running the application.",
            UserWarning,
            stacklevel=2,
        )

    # 4. 打印关键配置（隐藏 API Key）
    print("[env_loader] Key environment variables (API Key hidden):")
    print(f"  RAG_EMBEDDING_MODEL = {os.environ.get('RAG_EMBEDDING_MODEL')}")
    print(f"  RAG_CHUNK_JSONL     = {os.environ.get('RAG_CHUNK_JSONL')}")
    print(f"  RAG_FAISS_INDEX     = {os.environ.get('RAG_FAISS_INDEX')}")
    print(f"  RAG_BM25_INDEX      = {os.environ.get('RAG_BM25_INDEX')}")
    print(f"  RAG_LLM_MODEL       = {os.environ.get('RAG_LLM_MODEL')}")
    api_key_set = any(os.environ.get(k) for k in api_keys)
    print(f"  API Key             = {'[OK] Set' if api_key_set else '[WARN] Missing'}")


def create_env_template() -> None:
    """生成 .env.template 文件，供用户参考配置（不含敏感信息）"""
    template = """# Agentic RAG 环境变量配置文件（非敏感部分）
# 复制此文件为 .env 并修改对应的值
# 注意：API Key 请通过系统环境变量设置，不要写在 .env 中！

# Embedding 模型路径（本地绝对路径或 HuggingFace 模型名）
RAG_EMBEDDING_MODEL=E:/agentProject/embedding-model/bge-base-en-v1.5

# 索引文件路径（默认在项目根目录）
RAG_CHUNK_JSONL=./chunks.jsonl
RAG_FAISS_INDEX=./faiss.index
RAG_BM25_INDEX=./bm25_index.pkl

# LLM 配置
RAG_LLM_MODEL=deepseek-v4-pro
RAG_LLM_TEMPERATURE=0.0
RAG_TOP_K=5
RAG_MAX_RETRIES=3

# 其他路径配置（如有需要）
# RAG_DATA_DIR=./veracier-industries/by_entity/by_entity
"""
    template_path = Path(__file__).parent / ".env.template"
    template_path.write_text(template, encoding="utf-8")
    print(f"[env_loader] Created .env.template at {template_path}")


if __name__ == "__main__":
    create_env_template()