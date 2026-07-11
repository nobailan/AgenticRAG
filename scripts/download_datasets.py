"""
download_datasets.py — 下载 v0.8 测试数据集

数据集:
    1. EnterpriseRAG-Bench (5,000 文件 slice) — 多来源企业文本
    2. SEC 10-K 年报 (~100 PDFs) — 真实企业 PDF，含表格+图片

运行:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --sec-limit 50  --rag-limit 2000
"""

import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "datasets"
RAG_DIR = DATA_DIR / "enterprise_rag_bench"
SEC_DIR = DATA_DIR / "sec_10k"


# ======================================================================
# Dataset 1: EnterpriseRAG-Bench
# ======================================================================

def download_rag_bench(limit: int = 5000):
    """从 HuggingFace 下载 EnterpriseRAG-Bench 数据集。

    优先使用 huggingface_hub，不可用时从 GitHub Releases 下载 slice。

    Args:
        limit: 最大下载文件数
    """
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("下载 EnterpriseRAG-Bench (目标: %d 文件)...", limit)

    # 方法1: huggingface_hub
    try:
        from huggingface_hub import snapshot_download, list_repo_files

        logger.info("尝试通过 huggingface_hub 下载...")
        # 只下载文档，不下载问题集
        all_files = list_repo_files("onyx-dot-app/EnterpriseRAG-Bench", repo_type="dataset")
        doc_files = [f for f in all_files if f.startswith("documents/") and f.endswith(".txt")][:limit]

        if doc_files:
            # 下载并解压
            snapshot_download(
                "onyx-dot-app/EnterpriseRAG-Bench",
                repo_type="dataset",
                allow_patterns=doc_files[:100] + ["questions.jsonl"],
                local_dir=str(RAG_DIR),
                local_dir_use_symlinks=False,
            )
            count = len(list(RAG_DIR.rglob("*.txt")))
            logger.info("EnterpriseRAG-Bench 下载完成: %d 文件 -> %s", count, RAG_DIR)
            return count
    except ImportError:
        logger.warning("huggingface_hub 未安装，尝试 GitHub Releases...")
    except Exception as e:
        logger.warning("huggingface_hub 下载失败: %s，尝试 GitHub Releases...", str(e)[:80])

    # 方法2: GitHub Releases (slice download)
    try:
        import requests
        url = ("https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/"
               "v1.0.0/confluence_slice_0.zip")
        logger.info("尝试从 GitHub Releases 下载 slice...")

        # Try a few different slices
        slices = [
            "confluence_slice_0.zip",
            "github_slice_0.zip",
            "jira_slice_0.zip",
        ]

        for slice_name in slices:
            url = f"https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0/{slice_name}"
            zip_path = RAG_DIR / slice_name
            try:
                resp = requests.get(url, stream=True, timeout=30)
                if resp.status_code == 200:
                    with open(zip_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    # 解压
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(RAG_DIR)
                    zip_path.unlink()
                    logger.info("Slice '%s' 下载+解压完成", slice_name)
                else:
                    logger.debug("Slice '%s' 不存在 (HTTP %d)", slice_name, resp.status_code)
            except Exception as e:
                logger.debug("Slice '%s' 下载失败: %s", slice_name, str(e)[:60])

        count = len(list(RAG_DIR.rglob("*.txt")))
        if count > 0:
            logger.info("EnterpriseRAG-Bench: %d 文件 -> %s", count, RAG_DIR)
            return count
    except ImportError:
        logger.warning("requests 未安装")
    except Exception as e:
        logger.warning("GitHub Releases 下载失败: %s", e)

    # 方法3: 手动指引
    logger.warning(
        "自动下载失败。请手动下载:\n"
        "  https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench\n"
        "  解压到: %s", RAG_DIR
    )
    return 0


# ======================================================================
# Dataset 2: SEC 10-K 年报
# ======================================================================

# 10 家知名科技公司的 CIK (SEC Central Index Key)
TECH_COMPANIES = [
    ("AAPL", "0000320193", "Apple Inc."),
    ("MSFT", "0000789019", "Microsoft Corp."),
    ("GOOGL", "0001652044", "Alphabet Inc."),
    ("AMZN", "0001018724", "Amazon.com Inc."),
    ("META", "0001326801", "Meta Platforms Inc."),
    ("NVDA", "0001045810", "NVIDIA Corp."),
    ("TSLA", "0001318605", "Tesla Inc."),
    ("CRM", "0001108524", "Salesforce Inc."),
    ("ADBE", "0000796343", "Adobe Inc."),
    ("ORCL", "0001341439", "Oracle Corp."),
]


def download_sec_10k(limit: int = 100):
    """从 SEC EDGAR 下载 10-K 年报 PDF。

    使用 sec_edgar_downloader 或直接 HTTP 请求。

    Args:
        limit: 最大下载 PDF 数
    """
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("下载 SEC 10-K 年报 (目标: %d 份)...", limit)

    # 方法1: sec_edgar_downloader
    try:
        from sec_edgar_downloader import Downloader

        dl = Downloader(str(SEC_DIR), "your-email@example.com")
        for ticker, cik, name in TECH_COMPANIES:
            try:
                dl.get("10-K", ticker, after="2020-01-01", before="2025-12-31", limit=5)
                logger.info("  已下载 %s (%s) 10-K", ticker, name)
            except Exception as e:
                logger.warning("  %s 下载失败: %s", ticker, str(e)[:60])

        count = len(list(SEC_DIR.rglob("*.pdf")))
        if count > 0:
            logger.info("SEC 10-K 下载完成: %d PDFs -> %s", count, SEC_DIR)
            return count
    except ImportError:
        logger.warning("sec_edgar_downloader 未安装")
    except Exception as e:
        logger.warning("sec_edgar_downloader 失败: %s", str(e)[:80])

    # 方法2: 直接 HTTP 下载
    try:
        import requests
        downloaded = 0
        for ticker, cik, name in TECH_COMPANIES:
            if downloaded >= limit:
                break
            # SEC EDGAR submissions API
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            headers = {"User-Agent": "AgenticRAG/0.8 (research project; contact@example.com)"}
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    filings = data.get("filings", {}).get("recent", {})
                    forms = filings.get("form", [])
                    accessions = filings.get("accessionNumber", [])
                    docs = filings.get("primaryDocument", [])

                    for i, (form, acc, doc) in enumerate(zip(forms, accessions, docs)):
                        if form == "10-K" and downloaded < limit // len(TECH_COMPANIES) + 1:
                            # 构建 PDF URL
                            acc_clean = acc.replace("-", "")
                            pdf_url = (
                                f"https://www.sec.gov/Archives/edgar/data/"
                                f"{int(cik)}/{acc_clean}/{doc}"
                            )
                            pdf_path = SEC_DIR / f"{ticker}_{acc[:10]}_10K.pdf"
                            if not pdf_path.exists():
                                pdf_resp = requests.get(pdf_url, headers=headers, timeout=30)
                                if pdf_resp.status_code == 200:
                                    pdf_path.write_bytes(pdf_resp.content)
                                    downloaded += 1
            except Exception as e:
                logger.warning("  %s SEC API 失败: %s", ticker, str(e)[:60])

        count = len(list(SEC_DIR.rglob("*.pdf")))
        if count > 0:
            logger.info("SEC 10-K 下载完成: %d PDFs -> %s", count, SEC_DIR)
            return count
    except ImportError:
        pass
    except Exception as e:
        logger.warning("SEC EDGAR HTTP 下载失败: %s", e)

    logger.warning(
        "自动下载 SEC 10-K 失败。请手动从 EDGAR 下载:\n"
        "  https://www.sec.gov/edgar/search/\n"
        "  搜索 '10-K' 并下载 PDF 到: %s", SEC_DIR
    )
    return 0


# ======================================================================
# 主入口
# ======================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="下载 v0.8 测试数据集")
    parser.add_argument("--sec-limit", type=int, default=100, help="SEC 10-K 最大 PDF 数")
    parser.add_argument("--rag-limit", type=int, default=5000, help="RAG-Bench 最大文件数")
    parser.add_argument("--skip-sec", action="store_true", help="跳过 SEC 10-K")
    parser.add_argument("--skip-rag", action="store_true", help="跳过 EnterpriseRAG-Bench")
    args = parser.parse_args()

    print("=" * 50)
    print("AgenticRAG v0.8 数据集下载")
    print(f"存储路径: {DATA_DIR}")
    print("=" * 50)

    total = 0

    if not args.skip_rag:
        print("\n[1/2] EnterpriseRAG-Bench")
        n = download_rag_bench(limit=args.rag_limit)
        total += n
    else:
        print("\n[1/2] EnterpriseRAG-Bench: 跳过")

    if not args.skip_sec:
        print("\n[2/2] SEC 10-K 年报")
        n = download_sec_10k(limit=args.sec_limit)
        total += n
    else:
        print("\n[2/2] SEC 10-K: 跳过")

    print(f"\n总计: {total} 文件 -> {DATA_DIR}")
    print("下载完成。运行: python -m src.data.pipeline --input data/datasets")
