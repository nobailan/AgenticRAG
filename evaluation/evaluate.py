"""
evaluate.py — RAGAS 评测运行器

功能：加载测试集，逐条运行 RAG 流水线，计算 RAGAS 指标，生成对比报告。

运行方式：
    python evaluation/evaluate.py                    # 运行全部测试
    python evaluation/evaluate.py --limit 5         # 只跑前 5 道题
    python evaluation/evaluate.py --output my_report.md  # 指定输出路径

输出：
    - 终端打印每道题的三个指标分数和耗时
    - 汇总平均分
    - 生成 evaluation/comparison_report.md 评测报告
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# 确保项目根目录在 sys.path 中，使 from src.xxx import 能正常工作
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import config
from evaluation.ragas_metrics import (
    compute_ragas_metrics,
    compute_ragas_metrics_simple,
    RAGASScore,
)

logger = logging.getLogger(__name__)

# 默认路径
TESTSET_PATH = Path(__file__).parent / "testset.json"
REPORT_PATH = Path(__file__).parent / "comparison_report.md"


# ======================================================================
# 测试集加载
# ======================================================================

def load_testset(path: Path = TESTSET_PATH) -> List[Dict]:
    """从 JSON 文件加载评测测试集。

    Args:
        path: testset.json 路径

    Returns:
        测试用例列表，每项包含 question, ground_truth, contexts
    """
    if not path.exists():
        logger.error("测试集文件不存在: %s", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("已加载 %d 条测试用例 (%s)", len(data), path.name)
    return data


# ======================================================================
# 单条评测
# ======================================================================

def run_single_evaluation(test: Dict) -> Dict:
    """对一条测试用例跑完整流水线并计算指标。

    流程：
        1. 调 run_rag() 生成答案
        2. 计时
        3. 调 RAGAS 评分（优先用完整版，不可用时降级为简易版）

    Args:
        test: {"question": str, "ground_truth": str, "contexts": [str]}

    Returns:
        {"question", "answer", "ground_truth", "contexts", "scores", "elapsed_sec", "intent"}
    """
    from src.agents.workflow import run_rag

    question = test["question"]
    ground_truth = test.get("ground_truth", "")

    start = time.time()
    try:
        result = run_rag(question)
        answer = result.get("answer", "")
    except Exception as e:
        logger.error("流水线执行失败 '%s': %s", question[:60], e)
        answer = f"[ERROR] {e}"
        result = {}

    elapsed = time.time() - start

    # 评分：优先用完整 RAGAS，不可用时降级为简易版
    try:
        scores = compute_ragas_metrics(
            question, answer, test.get("contexts", []), ground_truth
        )
    except Exception:
        scores = compute_ragas_metrics_simple(
            question, answer, test.get("contexts", []), ground_truth
        )

    return {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "contexts": test.get("contexts", []),
        "scores": scores.to_dict(),
        "elapsed_sec": round(elapsed, 2),
        "intent": result.get("intent", "unknown"),
        "from_cache": result.get("from_cache", False),
    }


# ======================================================================
# 批量评测
# ======================================================================

def run_evaluation(
    limit: int = 0, output_path: Path = REPORT_PATH
) -> List[Dict]:
    """批量运行全部测试用例并生成报告。

    Args:
        limit: 限制测试条数（0 = 全部）
        output_path: 报告输出路径

    Returns:
        每条测试的结果列表
    """
    tests = load_testset()
    if not tests:
        return []

    if limit > 0:
        tests = tests[:limit]

    results = []
    total = len(tests)

    for i, test in enumerate(tests, 1):
        print(f"[{i}/{total}] {test['question'][:80]}...", flush=True)
        r = run_single_evaluation(test)
        results.append(r)

        s = r["scores"]
        print(
            f"  Faith={s['faithfulness']:.3f}  "
            f"AnswerR={s['answer_relevancy']:.3f}  "
            f"CtxR={s['context_relevancy']:.3f}  "
            f"({r['elapsed_sec']}s)"
        )

    # 汇总
    agg = aggregate_scores(results)
    print(f"\n{'='*50}")
    print(f"平均得分（{len(results)} 题）:")
    print(f"  Faithfulness:        {agg['faithfulness']:.3f}")
    print(f"  Answer Relevancy:    {agg['answer_relevancy']:.3f}")
    print(f"  Context Relevancy:   {agg['context_relevancy']:.3f}")
    print(f"  综合平均:             {agg['average']:.3f}")
    print(f"  平均耗时:             {agg['avg_time']:.1f}s")

    # 生成 Markdown 报告
    generate_report(results, agg, output_path)

    return results


def aggregate_scores(results: List[Dict]) -> Dict[str, float]:
    """计算所有测试结果的平均分。"""
    if not results:
        return {}
    n = len(results)
    agg = {
        "faithfulness": sum(r["scores"]["faithfulness"] for r in results) / n,
        "answer_relevancy": sum(r["scores"]["answer_relevancy"] for r in results) / n,
        "context_relevancy": sum(r["scores"]["context_relevancy"] for r in results) / n,
    }
    agg["average"] = sum(agg.values()) / 3
    agg["avg_time"] = sum(r["elapsed_sec"] for r in results) / n
    return agg


def generate_report(
    results: List[Dict], agg: Dict, path: Path
) -> None:
    """生成 Markdown 格式的评测报告。

    报告结构：
        1. 概览（日期、题数、配置状态）
        2. 汇总表
        3. 每题详细结果

    Args:
        results: 每题结果列表
        agg: 汇总统计
        path: 输出路径
    """
    lines = [
        "# AgenticRAG v0.4 — RAGAS 评测报告",
        "",
        f"**评测日期**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**测试题数**: {len(results)}",
        f"**当前配置**: 精排={'开启' if config.reranker_enabled else '关闭'}, "
        f"缓存={'开启' if config.cache_enabled else '关闭'}",
        "",
        "## 汇总得分",
        "",
        "| 指标 | 得分 | 说明 |",
        "|------|------|------|",
        f"| Faithfulness | {agg['faithfulness']:.3f} | 答案是否忠于检索文档 |",
        f"| Answer Relevancy | {agg['answer_relevancy']:.3f} | 答案是否切题 |",
        f"| Context Relevancy | {agg['context_relevancy']:.3f} | 检索文档是否相关 |",
        f"| **综合平均** | **{agg['average']:.3f}** | 三项均值 |",
        f"| 平均响应时间 | {agg['avg_time']:.1f}s | 端到端耗时 |",
        "",
        "## 逐题结果",
        "",
    ]

    for i, r in enumerate(results, 1):
        s = r["scores"]
        cache_tag = " (缓存命中)" if r.get("from_cache") else ""
        lines.append(f"### {i}. {r['question'][:100]}")
        lines.append(f"- **答案**: {r['answer'][:200]}...")
        lines.append(f"- **忠实度**: {s['faithfulness']:.3f}")
        lines.append(f"- **答案相关性**: {s['answer_relevancy']:.3f}")
        lines.append(f"- **上下文相关性**: {s['context_relevancy']:.3f}")
        lines.append(f"- **耗时**: {r['elapsed_sec']}s{cache_tag}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("评测报告已写入 %s", path)


# ======================================================================
# CLI 入口
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AgenticRAG RAGAS 评测运行器")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="限制测试条数（0 = 全部）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="报告输出路径（默认 evaluation/comparison_report.md）"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="打印详细日志"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output = Path(args.output) if args.output else REPORT_PATH
    run_evaluation(limit=args.limit, output_path=output)
