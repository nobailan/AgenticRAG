"""
regression_test.py — 自动化回归测试

功能：加载黄金测试集，逐条运行 RAG 流水线，计算关键指标，
      与基线指标对比。若任一指标退化超过阈值，退出码 1（CI 失败）。

运行方式：
    python evaluation/regression_test.py                    # 运行全部
    python evaluation/regression_test.py --limit 10        # 只跑前10题
    python evaluation/regression_test.py --baseline-only   # 存储当前指标为基线

输出：
    - 终端打印逐题结果
    - 与基线对比的退化报告
    - 退出码 0 = 全部通过，1 = 有退化
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import config

logger = logging.getLogger(__name__)

# 路径
GOLDEN_PATH = Path(__file__).parent / "golden_testset.json"
BASELINE_PATH = Path(__file__).parent / "baseline_metrics.json"

# 退化阈值：指标下降超过此比例视为退化
DEGRADATION_THRESHOLD = 0.05  # 5%


# ======================================================================
# 加载
# ======================================================================

def load_golden_testset() -> List[Dict]:
    """加载黄金测试集。"""
    if not GOLDEN_PATH.exists():
        logger.error("黄金测试集不存在: %s", GOLDEN_PATH)
        return []
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_baseline() -> Optional[Dict]:
    """加载基线指标。"""
    if not BASELINE_PATH.exists():
        return None
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_baseline(metrics: Dict) -> None:
    """保存当前指标为基线。"""
    BASELINE_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"基线已保存: {BASELINE_PATH}")


# ======================================================================
# 核心评测
# ======================================================================

def run_regression(limit: int = 0) -> Dict:
    """运行回归测试，返回汇总指标。

    Args:
        limit: 限制题数（0 = 全部）

    Returns:
        dict: {"total": N, "passed": N, "failed": N, "accuracy": float,
               "avg_time": float, "cache_hit_rate": float, "degradations": [...]}
    """
    tests = load_golden_testset()
    if not tests:
        return {}

    if limit > 0:
        tests = tests[:limit]

    from src.agents.workflow import run_rag
    from evaluation.ragas_metrics import compute_ragas_metrics_simple

    total = len(tests)
    passed = 0
    failed = 0
    times = []
    cache_hits = 0
    degradations = []
    per_question = []

    for i, test in enumerate(tests, 1):
        question = test["question"]
        ground_truth = test.get("ground_truth", "")
        qtype = test.get("type", "unknown")

        start = time.time()
        try:
            result = run_rag(question)
            answer = result.get("answer", "")
            from_cache = result.get("from_cache", False)
        except Exception as e:
            answer = f"[ERROR] {e}"
            from_cache = False
            result = {}
        elapsed = time.time() - start

        # 评分
        scores = compute_ragas_metrics_simple(
            question, answer, test.get("contexts", []), ground_truth
        )

        # 判定通过/失败：综合分 > 0.3 视为通过
        avg = scores.average
        ok = avg >= 0.3
        if ok:
            passed += 1
        else:
            failed += 1
            degradations.append({
                "question": question[:80],
                "type": qtype,
                "avg_score": round(avg, 3),
                "faithfulness": round(scores.faithfulness, 3),
                "answer_relevancy": round(scores.answer_relevancy, 3),
                "context_relevancy": round(scores.context_relevancy, 3),
            })

        if from_cache:
            cache_hits += 1
        times.append(elapsed)

        per_question.append({
            "question": question[:100],
            "type": qtype,
            "avg_score": round(avg, 3),
            "elapsed": round(elapsed, 2),
            "from_cache": from_cache,
            "passed": ok,
        })

        print(f"[{i}/{total}] {'✅' if ok else '❌'} avg={avg:.3f} "
              f"({elapsed:.1f}s) {'[cache]' if from_cache else ''} "
              f"{question[:60]}...")

    metrics = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy": passed / max(1, total),
        "avg_time": sum(times) / max(1, len(times)),
        "max_time": max(times) if times else 0,
        "min_time": min(times) if times else 0,
        "cache_hit_rate": cache_hits / max(1, total),
        "avg_faithfulness": sum(r["avg_score"] for r in per_question) / max(1, len(per_question)),
        "degradations": degradations,
        "per_question": per_question,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return metrics


# ======================================================================
# 基线对比
# ======================================================================

def compare_with_baseline(current: Dict, baseline: Dict) -> Dict:
    """对比当前指标与基线，检测退化。

    Args:
        current: 当前评测指标
        baseline: 基线指标

    Returns:
        dict: {"has_degradation": bool, "details": [...]}
    """
    details = []
    has_degradation = False

    checks = [
        ("accuracy", "准确率"),
        ("avg_faithfulness", "平均忠实度"),
        ("cache_hit_rate", "缓存命中率"),
    ]

    for key, label in checks:
        cur_val = current.get(key, 0)
        base_val = baseline.get(key, 1)  # 基线不存在时默认 1（不触发退化）
        if base_val > 0:
            change = (cur_val - base_val) / base_val
            if change < -DEGRADATION_THRESHOLD:
                has_degradation = True
                details.append(
                    f"⚠️ {label} 退化: {base_val:.3f} → {cur_val:.3f} "
                    f"({change*100:+.1f}%, 超过阈值 {DEGRADATION_THRESHOLD*100:.0f}%)"
                )
            else:
                details.append(
                    f"✅ {label}: {base_val:.3f} → {cur_val:.3f} "
                    f"({change*100:+.1f}%)"
                )

    # 时间检查（增加超过 20% 告警）
    cur_time = current.get("avg_time", 0)
    base_time = baseline.get("avg_time", cur_time + 0.01)
    time_change = (cur_time - base_time) / max(0.01, base_time)
    if time_change > 0.20:
        has_degradation = True
        details.append(f"⚠️ 平均耗时增加: {base_time:.1f}s → {cur_time:.1f}s (+{time_change*100:.0f}%)")
    else:
        details.append(f"✅ 平均耗时: {base_time:.1f}s → {cur_time:.1f}s ({time_change*100:+.0f}%)")

    return {"has_degradation": has_degradation, "details": details}


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AgenticRAG 回归测试")
    parser.add_argument("--limit", type=int, default=0, help="限制题数（0=全部）")
    parser.add_argument("--baseline-only", action="store_true", help="仅存储基线，不对比")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print(f"=" * 55)
    print(f"AgenticRAG v0.5 回归测试")
    print(f"测试集: {GOLDEN_PATH}")
    print(f"精排: {'开' if config.reranker_enabled else '关'}  "
          f"缓存: {'开' if config.cache_enabled else '关'}")
    print(f"=" * 55)

    metrics = run_regression(limit=args.limit)
    if not metrics:
        sys.exit(1)

    if args.baseline_only:
        save_baseline(metrics)
        print(f"\n基线已存储。成功: {metrics['passed']}/{metrics['total']}, "
              f"平均分: {metrics['avg_faithfulness']:.3f}")
        sys.exit(0)

    baseline = load_baseline()
    if baseline:
        comparison = compare_with_baseline(metrics, baseline)
        print(f"\n--- 基线对比 ---")
        for d in comparison["details"]:
            print(d)

        if comparison["has_degradation"]:
            print(f"\n❌ 检测到退化，CI 失败")
            sys.exit(1)
        else:
            print(f"\n✅ 所有指标正常，无退化")
    else:
        print(f"\n⚠️ 基线不存在，自动创建。下次运行将进行对比。")
        save_baseline(metrics)

    print(f"\n完成: {metrics['passed']}/{metrics['total']} 通过, "
          f"平均分 {metrics['avg_faithfulness']:.3f}, "
          f"缓存命中率 {metrics['cache_hit_rate']:.1%}")
