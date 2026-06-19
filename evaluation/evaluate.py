"""
evaluate.py -- RAGAS evaluation runner.

Loads testset.json, runs each question through the RAG pipeline,
and computes RAGAS metrics. Generates a comparison report.

Usage:
    python evaluation/evaluate.py                    # Run all tests
    python evaluation/evaluate.py --limit 10        # First 10 only
    python evaluation/evaluate.py --output report.md  # Custom output path
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import config
from evaluation.ragas_metrics import (
    compute_ragas_metrics,
    compute_ragas_metrics_simple,
    RAGASScore,
)

logger = logging.getLogger(__name__)

# Default testset path
TESTSET_PATH = Path(__file__).parent / "testset.json"
REPORT_PATH = Path(__file__).parent / "comparison_report.md"


def load_testset(path: Path = TESTSET_PATH) -> List[Dict]:
    """Load evaluation test set from JSON file."""
    if not path.exists():
        logger.error("Testset not found: %s", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d test cases from %s", len(data), path.name)
    return data


def run_single_evaluation(test: Dict) -> Dict:
    """Run the RAG pipeline on one test case and compute metrics.

    Args:
        test: Dict with 'question', 'ground_truth', 'contexts'.

    Returns:
        Dict with question, answer, contexts, ground_truth, scores, timing.
    """
    from src.agents.workflow import run_rag

    question = test["question"]
    ground_truth = test.get("ground_truth", "")
    expected_contexts = test.get("contexts", [])

    start = time.time()
    try:
        result = run_rag(question)
        answer = result.get("answer", "")
        # Get retrieved contexts (text of chunks)
        retrieved_contexts = []
        if "retrieved_sources" in result:
            # result only has chunk_ids; the full chunks are in final_state
            pass
        if "final_state" in result and result["final_state"]:
            pass
    except Exception as e:
        logger.error("Pipeline failed for '%s': %s", question[:60], e)
        answer = f"[ERROR] {e}"
        result = {}

    elapsed = time.time() - start

    # Compute metrics (use simple version if ragas is unavailable)
    try:
        scores = compute_ragas_metrics(question, answer, expected_contexts, ground_truth)
    except Exception:
        scores = compute_ragas_metrics_simple(question, answer, expected_contexts, ground_truth)

    return {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "contexts": expected_contexts,
        "scores": scores.to_dict(),
        "elapsed_sec": round(elapsed, 2),
        "intent": result.get("intent", "unknown"),
        "from_cache": result.get("from_cache", False),
    }


def run_evaluation(
    limit: int = 0, output_path: Path = REPORT_PATH
) -> List[Dict]:
    """Run full evaluation on the test set.

    Args:
        limit: If > 0, only evaluate the first N test cases.
        output_path: Path for the generated comparison report.

    Returns:
        List of per-question result dicts.
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

        # Quick per-result summary
        s = r["scores"]
        print(f"  Faith={s['faithfulness']:.3f}  "
              f"AnswerR={s['answer_relevancy']:.3f}  "
              f"CtxR={s['context_relevancy']:.3f}  "
              f"({r['elapsed_sec']}s)")

    # Aggregate scores
    agg = aggregate_scores(results)
    print(f"\n{'='*50}")
    print(f"Average Scores ({len(results)} tests):")
    print(f"  Faithfulness:       {agg['faithfulness']:.3f}")
    print(f"  Answer Relevancy:   {agg['answer_relevancy']:.3f}")
    print(f"  Context Relevancy:  {agg['context_relevancy']:.3f}")
    print(f"  Overall Average:    {agg['average']:.3f}")
    print(f"  Avg Time:           {agg['avg_time']:.1f}s")

    # Generate report
    generate_report(results, agg, output_path)

    return results


def aggregate_scores(results: List[Dict]) -> Dict[str, float]:
    """Compute average scores across all results."""
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
    """Generate a Markdown comparison report."""
    lines = [
        "# AgenticRAG v0.4 — RAGAS Evaluation Report",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Tests**: {len(results)}",
        f"**Config**: reranker={'on' if config.reranker_enabled else 'off'}, "
        f"cache={'on' if config.cache_enabled else 'off'}",
        "",
        "## Aggregate Scores",
        "",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Faithfulness | {agg['faithfulness']:.3f} |",
        f"| Answer Relevancy | {agg['answer_relevancy']:.3f} |",
        f"| Context Relevancy | {agg['context_relevancy']:.3f} |",
        f"| **Overall Average** | **{agg['average']:.3f}** |",
        f"| Avg Response Time | {agg['avg_time']:.1f}s |",
        "",
        "## Per-Question Results",
        "",
    ]

    for i, r in enumerate(results, 1):
        s = r["scores"]
        lines.append(f"### {i}. {r['question'][:100]}")
        lines.append(f"- **Answer**: {r['answer'][:200]}...")
        lines.append(f"- **Faithfulness**: {s['faithfulness']:.3f}")
        lines.append(f"- **Answer Relevancy**: {s['answer_relevancy']:.3f}")
        lines.append(f"- **Context Relevancy**: {s['context_relevancy']:.3f}")
        lines.append(f"- **Time**: {r['elapsed_sec']}s"
                     f"{' (cached)' if r.get('from_cache') else ''}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAGAS Evaluation Runner")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N tests (0 = all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Report output path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output = Path(args.output) if args.output else REPORT_PATH
    run_evaluation(limit=args.limit, output_path=output)
