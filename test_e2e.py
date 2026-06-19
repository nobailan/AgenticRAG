"""
test_e2e.py -- End-to-end integration tests for the Agentic RAG system.

Uses 5 simple questions from ANSWER_KEY.json (or built-in fallbacks) to
validate the complete pipeline end-to-end:
    - classify_intent → retrieve → check_sufficiency → generate_answer

Each test verifies:
    1. Final answer is non-empty and > 10 characters
    2. Answer does NOT contain placeholder/stub words
    3. At least 1 chunk was retrieved (retrieved_sources non-empty)
    4. Reasoning trace contains at least 3 decision steps

Requirements from v0_1_0.txt P0-5:
    - Use ANSWER_KEY.json or 5 manual simple questions
    - Call run_workflow / graph.invoke
    - Output "5/5 passed" or specific failure reason
    - Answer doesn't need to match ground truth, just be relevant

Usage:
    python test_e2e.py                # Run all 5 e2e tests
    python test_e2e.py --verbose      # Show detailed per-question output
    python test_e2e.py --limit 2      # Run only first 2 tests
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import config
from src.agents.workflow import run_rag, get_graph
from src.core.models import RAGState
from src.retrieval.retriever import is_loaded, get_chunk_count

logger = logging.getLogger(__name__)


# =============================================================================
# Question selection
# =============================================================================

# Built-in fallback questions (used if ANSWER_KEY.json is unavailable)
FALLBACK_QUESTIONS: List[str] = [
    "What is Veracier Industries and where is it headquartered?",
    "What are the main business segments of Veracier Industries?",
    "What was the R&D spending of Veracier in the most recent fiscal year?",
    "Which department is responsible for the carbon neutrality initiative?",
    "What are the key risk factors mentioned in Veracier's annual report?",
]


def load_answer_key() -> Dict:
    """Load ANSWER_KEY.json from the configured path.

    Returns:
        Parsed JSON dict, or empty dict if the file is not found.
    """
    path = config.answer_key_path
    if not path.exists():
        logger.warning(f"ANSWER_KEY.json not found at {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_questions(n: int = 5) -> List[Tuple[str, str]]:
    """Select N simple questions for end-to-end testing.

    Prefers questions from ANSWER_KEY.json tagged without 'cross_entity'
    or 'temporal_reasoning' (simpler, single-hop questions).

    Falls back to FALLBACK_QUESTIONS if ANSWER_KEY.json is unavailable.

    Args:
        n: Maximum number of questions to select (default 5).

    Returns:
        List of (question_id, question_text) tuples.
    """
    answer_key = load_answer_key()

    if answer_key:
        # Select simple questions (no cross_entity or temporal_reasoning)
        candidates = []
        for qid, qdata in answer_key.items():
            factors = qdata.get("difficulty_factors", [])
            if "cross_entity" not in factors and "temporal_reasoning" not in factors:
                candidates.append((qid, qdata["question"]))
        logger.info(f"Found {len(candidates)} simple questions in ANSWER_KEY.json")
        return candidates[:n]

    # Fallback to built-in questions
    logger.warning("Using built-in fallback questions (ANSWER_KEY.json unavailable)")
    return [(f"FALLBACK-{i + 1:02d}", q) for i, q in enumerate(FALLBACK_QUESTIONS[:n])]


# =============================================================================
# Test runner
# =============================================================================

def run_e2e_test(
    question_id: str,
    question: str,
    require_sources: bool = True,
) -> Tuple[bool, str, Dict]:
    """Run a single end-to-end test for one question.

    Args:
        question_id: Identifier for the question (e.g., "CEO-01").
        question: The natural language question text.
        require_sources: If True, require at least 1 retrieved chunk.
            Set to False when indexes are unavailable.

    Returns:
        Tuple of (passed: bool, message: str, result: dict).
        The result dict contains the full run_rag() output for inspection.
    """
    try:
        result = run_rag(question)
    except Exception as e:
        logger.error(f"E2E test [{question_id}] crashed: {e}", exc_info=True)
        return False, f"CRASH: {e}", {}

    answer = result.get("answer", "")
    sources = result.get("retrieved_sources", [])
    trace = result.get("reasoning_trace", [])
    intent = result.get("intent", "unknown")

    # ---- Check 1: Answer is non-empty and has reasonable length ----
    if not answer:
        return False, "FAIL: answer is empty", result
    if len(answer) < 10:
        return False, f"FAIL: answer too short ({len(answer)} chars): '{answer[:50]}...'", result

    # ---- Check 2: No placeholder or stub language ----
    lower_answer = answer.lower()
    placeholder_words = ["placeholder", "占位", "[stub]", "stub answer", "simulated result"]
    for word in placeholder_words:
        if word in lower_answer:
            return False, f"FAIL: answer contains placeholder word '{word}'", result

    # ---- Check 3: Retrieved sources (if required) ----
    if require_sources and not sources:
        return False, "FAIL: no retrieved sources (empty retrieved_sources)", result

    # ---- Check 4: Reasoning trace has sufficient detail ----
    if len(trace) < 3:
        return False, f"FAIL: reasoning_trace has only {len(trace)} entries (need ≥3)", result

    # ---- All checks passed ----
    msg = (
        f"PASS | intent={intent} | answer={len(answer)} chars | "
        f"sources={len(sources)} | trace={len(trace)} steps"
    )
    return True, msg, result


def print_detailed_result(
    question_id: str,
    question: str,
    passed: bool,
    message: str,
    result: Dict,
) -> None:
    """Print a detailed result for one e2e test question.

    Args:
        question_id: Question identifier.
        question: The question text.
        passed: Whether the test passed.
        message: Test result message.
        result: The full result dict from run_rag().
    """
    status = "[OK] PASS" if passed else "[FAIL] FAIL"
    print(f"\n{status} | {question_id}")
    print(f"  Q: {question[:100]}")
    print(f"  {message}")
    if result:
        answer = result.get("answer", "")
        print(f"  Answer preview: {answer[:150]}...")
        sources = result.get("retrieved_sources", [])
        if sources:
            print(f"  Sources: {', '.join(sources[:10])}")
        trace = result.get("reasoning_trace", [])
        if trace:
            print(f"  Trace ({len(trace)} steps):")
            for t in trace:
                print(f"    • {t[:120]}")


# =============================================================================
# Main test runner
# =============================================================================

def main() -> None:
    """Run all end-to-end tests and print a summary."""
    import argparse
    parser = argparse.ArgumentParser(
        description="End-to-end integration tests for the Agentic RAG system"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed per-question results",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=5,
        help="Number of questions to test (default: 5)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    print("=" * 70)
    print("  Agentic RAG — End-to-End Integration Tests (v0.1)")
    print("=" * 70)

    # Check index status
    indexes_available = is_loaded()
    if indexes_available:
        chunk_count = get_chunk_count()
        print(f"  Indexes: [OK] loaded ({chunk_count} chunks)")
    else:
        print(f"  Indexes: [WARN] not loaded — source retrieval check will be SKIPPED")
        print(f"  Run 'python data_prepare.py --limit 200' first for full e2e testing.")
    print(f"  LLM: {config.llm_provider}:{config.llm_model}")
    print("-" * 70)

    # Select questions
    questions = select_questions(n=args.limit)
    if not questions:
        print("ERROR: No questions available for testing.")
        sys.exit(1)

    print(f"  Testing {len(questions)} question(s)...")
    print("-" * 70)

    # Run tests
    results: List[Tuple[str, str, bool, str, Dict]] = []
    passed_count = 0

    for qid, qtext in questions:
        if args.verbose:
            print(f"\n  Running [{qid}]...", end=" ", flush=True)

        passed, message, result = run_e2e_test(
            question_id=qid,
            question=qtext,
            require_sources=indexes_available,  # Only require sources if indexes loaded
        )

        if passed:
            passed_count += 1

        results.append((qid, qtext, passed, message, result))

        if args.verbose:
            status = "[OK]" if passed else "[FAIL]"
            print(f"{status} {message}")

        # Always show detailed output for failures
        if not passed:
            print_detailed_result(qid, qtext, passed, message, result)

    # ---- Summary ----
    print()
    print("=" * 70)
    print(f"  E2E Test Results: {passed_count}/{len(questions)} passed")
    print("=" * 70)

    for qid, qtext, passed, message, _ in results:
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"  {status} | {qid}: {qtext[:80]}...")
        if not passed:
            print(f"         | {message}")

    print("-" * 70)

    if passed_count == len(questions):
        print(f"  {passed_count}/{len(questions)} 通过")
    else:
        print(f"  {len(questions) - passed_count} test(s) failed.")
        for qid, _, passed, message, _ in results:
            if not passed:
                print(f"    [{qid}] {message}")

    print()

    # Exit code
    if passed_count == len(questions):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
