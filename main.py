"""
main.py -- Command-line entry point for the Agentic RAG system.

Usage:
    python main.py "What was Veracier's R&D spending in 2022?"
    python main.py --file questions.txt
    python main.py --interactive
    python main.py --interactive --verbose

Output (to stdout):
    A JSON object with keys:
        - answer: str              — final answer
        - reasoning_trace: list    — step-by-step decision log
        - retrieved_sources: list  — chunk_ids used in answer
        - intent: str              — classified intent
        - final_state: dict        — complete final RAGState (summary)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from src.core.config import config
from src.agents.workflow import run_rag, get_graph
from src.core.models import RAGState
from src.retrieval.retriever import is_loaded, get_chunk_count

logger = logging.getLogger(__name__)


# =============================================================================
# Output formatting
# =============================================================================

def format_output(result: Dict[str, Any], pretty: bool = True) -> str:
    """Format the result dict as JSON.

    Args:
        result: Dict from run_rag().
        pretty: If True, indent the JSON for readability.

    Returns:
        JSON string.
    """
    indent = 2 if pretty else None
    return json.dumps(result, ensure_ascii=False, indent=indent, default=str)


# =============================================================================
# Interactive REPL
# =============================================================================

def run_interactive() -> None:
    """Run an interactive question-answering REPL.

    Commands:
        /help   — show help
        /quit   — exit
        /status — show index status
        any other input is treated as a question
    """
    print("=" * 60)
    print("  Agentic RAG System — Veracier Industries Knowledge Base")
    print("=" * 60)
    print(f"  Index: {'[OK] loaded' if is_loaded() else '[WARN] not loaded (stub mode)'}")
    if is_loaded():
        print(f"  Chunks: {get_chunk_count()}")
    print()
    print("  Type your question, or /help for commands, /quit to exit.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye.")
                break
            elif cmd in ("/help", "/h", "/?"):
                print("Commands:")
                print("  /help    — show this help")
                print("  /quit    — exit the REPL")
                print("  /status  — show index and config status")
                print("  /graph   — print the LangGraph topology summary")
                print("  Any other text is treated as a question.")
                continue
            elif cmd == "/status":
                print(f"  Index loaded: {is_loaded()}")
                print(f"  Chunks: {get_chunk_count()}")
                print(f"  LLM: {config.llm_provider}:{config.llm_model}")
                print(f"  Embedding: {config.embedding_model_name}")
                print(f"  Top-K: {config.top_k}, Max retries: {config.max_retries}")
                continue
            elif cmd == "/graph":
                graph = get_graph()
                print(f"  Graph compiled: {graph is not None}")
                print(f"  Nodes: classify_intent, ask_clarification, plan_sub_questions, "
                       "retrieve, check_sufficiency, refine_query, generate_answer")
                print("  Topology: START→classify_intent→[intent]→...→generate_answer→END")
                continue
            else:
                print(f"  Unknown command: {user_input}. Use /help for available commands.")
                continue

        # Process question
        print(f"\n  Processing: \"{user_input}\"...")
        try:
            result = run_rag(user_input)
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            print(f"\n  [ERR] Error: {e}")
            continue

        # Display result
        print(f"\n  Intent: {result['intent']}")
        print(f"  Answer: {result['answer']}")
        if result['retrieved_sources']:
            print(f"  Sources ({len(result['retrieved_sources'])}): "
                  f"{', '.join(result['retrieved_sources'][:10])}")
        print(f"  Trace ({len(result['reasoning_trace'])} steps):")
        for entry in result['reasoning_trace']:
            print(f"    • {entry}")

        # Print full JSON (can be redirected)
        print(f"\n  Full JSON output:\n{format_output(result)}")


# =============================================================================
# Batch processing from file
# =============================================================================

def run_batch(file_path: str) -> None:
    """Process multiple questions from a text file (one per line).

    Args:
        file_path: Path to a text file with one question per line.
            Blank lines and lines starting with '#' are skipped.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                questions.append(line)

    if not questions:
        print(f"No questions found in {file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(questions)} questions from {file_path}...")
    print("=" * 60)

    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q[:80]}...")
        try:
            result = run_rag(q)
            results.append({"question": q, **result})
            print(f"  → {result['answer'][:100]}...")
        except Exception as e:
            logger.error(f"Error on question {i}: {e}", exc_info=True)
            results.append({"question": q, "error": str(e)})
            print(f"  → [ERR] Error: {e}")

    # Write all results as JSON Lines
    output_path = path.with_suffix(".results.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Results written to: {output_path}")
    print(f"Success: {sum(1 for r in results if 'error' not in r)}/{len(results)}")


# =============================================================================
# CLI entry
# =============================================================================

def main() -> None:
    """Parse command-line arguments and dispatch to the appropriate mode."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG System — Veracier Industries Knowledge Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py "What was the R&D budget in 2022?"
    python main.py --file questions.txt
    python main.py --interactive
    python main.py -v "What is the carbon neutrality target?"
        """,
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Natural language question to answer",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a text file with one question per line (batch mode)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive question-answering REPL",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="Output compact JSON (no indentation)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Write JSON output to file instead of stdout",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else getattr(logging, config.log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Dispatch
    if args.interactive:
        run_interactive()
    elif args.file:
        run_batch(args.file)
    elif args.question:
        # Single question mode
        result = run_rag(args.question)
        output = format_output(result, pretty=not args.no_pretty)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Output written to: {args.output}")
        else:
            print(output)
    else:
        # No arguments: print help and start interactive
        parser.print_help()
        print()
        print("No question provided. Starting interactive mode...")
        print()
        run_interactive()


if __name__ == "__main__":
    main()
