import argparse
import textwrap
import sys
from pathlib import Path

# --- Add the project root to the Python path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
# --- End path addition ---

try:
    from agent.core.controller import run_agent_once
except ImportError:
    print(f"[ERROR] Could not import 'agent.core.controller'.")
    print(f"Make sure you are running from the project root, or check PYTHONPATH.")
    sys.exit(1)


def _print_result(result):
    """Prints the final agent result to the console."""
    print("\n" + "=" * 80)
    print("Repo Copilot – Agent Result")
    print("=" * 80)
    print(f"\nQuestion / goal:\n{result.question}\n")

    print("-" * 80)
    print("\n")
    print(result.answer)
    print("-" * 80)

    print(f"\nCritic scores:")
    print(f"  grounding : {result.critic.get('grounding', 0.0):.2f}")
    print(f"  usefulness: {result.critic.get('usefulness', 0.0):.2f}")
    print(f"  reflection: {result.critic.get('reflection', 0.0):.2f}")
    print(f"  overall   : {result.score:.2f}")
    print(f"  comments  : {result.critic.get('comments', '')}")

    if result.pr:
        print("\n" + "-" * 80)
        print("Generated Pull Request payload:\n")
        print(f"TITLE:\n{result.pr['title']}\n")
        print("BODY:\n")
        print(result.pr["body"])
        print("-" * 80)
    else:
        print("\n[INFO] No PR was generated (score below threshold).")


def main():
    parser = argparse.ArgumentParser(
        prog="repo-copilot",
        description="Repo Copilot – RAG-based agent for analyzing code repositories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python -m agent.cli analyze "Explain what build_vectorstore does."
              python -m agent.cli analyze "Refactor the main script in build_index.py" --llm gemini
              python -m agent.cli demo --llm ollama
            """
        ),
    )

    # Global options
    parser.add_argument(
        "--llm",
        choices=["gemini", "ollama"],
        default="gemini",
        help="Which LLM backend to use (default: gemini).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Generation temperature for the main agent (default: 0.4).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'analyze' command
    p_analyze = subparsers.add_parser(
        "analyze",
        help="Run the agent once for a given question/goal.",
    )
    p_analyze.add_argument(
        "question",
        type=str,
        help="Question or goal for the agent, e.g. 'Explain what build_vectorstore does.'",
    )
    p_analyze.add_argument(
        "--threshold",
        type=float,
        default=0.66,
        help="Score threshold to generate a PR payload (default: 0.66).",
    )
    p_analyze.add_argument(
        "--repo-hint",
        type=str,
        default=None,
        help="Optional textual hint about the target repo, e.g. 'github.com/user/repo'.",
    )

    # 'demo' command
    p_demo = subparsers.add_parser(
        "demo",
        help="Run a built-in demo scenario on this repository.",
    )
    p_demo.add_argument(
        "--threshold",
        type=float,
        default=0.66,
        help="Score threshold to generate a PR payload (default: 0.66).",
    )
    p_demo.add_argument(
        "--repo-hint",
        type=str,
        default=None,
        help="Optional textual hint about the target repo, e.g. 'github.com/user/repo'.",
    )

    args = parser.parse_args()

    # Choose scenario
    if args.command == "analyze":
        question = args.question
        threshold = args.threshold

        repo_hint = args.repo_hint

    elif args.command == "demo":
        question = (
            "Explain what build_vectorstore does in this repository, "
            "and suggest any safe improvements to the RAG setup or configuration."
        )
        threshold = args.threshold
        repo_hint = args.repo_hint

        print("\n[INFO] Running demo scenario with question:\n")
        print(" ", question)
        print("\n[INFO] This may take some time depending on the LLM and index size...\n")

    else:
        parser.error(f"Unknown command: {args.command}")

    # Run agent
    try:
        result = run_agent_once(
            question=question,
            llm_backend=args.llm,
            temperature=args.temperature,
            create_pr_threshold=threshold,
            repo_hint=repo_hint,
        )
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return
    except Exception as e:
        print(f"[ERROR] Agent failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return

    _print_result(result)


if __name__ == "__main__":
    main()