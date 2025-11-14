from __future__ import annotations

from pathlib import Path
import csv
import json
import datetime
from typing import List, Dict, Any
import yaml
import sys

# --- !! WAŻNA ZMIANA !! ---
# Add the project root (two levels up from 'eval') to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# --- Koniec zmiany ---

try:
    from agent.core.controller import run_agent_once
except ImportError:
    print(f"[ERROR] Could not import 'agent.core.controller'.")
    print(f"Make sure you are running from the project root, or check PYTHONPATH.")
    sys.exit(1)


BASE_DIR = project_root
EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "dataset.csv"
RUBRIC_PATH = EVAL_DIR / "rubric.yaml"
REPORT_DIR = BASE_DIR / "eval_reports"
REPORT_DIR.mkdir(exist_ok=True)


def load_dataset() -> List[Dict[str, str]]:
    """Loads evaluation questions from the CSV."""
    rows: List[Dict[str, str]] = []
    with DATASET_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_rubric() -> Dict[str, Any]:
    """Loads evaluation metrics from the YAML."""
    with RUBRIC_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_eval(llm_backend: str = "gemini", temperature: float = 0.4) -> Dict[str, Any]:
    """
    Runs the agent for each question from dataset.csv,
    uses the critic for evaluation (grounding/usefulness/reflection),
    and saves a JSON report in eval_reports/.
    """
    dataset = load_dataset()
    rubric = load_rubric()

    metric_weights = rubric.get("metrics", {})
    total_weight = sum(m.get("weight", 0.0) for m in metric_weights.values()) or 1.0
    pass_threshold = float(rubric.get("overall_pass_threshold", 0.7))

    runs: List[Dict[str, Any]] = []

    for row in dataset:
        qid = row.get("id") or f"q{len(runs) + 1}"
        question = row["question"]

        print(f"[EVAL] Running agent for {qid}: {question}")

        try:
            # Call the agent
            result = run_agent_once(
                question=question,
                llm_backend=llm_backend,
                temperature=temperature,
                create_pr_threshold=1.1, # Set high to prevent PR generation during eval
            )
            critic = result.critic

            # Get scores from the agent's self-evaluation
            grounding = float(critic.get("grounding", 0.0))
            usefulness = float(critic.get("usefulness", 0.0))
            reflection = float(critic.get("reflection", 0.0))

            # Calculate weighted score
            overall = (
                grounding * metric_weights.get("grounding", {}).get("weight", 0.0)
                + usefulness * metric_weights.get("usefulness", {}).get("weight", 0.0)
                + reflection * metric_weights.get("reflection", {}).get("weight", 0.0)
            ) / total_weight

            run_item = {
                "id": qid,
                "question": question,
                "grounding": grounding,
                "usefulness": usefulness,
                "reflection": reflection,
                "overall": overall,
                "pass": overall >= pass_threshold,
                "critic_comments": critic.get("comments", ""),
                "full_answer": result.answer, # Save the answer for review
            }

        except Exception as e:
            # e.g., missing API key / LLM problem
            run_item = {
                "id": qid,
                "question": question,
                "error": str(e),
                "pass": False,
            }

        runs.append(run_item)

    # Aggregation
    total = len(runs)
    passed = sum(1 for r in runs if r.get("pass"))
    overall_values = [r.get("overall", 0.0) for r in runs if "overall" in r]
    avg_overall = sum(overall_values) / max(1, len(overall_values))

    summary = {
        "total_questions": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "avg_overall": avg_overall,
        "pass_threshold": pass_threshold,
    }

    report = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "llm_backend": llm_backend,
        "temperature": temperature,
        "rubric": rubric,
        "summary": summary,
        "runs": runs,
    }

    out_path = REPORT_DIR / f"eval_report_{llm_backend}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[EVAL] Saved report to {out_path}")
    print(f"[EVAL] Summary: passed {passed}/{total} "
          f"({summary['pass_rate']*100:.1f}%), avg_overall={avg_overall:.2f}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run offline evaluation for Repo Copilot agent."
    )
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
        help="Generation temperature (default: 0.4).",
    )
    args = parser.parse_args()

    run_eval(llm_backend=args.llm, temperature=args.temperature)