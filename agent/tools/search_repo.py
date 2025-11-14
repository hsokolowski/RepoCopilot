from pathlib import Path
from typing import List, Dict
import os

_env_root = os.getenv("REPO_ROOT")
if _env_root:
    ROOT_DIR = Path(_env_root).resolve()
else:
    ROOT_DIR = Path(__file__).resolve().parents[2]

IGNORE_DIRS = {
    ".git", ".idea", ".venv", "__pycache__",
    "chroma_db", "node_modules", "bin", "obj", "eval_reports"
}

INCLUDE_EXT = {
    ".cs", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".html", ".json", ".md", ".txt", ".sh", ".yaml", ".yml"
}


def search_repo(query: str, max_results: int = 20) -> List[Dict]:
    """
    Simple text search in the repository.

    This tool NOW searches BOTH file paths and file contents.
    """
    q = query.lower()
    results: List[Dict] = []

    # --- NEW: Search by file path first ---
    for path in ROOT_DIR.rglob(f"*{q}*"):
        if not path.is_file():
            continue

        # Check ignores
        if any(part in IGNORE_DIRS for part in path.parts):
            continue

        rel_path_str = str(path.relative_to(ROOT_DIR)).replace("\\", "/")
        results.append({
            "path": rel_path_str,
            "line_no": 1,
            "snippet": "(File path matched the query)"
        })
        if len(results) >= max_results:
            return results
    # --- End of new search ---

    # --- Original content search ---
    for path in ROOT_DIR.rglob("*"):
        if not path.is_file():
            continue

        if any(part in IGNORE_DIRS for part in path.parts):
            continue

        if path.suffix.lower() not in INCLUDE_EXT:
            continue

        # Avoid double-adding files we already found
        if any(r['path'] == str(path.relative_to(ROOT_DIR)).replace("\\", "/") for r in results):
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for i, line in enumerate(text.splitlines(), start=1):
            if q in line.lower():
                snippet = line.strip()
                if len(snippet) > 200:
                    snippet = snippet[:200] + "…"

                results.append(
                    {
                        "path": str(path.relative_to(ROOT_DIR)).replace("\\", "/"),
                        "line_no": i,
                        "snippet": snippet,
                    }
                )
                if len(results) >= max_results:
                    return results
                break  # Only add first match per file

    return results