from pathlib import Path
from typing import List, Dict
import os
import re

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

    This tool searches BOTH file paths and file contents.
    It jest teraz bardziej wyrozumiały dla zapytań typu
    "scripts/build_index.py main" itd.
    """
    terms = _candidate_terms(query)
    results: List[Dict] = []
    seen_paths = set()

    # --- 1) Search by file path first (po kolei po termach) ---
    for term in terms:
        if not term:
            continue

        for path in ROOT_DIR.rglob(f"*{term}*"):
            if not path.is_file():
                continue

            # Check ignores
            if any(part in IGNORE_DIRS for part in path.parts):
                continue

            rel_path_str = str(path.relative_to(ROOT_DIR)).replace("\\", "/")
            if rel_path_str in seen_paths:
                continue

            results.append({
                "path": rel_path_str,
                "line_no": 1,
                "snippet": "(File path matched the query)"
            })
            seen_paths.add(rel_path_str)

            if len(results) >= max_results:
                return results

        # jeżeli coś znaleźliśmy dla tego termu – nie musimy iść dalej po ścieżkach
        if results:
            break

    # --- 2) Content search (jeśli nadal jest miejsce w results) ---
    for term in terms:
        if not term:
            continue

        for path in ROOT_DIR.rglob("*"):
            if not path.is_file():
                continue

            if any(part in IGNORE_DIRS for part in path.parts):
                continue

            if path.suffix.lower() not in INCLUDE_EXT:
                continue

            rel_path_str = str(path.relative_to(ROOT_DIR)).replace("\\", "/")
            # unikamy duplikatów
            if rel_path_str in seen_paths:
                continue

            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for i, line in enumerate(text.splitlines(), start=1):
                if term in line.lower():
                    snippet = line.strip()
                    if len(snippet) > 200:
                        snippet = snippet[:200] + "…"

                    results.append(
                        {
                            "path": rel_path_str,
                            "line_no": i,
                            "snippet": snippet,
                        }
                    )
                    seen_paths.add(rel_path_str)

                    if len(results) >= max_results:
                        return results

                    # tylko pierwszy match na plik
                    break

        if results and len(results) >= max_results:
            break

    return results

def _candidate_terms(raw: str) -> List[str]:
    q = raw.lower().strip()
    tokens = [t for t in re.split(r"[\s,]+", q) if t]

    # "scripts/build_index.py main"
    # ["scripts/build_index.py main", "scripts/build_index.py", "main"]
    candidates: List[str] = [q]
    candidates.extend(sorted(tokens, key=lambda t: (".py" not in t, len(t) * -1)))
    seen = set()
    uniq: List[str] = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq