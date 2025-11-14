from pathlib import Path
from typing import Dict, List
import os

_env_root = os.getenv("REPO_ROOT")
if _env_root:
    ROOT_DIR = Path(_env_root).resolve()
else:
    ROOT_DIR = Path(__file__).resolve().parents[2]

# Musimy znać ignorowane foldery, aby wyszukiwanie ich unikało
IGNORE_DIRS = {
    ".git", ".idea", ".venv", "__pycache__",
    "chroma_db", "node_modules", "bin", "obj", "eval_reports"
}


def inspect_file(
        relative_path: str,
        center_line: int | None = None,
        window: int = 15,
) -> Dict:
    """
    Inspects a snippet of a file with line numbers.

    This tool is now 'forgiving': if the exact path isn't found,
    it will search for a unique file matching the name.
    """

    # --- NOWA, ODPORNA LOGIKA WYSZUKIWANIA ---

    file_path = (ROOT_DIR / relative_path).resolve()

    if not file_path.exists():
        print(f"[WARN] File not found at exact path: {file_path}. Searching for alternative...")

        # We search for any file ending with this name
        # (e.g., agent asks for "build_index.py", we find "scripts/build_index.py")

        # os.path.basename handles both / and \ separators
        base_name = os.path.basename(relative_path)

        # Use rglob to find all possible matches
        matches = list(ROOT_DIR.rglob(f"**/{base_name}"))

        # Filter out any matches in ignored directories
        filtered_matches = []
        for m in matches:
            if not any(part in IGNORE_DIRS for part in m.parts):
                filtered_matches.append(m)

        if len(filtered_matches) == 1:
            file_path = filtered_matches[0]
            # Update relative_path to the *correct* one we found
            relative_path = str(file_path.relative_to(ROOT_DIR)).replace("\\", "/")
            print(f"[INFO] Found unique match. Using: {file_path}")

        elif len(filtered_matches) > 1:
            paths = [str(m.relative_to(ROOT_DIR)).replace("\\", "/") for m in filtered_matches]
            raise FileNotFoundError(f"Ambiguous file path '{relative_path}'. Found multiple matches: {paths}")

        else:
            # Original error
            raise FileNotFoundError(f"File not found: {relative_path} (and no alternatives were found)")

    # --- KONIEC NOWEJ LOGIKI ---

    # (Reszta funkcji jest taka sama)
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    if center_line is None:
        start = 0
        end = min(len(lines), window * 2)
    else:
        # 1-based indexing for line numbers
        start = max(center_line - window - 1, 0)
        end = min(center_line + window, len(lines))

    snippet_lines: List[str] = []
    for i in range(start, end):
        ln = i + 1  # line numbers are 1-based
        snippet_lines.append(f"{ln:4} | {lines[i]}")

    return {
        "path": str(relative_path),
        "start_line": start + 1,
        "end_line": end,
        "snippet": "\n".join(snippet_lines),
    }