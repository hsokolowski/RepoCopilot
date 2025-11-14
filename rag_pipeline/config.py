import os
from pathlib import Path
from langchain.prompts import PromptTemplate

# -------------------------------------------------------------------
# Path Configuration
# -------------------------------------------------------------------

# BASE_DIR is the project root (one level up from this package)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = BASE_DIR / "chroma_db"

# Check for REPO_ROOT in .env
_env_root = os.getenv("REPO_ROOT")
if _env_root:
    REPO_ROOT_PATH = Path(_env_root).resolve()
else:
    # Fallback to the project root
    REPO_ROOT_PATH = BASE_DIR

# -------------------------------------------------------------------
# Embedding Config
# -------------------------------------------------------------------
DEFAULT_EMBED = "intfloat/multilingual-e5-base"
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

# -------------------------------------------------------------------
# Code Loading Config
# -------------------------------------------------------------------
SUPPORTED_CODE_FILES = {
    ".cs", ".py", ".js", ".ts", ".java", ".html",
    ".json", ".md", ".txt", ".sh", ".bash", ".yaml", ".yml", ".mmd"
}

DEFAULT_EXCLUDED_DIRS = {
    ".venv", "venv", "env",
    "__pycache__",
    ".git",
    ".idea",
    "chroma_db", # Don't index the index
    "eval_reports",
    "node_modules",
    "bin",
    "obj",
    "site-packages",
}

# -------------------------------------------------------------------
# RAG Prompt
# -------------------------------------------------------------------
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an AI reasoning agent. You answer ONLY from the provided context.\n"
        "If the context is insufficient, say 'I don't know based on the repository context'.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    ),
)