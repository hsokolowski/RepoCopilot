import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document

# Import config from this package
from .config import DATA_DIR, SUPPORTED_CODE_FILES, DEFAULT_EXCLUDED_DIRS


# --- Helpers ---
def info(msg: str):
    print(f"[INFO] {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def ensure_ffmpeg():
    """Check if ffmpeg is installed."""
    if shutil.which("ffmpeg") is None:
        warn("ffmpeg not found. Audio transcription will be skipped.")
        return False
    return True


# --- Transcription ---
def transcribe_audio(file_path: Path) -> str:
    """Transcribes an audio file, caching the result."""
    if not ensure_ffmpeg():
        return f"(Audio file {file_path.name} skipped: ffmpeg not found)"

    try:
        import whisper
    except ImportError:
        warn("Whisper package not found. Skipping audio transcription.")
        return f"(Audio file {file_path.name} skipped: whisper not installed)"

    # Use .txt suffix for the transcript cache
    transcript_path = file_path.with_suffix(file_path.suffix + ".txt")
    if transcript_path.exists():
        info(f"Loading cached transcript for {file_path.name}")
        return transcript_path.read_text(encoding="utf-8")

    info(f"Transcribing {file_path.name} (this may take a moment)...")
    model = whisper.load_model("base")
    result = model.transcribe(str(file_path))
    text = result["text"].strip()
    transcript_path.write_text(text, encoding="utf-8")
    info(f"Finished transcribing {file_path.name}")
    return text


# --- Document Loaders ---
def load_data_documents() -> list[Document]:
    """Loads documents from the /data folder (PDF, Audio, TXT, MD)."""
    if not DATA_DIR.exists():
        warn(f"Missing data folder: {DATA_DIR.resolve()}. Skipping data docs.")
        return []

    docs: list[Document] = []
    info(f"Loading documents from {DATA_DIR}...")

    for f in sorted(DATA_DIR.iterdir()):
        suf = f.suffix.lower()
        try:
            if suf == ".pdf":
                loader = PyMuPDFLoader(str(f))
                docs.extend(loader.load())
            elif suf in {".mp3", ".wav", ".m4a", ".mp4"}:
                text = transcribe_audio(f)
                docs.append(Document(page_content=text, metadata={"file": f.name, "source": "audio"}))
            elif suf in {".txt", ".md"}:
                docs.append(
                    Document(page_content=f.read_text(encoding="utf-8"), metadata={"file": f.name, "source": "doc"}))
        except Exception as e:
            warn(f"Failed to load {f.name}: {e}")

    info(f"Loaded {len(docs)} documents from /data.")
    return docs


def load_repo_code_documents(root: Path) -> list[Document]:
    """Loads code files from the target repository path."""
    info(f"Loading code documents from {root}...")
    code_docs = []

    extra = os.getenv("EXCLUDE_DIRS", "")
    ENV_EXCLUDED = {x.strip() for x in extra.split(",") if x.strip()}
    EXCLUDED = DEFAULT_EXCLUDED_DIRS | ENV_EXCLUDED

    for f in root.rglob("*"):
        # Check if any part of the path is in the excluded set
        if any(part in EXCLUDED for part in f.parts):
            continue

        if f.is_file() and f.suffix.lower() in SUPPORTED_CODE_FILES:
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                # Standardize on forward slashes
                rel_path = str(f.relative_to(root)).replace("\\", "/")
                code_docs.append(Document(
                    page_content=content,
                    metadata={"file": rel_path, "source": "repo_code"}
                ))
            except Exception:
                continue  # Skip files that can't be read

    info(f"Loaded {len(code_docs)} code documents from repo.")
    return code_docs