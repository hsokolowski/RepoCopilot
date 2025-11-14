import os
import shutil
from pathlib import Path
import ctypes.util

from dotenv import load_dotenv

os.environ["PATH"] += os.pathsep + str(Path(os.getcwd()))

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = "chroma_db"
DEFAULT_EMBED = "intfloat/multilingual-e5-base"   # nowy default
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")


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


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def info(msg: str):
    print(f"[INFO] {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg not found. Install via `winget install ffmpeg` or from https://ffmpeg.org/download.html"
        )


# -------------------------------------------------------------------
# Transcription
# -------------------------------------------------------------------

def transcribe_audio(file_path: Path) -> str:
    import whisper
    transcript_path = file_path.with_suffix(file_path.suffix + ".txt")
    if transcript_path.exists():
        return transcript_path.read_text(encoding="utf-8")

    model = whisper.load_model("base")
    result = model.transcribe(str(file_path))
    text = result["text"].strip()
    transcript_path.write_text(text, encoding="utf-8")
    return text


# -------------------------------------------------------------------
# Load documents
# -------------------------------------------------------------------

SUPPORTED_CODE_FILES = {".cs", ".py", ".js", ".ts", ".java", ".html", ".json"}


def load_documents() -> list[Document]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {DATA_DIR.resolve()}")

    docs: list[Document] = []

    for f in sorted(DATA_DIR.iterdir()):
        suf = f.suffix.lower()

        # PDFs
        if suf == ".pdf":
            loader = PyMuPDFLoader(str(f))
            docs.extend(loader.load())

        # Audio
        elif suf in {".mp3", ".wav", ".m4a", ".mp4"}:
            text = transcribe_audio(f)
            docs.append(Document(page_content=text, metadata={"file": f.name, "source": "audio",}))

        # Text & markdown
        elif suf in {".txt", ".md"}:
            docs.append(Document(page_content=f.read_text(encoding="utf-8"), metadata={"file": f.name, "source": "doc"}))

        # Code files
        elif suf in SUPPORTED_CODE_FILES:
            content = f.read_text(encoding="utf-8", errors="ignore")
            docs.append(Document(page_content=content, metadata={"file": f.name, "source": "code_snippet"}))

        else:
            continue

    return docs

def load_repo_code(root: Path) -> list[Document]:
    code_docs = []

    DEFAULT_EXCLUDED_DIRS = {
        ".venv", "venv", "env",
        "__pycache__",
        ".git",
        "chroma_db",
        "site-packages",
    }

    extra = os.getenv("EXCLUDE_DIRS", "")
    ENV_EXCLUDED = {x.strip() for x in extra.split(",") if x.strip()}

    EXCLUDED = DEFAULT_EXCLUDED_DIRS | ENV_EXCLUDED

    for f in root.rglob("*"):
        if any(part in EXCLUDED for part in f.parts):
            continue

        if f.is_file() and f.suffix.lower() in SUPPORTED_CODE_FILES:
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            rel = str(f.relative_to(root))
            #print(f"[repo_code] {rel}")

            code_docs.append(Document(
                page_content=content,
                metadata={"file": rel, "source": "repo_code"}
            ))

    return code_docs

# -------------------------------------------------------------------
# Vector store
# -------------------------------------------------------------------

def build_vectorstore(embed_model: str = DEFAULT_EMBED, rebuild: bool = False) -> Chroma:
    if rebuild and Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR)

    if Path(PERSIST_DIR).exists():
        embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    docs = load_documents()

    repo_root = os.getenv("REPO_ROOT")
    print(f"[INFO] Repo root: {repo_root}")
    if repo_root:
        code_docs = load_repo_code(Path(repo_root))
        docs.extend(code_docs)
    print(f"[INFO] after loading repo code: {len(docs)}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vs = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    return vs


# -------------------------------------------------------------------
# LLM wrapper
# -------------------------------------------------------------------

def get_llm(name: str = "gemini", temperature: float = 0.5):
    name = name.lower()

    if name == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3:8b", temperature=temperature)

    # Gemini
    ENV_PATH = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Provide GOOGLE_API_KEY or GEMINI_API_KEY in .env")

    from langchain_google_genai import ChatGoogleGenerativeAI
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature
    )


# -------------------------------------------------------------------
# RetrievalQA (wrapped for tools)
# -------------------------------------------------------------------

def create_rag_qa(llm, vectorstore) -> RetrievalQA:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

from typing import List, Dict, Any

from langchain.schema import Document

from rag_pipeline import build_vectorstore, get_llm, create_rag_qa


def rag_retrieve(
    question: str,
    k: int = 5,
    llm_backend: str = "gemini",
    temperature: float = 0.5,
    rebuild_index: bool = False,
) -> Dict[str, Any]:
    """
    Run a RAG query over the local vector store.

    Returns:
        {
          "answer": str,
          "sources": List[Dict[file, page, snippet]]
        }
    """
    llm = get_llm(name=llm_backend, temperature=temperature)
    vs = build_vectorstore(rebuild=rebuild_index)

    qa = create_rag_qa(llm=llm, vectorstore=vs)

    res = qa.invoke({"query": question})
    answer: str = res["result"]
    docs: List[Document] = res.get("source_documents", []) or []

    sources = []
    for d in docs[:k]:
        meta = d.metadata or {}
        file_ = meta.get("file") or meta.get("source") or "doc"
        page = meta.get("page", None)
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 260:
            snippet = snippet[:260] + "…"
        sources.append(
            {
                "file": file_,
                "page": page,
                "snippet": snippet,
            }
        )

    return {
        "answer": answer,
        "sources": sources,
    }