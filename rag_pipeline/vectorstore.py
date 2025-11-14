import shutil
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema.vectorstore import VectorStore

# Import from this package
from .config import PERSIST_DIR, DEFAULT_EMBED, REPO_ROOT_PATH, RAG_PROMPT
from .loaders import load_data_documents, load_repo_code_documents


def info(msg: str):
    print(f"[INFO] {msg}")


def get_embeddings(embed_model: str = DEFAULT_EMBED):
    """Creates the embedding function."""
    # Specify cache_folder for HuggingFace embeddings
    cache_dir = Path(__file__).resolve().parent.parent / ".hf_cache"
    cache_dir.mkdir(exist_ok=True)

    return HuggingFaceEmbeddings(
        model_name=embed_model,
        cache_folder=str(cache_dir)
    )


def build_vectorstore(rebuild: bool = False) -> VectorStore:
    """Builds or loads the Chroma vector store."""
    persist_path = str(PERSIST_DIR)

    if rebuild and Path(persist_path).exists():
        info(f"Rebuilding index: Deleting old {persist_path}...")
        shutil.rmtree(persist_path)

    embeddings = get_embeddings()

    if Path(persist_path).exists():
        info(f"Loading existing vector store from {persist_path}...")
        return Chroma(
            persist_directory=persist_path,
            embedding_function=embeddings
        )

    info("Building new vector store...")
    # 1. Load docs from /data
    docs = load_data_documents()

    # 2. Load code from REPO_ROOT
    if REPO_ROOT_PATH and REPO_ROOT_PATH.exists():
        info(f"Loading code from {REPO_ROOT_PATH}...")
        code_docs = load_repo_code_documents(REPO_ROOT_PATH)
        docs.extend(code_docs)
    else:
        info(f"REPO_ROOT_PATH not set or invalid. Skipping code indexing.")

    if not docs:
        raise ValueError("No documents found to build vector store. Check /data folder and REPO_ROOT.")

    # 3. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    info(f"Split {len(docs)} documents into {len(chunks)} chunks.")

    # 4. Build and persist
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_path
    )
    info(f"Vector store built and persisted to {persist_path}.")
    return vs


def create_rag_qa_chain(llm, vectorstore: VectorStore) -> RetrievalQA:
    """Creates the RetrievalQA chain with our custom prompt."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )