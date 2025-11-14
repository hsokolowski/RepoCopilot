from typing import List, Dict, Any
from langchain.schema import Document

# Import logic from our new, clean package
from rag_pipeline.llm_factory import get_llm
from rag_pipeline.vectorstore import build_vectorstore, create_rag_qa_chain

# Import agent prompts for augmentation
from agent.core.prompts import QUERY_AUGMENT_PROMPT_STR
from langchain.prompts import PromptTemplate


def _augment_query(question: str, llm) -> str:
    """Rewrites the user's query to be better for RAG."""
    try:
        aug_prompt = PromptTemplate.from_template(QUERY_AUGMENT_PROMPT_STR)
        aug_chain = aug_prompt | llm
        resp = aug_chain.invoke({"question": question})
        return (getattr(resp, "content", None) or str(resp)).strip()
    except Exception as e:
        print(f"[WARN] Query augmentation failed: {e}. Using original query.")
        return question


def rag_retrieve(
        question: str,
        k: int = 5,
        llm_backend: str = "gemini",
        temperature: float = 0.5,
        rebuild_index: bool = False,
        use_augmentation: bool = False,
) -> Dict[str, Any]:
    """
    Agent Tool: Runs a RAG query against the local vector store.
    """
    llm = get_llm(name=llm_backend, temperature=temperature)
    vs = build_vectorstore(rebuild=rebuild_index)

    effective_query = question
    if use_augmentation:
        effective_query = _augment_query(question, llm)

    qa = create_rag_qa_chain(llm=llm, vectorstore=vs)

    res = qa.invoke({"query": effective_query})
    answer: str = res["result"]
    docs: List[Document] = res.get("source_documents", []) or []

    sources = []
    for d in docs[:k]:
        meta = d.metadata or {}
        file_ = meta.get("file") or meta.get("source") or "doc"
        page = meta.get("page", None)
        source_type = meta.get("source", "doc")  # For the critic

        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 260:
            snippet = snippet[:260] + "…"

        sources.append(
            {
                "file": file_,
                "page": page,
                "snippet": snippet,
                "source": source_type,
            }
        )

    return {
        "answer": answer,
        "sources": sources,
        "effective_query": effective_query,
    }