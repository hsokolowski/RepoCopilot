from typing import List, Dict, Any
from langchain.schema import Document
import ast  # Użyjemy do bezpiecznego parsowania

# Importy komponentów RAG
from rag_pipeline.llm_factory import get_llm
from rag_pipeline.vectorstore import build_vectorstore
from rag_pipeline.config import RAG_PROMPT

# Importy do augmentacji i syntezy
from agent.core.prompts import QUERY_AUGMENT_PROMPT_STR
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# --- !! NOWY IMPORT DLA RERANKERA !! ---
from sentence_transformers.cross_encoder import CrossEncoder


# ----------------------------------------

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
        k: int = 4,  # k to teraz liczba DOKŁADNYCH wyników po rerankingu
        llm_backend: str = "gemini",
        temperature: float = 0.5,
        rebuild_index: bool = False,
        use_augmentation: bool = False,
) -> Dict[str, Any]:
    """
    Agent Tool: Runs a RAG query against the local vector store,
    now with a Reranking step for improved accuracy.
    """
    llm = get_llm(name=llm_backend, temperature=temperature)
    vs = build_vectorstore(rebuild=rebuild_index)

    # --- ETAP 1: AUGMENTACJA (bez zmian) ---
    effective_query = question
    if use_augmentation:
        effective_query = _augment_query(question, llm)
        print(f"[INFO] RAG Effective Query: {effective_query}")

    # --- ETAP 2: POBRANIE (Retrieve) ---
    # Pobieramy więcej (np. 15) "brudnych" dokumentów z bazy wektorowej
    k_retrieval = 15
    retriever = vs.as_retriever(search_kwargs={"k": k_retrieval})
    dirty_docs: List[Document] = retriever.get_relevant_documents(effective_query)

    if not dirty_docs:
        return {
            "answer": "I found no relevant documents in the repository context.",
            "sources": [],
            "effective_query": effective_query,
        }

    # --- !! ETAP 3: RERANKING !! ---
    print(f"[INFO] Reranking {len(dirty_docs)} documents...")

    # 1. Inicjalizujemy model Rerankera (pobierze się automatycznie za pierwszym razem)
    reranker = CrossEncoder('BAAI/bge-reranker-base')

    # 2. Tworzymy pary (zapytanie, treść_dokumentu) do oceny
    pairs = [(effective_query, doc.page_content) for doc in dirty_docs]

    # 3. Model ocenia wszystkie pary na raz
    scores = reranker.predict(pairs)

    # 4. Łączymy dokumenty z ich nowymi wynikami i sortujemy
    doc_with_scores = list(zip(dirty_docs, scores))
    sorted_docs = sorted(doc_with_scores, key=lambda x: x[1], reverse=True)

    # 5. Bierzemy 'k' (np. 4) najlepszych dokumentów ("czyste" wyniki)
    clean_docs: List[Document] = [doc for doc, score in sorted_docs[:k]]

    # --- ETAP 4: SYNTEZA (Generate) ---
    # Używamy tylko "czystych" dokumentów jako kontekstu
    context = "\n\n---\n\n".join([doc.page_content for doc in clean_docs])

    # Tworzymy prosty łańcuch do syntezy odpowiedzi
    synthesis_chain = LLMChain(llm=llm, prompt=RAG_PROMPT)

    # Wywołujemy LLM z naszym czystym kontekstem
    res = synthesis_chain.invoke({"context": context, "question": effective_query})
    answer: str = res.get("text", "Error synthesizing answer.")

    # --- ETAP 5: Formatowanie źródeł (bez zmian) ---
    sources = []
    for d in clean_docs:  # Używamy 'clean_docs'
        meta = d.metadata or {}
        file_ = meta.get("file") or meta.get("source") or "doc"
        page = meta.get("page", None)
        source_type = meta.get("source", "doc")

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