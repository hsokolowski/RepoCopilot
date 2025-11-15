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
from collections import OrderedDict

# ----------------------------------------

def _automerge_by_file(
    docs: List[Document],
    scores: List[float],
    max_chars_per_file: int = 6000,
    max_files: int = 4,
) -> List[Document]:
    """
    Łączy chanki z tego samego pliku w większe bloki tekstu.
    Dzięki temu LLM widzi spójny kontekst z README / docs / kodu,
    zamiast porwanych kawałków.

    - Grupuje po metadata["file"] (np. README.md, agent/core/controller.py).
    - Dla każdego pliku skleja tekst aż do max_chars_per_file.
    - Zwraca najwyżej max_files plików, posortowanych po najlepszym score.
    """
    grouped: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    for doc, score in zip(docs, scores):
        meta = doc.metadata or {}
        file_id = meta.get("file") or meta.get("source") or "unknown"

        if file_id not in grouped:
            # Pierwszy raz widzimy ten plik – inicjalizujemy entry
            grouped[file_id] = {
                "score": float(score),
                "meta": meta.copy(),
                "text": "",
            }

        text = grouped[file_id]["text"]
        chunk = (doc.page_content or "").strip()
        if not chunk:
            continue

        if text:
            candidate = text + "\n\n---\n\n" + chunk
        else:
            candidate = chunk

        # Nie przekraczaj limitu znaków na plik
        if len(candidate) <= max_chars_per_file:
            grouped[file_id]["text"] = candidate
        # Jeśli byśmy przekroczyli, po prostu pomijamy ten dodatkowy chunk

    # Budujemy listę Document-ów, sortując po najlepszym score dla pliku
    merged_docs: List[Document] = []
    for file_id, info in sorted(
        grouped.items(),
        key=lambda kv: kv[1]["score"],
        reverse=True,
    ):
        if not info["text"]:
            continue
        merged_docs.append(
            Document(
                page_content=info["text"],
                metadata=info["meta"],
            )
        )
        if len(merged_docs) >= max_files:
            break

    return merged_docs

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
    dirty_docs: List[Document] = retriever.invoke(effective_query)

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

    # 5. Automerge po plikach – łączymy chanki z tego samego pliku
    all_docs_sorted: List[Document] = [doc for doc, _ in sorted_docs]
    all_scores_sorted: List[float] = [float(score) for _, score in sorted_docs]

    merged_docs: List[Document] = _automerge_by_file(
        all_docs_sorted,
        all_scores_sorted,
        max_chars_per_file=6000,
        max_files=k,
    )

    # Fallback: gdyby coś poszło nie tak, użyj po prostu top-k chunków
    if not merged_docs:
        merged_docs = [doc for doc, _ in sorted_docs[:k]]

    # --- ETAP 4: SYNTEZA (Generate) ---
    # Używamy tylko "czystych" dokumentów jako kontekstu
    context = "\n\n---\n\n".join([doc.page_content for doc in merged_docs])

    # Tworzymy prosty łańcuch do syntezy odpowiedzi
    synthesis_chain = LLMChain(llm=llm, prompt=RAG_PROMPT)

    # Wywołujemy LLM z naszym czystym kontekstem
    res = synthesis_chain.invoke({"context": context, "question": effective_query})
    answer: str = res.get("text", "Error synthesizing answer.")

    # --- ETAP 5: Formatowanie źródeł (bez zmian) ---
    sources = []
    for d in merged_docs:  # Używamy 'clean_docs'
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