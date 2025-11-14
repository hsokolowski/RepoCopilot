from typing import Dict, List, Optional

# Import from the refactored package
from rag_pipeline.llm_factory import get_llm

# Import the prompt from the central prompt library
from agent.core.prompts import PATCH_SYSTEM_PROMPT


def propose_patch(
        issue_description: str,
        evidence: List[Dict],  # Oczekujemy listy dictów: [{"snippet": "..."}]
        llm_backend: str = "gemini",
        temperature: float = 0.4,
        allowed_paths: Optional[List[str]] = None,
) -> str:
    """
    Generates a 'dry-run' patch in Markdown based on evidence.
    """

    # --- !! NOWY STRAŻNIK (GUARD CLAUSE) !! ---
    # Sprawdź, czy dowody nie są puste
    if not evidence or not any(ev.get("snippet") for ev in evidence):
        return (
            "Error: Evidence is missing or empty. "
            "I cannot propose a patch without a valid code snippet. "
            "Please use `inspect_file` first to get the code."
        )
    # --- Koniec Strażnika ---

    llm = get_llm(name=llm_backend, temperature=temperature)

    evidence_text_parts = []
    for ev in evidence:
        path = ev.get("path", "?")
        snippet = ev.get("snippet") or ev.get("text") or ""

        # Nie dodawaj pustych
        if not snippet:
            continue

        header = f"FILE: {path}"
        evidence_text_parts.append(header + "\n" + snippet)

    if not evidence_text_parts:
        return (
            "Error: Evidence is missing or empty. "
            "I cannot propose a patch without a valid code snippet. "
            "Please use `inspect_file` first to get the code."
        )

    evidence_text = "\n\n---\n\n".join(evidence_text_parts)

    allowed_paths = allowed_paths or []
    if not allowed_paths:
        # Weź ścieżki z dowodów
        allowed_from_evidence = list(set(ev.get("path") for ev in evidence if ev.get("path")))
        if allowed_from_evidence:
            allowed_block = "\n".join(f"- {p}" for p in allowed_from_evidence)
        else:
            allowed_block = "(none – use file paths found in evidence)"
    else:
        allowed_block = "\n".join(f"- {p}" for p in allowed_paths)

    prompt = (
            PATCH_SYSTEM_PROMPT
            + "\n\nIssue description:\n"
            + issue_description
            + "\n\nAllowed files:\n"
            + allowed_block
            + "\n\nEvidence from repo (use this code for 'BEFORE' block):\n"
            + evidence_text
            + "\n\nNow propose a dry-run Markdown patch."
    )

    response = llm.invoke(prompt)
    text = getattr(response, "content", None) or getattr(response, "text", None) or str(response)
    return text.strip()