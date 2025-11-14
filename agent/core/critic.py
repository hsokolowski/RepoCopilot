import json
import re
from typing import Dict, Any, Optional

# Import from the new refactored package
from rag_pipeline.llm_factory import get_llm
from .prompts import CRITIC_SYSTEM_PROMPT


def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Tries to safely parse JSON returned by the LLM.
    Handles: ```json ... ``` fences and extra text.
    """
    if not text:
        return None

    raw = text.strip()

    # 1) Remove markdown fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = lines[1:] if lines else []
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    # 2) Find the first { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    candidate = match.group(0) if match else raw

    # 3) Try json.loads
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def evaluate_step(
    question: str,
    answer: str,
    sources_summary: str,
    llm_backend: str = "gemini",
    temperature: float = 0.5,
) -> Dict[str, Any]:
    """
    Uses an LLM as a critic to evaluate the quality of the agent's response.
    """
    llm = get_llm(name=llm_backend, temperature=temperature)

    prompt = (
        CRITIC_SYSTEM_PROMPT
        + "\n\nUser question or goal:\n"
        + question
        + "\n\nAgent answer (including any proposed patch):\n"
        + answer
        + "\n\nSources / evidence summary:\n"
        + (sources_summary or "(none)")
        + "\n\nNow return ONLY a valid JSON object with the fields "
          "`grounding`, `usefulness`, `reflection`, `comments`."
          " Do NOT wrap it in markdown fences. Do NOT add any extra text."
    )

    resp = llm.invoke(prompt)
    raw = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)

    data = _safe_parse_json(raw)
    if not data:
        # fallback with raw response preview
        return {
            "grounding": 0.5,
            "usefulness": 0.0, # Be harsh on parse failures
            "reflection": 0.0,
            "comments": f"Failed to parse critic JSON. Raw response: {raw[:300]}",
        }

    # ensure all fields are present
    return {
        "grounding": float(data.get("grounding", 0.5)),
        "usefulness": float(data.get("usefulness", 0.5)),
        "reflection": float(data.get("reflection", 0.5)),
        "comments": str(data.get("comments", "")),
    }