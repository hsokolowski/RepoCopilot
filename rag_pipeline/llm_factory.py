import os
from pathlib import Path
from dotenv import load_dotenv

# Import config from this package
from .config import BASE_DIR


def get_llm(name: str = "gemini", temperature: float = 0.5):
    """
    Creates and returns an LLM instance.
    """
    name = name.lower()

    if name == "ollama":
        from langchain_ollama import ChatOllama
        # Assumes ollama is running with llama3:8b
        return ChatOllama(model="llama3:8b", temperature=temperature)

    # Default to Gemini
    ENV_PATH = BASE_DIR / ".env"
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(f"Provide GOOGLE_API_KEY or GEMINI_API_KEY in .env (path: {ENV_PATH})")

    from langchain_google_genai import ChatGoogleGenerativeAI
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature
    )