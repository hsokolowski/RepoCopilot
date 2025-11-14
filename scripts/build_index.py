import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the path so we can import 'rag_pipeline'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    # Import from the new package
    from rag_pipeline.vectorstore import build_vectorstore
except ImportError:
    print("[ERROR] Could not import 'rag_pipeline'. Make sure it is in your PYTHONPATH.")
    sys.exit(1)


def main():
    """
    Builds or rebuilds the vector index from the data/ directory and repo code.
    """
    print("[INFO] Building vector index...")

    # Go two levels up (scripts/ -> root) to find .env
    env_path = project_root / ".env"
    if not env_path.exists():
        print(f"[WARN] .env file not found at {env_path}. REPO_ROOT might not be set.")

    load_dotenv(dotenv_path=env_path, override=True)

    try:
        # Call the refactored function
        build_vectorstore(rebuild=True)
        print("[INFO] Vector index build complete.")
    except Exception as e:
        print(f"[ERROR] Failed to build vector store: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()