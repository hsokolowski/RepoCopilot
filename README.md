# Repo Copilot – AI-powered repository assistant

Repo Copilot is a local AI agent that analyzes a code repository together with its documentation and proposes **safe, dry-run improvements**:

- explains what parts of the code do,
- suggests small refactorings and comment changes,
- proposes Markdown patches (without touching files),
- self-evaluates its own answers (grounding/usefulness/reflection),
- optionally generates a short Pull Request description.

The project is implemented as the **Engineers Capstone Project** for the **Ciklum AI Academy** and focuses on modern agentic patterns: **Router, ReAct (Reason-Act) Loop, Tool Calling, Self-Reflection, and Grounded Evaluation.**

---

## 1. Features

- **RAG over docs + code**
  - Indexes PDF slides/notes and optional audio/text files from `./data`.
  - Indexes source files from a local repository (via `REPO_ROOT`).
  - Uses multilingual sentence embeddings (`intfloat/multilingual-e5-base`).

- **Query augmentation**
  - Rewrites the user question into a better retrieval query (LLM-based).

- **Agentic Core (Router + ReAct)**
  - **Router:** Classifies user intent (`question` vs. `task_patch`) to use the correct workflow.
  - **ReAct Loop:** For tasks, the agent *thinks*, *acts* by calling a tool, and *observes* the result, repeating until the task is done.

- **Robust Tooling**
  - `search_repo`: "Forgiving" search that finds files by path *and* content.
  - `inspect_file`: "Forgiving" tool that finds files even if the agent provides a partial path.
  - `propose_patch`: Generates Markdown-only patches, with "guard rails" to prevent hallucination.

- **Self-reflection & Grounded Evaluation**
  - A separate critic LLM (`evaluate_step`) scores the *final answer* against the *evidence* (search hits, inspected code) collected during the ReAct loop.
  - This "Grounded Evaluation" prevents the critic from hallucinating and agreeing with a bad answer.

- **PR suggestion**
  - If the score is above a threshold, the agent generates a PR payload.

---

## 2. High-level architecture

A detailed diagram is available in [`architecture.mmd`](architecture.mmd).

At a high level, the `controller` acts as the agent's brain:
1.  **Router:** It first classifies the user's intent (`question` or `task_patch`).
2.  **Path A (`question`):** It uses a simple RAG handler (`rag_retrieve`) to answer.
3.  **Path B (`task_patch`):** It starts a **ReAct Loop**.
4.  **ReAct Loop:** The agent `Thinks`, `Acts` (by calling tools like `search_repo` or `inspect_file`), and `Observes` the results, repeating until it can `propose_patch` and `finish`.
5.  **Critic:** The final result is scored by a "grounded" critic that sees all the evidence the agent collected.

---

## 3. Requirements

- **Python**: 3.11+
- **ffmpeg** (if you want audio transcription via Whisper)
  - Windows: `winget install ffmpeg`
  - Or download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **LLM backend**
  - Google Gemini (via `langchain-google-genai`) – default
  - OR local Ollama (`llama3:8b`) – optional

Python deps are listed in `requirements.txt`.

---

## 4. Setup

### 4.1. Clone & create virtualenv

```bash
git clone https://github.com/hsokolowski/RepoCopilot.git RepoCopilot
cd RepoCopilot

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
````

### 4.2. Configure `.env`

Create a `.env` file in the project root, for example:

```dotenv
# LLM setup (Gemini cloud)
GOOGLE_API_KEY=your_gemini_or_google_api_key_here
GEMINI_MODEL=gemini-1.5-flash

# Repository to analyze (absolute path)
REPO_ROOT=G:\PycharmProjects\RepoCopilot

# Optional: tweak RAG parameters here later if needed
```

-----

## 5\. Preparing data (RAG index)

Repo Copilot uses a vector index over files from `./data` and source files from `REPO_ROOT`.

### 5.1. Put documents into `data/`

Create a `data/` folder and copy materials there (PDFs, .txt, .md, .mp3, etc.).

### 5.2. Build / rebuild the index

```bash
 python scripts/build_index.py
```

This script calls the `rag_pipeline.vectorstore.build_vectorstore` function, which:

1.  Loads all documents from `data/`.
2.  Loads all code from `REPO_ROOT`.
3.  Splits, embeds, and persists them into `./chroma_db`.

-----

## 6\. Running the agent

```bash
# Run a specific analysis (task)
python -m agent.cli --llm gemini analyze "Refactor the main function in scripts/build_index.py"

# Ask a question (RAG)
python -m agent.cli --llm gemini analyze "What is this project's architecture?"
```

The CLI passes the query to the main controller.

-----

## 7\. What happens in one agent run? (New Architecture)

The agent is no longer a static 6-step pipeline. It now operates as a true agent using a **Router + ReAct (Reason-Act) loop**, orchestrated by `agent/core/controller.py`.

**1. Intent Classification (Router)**

  - The controller first classifies the user's intent using an LLM (`INTENT_CLASSIFICATION_PROMPT`).
  - It decides if the user has a simple `question` (e.g., "what does this do?") or a complex `task_patch` (e.g., "refactor this file").

**2. Path A: `question` Intent**

  - If it's a `question`, the controller routes to a simple RAG-only handler (`_handle_question_intent`).
  - This handler calls `rag_retrieve`, gets an answer from the vector store, formats it, and finishes.
  - **No patch is ever proposed.**

**3. Path B: `task_patch` Intent (The ReAct Loop)**

  - If it's a `task_patch`, the controller starts the `_handle_task_intent_react` loop.

  - The agent (driven by `AGENT_TASK_PROMPT`) repeats the following cycle (up to 10 steps):

      - **a. Thought:** The LLM analyzes the goal and the history.
        *(e.g., "I need to find the file `build_index.py`.")*

      - **b. Action:** The LLM decides which *one* tool to call.
        *(e.g., `Action: search_repo(query="build_index.py")`)*

      - **c. Observation:** The `controller.py` parses the action, executes the tool, and feeds the result back to the agent.
        *(e.g., `Observation: [{"path": "scripts/build_index.py", ...}]`)*

      - **d. Repeat:** The loop continues.
        *(e.g., `Thought: OK, I found the file. Now I must inspect its full content.` -\> `Action: inspect_file(relative_path="scripts/build_index.py", window=999)`)*

      - **e. Finish:** When the agent has all the evidence, it calls `propose_patch` to get a patch, and then calls the special `finish(...)` tool to exit the loop.

**4. Grounded Evaluation (Critic)**

  - After the agent finishes (from either path), the controller collects all `Observations` (search hits, inspected code) into a `sources_summary`.
  - It sends this summary, along with the final answer, to the `evaluate_step` (Critic).
  - The Critic scores the answer *against the evidence*, preventing it from rewarding hallucinations.

-----

## 8\. Project context (Ciklum AI Academy)

This project was built as the final assignment in the Ciklum AI Academy – Engineering Track.

The goal is to demonstrate:

  * Practical use of RAG for real developer workflows.
  * Agentic orchestration (Router + ReAct multi-tool loop).
  * Self-reflection and grounded evaluation of LLM outputs.
  * Safe, review-first integration with Git repositories.

-----

## 9\. Component Overview (Updated)

**1. CLI (`agent.cli`)**

  - Parses commands and calls the controller.
  - Prints the final `AgentResult` to the console.

**2. Controller (`agent/core/controller.py`)**

  - The **heart and brain** of the system.
  - **Router:** Classifies intent (`_get_intent`).
  - **Orchestrator:** Manages the ReAct loop (`_handle_task_intent_react`) and the RAG handler (`_handle_question_intent`).
  - **Tool Executor:** Parses the agent's actions and calls the correct tool (`_parse_and_execute_tool`).
  - **Evidence Collector:** Gathers all `raw_data` from tool calls to feed to the Critic.

**3. Prompts (`agent/core/prompts.py`)**

  - The "personality" and "rules" of the agent.
  - `INTENT_CLASSIFICATION_PROMPT`: The brain for the Router.
  - `AGENT_TASK_PROMPT`: The core logic and rules for the ReAct loop.
  - `PATCH_SYSTEM_PROMPT`: The rules for the Patcher tool, with anti-hallucination guards.
  - `CRITIC_SYSTEM_PROMPT`: The rules for the Critic, forcing "grounded evaluation".

**4. RAG Pipeline (`rag_pipeline/` package)**

  - A refactored package for all data ingestion and retrieval.
  - `config.py`: Centralized paths and prompts.
  - `llm_factory.py`: Creates all LLM instances (`get_llm`).
  - `loaders.py`: Handles loading all documents (PDF, audio, code).
  - `vectorstore.py`: Manages building and querying the ChromaDB index.

**5. Tools (`agent/tools/*.py`)**

  - `search_repo`: "Forgiving" search for file paths and content.
  - `inspect_file`: "Forgiving" tool to read file content, even with partial paths.
  - `rag_retrieve`: The tool used by the `question` handler.
  - `propose_patch`: The tool used by the `task_patch` agent.
  - `create_pr`: Generates the PR payload text.

**6. Critic (`agent/core/critic.py`)**

  - `evaluate_step`: Receives the final answer *and* a summary of all evidence (`sources_summary`) from the controller.
  - This "grounded" approach makes its scores much more reliable.

**7. Evaluation (`agent/eval/`)**

  - `evaluator.py`: A script to run the agent against a test set (`dataset.csv`).
  - `rubric.yaml`: Defines the metrics and weights for evaluation.
