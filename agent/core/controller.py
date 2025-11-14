from __future__ import annotations
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Imports from our refactored packages
from rag_pipeline.llm_factory import get_llm
from langchain.prompts import PromptTemplate

# Core components
from .critic import evaluate_step
from .prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    AGENT_TASK_PROMPT,
    QUERY_AUGMENT_PROMPT_STR
)

# All available tools
from ..tools.search_repo import search_repo
from ..tools.inspect_file import inspect_file
from ..tools.rag_retrieve import rag_retrieve
from ..tools.propose_patch import propose_patch
from ..tools.create_pr import generate_pr_payload

# Max steps for the agent loop
MAX_STEPS = 10


@dataclass
class AgentResult:
    """Dataclass for holding the final agent result."""
    question: str
    answer: str
    patch_markdown: str
    search_hits: List[Dict[str, Any]]
    inspected_snippets: List[Dict[str, Any]]  # **ZMIENIONE NA LISTĘ**
    rag_answer: str
    rag_sources: List[Dict[str, Any]]
    critic: Dict[str, Any]
    score: float
    pr: Optional[Dict[str, str]] = None

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__


def _summarize_sources_for_critic(
        search_hits: List[Dict[str, Any]],
        inspected_snippets: List[Dict[str, Any]],  # **ZMIENIONE NA LISTĘ**
        rag_sources: List[Dict[str, Any]],
) -> str:
    """Helper to create a concise summary for the critic."""
    parts: List[str] = []
    if search_hits:
        parts.append("Search hits:")
        for h in search_hits[:5]:
            parts.append(f"- {h['path']}:{h.get('line_no', 1)} → {h['snippet'][:120]}")

    if inspected_snippets:
        parts.append("Inspected File Snippets (Evidence):")
        # Pokaż 2 ostatnie (najważniejsze)
        for s in inspected_snippets[-2:]:
            parts.append(f"--- (from {s['path']}) ---\n{s['snippet'][:300]}\n---")

    if rag_sources:
        parts.append("RAG sources:")
        for s in rag_sources[:5]:
            parts.append(f"- {s['file']} (page={s.get('page')}) → {s['snippet'][:120]}")

    summary = "\n".join(parts)
    return summary if summary else "(No sources or evidence collected by agent)"


def _get_intent(question: str, llm_backend: str, temperature: float) -> str:
    """
    ROUTER: Classifies the user's intent.
    """
    try:
        llm = get_llm(name=llm_backend, temperature=temperature)
        prompt = PromptTemplate.from_template(INTENT_CLASSIFICATION_PROMPT)
        chain = prompt | llm

        resp = chain.invoke({"question": question})
        raw_intent = (getattr(resp, "content", None) or str(resp))

        intent = raw_intent.strip().strip(" \n\t*`'\"").lower()

        valid_intents = ["question", "task_patch", "search"]
        if intent in valid_intents:
            return intent

        print(f"[WARN] Unknown intent: '{intent}' (Raw: '{raw_intent}'). Defaulting to 'question'.")
        return "question"  # Safe fallback

    except Exception as e:
        print(f"[ERROR] Intent classification failed: {e}. Defaulting to 'question'.")
        return "question"


def run_agent_once(
        question: str,
        llm_backend: str = "gemini",
        temperature: float = 0.4,
        create_pr_threshold: float = 0.75,
        repo_hint: Optional[str] = None,
) -> AgentResult:
    """
    MAIN CONTROLLER (ROUTER)
    """

    intent = _get_intent(question, llm_backend, temperature)
    print(f"[INFO] Classified intent: {intent}")

    if intent == "question" or intent == "search":
        return _handle_question_intent(
            question=question,
            llm_backend=llm_backend,
            temperature=temperature
        )
    elif intent == "task_patch":
        return _handle_task_intent_react(
            question=question,
            llm_backend=llm_backend,
            temperature=temperature,
            create_pr_threshold=create_pr_threshold,
            repo_hint=repo_hint
        )
    else:  # intent == "unknown"
        return AgentResult(
            question=question,
            answer="I'm sorry, I'm not sure how to handle that request. I am designed to answer questions about code or perform refactoring tasks.",
            patch_markdown="", search_hits=[], inspected_snippets=[],
            rag_answer="", rag_sources=[], critic={}, score=0.0, pr=None
        )


def _handle_question_intent(
        question: str, llm_backend: str, temperature: float
) -> AgentResult:
    """
    Handler for simple "question" intents. Uses RAG only.
    """
    print("[INFO] Handling 'question' intent with RAG.")

    rag_res = rag_retrieve(
        question=question, k=5, llm_backend=llm_backend,
        temperature=temperature, use_augmentation=True
    )
    rag_answer = rag_res["answer"]
    rag_sources = rag_res["sources"]
    effective_query = rag_res.get("effective_query", question)

    final_answer = (
        f"**Effective retrieval query:** `{effective_query}`\n\n"
        f"#### Reasoning based on repository context\n\n"
        f"{rag_answer}\n"
    )

    src_summary = _summarize_sources_for_critic([], [], rag_sources)
    try:
        critic = evaluate_step(
            question=question, answer=final_answer,
            sources_summary=src_summary, llm_backend=llm_backend
        )
        score = (critic.get("grounding", 0) + critic.get("usefulness", 0) + critic.get("reflection", 0)) / 3.0
    except Exception as e:
        critic = {"comments": f"Critic failed: {e}", "grounding": 0.5, "usefulness": 0.0, "reflection": 0.0}
        score = 0.16  # Penalize failure

    return AgentResult(
        question=question, answer=final_answer, patch_markdown="",
        search_hits=[], inspected_snippets=[], rag_answer=rag_answer,
        rag_sources=rag_sources, critic=critic, score=score, pr=None
    )


# -------------------------------------------------------------------
# ReAct Agent Loop
# -------------------------------------------------------------------

def _robust_tool_parser(action_str: str) -> (str, Dict):
    """
    Parses 'tool_name("arg1", key="arg2")' into (tool_name, kwargs)
    Handles multiline strings in arguments.
    """
    action_str = action_str.strip()

    # 1. Extract tool name
    match = re.match(r'(\w+)\s*\((.*)\)\s*$', action_str, re.DOTALL)
    if not match:
        raise ValueError(f"Invalid action format. Expected tool_name(...). Got: {action_str}")

    tool_name = match.group(1).strip()
    args_str = match.group(2).strip()

    kwargs = {}
    pos_args = []

    if not args_str: return tool_name, kwargs

    # 2. Split arguments by comma, but respect quotes AND brackets
    # This is a simplified parser; it will fail on nested brackets/quotes.
    args = re.split(
        r',(?=(?:[^"]*"[^"]*")*[^"]*$)(?=(?:[^\']*"[^\']*\')*[^\']*$)(?=(?:[^\[\]]*\[[^\[\]]*\])*[^\[\]]*$)', args_str)

    for arg in args:
        arg = arg.strip()
        if not arg: continue

        # 3. Check for keyword arguments (key=value)
        kv_match = re.match(r'(\w+)\s*=\s*(.*)', arg, re.DOTALL)
        if kv_match:
            key = kv_match.group(1).strip()
            value = kv_match.group(2).strip()
            # De-quote simple strings
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            kwargs[key] = value
        else:
            # 4. Otherwise, it's a positional argument
            pos_args.append(arg.strip('"\' '))

    # 5. Map positional args
    if tool_name == 'inspect_file':
        if pos_args and 'relative_path' not in kwargs:
            kwargs['relative_path'] = pos_args.pop(0)
        if pos_args and 'center_line' not in kwargs:
            kwargs['center_line'] = pos_args.pop(0)
    elif tool_name == 'search_repo':
        if pos_args and 'query' not in kwargs:
            kwargs['query'] = pos_args[0]
    elif tool_name == 'rag_retrieve':
        if pos_args and 'question' not in kwargs:
            kwargs['question'] = pos_args[0]

    # --- !! POPRAWKA PARSERA propose_patch I finish !! ---
    # Te narzędzia mają specjalne, wieloliniowe argumenty

    elif tool_name == 'propose_patch':
        # Regex to find issue_description="...", evidence_snippets=[...]
        issue_match = re.search(r'issue_description\s*=\s*(["\'])(.*?)\1', args_str, re.DOTALL)
        evidence_match = re.search(r'evidence_snippets\s*=\s*(\[.*?\])', args_str, re.DOTALL)

        if issue_match:
            kwargs['issue_description'] = issue_match.group(2)
        elif pos_args:
            kwargs['issue_description'] = pos_args.pop(0)

        if evidence_match:
            # Safely evaluate the list string
            try:
                # Use json.loads for robust parsing of list-of-strings
                snippet_list_str = evidence_match.group(1).replace('"""', '"')
                kwargs['evidence_snippets'] = json.loads(snippet_list_str)
            except json.JSONDecodeError:
                # Fallback for simple case like ["snippet"]
                simple_match = re.search(r'\[\s*(["\'])(.*?)\1\s*\]', evidence_match.group(1), re.DOTALL)
                if simple_match:
                    kwargs['evidence_snippets'] = [simple_match.group(2)]
                else:
                    kwargs['evidence_snippets'] = []  # Failed to parse
        elif pos_args:
            kwargs['evidence_snippets'] = pos_args  # Assume remaining args are snippets

    elif tool_name == 'finish':
        # Regex to find reasoning_summary="...", patch_markdown="..."
        reason_match = re.search(r'reasoning_summary\s*=\s*(["\'])(.*?)\1', args_str, re.DOTALL)
        patch_match = re.search(r'patch_markdown\s*=\s*(["\'])(.*?)\1', args_str, re.DOTALL)

        if reason_match:
            kwargs['reasoning_summary'] = reason_match.group(2)
        elif pos_args:
            kwargs['reasoning_summary'] = pos_args.pop(0)  # Fallback

        if patch_match:
            kwargs['patch_markdown'] = patch_match.group(2)
        elif pos_args:
            kwargs['patch_markdown'] = pos_args.pop(0)  # Fallback
    # --- Koniec Poprawki ---

    # 6. Convert types
    # --- !! POPRAWKA: Przeniesione do bloku tool_name == 'inspect_file' !! ---
    if tool_name == 'inspect_file':
        if 'center_line' in kwargs and kwargs['center_line'] is not None and kwargs['center_line'] != 'None':
            kwargs['center_line'] = int(kwargs['center_line'])
        elif 'center_line' in kwargs:
            kwargs['center_line'] = None
        if 'window' in kwargs and kwargs['window'] is not None:
            kwargs['window'] = int(kwargs['window'])

    return tool_name, kwargs


def _parse_and_execute_tool(action_str: str, llm_backend: str, temperature: float) -> (bool, str, Any):
    """
    Parses and executes a tool.
    Returns (is_finished, tool_output_string, raw_data_or_none)
    """
    action_str = action_str.strip().replace("`", "")
    print(f"  [Action]  {action_str}")

    try:
        tool_name, kwargs = _robust_tool_parser(action_str)

        if tool_name == "finish":
            return True, "Observation: finish() called.", kwargs

        elif tool_name == "search_repo":
            if 'query' not in kwargs: raise ValueError("requires 'query'")
            result = search_repo(**kwargs)
            return False, f"Observation: {json.dumps(result)}", result  # Return raw list

        elif tool_name == "inspect_file":
            if 'relative_path' not in kwargs: raise ValueError("requires 'relative_path'")
            result = inspect_file(**kwargs)
            return False, f"Observation: (Content of {kwargs['relative_path']})\n{result['snippet']}", result  # Return raw dict

        elif tool_name == "rag_retrieve":
            if 'question' not in kwargs: raise ValueError("requires 'question'")
            result = rag_retrieve(
                llm_backend=llm_backend, temperature=temperature, **kwargs
            )
            return False, f"Observation: {result['answer']} (Sources: {json.dumps(result['sources'])})", result  # Return raw dict

        elif tool_name == "propose_patch":
            if 'issue_description' not in kwargs: raise ValueError("requires 'issue_description'")

            evidence_list = [{"snippet": ev} for ev in kwargs.get("evidence_snippets", [])]

            result_patch = propose_patch(
                issue_description=kwargs['issue_description'],
                evidence=evidence_list,
                llm_backend=llm_backend,
                temperature=temperature
            )
            return False, f"Observation: {result_patch}", result_patch  # Return raw string

        else:
            return False, f"Observation: Unknown tool '{tool_name}'.", None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Observation: Tool Error: {e}. Check tool name and arguments.", None


def _handle_task_intent_react(
        question: str,
        llm_backend: str,
        temperature: float,
        create_pr_threshold: float,
        repo_hint: Optional[str]
) -> AgentResult:
    """
    MAIN AGENT LOOP (ReAct)
    """
    print(f"[INFO] Handling 'task_patch' intent with ReAct loop.")

    llm = get_llm(name=llm_backend, temperature=temperature)
    history: List[str] = []  # Stores the Thought/Action/Observation history
    prompt_template = PromptTemplate.from_template(AGENT_TASK_PROMPT)

    # --- !! ZBIERAMY DOWODY DLA KRYTYKA !! ---
    all_search_hits = []
    all_inspected_snippets = []
    all_rag_sources = []
    # --- Koniec Zbierania ---

    for i in range(MAX_STEPS):
        print(f"[INFO] ReAct Step {i + 1}/{MAX_STEPS}")

        current_prompt = prompt_template.invoke({
            "history": "\n".join(history),
            "question": question
        })

        resp = llm.invoke(current_prompt)
        full_response = (getattr(resp, "content", None) or str(resp)).strip()

        try:
            # --- !! NOWY, ODPORNY PARSER MYŚLI/AKCJI !! ---
            # Znajdź 'Thought:', a potem 'Action:', które może być wieloliniowe
            thought_match = re.search(r"Thought: (.*?)(?=Action:|$)", full_response, re.DOTALL)
            action_match = re.search(r"Action: (.*)", full_response, re.DOTALL)

            if not thought_match or not action_match:
                raise ValueError("Missing Thought or Action")

            thought = thought_match.group(1).strip()
            # Cały blok akcji (może być wieloliniowy)
            action = action_match.group(1).strip()
            # --- Koniec Poprawki ---

            history.append(f"Thought: {thought}\nAction: {action}")
            print(f"  [Thought] {thought}")
        except Exception:
            history.append(f"Observation: Invalid response format. {full_response}")
            print(f"  [Observe] Invalid response format. Trying again.")
            continue

            # Przechwytujemy 'raw_data'
        is_finished, result_str, raw_data = _parse_and_execute_tool(action, llm_backend, temperature)

        # Kolekcjonuj dowody
        if raw_data is not None:  # **POPRAWKA: Sprawdzaj czy nie jest None**
            if action.startswith("search_repo"):
                all_search_hits.extend(raw_data)  # raw_data to lista
            elif action.startswith("inspect_file"):
                all_inspected_snippets.append(raw_data)  # raw_data to dict
            elif action.startswith("rag_retrieve"):
                all_rag_sources.extend(raw_data.get("sources", []))

        if is_finished:
            print("[INFO] ReAct loop finished.")
            final_data = raw_data  # raw_data from finish() is the dict
            break
        else:
            history.append(result_str)  # Add Observation to history
            print(f"  [Observe] {result_str[:300]}...")
    else:
        print("[WARN] ReAct loop reached max steps.")
        final_data = {
            "reasoning": "Agent reached maximum steps without calling finish().",
            "patch": ""
        }

    # Assemble Final Answer
    patch_markdown = final_data.get("patch", "")
    reasoning_summary = final_data.get("reasoning", "No summary provided.")

    final_answer = (
        f"#### Agent Reasoning\n"
        f"{reasoning_summary}\n\n"
        "#### Proposed patch (dry-run)\n"
        f"{patch_markdown if patch_markdown else '(No patch proposed)'}\n"
    )

    # --- !! PODAJEMY DOWODY DO KRYTYKA !! ---
    src_summary = _summarize_sources_for_critic(
        all_search_hits,
        all_inspected_snippets,
        all_rag_sources
    )
    print("\n" + "-" * 30 + " CRITIC SUMMARY " + "-" * 30)
    print(src_summary)
    print("-" * 80)
    # --- Koniec Podawania ---

    try:
        critic = evaluate_step(
            question=question,
            answer=final_answer,
            sources_summary=src_summary,
            llm_backend=llm_backend,
            temperature=0.5,
        )
        score = (critic.get("grounding", 0) + critic.get("usefulness", 0) + critic.get("reflection", 0)) / 3.0
    except Exception as e:
        critic = {"comments": f"Critic failed: {e}", "grounding": 0.5, "usefulness": 0.0, "reflection": 0.0}
        score = 0.16

    # Optional PR
    pr_payload: Optional[Dict[str, str]] = None
    if patch_markdown and score >= create_pr_threshold:
        pr_payload = generate_pr_payload(
            question=question,
            patch_markdown=patch_markdown,
            critic=critic,
            repo_hint=repo_hint,
        )

    return AgentResult(
        question=question,
        answer=final_answer,
        patch_markdown=patch_markdown,
        search_hits=all_search_hits,
        inspected_snippets=all_inspected_snippets,  # **ZMIENIONE NA LISTĘ**
        rag_answer=reasoning_summary,
        rag_sources=all_rag_sources,
        critic=critic,
        score=score,
        pr=pr_payload,
    )