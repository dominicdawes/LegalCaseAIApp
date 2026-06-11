# agents/exam_questions/graph.py
"""
LangGraph StateGraph for the exam-questions agent.

Graph topology:
  planner → [Send → source_profiler] → concept_synthesizer → issue_clusterer
  issue_clusterer → [Send → retriever] → [Send → question_drafter]
  question_drafter → [Send → answer_key_builder] → [Send → grounder]
  grounder → critic → (should_revise?) → reviser ↺ | final_drafter → assembler

Checkpointing:
  Uses AsyncPostgresSaver (langgraph-checkpoint-postgres) for durable state
  across worker restarts.  Falls back to MemorySaver if the package is not
  installed, preserving the original in-process behaviour.

Run with:
    result = await run_exam_agent(request, project_id, source_ids, n_questions)
    markdown = result["final_output"]
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv

# ——— Logging & Env Load ───────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
load_dotenv()

USE_LANGGRAPH_AGENT = os.getenv("USE_LANGGRAPH_AGENT", "false").lower() == "true"


# ── Checkpointer factory ───────────────────────────────────────────────────────

@asynccontextmanager
async def _checkpointer_ctx():
    """
    Async context manager that yields a configured LangGraph checkpointer.

    Uses POSTGRES_DSN (direct connection, port 5432) — not the pgbouncer
    transaction-mode pool — so prepared statements work without any special flags.

    Falls back to MemorySaver only when setup fails (before yielding).
    Errors that occur AFTER yielding (during graph.ainvoke) are re-raised
    so they propagate normally — a second yield from an asynccontextmanager
    generator raises RuntimeError.
    """
    _setup_ok = False
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        _dsn = (os.getenv("POSTGRES_DSN_SESSION") or "").strip()
        async with AsyncPostgresSaver.from_conn_string(_dsn) as saver:
            await saver.setup()
            _setup_ok = True
            yield saver
            return
    except (ImportError, Exception) as exc:
        if _setup_ok:
            raise  # error came from inside graph.ainvoke — propagate, don't mask
        if not isinstance(exc, ImportError):
            logger.warning(
                "AsyncPostgresSaver failed (%s) — falling back to MemorySaver", exc
            )

    from langgraph.checkpoint.memory import MemorySaver
    yield MemorySaver()


# ── Graph builder ──────────────────────────────────────────────────────────────

def _build_graph(checkpointer):
    """
    Compile the StateGraph with the supplied checkpointer.
    Separated from the async runner so the topology is easy to read.
    """
    from langgraph.graph import StateGraph, END

    from .state import AgentState
    from .nodes import (
        planner,
        planner_to_profiler,
        source_profiler,
        concept_synthesizer,
        issue_clusterer,
        clusterer_to_retriever,
        retriever,
        retriever_to_drafter,
        question_drafter,
        drafter_to_answerkey,
        answer_key_builder,
        grounder_dispatcher,
        grounder_dispatcher_to_grounder,
        grounder,
        critic,
        should_revise,
        reviser,
        final_drafter,
        final_drafter_to_writer,
        exam_card_writer,
    )

    builder = StateGraph(AgentState)

    # ── nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("planner",             planner)
    builder.add_node("source_profiler",     source_profiler)
    builder.add_node("concept_synthesizer", concept_synthesizer)
    builder.add_node("issue_clusterer",     issue_clusterer)
    builder.add_node("retriever",           retriever)
    builder.add_node("question_drafter",    question_drafter)
    builder.add_node("answer_key_builder",  answer_key_builder)
    builder.add_node("grounder_dispatcher", grounder_dispatcher)
    builder.add_node("grounder",            grounder)
    builder.add_node("critic",              critic)
    builder.add_node("reviser",             reviser)
    builder.add_node("final_drafter",       final_drafter)
    builder.add_node("exam_card_writer",    exam_card_writer)

    # ── entry ──────────────────────────────────────────────────────────────────
    builder.set_entry_point("planner")

    # ── edges ──────────────────────────────────────────────────────────────────
    # planner → parallel source_profiler (one per doc)
    builder.add_conditional_edges("planner", planner_to_profiler)

    # all source_profiler branches → concept_synthesizer (single sequential node)
    builder.add_edge("source_profiler", "concept_synthesizer")

    # concept_synthesizer → issue_clusterer
    builder.add_edge("concept_synthesizer", "issue_clusterer")

    # issue_clusterer → parallel retriever (one per issue)
    builder.add_conditional_edges("issue_clusterer", clusterer_to_retriever)

    # retriever → parallel question_drafter (one per bundle)
    builder.add_conditional_edges("retriever", retriever_to_drafter)

    # question_drafter → parallel answer_key_builder
    builder.add_conditional_edges("question_drafter", drafter_to_answerkey)

    # answer_key_builder → grounder_dispatcher (barrier: waits for ALL AKBs)
    # then fans out to one grounder per finished draft
    builder.add_edge("answer_key_builder", "grounder_dispatcher")
    builder.add_conditional_edges("grounder_dispatcher", grounder_dispatcher_to_grounder)

    # grounder (all branches) → critic
    builder.add_edge("grounder", "critic")

    # critic → conditional: reviser or final_drafter
    builder.add_conditional_edges(
        "critic",
        should_revise,
        {"reviser": "reviser", "final_drafter": "final_drafter"},
    )

    # reviser loops back to critic for another grounding pass
    builder.add_edge("reviser", "critic")

    # final_drafter → [Send → exam_card_writer ×N (parallel, one per question)] → END
    # Each branch owns one DB write (exam_questions + exam_answers row).
    # LangGraph waits for all N branches before returning to the caller.
    # note_tasks._generate_exam_questions_agent then stamps notes COMPLETE.
    builder.add_conditional_edges("final_drafter", final_drafter_to_writer)
    builder.add_edge("exam_card_writer", END)

    return builder.compile(checkpointer=checkpointer)


# ── Public run functions ───────────────────────────────────────────────────────

async def run_exam_agent(
    request: str,
    project_id: str,
    source_ids: List[str],
    n_questions: int = 5,
    use_voyage: bool = False,
    thread_id: Optional[str] = None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict:
    """
    Run the exam-questions agent to completion and return the final state.

    Args:
        request      — user's exam request string
        project_id   — Supabase project UUID
        source_ids   — list of document UUIDs to scope retrieval
        n_questions  — number of exam questions to generate
        use_voyage   — True if documents were ingested with voyage-law-2
        thread_id    — LangGraph checkpoint thread ID (from AgentLedgerService.initialize_run)
        job_id       — agent_jobs.id / notes.id for artifact + exam card persistence
        run_id       — agent_runs.id for progress tracking (optional)
        user_id      — auth.users.id; required for exam_questions.user_id FK

    Returns:
        Final AgentState dict.  Keys: state["final_output"] (Markdown),
        state["persisted_question_ids"] (list of exam_questions UUIDs written).
    """
    initial_state = {
        "request":        request,
        "project_id":     project_id,
        "source_ids":     source_ids,
        "n_questions":    n_questions,
        "use_voyage":     use_voyage,
        "revision_count": 0,
        "budget":         {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        "job_id":         job_id or "",
        "run_id":         run_id or "",
        "user_id":        user_id or "",
    }
    config = {"configurable": {"thread_id": thread_id or "exam-agent"}}

    async with _checkpointer_ctx() as checkpointer:
        graph = _build_graph(checkpointer)
        return await graph.ainvoke(initial_state, config=config)


async def run_exam_agent_stream(
    request: str,
    project_id: str,
    source_ids: List[str],
    n_questions: int = 5,
    use_voyage: bool = False,
    thread_id: Optional[str] = None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Stream partial state updates from the exam agent.

    Yields state snapshot dicts as each node completes.  The first dict
    containing "draft_questions" (with at least one entry) signals that
    the first question draft is ready.

    Usage:
        async for state in run_exam_agent_stream(...):
            if state.get("final_output"):
                print(state["final_output"])
                break
    """
    initial_state = {
        "request":        request,
        "project_id":     project_id,
        "source_ids":     source_ids,
        "n_questions":    n_questions,
        "use_voyage":     use_voyage,
        "revision_count": 0,
        "budget":         {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        "job_id":         job_id or "",
        "run_id":         run_id or "",
        "user_id":        user_id or "",
    }
    config = {"configurable": {"thread_id": thread_id or "exam-agent-stream"}}

    async with _checkpointer_ctx() as checkpointer:
        graph = _build_graph(checkpointer)
        async for event in graph.astream(initial_state, config=config, stream_mode="values"):
            yield event
