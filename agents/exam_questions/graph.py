# agents/exam_questions/graph.py
"""
LangGraph StateGraph for the exam-questions agent.

Graph topology:
  planner → [Send → source_profiler] → issue_clusterer
  issue_clusterer → [Send → retriever] → [Send → question_drafter]
  question_drafter → [Send → answer_key_builder] → [Send → grounder]
  grounder → critic → (should_revise?) → reviser ↺ | assembler

Run with:
    result = await run_exam_agent(request, project_id, source_ids, n_questions)
    markdown = result["final_output"]
"""

import logging
import os
from typing import AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)

USE_LANGGRAPH_AGENT = os.getenv("USE_LANGGRAPH_AGENT", "false").lower() == "true"


def create_exam_agent():
    """
    Build and compile the LangGraph exam-questions StateGraph.

    Lazy import so that importing this module never fails even if
    langgraph is not installed (returns None, checked by caller).
    """
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:
        logger.error("langgraph not installed — cannot create exam agent")
        return None

    from .state import AgentState
    from .nodes import (
        planner,
        planner_to_profiler,
        source_profiler,
        issue_clusterer,
        clusterer_to_retriever,
        retriever,
        retriever_to_drafter,
        question_drafter,
        drafter_to_answerkey,
        answer_key_builder,
        answerkey_to_grounder,
        grounder,
        critic,
        should_revise,
        reviser,
        assembler,
    )

    builder = StateGraph(AgentState)

    # ── add nodes ─────────────────────────────────────────────────────────────
    builder.add_node("planner", planner)
    builder.add_node("source_profiler", source_profiler)
    builder.add_node("issue_clusterer", issue_clusterer)
    builder.add_node("retriever", retriever)
    builder.add_node("question_drafter", question_drafter)
    builder.add_node("answer_key_builder", answer_key_builder)
    builder.add_node("grounder", grounder)
    builder.add_node("critic", critic)
    builder.add_node("reviser", reviser)
    builder.add_node("assembler", assembler)

    # ── entry ──────────────────────────────────────────────────────────────────
    builder.set_entry_point("planner")

    # ── sequential / fan-out edges ────────────────────────────────────────────
    # planner → parallel source_profiler (one per doc)
    builder.add_conditional_edges("planner", planner_to_profiler)

    # source_profiler (all branches) → issue_clusterer
    builder.add_edge("source_profiler", "issue_clusterer")

    # issue_clusterer → parallel retriever (one per issue)
    builder.add_conditional_edges("issue_clusterer", clusterer_to_retriever)

    # retriever (all branches) → parallel question_drafter
    builder.add_conditional_edges("retriever", retriever_to_drafter)

    # question_drafter → parallel answer_key_builder
    builder.add_conditional_edges("question_drafter", drafter_to_answerkey)

    # answer_key_builder → parallel grounder
    builder.add_conditional_edges("answer_key_builder", answerkey_to_grounder)

    # grounder (all branches) → critic
    builder.add_edge("grounder", "critic")

    # critic → conditional: reviser or assembler
    builder.add_conditional_edges(
        "critic",
        should_revise,
        {"reviser": "reviser", "assembler": "assembler"},
    )

    # reviser loops back to critic for another grounding pass
    builder.add_edge("reviser", "critic")

    # assembler → END
    builder.add_edge("assembler", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


async def run_exam_agent(
    request: str,
    project_id: str,
    source_ids: List[str],
    n_questions: int = 3,
    use_voyage: bool = False,
    thread_id: Optional[str] = None,
    stream_first: bool = False,
) -> Dict:
    """
    Run the exam-questions agent to completion and return the final state.

    Args:
        request      — user's exam request string
        project_id   — Supabase project UUID
        source_ids   — list of document UUIDs to scope retrieval
        n_questions  — number of exam questions to generate
        use_voyage   — True if documents were ingested with voyage-law-2
        thread_id    — optional LangGraph checkpoint thread ID for resumption
        stream_first — if True, yield partial state dicts as they arrive
                       (caller must use run_exam_agent_stream instead)

    Returns:
        Final AgentState dict.  Key: state["final_output"] is the Markdown.
    """
    graph = create_exam_agent()
    if graph is None:
        raise RuntimeError("langgraph not available — install langgraph>=0.2.0")

    initial_state = {
        "request": request,
        "project_id": project_id,
        "source_ids": source_ids,
        "n_questions": n_questions,
        "use_voyage": use_voyage,
        "revision_count": 0,
        "budget": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
    }

    config = {"configurable": {"thread_id": thread_id or "exam-agent"}}

    final_state = await graph.ainvoke(initial_state, config=config)
    return final_state


async def run_exam_agent_stream(
    request: str,
    project_id: str,
    source_ids: List[str],
    n_questions: int = 3,
    use_voyage: bool = False,
    thread_id: Optional[str] = None,
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
    graph = create_exam_agent()
    if graph is None:
        raise RuntimeError("langgraph not available — install langgraph>=0.2.0")

    initial_state = {
        "request": request,
        "project_id": project_id,
        "source_ids": source_ids,
        "n_questions": n_questions,
        "use_voyage": use_voyage,
        "revision_count": 0,
        "budget": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
    }

    config = {"configurable": {"thread_id": thread_id or "exam-agent-stream"}}

    async for event in graph.astream(initial_state, config=config, stream_mode="values"):
        yield event
