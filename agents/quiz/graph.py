# agents/quiz/graph.py
"""
LangGraph StateGraph for the quiz agent.

Graph topology:
  head_orchestrator → [Send → source_profiler ×N] → case_rule_extractor
  case_rule_extractor → cross_doc_concepts_synthesis → quiz_blueprint_planner
  quiz_blueprint_planner → question_drafter  ← ─────────────────────────┐
  question_drafter → false_trap_red_herring_generator                    │
  false_trap_red_herring_generator → question_evaluator                  │
  question_evaluator → (should_revise_batch?)                            │
      → reviser → question_evaluator  (revision loop, max 2 passes)      │
      → grounder → batch_commit                                           │
  batch_commit → (batch_router?) ─────────────────────────────────────── ┘
      → critic → final_formatter → END

Batch loop:
  batch_commit increments current_batch_index.
  batch_router routes back to question_drafter while batches remain and
  accepted_count < num_questions; otherwise routes to critic.

Revision loop:
  should_revise_batch routes to reviser when current_batch_eval.passes is
  False AND batch_revision_count < 2; otherwise routes to grounder.
  reviser increments batch_revision_count on each pass.

Enabled via USE_QUIZ_AGENT env var (default: false).
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)

USE_QUIZ_AGENT = os.getenv("USE_QUIZ_AGENT", "false").lower() == "true"


@asynccontextmanager
async def _checkpointer_ctx():
    """Yield a durable AsyncPostgresSaver or fall back to MemorySaver."""
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from tasks.database import DB_DSN
        async with AsyncPostgresSaver.from_conn_string(DB_DSN) as saver:
            await saver.setup()
            yield saver
    except (ImportError, Exception) as exc:
        if not isinstance(exc, ImportError):
            logger.warning(
                "AsyncPostgresSaver failed (%s) — falling back to MemorySaver", exc
            )
        from langgraph.checkpoint.memory import MemorySaver
        yield MemorySaver()


def _build_graph(checkpointer):
    from langgraph.graph import StateGraph, END

    from .state import AgentState
    from .nodes import (
        head_orchestrator,
        head_orchestrator_to_profiler,
        source_profiler,
        case_rule_extractor,
        cross_doc_concepts_synthesis,
        quiz_blueprint_planner,
        question_drafter,
        false_trap_red_herring_generator,
        question_evaluator,
        should_revise_batch,
        reviser,
        grounder,
        batch_commit,
        batch_router,
        critic,
        final_formatter,
    )

    builder = StateGraph(AgentState)

    # ── nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("head_orchestrator",               head_orchestrator)
    builder.add_node("source_profiler",                 source_profiler)
    builder.add_node("case_rule_extractor",             case_rule_extractor)
    builder.add_node("cross_doc_concepts_synthesis",    cross_doc_concepts_synthesis)
    builder.add_node("quiz_blueprint_planner",          quiz_blueprint_planner)
    builder.add_node("question_drafter",                question_drafter)
    builder.add_node("false_trap_red_herring_generator", false_trap_red_herring_generator)
    builder.add_node("question_evaluator",              question_evaluator)
    builder.add_node("reviser",                         reviser)
    builder.add_node("grounder",                        grounder)
    builder.add_node("batch_commit",                    batch_commit)
    builder.add_node("critic",                          critic)
    builder.add_node("final_formatter",                 final_formatter)

    # ── entry ──────────────────────────────────────────────────────────────────
    builder.set_entry_point("head_orchestrator")

    # ── Phase 1: Source understanding ─────────────────────────────────────────
    # head_orchestrator → parallel source_profiler (one per source document)
    builder.add_conditional_edges("head_orchestrator", head_orchestrator_to_profiler)

    # All source_profiler branches merge into case_rule_extractor
    builder.add_edge("source_profiler", "case_rule_extractor")

    builder.add_edge("case_rule_extractor",          "cross_doc_concepts_synthesis")
    builder.add_edge("cross_doc_concepts_synthesis", "quiz_blueprint_planner")

    # ── Phase 2: Batch loop entry ─────────────────────────────────────────────
    builder.add_edge("quiz_blueprint_planner", "question_drafter")

    # ── Batch pipeline ────────────────────────────────────────────────────────
    builder.add_edge("question_drafter", "false_trap_red_herring_generator")
    builder.add_edge("false_trap_red_herring_generator", "question_evaluator")

    # question_evaluator → conditional: reviser or grounder
    builder.add_conditional_edges(
        "question_evaluator",
        should_revise_batch,
        {"reviser": "reviser", "grounder": "grounder"},
    )

    # Revision loop: reviser loops back to question_evaluator (max 2 via batch_revision_count)
    builder.add_edge("reviser", "question_evaluator")

    builder.add_edge("grounder", "batch_commit")

    # Batch loop: batch_commit routes back to question_drafter or advances to critic
    builder.add_conditional_edges(
        "batch_commit",
        batch_router,
        {"question_drafter": "question_drafter", "critic": "critic"},
    )

    # ── Phase 3: Final QA + formatting ────────────────────────────────────────
    builder.add_edge("critic",          "final_formatter")
    builder.add_edge("final_formatter", END)

    return builder.compile(checkpointer=checkpointer)


# ── Public run functions ───────────────────────────────────────────────────────

async def run_quiz_agent(
    request: str,
    project_id: str,
    source_ids: List[str],
    num_questions: int = 10,
    batch_size: int = 5,
    quiz_mode: str = "mixed",
    target_difficulty: str = "application",
    use_voyage: bool = False,
    thread_id: Optional[str] = None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict:
    """
    Run the quiz agent to completion and return the final state.

    Args:
        request          — user's quiz request string
        project_id       — Supabase project UUID
        source_ids       — document UUIDs to scope retrieval
        num_questions    — total questions to generate (default 10, supports 50+)
        batch_size       — questions per batch (default 5, max 10)
        quiz_mode        — "recall" | "application" | "exam-style" | "mixed"
        target_difficulty— "recall" | "application" | "analysis"
        use_voyage       — True if documents were ingested with voyage-law-2
        thread_id        — LangGraph checkpoint thread ID
        job_id           — notes.id (parent quiz container; quiz_questions.quiz_id FK)
        run_id           — agent_runs.id for progress tracking
        user_id          — auth.users.id; required for quiz_questions.user_id FK

    Returns:
        Final AgentState.  Key fields:
          state["persisted_question_ids"] — list of quiz_questions UUIDs
          state["final_output"]           — markdown summary
    """
    initial_state = {
        "request":            request,
        "project_id":         project_id,
        "source_ids":         source_ids,
        "num_questions":      num_questions,
        "batch_size":         batch_size,
        "quiz_mode":          quiz_mode,
        "target_difficulty":  target_difficulty,
        "use_voyage":         use_voyage,
        "job_id":             job_id or "",
        "run_id":             run_id or "",
        "user_id":            user_id or "",
        # Annotated accumulator — must be initialised before parallel fan-out
        "source_profiles":    [],
        # Batch loop initial state
        "current_batch_index":  0,
        "batch_revision_count": 0,
        "accepted_question_ids":      [],
        "rejected_question_metadata": [],
        "used_question_signatures":   [],
        "coverage_summary":           "{}",
        "budget": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
    }
    config = {"configurable": {"thread_id": thread_id or f"quiz-agent-{job_id or 'local'}"}}

    async with _checkpointer_ctx() as checkpointer:
        graph = _build_graph(checkpointer)
        return await graph.ainvoke(initial_state, config=config)


async def run_quiz_agent_stream(
    request: str,
    project_id: str,
    source_ids: List[str],
    num_questions: int = 10,
    batch_size: int = 5,
    quiz_mode: str = "mixed",
    target_difficulty: str = "application",
    use_voyage: bool = False,
    thread_id: Optional[str] = None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Stream partial state updates from the quiz agent.

    Yields state snapshots as each node completes.  Monitor
    state.get("current_batch_index") for batch progress and
    state.get("accepted_question_ids") for accumulating IDs.

    Usage:
        async for state in run_quiz_agent_stream(...):
            accepted = len(state.get("accepted_question_ids") or [])
            print(f"Accepted so far: {accepted}")
            if state.get("final_output"):
                break
    """
    initial_state = {
        "request":            request,
        "project_id":         project_id,
        "source_ids":         source_ids,
        "num_questions":      num_questions,
        "batch_size":         batch_size,
        "quiz_mode":          quiz_mode,
        "target_difficulty":  target_difficulty,
        "use_voyage":         use_voyage,
        "job_id":             job_id or "",
        "run_id":             run_id or "",
        "user_id":            user_id or "",
        "source_profiles":    [],
        "current_batch_index":  0,
        "batch_revision_count": 0,
        "accepted_question_ids":      [],
        "rejected_question_metadata": [],
        "used_question_signatures":   [],
        "coverage_summary":           "{}",
        "budget": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
    }
    config = {
        "configurable": {"thread_id": thread_id or f"quiz-agent-stream-{job_id or 'local'}"}
    }

    async with _checkpointer_ctx() as checkpointer:
        graph = _build_graph(checkpointer)
        async for event in graph.astream(initial_state, config=config, stream_mode="values"):
            yield event
