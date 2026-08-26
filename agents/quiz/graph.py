# agents/quiz/graph.py
"""
LangGraph StateGraph for the quiz agent.

Graph topology:
  head_orchestrator → [Send → source_profiler ×N] → case_rule_extractor
  case_rule_extractor → cross_doc_concepts_synthesis → quiz_blueprint_planner
  quiz_blueprint_planner → [Send → process_batch ×N_batches]   ← parallel fan-out
  process_batch (×N, parallel) → critic                         ← merge point
  critic → final_formatter → END

Phase 2 — Parallel batch execution:
  quiz_blueprint_planner fans out all batches simultaneously via Send.
  Each process_batch runs internally:
    question_drafter (worker_mid) →
    false_trap_red_herring_generator (worker_mid, parallel per-question) →
    question_evaluator → reviser loop (max 2) → grounder → DB write.
  Results accumulate into accepted_question_ids, rejected_question_metadata,
  used_question_signatures, batch_results via operator.add reducers.

Enabled via USE_QUIZ_AGENT env var (default: false).
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv

# ——— Logging & Env Load ───────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
load_dotenv()

USE_QUIZ_AGENT = os.getenv("USE_QUIZ_AGENT", "false").lower() == "true"

# Compact 4-stage pipeline (compact_nodes.py): plan ∥ research (tool-loop) →
# sync_barrier → [Send×N_batches] question_batch_generator → deterministic
# formatter. Default ON; set QUIZ_COMPACT=false for the legacy graph.
QUIZ_COMPACT = os.getenv("QUIZ_COMPACT", "true").lower() == "true"


@asynccontextmanager
async def _checkpointer_ctx():
    """Yield a durable AsyncPostgresSaver or fall back to MemorySaver.

    Uses POSTGRES_DSN (session-mode pooler, port 5432) rather than the
    transaction-mode pool (port 6543), which does not support prepared
    statements.  The _setup_ok flag ensures errors inside graph.ainvoke
    are re-raised instead of triggering an illegal second yield.
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
            raise
        if not isinstance(exc, ImportError):
            logger.warning(
                "AsyncPostgresSaver failed (%s) — falling back to MemorySaver", exc
            )
    from langgraph.checkpoint.memory import MemorySaver
    yield MemorySaver()


def _build_compact_graph(checkpointer):
    """
    Compact 4-stage topology (QUIZ_COMPACT=true, the default):

      START ──┬─▶ plan_agent ─────┐
              └─▶ research_agent ─┴─▶ sync_barrier ─▶ [Send×N] question_batch_generator
                                                        → final_formatter → END
                                    └──(no specs)────────→ final_formatter

    The legacy graph was already blueprint→parallel-batch; the compression is
    in the research stage (case_rule_extractor made up to 15 orchestrator calls)
    and in folding the per-question distractor/evaluator/grounder calls into
    the batch generator.
    """
    from langgraph.graph import StateGraph, START, END

    from .state import AgentState
    from .compact_nodes import (
        plan_agent,
        research_agent,
        sync_barrier,
        question_batch_generator,
        final_formatter,
        dossier_to_batches,
    )

    builder = StateGraph(AgentState)
    builder.add_node("plan_agent",               plan_agent)
    builder.add_node("research_agent",           research_agent)
    builder.add_node("sync_barrier",             sync_barrier)
    builder.add_node("question_batch_generator", question_batch_generator)
    builder.add_node("final_formatter",          final_formatter)

    builder.add_edge(START, "plan_agent")
    builder.add_edge(START, "research_agent")
    builder.add_edge("plan_agent", "sync_barrier")
    builder.add_edge("research_agent", "sync_barrier")
    builder.add_conditional_edges(
        "sync_barrier",
        dossier_to_batches,
        {"final_formatter": "final_formatter"},
    )
    builder.add_edge("question_batch_generator", "final_formatter")
    builder.add_edge("final_formatter", END)

    return builder.compile(checkpointer=checkpointer)


def _build_graph(checkpointer):
    if QUIZ_COMPACT:
        return _build_compact_graph(checkpointer)
    from langgraph.graph import StateGraph, END

    from .state import AgentState
    from .nodes import (
        head_orchestrator,
        head_orchestrator_to_profiler,
        source_profiler,
        case_rule_extractor,
        cross_doc_concepts_synthesis,
        quiz_blueprint_planner,
        quiz_blueprint_planner_to_batch,
        process_batch,
        critic,
        final_formatter,
    )

    builder = StateGraph(AgentState)

    # ── nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("head_orchestrator",            head_orchestrator)
    builder.add_node("source_profiler",              source_profiler)
    builder.add_node("case_rule_extractor",          case_rule_extractor)
    builder.add_node("cross_doc_concepts_synthesis", cross_doc_concepts_synthesis)
    builder.add_node("quiz_blueprint_planner",       quiz_blueprint_planner)
    builder.add_node("process_batch",                process_batch)
    builder.add_node("critic",                       critic)
    builder.add_node("final_formatter",              final_formatter)

    # ── entry ──────────────────────────────────────────────────────────────────
    builder.set_entry_point("head_orchestrator")

    # ── Phase 1: Source understanding ─────────────────────────────────────────
    builder.add_conditional_edges("head_orchestrator", head_orchestrator_to_profiler)
    builder.add_edge("source_profiler",          "case_rule_extractor")
    builder.add_edge("case_rule_extractor",      "cross_doc_concepts_synthesis")
    builder.add_edge("cross_doc_concepts_synthesis", "quiz_blueprint_planner")

    # ── Phase 2: Parallel batch fan-out ───────────────────────────────────────
    # All batches run concurrently; results merge into critic via operator.add
    builder.add_conditional_edges("quiz_blueprint_planner", quiz_blueprint_planner_to_batch)
    builder.add_edge("process_batch", "critic")

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
    # 3 (not 5) so a 10-question quiz fans out to 4 Sends rather than 2:
    # generator latency under thinking=True varies wildly per call, and
    # with only 2 batches one slow draw sets the whole stage (observed
    # 137s vs 351s on identically-shaped batches).
    batch_size: int = 3,
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
        # Annotated operator.add accumulators — must be initialised before fan-outs
        "source_profiles":          [],
        "accepted_question_ids":    [],
        "rejected_question_metadata": [],
        "used_question_signatures": [],
        "batch_results":            [],
        "generated_questions":      [],
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
    # 3 (not 5) so a 10-question quiz fans out to 4 Sends rather than 2:
    # generator latency under thinking=True varies wildly per call, and
    # with only 2 batches one slow draw sets the whole stage (observed
    # 137s vs 351s on identically-shaped batches).
    batch_size: int = 3,
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
        "source_profiles":          [],
        "accepted_question_ids":    [],
        "rejected_question_metadata": [],
        "used_question_signatures": [],
        "batch_results":            [],
        "generated_questions":      [],
        "budget": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
    }
    config = {
        "configurable": {"thread_id": thread_id or f"quiz-agent-stream-{job_id or 'local'}"}
    }

    async with _checkpointer_ctx() as checkpointer:
        graph = _build_graph(checkpointer)
        async for event in graph.astream(initial_state, config=config, stream_mode="values"):
            yield event
