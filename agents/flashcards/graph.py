# agents/flashcards/graph.py
"""
LangGraph StateGraph for the flashcard agent.

Graph topology:
  head_orchestrator → [Send → source_profiler ×N] → concept_extractor
  concept_extractor → card_blueprint_planner
  card_blueprint_planner → [Send → process_batch ×N_batches]  ← parallel fan-out
  process_batch (×N, parallel) → global_deck_critic           ← merge point
  global_deck_critic → deterministic_formatter_persister → END

Phase 1 — Source understanding (unchanged):
  head_orchestrator fans out to one source_profiler per document; all branches
  merge into concept_extractor via operator.add on source_profiles.

Phase 2 — Parallel batch execution (new):
  card_blueprint_planner pre-assigns disjoint spec_index ranges to each batch.
  card_blueprint_planner_to_batch fans out via Send so all N batches run
  concurrently. Each process_batch runs: draft → critique → repair (max 1) → DB write.
  Results accumulate into accepted_card_ids, used_card_signatures, batch_results via
  operator.add reducers; no cross-batch dedup lock needed.

Phase 3 — Final QA (unchanged):
  global_deck_critic reads accepted_card_ids + used_card_signatures from merged state.
  deterministic_formatter_persister updates the notes row.

Enabled via USE_FLASHCARD_AGENT env var (default: false).
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv

# ——— Logging & Env Load ───────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
load_dotenv()

USE_FLASHCARD_AGENT = os.getenv("USE_FLASHCARD_AGENT", "false").lower() == "true"


# ── Checkpointer factory ───────────────────────────────────────────────────────

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
        _dsn = (os.getenv("POSTGRES_DSN") or "").strip()
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


# ── Graph builder ──────────────────────────────────────────────────────────────

def _build_graph(checkpointer):
    from langgraph.graph import StateGraph, END

    from .state import AgentState
    from .nodes import (
        head_orchestrator,
        head_orchestrator_to_profiler,
        source_profiler,
        concept_extractor,
        card_blueprint_planner,
        card_blueprint_planner_to_batch,
        process_batch,
        global_deck_critic,
        deterministic_formatter_persister,
    )

    builder = StateGraph(AgentState)

    # ── nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("head_orchestrator",                 head_orchestrator)
    builder.add_node("source_profiler",                   source_profiler)
    builder.add_node("concept_extractor",                 concept_extractor)
    builder.add_node("card_blueprint_planner",            card_blueprint_planner)
    builder.add_node("process_batch",                     process_batch)
    builder.add_node("global_deck_critic",                global_deck_critic)
    builder.add_node("deterministic_formatter_persister", deterministic_formatter_persister)

    # ── entry ──────────────────────────────────────────────────────────────────
    builder.set_entry_point("head_orchestrator")

    # ── Phase 1: Source understanding ─────────────────────────────────────────
    builder.add_conditional_edges("head_orchestrator", head_orchestrator_to_profiler)
    builder.add_edge("source_profiler",      "concept_extractor")
    builder.add_edge("concept_extractor",    "card_blueprint_planner")

    # ── Phase 2: Parallel batch execution ────────────────────────────────────
    # All batches run concurrently; results merge into global_deck_critic
    builder.add_conditional_edges("card_blueprint_planner", card_blueprint_planner_to_batch)
    builder.add_edge("process_batch", "global_deck_critic")

    # ── Phase 3: Final QA + persistence ───────────────────────────────────────
    builder.add_edge("global_deck_critic",               "deterministic_formatter_persister")
    builder.add_edge("deterministic_formatter_persister", END)

    return builder.compile(checkpointer=checkpointer)


# ── Public run functions ───────────────────────────────────────────────────────

def _base_initial_state(
    request: str,
    project_id: str,
    source_ids: List[str],
    num_cards: int,
    batch_size: int,
    use_voyage: bool,
    is_essential: bool,
    job_id: Optional[str],
    run_id: Optional[str],
    user_id: Optional[str],
) -> Dict:
    return {
        "request":        request,
        "project_id":     project_id,
        "source_ids":     source_ids,
        "num_cards":      num_cards,
        "batch_size":     batch_size,
        "use_voyage":     use_voyage,
        "is_essential":   is_essential,
        "job_id":         job_id or "",
        "run_id":         run_id or "",
        "user_id":        user_id or "",
        # Annotated operator.add accumulators — must be initialised before fan-outs
        "source_profiles":      [],
        "accepted_card_ids":    [],
        "used_card_signatures": [],
        "rejected_card_metadata": [],
        "batch_results":        [],
        "budget": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
    }


async def run_flashcard_agent(
    request: str,
    project_id: str,
    source_ids: List[str],
    num_cards: int = 10,
    batch_size: int = 5,
    use_voyage: bool = False,
    is_essential: bool = False,
    thread_id: Optional[str] = None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict:
    """
    Run the flashcard agent to completion and return the final state.

    Returns:
        state["persisted_card_ids"]  — list of individual_cards UUIDs
        state["final_output"]        — markdown deck summary
    """
    initial_state = _base_initial_state(
        request, project_id, source_ids, num_cards, batch_size,
        use_voyage, is_essential, job_id, run_id, user_id,
    )
    config = {
        "configurable": {"thread_id": thread_id or f"flashcard-agent-{job_id or 'local'}"}
    }
    async with _checkpointer_ctx() as checkpointer:
        graph = _build_graph(checkpointer)
        return await graph.ainvoke(initial_state, config=config)


async def run_flashcard_agent_stream(
    request: str,
    project_id: str,
    source_ids: List[str],
    num_cards: int = 10,
    batch_size: int = 5,
    use_voyage: bool = False,
    is_essential: bool = False,
    thread_id: Optional[str] = None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Stream partial state updates from the flashcard agent.

    Monitor state.get("accepted_card_ids") for accumulating card IDs
    as parallel batches complete.
    """
    initial_state = _base_initial_state(
        request, project_id, source_ids, num_cards, batch_size,
        use_voyage, is_essential, job_id, run_id, user_id,
    )
    config = {
        "configurable": {
            "thread_id": thread_id or f"flashcard-agent-stream-{job_id or 'local'}"
        }
    }
    async with _checkpointer_ctx() as checkpointer:
        graph = _build_graph(checkpointer)
        async for event in graph.astream(initial_state, config=config, stream_mode="values"):
            yield event
