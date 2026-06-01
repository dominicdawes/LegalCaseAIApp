# agents/flashcards/graph.py
"""
LangGraph StateGraph for the flashcard agent.

Graph topology:
  head_orchestrator → [Send → source_profiler ×N] → concept_extractor
  concept_extractor → card_blueprint_planner
  card_blueprint_planner → flashcard_drafter  ← ───────────────────────────┐
  flashcard_drafter → answer_backside_enricher                              │
  answer_backside_enricher → local_card_critic                              │
  local_card_critic → (should_repair_batch?)                                │
      → card_repair_agent → local_card_critic  (repair loop, max 2 passes)  │
      → batch_commit                                                          │
  batch_commit → (batch_router?) ──────────────────────────────────────────── ┘
      → global_deck_critic → deterministic_formatter_persister → END

Batch loop:
  batch_commit increments current_batch_index.
  batch_router routes back to flashcard_drafter while batches remain and
  accepted_count < num_cards; otherwise routes to global_deck_critic.

Repair loop:
  should_repair_batch routes to card_repair_agent when current_batch_eval.passes is
  False AND batch_revision_count < 2; otherwise routes to batch_commit.
  card_repair_agent increments batch_revision_count on each pass.

Checkpointing:
  Uses AsyncPostgresSaver (langgraph-checkpoint-postgres) for durable state
  across worker restarts.  Falls back to MemorySaver for local dev.

Enable via USE_FLASHCARD_AGENT env var (default: false).
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
        flashcard_drafter,
        local_card_critic,
        should_repair_batch,
        card_repair_agent,
        batch_commit,
        batch_router,
        global_deck_critic,
        deterministic_formatter_persister,
    )

    builder = StateGraph(AgentState)

    # ── nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("head_orchestrator",                head_orchestrator)
    builder.add_node("source_profiler",                  source_profiler)
    builder.add_node("concept_extractor",                concept_extractor)
    builder.add_node("card_blueprint_planner",           card_blueprint_planner)
    builder.add_node("flashcard_drafter",                flashcard_drafter)
    builder.add_node("local_card_critic",                local_card_critic)
    builder.add_node("card_repair_agent",                card_repair_agent)
    builder.add_node("batch_commit",                     batch_commit)
    builder.add_node("global_deck_critic",               global_deck_critic)
    builder.add_node("deterministic_formatter_persister", deterministic_formatter_persister)

    # ── entry ──────────────────────────────────────────────────────────────────
    builder.set_entry_point("head_orchestrator")

    # ── Phase 1: Source understanding ─────────────────────────────────────────
    # head_orchestrator → parallel source_profiler (one per source document)
    builder.add_conditional_edges("head_orchestrator", head_orchestrator_to_profiler)

    # All source_profiler branches merge into concept_extractor
    builder.add_edge("source_profiler", "concept_extractor")

    builder.add_edge("concept_extractor",    "card_blueprint_planner")

    # ── Phase 2: Batch loop entry ─────────────────────────────────────────────
    builder.add_edge("card_blueprint_planner", "flashcard_drafter")

    # ── Batch pipeline ────────────────────────────────────────────────────────
    builder.add_edge("flashcard_drafter",        "local_card_critic")

    # local_card_critic → conditional: repair or commit
    builder.add_conditional_edges(
        "local_card_critic",
        should_repair_batch,
        {"card_repair_agent": "card_repair_agent", "batch_commit": "batch_commit"},
    )

    # Repair loop: card_repair_agent routes back to local_card_critic
    builder.add_edge("card_repair_agent", "local_card_critic")

    # Batch loop: batch_commit routes back to drafter or advances to global critic
    builder.add_conditional_edges(
        "batch_commit",
        batch_router,
        {"flashcard_drafter": "flashcard_drafter", "global_deck_critic": "global_deck_critic"},
    )

    # ── Phase 3: Final QA + persistence ───────────────────────────────────────
    builder.add_edge("global_deck_critic",               "deterministic_formatter_persister")
    builder.add_edge("deterministic_formatter_persister", END)

    return builder.compile(checkpointer=checkpointer)


# ── Public run functions ───────────────────────────────────────────────────────

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

    Args:
        request        — user's flashcard request string
        project_id     — Supabase project UUID
        source_ids     — document UUIDs to scope retrieval
        num_cards      — total cards to generate (default 10, supports 50+)
        batch_size     — cards per batch (default 5, max 10)
        use_voyage     — True if documents were ingested with voyage-law-2
        is_essential   — forwarded to notes.is_essential column
        thread_id      — LangGraph checkpoint thread ID
        job_id         — notes.id (parent deck container; individual_cards.deck_id FK)
        run_id         — agent_runs.id for progress tracking
        user_id        — auth.users.id; required for individual_cards.user_id FK

    Returns:
        Final AgentState.  Key fields:
          state["persisted_card_ids"]  — list of individual_cards UUIDs
          state["final_output"]        — markdown deck summary
    """
    initial_state = {
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
        # Annotated accumulator — must be initialised before parallel fan-out
        "source_profiles":              [],
        # Batch loop initial state
        "current_batch_index":          0,
        "batch_revision_count":         0,
        "accepted_card_ids":            [],
        "rejected_card_metadata":       [],
        "used_card_signatures":         [],
        "coverage_summary":             "{}",
        "budget": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
    }
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

    Yields state snapshots as each node completes.  Monitor
    state.get("current_batch_index") for batch progress and
    state.get("accepted_card_ids") for accumulating IDs.

    Usage:
        async for state in run_flashcard_agent_stream(...):
            accepted = len(state.get("accepted_card_ids") or [])
            print(f"Cards committed so far: {accepted}")
            if state.get("final_output"):
                break
    """
    initial_state = {
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
        "source_profiles":              [],
        "current_batch_index":          0,
        "batch_revision_count":         0,
        "accepted_card_ids":            [],
        "rejected_card_metadata":       [],
        "used_card_signatures":         [],
        "coverage_summary":             "{}",
        "budget": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
    }
    config = {
        "configurable": {
            "thread_id": thread_id or f"flashcard-agent-stream-{job_id or 'local'}"
        }
    }

    async with _checkpointer_ctx() as checkpointer:
        graph = _build_graph(checkpointer)
        async for event in graph.astream(initial_state, config=config, stream_mode="values"):
            yield event
