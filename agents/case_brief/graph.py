# agents/case_brief/graph.py
"""
LangGraph StateGraph for the case-brief agent.

Graph topology (14 nodes):
  head_orchestrator       → [Send×N] source_profiler
  source_profiler         → retrieval_planner                   (merge all N; inlines corpus orientation)
  retrieval_planner       → [Send×T] planned_retriever          (one per target)
  planned_retriever       → [Send×B] evidence_card_builder      (one per bundle)
  evidence_card_builder   → [Send×4] legal_artifact_extractor   (4 parallel types)
  legal_artifact_extractor → doctrinal_synthesizer              (merge all 4)
  doctrinal_synthesizer   → brief_drafter
  brief_drafter           → [Send×10] section_writer            (one per section type)
  section_writer          → [Send×N] section_grounder           (one per drafted section)
  section_grounder        → brief_assembler                     (merge all; raw_sections fallback)
  brief_assembler         → critic
  critic                  → (should_revise?)
    → brief_revision_agent → critic                             (loop, max 3 passes)
    → final_formatter → END

Checkpointing:
  Uses AsyncPostgresSaver for durable state across worker restarts.
  Falls back to MemorySaver when unavailable.

Usage:
  result = await run_case_brief_agent(request, project_id, source_ids)
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

USE_CASE_BRIEF_AGENT = os.getenv("USE_CASE_BRIEF_AGENT", "false").lower() == "true"

# Compact 4-stage pipeline (compact_nodes.py): plan ∥ research (tool-loop) →
# sync_barrier → [Send×N] section_generator → deterministic formatter.
# Default ON; set CASE_BRIEF_COMPACT=false to fall back to the legacy 14-node
# graph, which is retained in nodes.py for rollback.
CASE_BRIEF_COMPACT = os.getenv("CASE_BRIEF_COMPACT", "true").lower() == "true"


# ── Checkpointer factory (mirrors attack_outline/graph.py) ────────────────────

@asynccontextmanager
async def _checkpointer_ctx():
    """
    Yields a configured LangGraph checkpointer.
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


# ── Graph builder ──────────────────────────────────────────────────────────────

def _build_compact_graph(checkpointer):
    """
    Compact 4-stage topology (CASE_BRIEF_COMPACT=true, the default):

      START ──┬─▶ plan_agent ─────┐
              └─▶ research_agent ─┴─▶ sync_barrier ─▶ [Send×N] section_generator
                                                        → final_formatter → END
                                    └──(no dossier)──────→ final_formatter

    plan_agent and research_agent run in PARALLEL — research_agent does its own
    corpus survey rather than waiting on job_plan. sync_barrier is a no-op join:
    a node with two incoming static edges waits for both predecessors, so every
    section writer sees both job_plan and the dossier.

    The join also keeps conditional edges OFF fan-out nodes. The legacy graph
    hung routers directly on Send-parallel nodes (planned_retriever,
    evidence_card_builder, section_writer), where the branch re-evaluates once
    per parallel task over an accumulating list.
    """
    from langgraph.graph import StateGraph, START, END

    from .state import AgentState
    from .compact_nodes import (
        plan_agent,
        research_agent,
        sync_barrier,
        section_generator,
        final_formatter,
        dossier_to_generators,
    )

    builder = StateGraph(AgentState)
    builder.add_node("plan_agent",        plan_agent)
    builder.add_node("research_agent",    research_agent)
    builder.add_node("sync_barrier",      sync_barrier)
    builder.add_node("section_generator", section_generator)
    builder.add_node("final_formatter",   final_formatter)

    # Parallel entry — neither branch waits on the other.
    builder.add_edge(START, "plan_agent")
    builder.add_edge(START, "research_agent")

    # Join: sync_barrier runs only once BOTH branches have completed.
    builder.add_edge("plan_agent", "sync_barrier")
    builder.add_edge("research_agent", "sync_barrier")

    # List[Send] fan-out over the fixed brief-unit table, or the string route
    # straight to the formatter when research produced no dossier.
    builder.add_conditional_edges(
        "sync_barrier",
        dossier_to_generators,
        {"final_formatter": "final_formatter"},
    )
    builder.add_edge("section_generator", "final_formatter")
    builder.add_edge("final_formatter", END)

    return builder.compile(checkpointer=checkpointer)


def _build_graph(checkpointer):
    if CASE_BRIEF_COMPACT:
        return _build_compact_graph(checkpointer)
    from langgraph.graph import StateGraph, END

    from .state import AgentState
    from .nodes import (
        # nodes
        head_orchestrator,
        source_profiler,
        retrieval_planner,
        planned_retriever,
        evidence_card_builder,
        legal_artifact_extractor,
        doctrinal_synthesizer,
        brief_drafter,
        section_writer,
        section_grounder,
        brief_assembler,
        critic,
        brief_revision_agent,
        final_formatter,
        # routing helpers
        head_orchestrator_to_profiler,
        retrieval_planner_to_retriever,
        retriever_to_card_builder,
        card_builder_to_extractors,
        drafter_to_writers,
        writers_to_grounders,
        should_revise,
    )

    builder = StateGraph(AgentState)

    # ── nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("head_orchestrator",        head_orchestrator)
    builder.add_node("source_profiler",          source_profiler)
    builder.add_node("retrieval_planner",        retrieval_planner)
    builder.add_node("planned_retriever",        planned_retriever)
    builder.add_node("evidence_card_builder",    evidence_card_builder)
    builder.add_node("legal_artifact_extractor", legal_artifact_extractor)
    builder.add_node("doctrinal_synthesizer",    doctrinal_synthesizer)
    builder.add_node("brief_drafter",            brief_drafter)
    builder.add_node("section_writer",           section_writer)
    builder.add_node("section_grounder",         section_grounder)
    builder.add_node("brief_assembler",          brief_assembler)
    builder.add_node("critic",                   critic)
    builder.add_node("brief_revision_agent",     brief_revision_agent)
    builder.add_node("final_formatter",          final_formatter)

    # ── entry ──────────────────────────────────────────────────────────────────
    builder.set_entry_point("head_orchestrator")

    # ── edges ──────────────────────────────────────────────────────────────────

    # head_orchestrator → parallel source_profiler (one per source doc)
    builder.add_conditional_edges("head_orchestrator", head_orchestrator_to_profiler)

    # all source_profiler branches merge → retrieval_planner (inlines corpus orientation)
    builder.add_edge("source_profiler", "retrieval_planner")

    # retrieval_planner → parallel planned_retriever (one per retrieval target)
    builder.add_conditional_edges("retrieval_planner", retrieval_planner_to_retriever)

    # all planned_retriever branches merge → parallel evidence_card_builder (one per bundle)
    builder.add_conditional_edges("planned_retriever", retriever_to_card_builder)

    # all evidence_card_builder branches merge → 4 parallel legal_artifact_extractors
    builder.add_conditional_edges("evidence_card_builder", card_builder_to_extractors)

    # all 4 legal_artifact_extractor branches merge → doctrinal_synthesizer
    builder.add_edge("legal_artifact_extractor", "doctrinal_synthesizer")

    # doctrinal_synthesizer → brief_drafter
    builder.add_edge("doctrinal_synthesizer", "brief_drafter")

    # brief_drafter → parallel section_writer (one per section type)
    builder.add_conditional_edges("brief_drafter", drafter_to_writers)

    # all section_writer branches merge → parallel section_grounder (one per section)
    builder.add_conditional_edges("section_writer", writers_to_grounders)

    # all section_grounder branches merge → brief_assembler
    builder.add_edge("section_grounder", "brief_assembler")

    # brief_assembler → critic
    builder.add_edge("brief_assembler", "critic")

    # critic → conditional: brief_revision_agent (loop) or final_formatter
    builder.add_conditional_edges(
        "critic",
        should_revise,
        {
            "brief_revision_agent": "brief_revision_agent",
            "final_formatter":      "final_formatter",
        },
    )

    # brief_revision_agent loops back to critic for re-evaluation
    builder.add_edge("brief_revision_agent", "critic")

    # final_formatter → END
    builder.add_edge("final_formatter", END)

    return builder.compile(checkpointer=checkpointer)


# ── Public run functions ───────────────────────────────────────────────────────

async def run_case_brief_agent(
    request: str,
    project_id: str,
    source_ids: List[str],
    use_voyage: bool = False,
    thread_id: Optional[str] = None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict:
    """
    Run the case-brief agent to completion and return the final state.

    Args:
        request    — user request string (e.g. "Create a case brief for these sources")
        project_id — Supabase project UUID
        source_ids — list of document_sources UUIDs to scope retrieval
        use_voyage — True if documents were ingested with voyage-law-2
        thread_id  — LangGraph checkpoint thread ID (from AgentLedgerService.initialize_run)
        job_id     — agent_jobs.id / notes.id for artifact persistence
        run_id     — agent_runs.id for progress tracking
        user_id    — auth.users.id (stored on ledger artifacts)

    Returns:
        Final AgentState dict. Key: state["final_output"] (Markdown case brief).
    """
    initial_state = {
        "request":             request,
        "project_id":          project_id,
        "source_ids":          source_ids,
        "use_voyage":          use_voyage,
        "revision_count":      0,
        "budget":              {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        "job_id":              job_id or "",
        "run_id":              run_id or "",
        "user_id":             user_id or "",
        # initialise Annotated list fields so operator.add has a base
        "source_profiles":     [],
        "retrieval_bundles":   [],
        "evidence_cards":      [],
        "extracted_artifacts": [],
        "raw_sections":        [],
        "grounding_reports":   [],
        "brief_units":         [],
    }
    config = {"configurable": {"thread_id": thread_id or "case-brief-agent"}}

    async with _checkpointer_ctx() as checkpointer:
        graph = _build_graph(checkpointer)
        return await graph.ainvoke(initial_state, config=config)


async def run_case_brief_agent_stream(
    request: str,
    project_id: str,
    source_ids: List[str],
    use_voyage: bool = False,
    thread_id: Optional[str] = None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Stream partial state updates from the case-brief agent.

    Yields state snapshot dicts as each node completes. The first dict
    containing "assembled_brief" signals assembly is complete.
    The dict containing "final_output" signals the pipeline is done.

    Usage:
        async for state in run_case_brief_agent_stream(...):
            if state.get("final_output"):
                print(state["final_output"])
                break
    """
    initial_state = {
        "request":             request,
        "project_id":          project_id,
        "source_ids":          source_ids,
        "use_voyage":          use_voyage,
        "revision_count":      0,
        "budget":              {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        "job_id":              job_id or "",
        "run_id":              run_id or "",
        "user_id":             user_id or "",
        "source_profiles":     [],
        "retrieval_bundles":   [],
        "evidence_cards":      [],
        "extracted_artifacts": [],
        "raw_sections":        [],
        "grounding_reports":   [],
        "brief_units":         [],
    }
    config = {"configurable": {"thread_id": thread_id or "case-brief-agent-stream"}}

    async with _checkpointer_ctx() as checkpointer:
        graph = _build_graph(checkpointer)
        async for event in graph.astream(initial_state, config=config, stream_mode="values"):
            yield event
