# agents/attack_outline/graph.py
"""
LangGraph StateGraph for the attack-outline agent.

Graph topology (15 nodes):
  head_orchestrator → [Send×N] source_profiler
  source_profiler   → corpus_topic_mapper          (waits for all N branches)
  corpus_topic_mapper → retrieval_planner
  retrieval_planner → [Send×C] planned_retriever   (one per concept plan)
  planned_retriever → [Send×C] legal_artifact_extractor  (one per bundle)
  legal_artifact_extractor → artifact_normalizer   (waits for all C branches)
  artifact_normalizer → concept_clusterer
  concept_clusterer → doctrine_graph_builder
  doctrine_graph_builder → [Send×D] attack_block_builder  (one per cluster)
  attack_block_builder → attack_outline_assembler  (waits for all D branches)
  attack_outline_assembler → [Send×B] grounding_verifier  (one per block)
  grounding_verifier → attack_outline_critic        (waits for all B branches)
  attack_outline_critic → (should_revise?)
    → revision_agent → attack_outline_critic         (revision loop, max 2 passes)
    → final_compressor_formatter → END

Checkpointing:
  Uses AsyncPostgresSaver (langgraph-checkpoint-postgres) for durable state
  across worker restarts. Falls back to MemorySaver when unavailable.

Run with:
    result = await run_attack_outline_agent(request, project_id, source_ids)
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

USE_ATTACK_OUTLINE_AGENT = os.getenv("USE_ATTACK_OUTLINE_AGENT", "false").lower() == "true"

# Compact 4-stage pipeline (compact_nodes.py): plan → research (tool-loop) →
# [Send×N] block_generator → deterministic formatter. Default ON; set
# ATTACK_OUTLINE_COMPACT=false to fall back to the legacy 15-node graph.
ATTACK_OUTLINE_COMPACT = os.getenv("ATTACK_OUTLINE_COMPACT", "true").lower() == "true"


# ── Checkpointer factory (mirrors exam_questions/graph.py) ────────────────────

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
    Compact 4-stage topology (ATTACK_OUTLINE_COMPACT=true, the default):

      START ──┬─▶ plan_agent ─────┐
              └─▶ research_agent ─┴─▶ sync_barrier ─▶ [Send×N] block_generator → final_formatter → END
                                                    └──────(no clusters)───────→ final_formatter

    plan_agent and research_agent run in PARALLEL — research_agent does its
    own survey (list_sources/get_doc_outline) rather than waiting on
    plan_agent's job_plan, so there's no real dependency between them.
    sync_barrier is a no-op join: a node with two incoming static edges waits
    for both predecessors (standard LangGraph fan-in), guaranteeing
    block_generator's Send payload always has both job_plan and the research
    dossier, regardless of which branch finishes first.

    research_agent is a bounded multistep tool-calling loop (its own internal
    LLM turns), so the graph itself stays tiny; state carries the dossier and
    the harvested evidence_store between stages.
    """
    from langgraph.graph import StateGraph, START, END

    from .state import AgentState
    from .compact_nodes import (
        plan_agent,
        research_agent,
        sync_barrier,
        block_generator,
        final_formatter,
        research_to_generators,
    )

    builder = StateGraph(AgentState)
    builder.add_node("plan_agent",      plan_agent)
    builder.add_node("research_agent",  research_agent)
    builder.add_node("sync_barrier",    sync_barrier)
    builder.add_node("block_generator", block_generator)
    builder.add_node("final_formatter", final_formatter)

    # Parallel entry — both branches start immediately, neither waits on the other.
    builder.add_edge(START, "plan_agent")
    builder.add_edge(START, "research_agent")

    # Join: sync_barrier only runs once BOTH branches have completed.
    builder.add_edge("plan_agent", "sync_barrier")
    builder.add_edge("research_agent", "sync_barrier")

    # List[Send] fan-out, or the string route straight to the formatter when
    # research produced no clusters (same pattern as assembler_to_verifier).
    builder.add_conditional_edges(
        "sync_barrier",
        research_to_generators,
        {"final_formatter": "final_formatter"},
    )
    builder.add_edge("block_generator", "final_formatter")
    builder.add_edge("final_formatter", END)

    return builder.compile(checkpointer=checkpointer)


def _build_graph(checkpointer):
    if ATTACK_OUTLINE_COMPACT:
        return _build_compact_graph(checkpointer)
    from langgraph.graph import StateGraph, END

    from .state import AgentState
    from .nodes import (
        # nodes
        head_orchestrator,
        source_profiler,
        corpus_topic_mapper,
        retrieval_planner,
        planned_retriever,
        legal_artifact_extractor,
        artifact_normalizer,
        concept_clusterer,
        doctrine_graph_builder,
        attack_block_builder,
        attack_outline_assembler,
        grounding_verifier,
        attack_outline_critic,
        revision_agent,
        final_compressor_formatter,
        # routing helpers
        head_orchestrator_to_profiler,
        retrieval_planner_to_retriever,
        retriever_to_extractor,
        doctrine_to_block_builder,
        assembler_to_verifier,
        should_revise,
    )

    builder = StateGraph(AgentState)

    # ── nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("head_orchestrator",         head_orchestrator)
    builder.add_node("source_profiler",           source_profiler)
    builder.add_node("corpus_topic_mapper",       corpus_topic_mapper)
    builder.add_node("retrieval_planner",         retrieval_planner)
    builder.add_node("planned_retriever",         planned_retriever)
    builder.add_node("legal_artifact_extractor",  legal_artifact_extractor)
    builder.add_node("artifact_normalizer",       artifact_normalizer)
    builder.add_node("concept_clusterer",         concept_clusterer)
    builder.add_node("doctrine_graph_builder",    doctrine_graph_builder)
    builder.add_node("attack_block_builder",      attack_block_builder)
    builder.add_node("attack_outline_assembler",  attack_outline_assembler)
    builder.add_node("grounding_verifier",        grounding_verifier)
    builder.add_node("attack_outline_critic",     attack_outline_critic)
    builder.add_node("revision_agent",            revision_agent)
    builder.add_node("final_compressor_formatter", final_compressor_formatter)

    # ── entry ──────────────────────────────────────────────────────────────────
    builder.set_entry_point("head_orchestrator")

    # ── edges ──────────────────────────────────────────────────────────────────

    # head_orchestrator → parallel source_profiler (one per source doc)
    builder.add_conditional_edges("head_orchestrator", head_orchestrator_to_profiler)

    # all source_profiler branches merge → corpus_topic_mapper
    builder.add_edge("source_profiler", "corpus_topic_mapper")

    # corpus_topic_mapper → retrieval_planner
    builder.add_edge("corpus_topic_mapper", "retrieval_planner")

    # retrieval_planner → parallel planned_retriever (one per concept plan)
    builder.add_conditional_edges("retrieval_planner", retrieval_planner_to_retriever)

    # all planned_retriever branches merge → parallel legal_artifact_extractor (one per bundle)
    builder.add_conditional_edges("planned_retriever", retriever_to_extractor)

    # all legal_artifact_extractor branches merge → artifact_normalizer
    builder.add_edge("legal_artifact_extractor", "artifact_normalizer")

    # artifact_normalizer → concept_clusterer
    builder.add_edge("artifact_normalizer", "concept_clusterer")

    # concept_clusterer → doctrine_graph_builder
    builder.add_edge("concept_clusterer", "doctrine_graph_builder")

    # doctrine_graph_builder → parallel attack_block_builder (one per cluster)
    builder.add_conditional_edges("doctrine_graph_builder", doctrine_to_block_builder)

    # all attack_block_builder branches merge → attack_outline_assembler
    builder.add_edge("attack_block_builder", "attack_outline_assembler")

    # attack_outline_assembler → parallel grounding_verifier (one per block)
    # OR → final_compressor_formatter directly when no blocks were produced.
    # The path_map entry tells LangGraph that the string route is valid;
    # the List[Send] branch needs no entry (LangGraph handles it automatically).
    builder.add_conditional_edges(
        "attack_outline_assembler",
        assembler_to_verifier,
        {"final_compressor_formatter": "final_compressor_formatter"},
    )

    # all grounding_verifier branches merge → attack_outline_critic
    builder.add_edge("grounding_verifier", "attack_outline_critic")

    # attack_outline_critic → conditional: revision_agent or final_compressor_formatter
    builder.add_conditional_edges(
        "attack_outline_critic",
        should_revise,
        {
            "revision_agent":            "revision_agent",
            "final_compressor_formatter": "final_compressor_formatter",
        },
    )

    # revision_agent loops back to attack_outline_critic for re-evaluation
    builder.add_edge("revision_agent", "attack_outline_critic")

    # final_compressor_formatter → END
    builder.add_edge("final_compressor_formatter", END)

    return builder.compile(checkpointer=checkpointer)


# ── Public run functions ───────────────────────────────────────────────────────

async def run_attack_outline_agent(
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
    Run the attack-outline agent to completion and return the final state.

    Args:
        request    — user request string (e.g. "Create an attack outline for these sources")
        project_id — Supabase project UUID
        source_ids — list of document_sources UUIDs to scope retrieval
        use_voyage — True if documents were ingested with voyage-law-2
        thread_id  — LangGraph checkpoint thread ID (from AgentLedgerService.initialize_run)
        job_id     — agent_jobs.id / notes.id for artifact persistence
        run_id     — agent_runs.id for progress tracking
        user_id    — auth.users.id (stored on ledger artifacts)

    Returns:
        Final AgentState dict. Key: state["final_output"] (Markdown attack outline).
    """
    initial_state = {
        "request":            request,
        "project_id":         project_id,
        "source_ids":         source_ids,
        "use_voyage":         use_voyage,
        "revision_count":     0,
        "budget":             {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        "job_id":             job_id or "",
        "run_id":             run_id or "",
        "user_id":            user_id or "",
        # initialise Annotated list fields so operator.add has a base
        "source_profiles":    [],
        "retrieval_bundles":  [],
        "raw_artifacts":      [],
        "raw_blocks":         [],
        "verification_reports": [],
        "compact_blocks":     [],
    }
    config = {"configurable": {"thread_id": thread_id or "attack-outline-agent"}}

    async with _checkpointer_ctx() as checkpointer:
        graph = _build_graph(checkpointer)
        return await graph.ainvoke(initial_state, config=config)


async def run_attack_outline_agent_stream(
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
    Stream partial state updates from the attack-outline agent.

    Yields state snapshot dicts as each node completes. The first dict
    containing "assembled_outline" signals that assembly is complete.
    The dict containing "final_output" signals the pipeline is done.

    Usage:
        async for state in run_attack_outline_agent_stream(...):
            if state.get("final_output"):
                print(state["final_output"])
                break
    """
    initial_state = {
        "request":            request,
        "project_id":         project_id,
        "source_ids":         source_ids,
        "use_voyage":         use_voyage,
        "revision_count":     0,
        "budget":             {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        "job_id":             job_id or "",
        "run_id":             run_id or "",
        "user_id":            user_id or "",
        "source_profiles":    [],
        "retrieval_bundles":  [],
        "raw_artifacts":      [],
        "raw_blocks":         [],
        "verification_reports": [],
        "compact_blocks":     [],
    }
    config = {"configurable": {"thread_id": thread_id or "attack-outline-agent-stream"}}

    async with _checkpointer_ctx() as checkpointer:
        graph = _build_graph(checkpointer)
        async for event in graph.astream(initial_state, config=config, stream_mode="values"):
            yield event
