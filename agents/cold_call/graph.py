# agents/cold_call/graph.py
"""
LangGraph StateGraph for the cold-call agent.

Graph topology (14 nodes):

Phase 1 — Source Understanding:
  head_orchestrator          → [Send×N] source_profiler
  source_profiler            → corpus_synthesizer              (merge all N)
  corpus_synthesizer         → [Send×N] case_rule_extractor    (one per identified case)
  case_rule_extractor        → doctrine_mapper                  (merge all N)
  doctrine_mapper            → compare_distinguish_mapper       (sequential)
  compare_distinguish_mapper → question_type_bank_selector

Phase 2 — Seed Generation with Diversity Retry:
  question_type_bank_selector → [Send×N] cold_call_seed_generator (one per case)
  cold_call_seed_generator    → seed_diversity_agent              (merge all N)
  seed_diversity_agent        → (seeds_routing?)
    → [Send×N] cold_call_seed_generator  (retry, max MAX_SEED_RETRIES=2)
    → [Send×N] socratic_thread_builder   (one per approved seed + compare prompts)

Phase 3 — Answer + QA:
  socratic_thread_builder → [Send×N] socratic_answer_agent   (one per sequence)
  socratic_answer_agent   → [Send×N] grounder_agent          (one per answer seq)
  grounder_agent          → critic_coverage_agent             (merge all N)
  critic_coverage_agent   → formatter_export_agent
  formatter_export_agent  → END

Checkpointing:
  AsyncPostgresSaver (durable) with MemorySaver fallback.

Usage:
  result = await run_cold_call_agent(request, project_id, source_ids, ...)
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

USE_COLD_CALL_AGENT = os.getenv("USE_COLD_CALL_AGENT", "false").lower() == "true"

# Compact 4-stage pipeline (compact_nodes.py): plan ∥ research (tool-loop) →
# sync_barrier → [Send×N] sequence_generator → deterministic formatter.
# Default ON; set COLD_CALL_COMPACT=false to fall back to the legacy 14-node
# graph, retained in nodes.py for rollback.
COLD_CALL_COMPACT = os.getenv("COLD_CALL_COMPACT", "true").lower() == "true"


# ── Checkpointer factory ───────────────────────────────────────────────────────

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
    Compact 4-stage topology (COLD_CALL_COMPACT=true, the default):

      START ──┬─▶ plan_agent ─────┐
              └─▶ research_agent ─┴─▶ sync_barrier ─▶ [Send×N] sequence_generator
                                                        → final_formatter → END
                                    └──(no cases)────────→ final_formatter

    plan_agent and research_agent run in PARALLEL (research does its own corpus
    survey rather than waiting on job_plan). sync_barrier is a no-op join so
    every generator sees both, and it keeps conditional edges off Send-parallel
    nodes — the legacy graph hung routers directly on fan-out nodes.

    Sequence diversity is assigned deterministically at fan-out (theme + case
    round-robin) rather than by generating 3x the seeds and paying an LLM to
    prune them.
    """
    from langgraph.graph import StateGraph, START, END

    from .state import AgentState
    from .compact_nodes import (
        plan_agent,
        research_agent,
        sync_barrier,
        sequence_generator,
        final_formatter,
        dossier_to_generators,
    )

    builder = StateGraph(AgentState)
    builder.add_node("plan_agent",         plan_agent)
    builder.add_node("research_agent",     research_agent)
    builder.add_node("sync_barrier",       sync_barrier)
    builder.add_node("sequence_generator", sequence_generator)
    builder.add_node("final_formatter",    final_formatter)

    builder.add_edge(START, "plan_agent")
    builder.add_edge(START, "research_agent")
    builder.add_edge("plan_agent", "sync_barrier")
    builder.add_edge("research_agent", "sync_barrier")
    builder.add_conditional_edges(
        "sync_barrier",
        dossier_to_generators,
        {"final_formatter": "final_formatter"},
    )
    builder.add_edge("sequence_generator", "final_formatter")
    builder.add_edge("final_formatter", END)

    return builder.compile(checkpointer=checkpointer)


def _build_graph(checkpointer):
    if COLD_CALL_COMPACT:
        return _build_compact_graph(checkpointer)
    from langgraph.graph import StateGraph, END

    from .state import AgentState
    from .nodes import (
        # ── nodes ──────────────────────────────────────────────────────────
        head_orchestrator,
        source_profiler,
        corpus_synthesizer,
        case_rule_extractor,
        doctrine_mapper,
        compare_distinguish_mapper,
        question_type_bank_selector,
        cold_call_seed_generator,
        seed_diversity_agent,
        socratic_thread_builder,
        socratic_answer_agent,
        grounder_agent,
        critic_coverage_agent,
        formatter_export_agent,
        # ── routing helpers ────────────────────────────────────────────────
        head_orchestrator_to_profiler,
        corpus_synthesizer_to_extractor,
        type_selector_to_seed_generators,
        seeds_routing,
        thread_builders_to_answer_agents,
        answer_agents_to_grounders,
    )

    builder = StateGraph(AgentState)

    # ── register nodes ────────────────────────────────────────────────────────
    builder.add_node("head_orchestrator",           head_orchestrator)
    builder.add_node("source_profiler",             source_profiler)
    builder.add_node("corpus_synthesizer",          corpus_synthesizer)
    builder.add_node("case_rule_extractor",         case_rule_extractor)
    builder.add_node("doctrine_mapper",             doctrine_mapper)
    builder.add_node("compare_distinguish_mapper",  compare_distinguish_mapper)
    builder.add_node("question_type_bank_selector", question_type_bank_selector)
    builder.add_node("cold_call_seed_generator",    cold_call_seed_generator)
    builder.add_node("seed_diversity_agent",        seed_diversity_agent)
    builder.add_node("socratic_thread_builder",     socratic_thread_builder)
    builder.add_node("socratic_answer_agent",       socratic_answer_agent)
    builder.add_node("grounder_agent",              grounder_agent)
    builder.add_node("critic_coverage_agent",       critic_coverage_agent)
    builder.add_node("formatter_export_agent",      formatter_export_agent)

    # ── entry ─────────────────────────────────────────────────────────────────
    builder.set_entry_point("head_orchestrator")

    # ── Phase 1: Source Understanding ─────────────────────────────────────────

    # head_orchestrator → parallel source_profiler (one per source doc)
    builder.add_conditional_edges("head_orchestrator", head_orchestrator_to_profiler)

    # all source_profiler branches merge → corpus_synthesizer
    builder.add_edge("source_profiler", "corpus_synthesizer")

    # corpus_synthesizer → parallel case_rule_extractor (one per identified case)
    builder.add_conditional_edges("corpus_synthesizer", corpus_synthesizer_to_extractor)

    # all case_rule_extractor branches merge → doctrine_mapper
    builder.add_edge("case_rule_extractor", "doctrine_mapper")

    # doctrine_mapper → compare_distinguish_mapper (sequential)
    builder.add_edge("doctrine_mapper", "compare_distinguish_mapper")

    # compare_distinguish_mapper → question_type_bank_selector
    builder.add_edge("compare_distinguish_mapper", "question_type_bank_selector")

    # ── Phase 2: Seed Generation with Diversity Retry ─────────────────────────

    # question_type_bank_selector → parallel cold_call_seed_generator (one per case)
    builder.add_conditional_edges("question_type_bank_selector", type_selector_to_seed_generators)

    # all cold_call_seed_generator branches merge → seed_diversity_agent
    builder.add_edge("cold_call_seed_generator", "seed_diversity_agent")

    # seed_diversity_agent → conditional:
    #   needs retry → cold_call_seed_generator (fan-out again)
    #   approved    → socratic_thread_builder (fan-out per seed)
    builder.add_conditional_edges("seed_diversity_agent", seeds_routing)

    # ── Phase 3: Answer + QA ──────────────────────────────────────────────────

    # all socratic_thread_builder branches merge → parallel socratic_answer_agent
    builder.add_conditional_edges("socratic_thread_builder", thread_builders_to_answer_agents)

    # all socratic_answer_agent branches merge → parallel grounder_agent
    builder.add_conditional_edges("socratic_answer_agent", answer_agents_to_grounders)

    # all grounder_agent branches merge → critic_coverage_agent
    builder.add_edge("grounder_agent", "critic_coverage_agent")

    # critic_coverage_agent → formatter_export_agent
    builder.add_edge("critic_coverage_agent", "formatter_export_agent")

    # formatter_export_agent → END
    builder.add_edge("formatter_export_agent", END)

    return builder.compile(checkpointer=checkpointer)


# ── Public run functions ───────────────────────────────────────────────────────

async def run_cold_call_agent(
    request: str,
    project_id: str,
    source_ids: List[str],
    requested_sequence_count: int = 3,
    target_difficulty: str = "day_one_t14",
    use_voyage: bool = False,
    thread_id: Optional[str] = None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    note_id: Optional[str] = None,
) -> Dict:
    """
    Run the cold-call agent to completion and return the final state.

    Args:
        request                 — user request string
        project_id              — Supabase project UUID
        source_ids              — list of document_sources UUIDs to scope retrieval
        requested_sequence_count — number of question sequences to generate (default 3)
        target_difficulty       — 'law_1l' | 'day_one_t14' | 'advanced'
        use_voyage              — True if documents were ingested with voyage-law-2
        thread_id               — LangGraph checkpoint thread ID
        job_id                  — agent_jobs.id / notes.id for artifact persistence
        run_id                  — agent_runs.id for progress tracking
        user_id                 — auth.users.id
        note_id                 — notes.id to link cold_call_sequences rows

    Returns:
        Final AgentState dict. Key: state["final_output"] (Markdown summary).
    """
    initial_state = {
        "request":                  request,
        "project_id":               project_id,
        "source_ids":               source_ids,
        "note_id":                  note_id or job_id or "",
        "requested_sequence_count": requested_sequence_count,
        "target_difficulty":        target_difficulty,
        "use_voyage":               use_voyage,
        "job_id":                   job_id or "",
        "run_id":                   run_id or "",
        "user_id":                  user_id or "",
        "revision_count":           0,
        "seed_attempt_count":       0,
        "budget":                   {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        # initialise Annotated list fields so operator.add has a base
        "source_profiles":          [],
        "case_rule_objects":        [],
        "seeds":                    [],
        "socratic_sequences":       [],
        "answer_sequences":         [],
        "grounding_results":        [],
        "generated_sequences":      [],
    }
    config = {"configurable": {"thread_id": thread_id or "cold-call-agent"}}

    async with _checkpointer_ctx() as checkpointer:
        graph = _build_graph(checkpointer)
        return await graph.ainvoke(initial_state, config=config)


async def run_cold_call_agent_stream(
    request: str,
    project_id: str,
    source_ids: List[str],
    requested_sequence_count: int = 3,
    target_difficulty: str = "day_one_t14",
    use_voyage: bool = False,
    thread_id: Optional[str] = None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    note_id: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Stream partial state updates from the cold-call agent.

    Yields state snapshot dicts as each node completes.
    The dict containing "final_output" signals the pipeline is done.
    """
    initial_state = {
        "request":                  request,
        "project_id":               project_id,
        "source_ids":               source_ids,
        "note_id":                  note_id or job_id or "",
        "requested_sequence_count": requested_sequence_count,
        "target_difficulty":        target_difficulty,
        "use_voyage":               use_voyage,
        "job_id":                   job_id or "",
        "run_id":                   run_id or "",
        "user_id":                  user_id or "",
        "seed_attempt_count":       0,
        "budget":                   {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        "source_profiles":          [],
        "case_rule_objects":        [],
        "seeds":                    [],
        "socratic_sequences":       [],
        "answer_sequences":         [],
        "grounding_results":        [],
        "generated_sequences":      [],
    }
    config = {"configurable": {"thread_id": thread_id or "cold-call-agent-stream"}}

    async with _checkpointer_ctx() as checkpointer:
        graph = _build_graph(checkpointer)
        async for event in graph.astream(initial_state, config=config, stream_mode="values"):
            yield event
