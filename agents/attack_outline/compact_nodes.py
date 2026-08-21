# agents/attack_outline/compact_nodes.py
"""
Compact 4-stage attack-outline pipeline ("the haircut").

Replaces the 15-node graph's five serial LLM stages with one multistep
tool-calling research agent, cutting LLM roundtrips from ~9 serial stages to:

  1. plan_agent        — one worker_mid call → master job plan. Runs in
                         PARALLEL with research_agent (see graph.py) — neither
                         depends on the other's output.
  2. research_agent    — ONE agent, native tool-calling ReAct loop (bounded
                         turns, parallel tool execution). Does its own survey
                         (list_sources/get_doc_outline) rather than waiting on
                         plan_agent's job_plan. Compresses: source_profiler +
                         corpus_topic_mapper + retrieval_planner +
                         legal_artifact_extractor + concept_clusterer. Emits a
                         research dossier: clusters + per-cluster legal
                         artifact cards + evidence chunk references. Every
                         retrieval tool result is harvested into
                         state["evidence_store"] so downstream generators
                         never re-retrieve.
     sync_barrier      — no-op join: waits for BOTH plan_agent and
                         research_agent (standard LangGraph fan-in) before
                         fanning out to block_generator, so every block has
                         both job_plan and the research dossier available.
  3. block_generator   — [Send×N] one per cluster. One-stop section builder
                         (compresses attack_block_builder + doctrine_graph_
                         builder's if/then logic + the critic's grounding
                         pass): drafts the full T-14 block, framed by
                         job_plan's course structure, and self-checks every
                         rule claim against the evidence in its prompt — with
                         bounded tool access (a free local grep over the
                         research corpus, then a live DB fallback) for claims
                         the prompt's evidence doesn't cover.
  4. final_formatter   — pure Python. Deterministic assembly, blank-section
                         checks, TOC. Zero LLM calls.

Model selection for research_agent/block_generator goes through the repo's
tiered WORKER_MODEL_MAP framework (_fetch_worker_model in worker_config.py)
via _build_tool_model() below, rather than hardcoding a provider's client —
escalating to a flagship model is a provider/env-var change, not a code change.

Intermediate products live in LangGraph state (durable via the Postgres
checkpointer; observable via ledger artifacts) — no temp files, no extra
tables.

Legacy 15-node pipeline remains in nodes.py; graph.py selects between them
via ATTACK_OUTLINE_COMPACT (default true).
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Send

# Reuse the battle-tested helpers from the legacy module (same package —
# telemetry, JSON parsing, ledger writes, DeepSeek concurrency cap).
from .nodes import (
    _llm,
    _parse_json,
    _try_save_artifact,
    _node_start,
    _node_done,
    _node_warn,
    _dbg_artifact,
    _get_deepseek_semaphore,
)
from .state import AgentState

logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────

# Max RETRIEVAL turns in the research loop. Each turn may issue MANY tool calls
# (executed in parallel). The final dossier emission happens in an additional
# reserved turn on top of these — see _run_bounded_tool_loop.
MAX_RESEARCH_TURNS = int(os.getenv("ATTACK_RESEARCH_MAX_TURNS", "5"))

# Max model turns per block_generator call. One drafting turn with tool
# access, one forced-final if it reached for evidence — keeps the new tool
# capability from compounding latency across the parallel fan-out.
ATTACK_BLOCK_MAX_TURNS = int(os.getenv("ATTACK_BLOCK_MAX_TURNS", "2"))

# Per-tool-result cap injected back into the conversation. Full chunk payloads
# are harvested separately into evidence_store, so the transcript stays lean.
TOOL_RESULT_CHAR_CAP = 6000

# ── Output budget ─────────────────────────────────────────────────────────────
# Budgeted in WORDS, not tokens — a token budget is invisible to the model and
# the previous 10k-token target never bound (sections were ~3k tokens, so the
# cap sat 3x above actual output and did nothing).
#
# The numbers come from what an attack outline actually IS. Law-school prep
# guidance is consistent: an attack outline is 1-5 pages, or "up to about 10%
# of your comprehensive outline" — a memorisable checklist, not an exposition.
# A prior run produced ~20,000 words (~40+ pages) for two documents, i.e. longer
# than a full-semester comprehensive outline. That is not an attack outline by
# definition, so the budget is a PRODUCT requirement, not a latency knob (though
# it cuts latency hard as a side effect).
#
#   ~300 words/section x ~10 sections ≈ 3,000 words ≈ 5-6 pages.
BLOCK_TARGET_WORDS = int(os.getenv("ATTACK_BLOCK_TARGET_WORDS", "300"))
BLOCK_MAX_WORDS = int(os.getenv("ATTACK_BLOCK_MAX_WORDS", "400"))

# The dossier is an internal hand-off, not user-facing prose; it only needs to
# carry enough for each generator to write its section.
DOSSIER_TARGET_WORDS = int(os.getenv("ATTACK_DOSSIER_TARGET_WORDS", "1200"))

# Runaway guard only. Deliberately far above both budgets so nothing clips
# mid-structure — the prompt does the budgeting, this just bounds pathology.
HARD_OUTPUT_TOKEN_CAP = int(os.getenv("ATTACK_HARD_OUTPUT_TOKENS", "14000"))


def _word_budget_line(target: int, cap: int, kind: str = "outline") -> str:
    """Soft word budget stated in the system prompt so the model self-regulates."""
    if kind == "dossier":
        tail = (
            "This is an internal hand-off, not prose for a reader — terse "
            "fragments only, no explanation, no restating the same rule twice."
        )
    else:
        tail = (
            "This is an attack outline — a memorisable checklist a student scans "
            "under exam pressure, NOT an essay or a case brief. Compress "
            "ruthlessly: fragments, not full sentences; bullets, not paragraphs. "
            "If you are running long, cut explanation and keep the rule, the "
            "triggers, and the IF/THEN logic."
        )
    return (
        f"\n\n**LENGTH LIMIT (hard requirement)**: aim for ~{target} words; "
        f"never exceed {cap}. {tail}"
    )

# ── Corpus survey pre-fetch (R1) ──────────────────────────────────────────────
# Caps on the deterministically-injected survey, so a large project can't blow
# up the opening prompt.
PREFETCH_MAX_OUTLINES = int(os.getenv("ATTACK_PREFETCH_MAX_OUTLINES", "10"))
PREFETCH_MAX_SECTIONS = int(os.getenv("ATTACK_PREFETCH_MAX_SECTIONS", "40"))
PREFETCH_SECTION_SUMMARY_CHARS = 300

# Evidence included in each generator prompt.
GENERATOR_MAX_CHUNKS = 12
GENERATOR_CHUNK_CHAR_CAP = 1400

# NOTE: the old 9-type ARTIFACT_CARD_TYPES taxonomy is gone. research_agent no
# longer pre-extracts cards (that inflated the serial dossier and duplicated
# work); block_generator derives rule / triggers / elements / exceptions /
# defenses / traps directly from evidence, guided by its section template.


# ─────────────────────────────────────────────────────────────────────────────
# 1. plan_agent
# ─────────────────────────────────────────────────────────────────────────────

async def plan_agent(state: AgentState) -> Dict:
    """Master planner — one fast call. Same job as the legacy head_orchestrator."""
    from agents.tools.base import make_tools
    from agents.tools.registry import ATTACK_PLANNER_TOOLS

    logger.info("🔀 [%s] PARALLEL fire → plan_agent (alongside research_agent)",
                (state.get("job_id") or "")[:8] or "no-job")
    _node_start("plan_agent", state,
                n_sources=len(state.get("source_ids") or []),
                request=state.get("request", "")[:60])

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=ATTACK_PLANNER_TOOLS,
    )
    list_tool = next((t for t in tools if t.name == "list_sources"), None)
    sources_json = "[]"
    if list_tool:
        try:
            sources_json = await list_tool.ainvoke({})
        except Exception as exc:
            _node_warn("plan_agent", state, f"list_sources failed: {exc}")
    logger.info("  📋 [plan_agent] list_sources → %d chars", len(sources_json))

    system = (
        "You are the planner for a T-14 law-school attack-outline generator. "
        "Given the available source documents, create a job plan. Identify: "
        "(1) how many distinct course areas are covered, "
        "(2) the likely depth of doctrine coverage, "
        "(3) whether the outline should be single-course or multi-course, "
        "(4) the doctrine areas a research agent should investigate first.\n\n"
        "Return JSON with keys: job_type, source_ids, "
        "outline_mode (single_course|multi_course), course_areas (list), "
        "priority_doctrines (list of strings), "
        "target_format (always 'T-14 comprehensive attack outline')."
    )
    raw = await _llm(
        "worker_mid",
        f"Sources available:\n{sources_json}\n\nUser request: {state['request']}",
        system=system,
        max_tokens=1500,
        _node="plan_agent",
    )
    try:
        job_plan = _parse_json(raw)
        _dbg_artifact("plan_agent", job_plan)
    except Exception as exc:
        _node_warn("plan_agent", state, f"JSON parse failed ({exc}) — minimal plan")
        job_plan = {
            "job_type": "attack_outline",
            "source_ids": state.get("source_ids", []),
            "outline_mode": "single_course",
            "course_areas": [],
            "priority_doctrines": [],
            "target_format": "T-14 comprehensive attack outline",
        }

    await _try_save_artifact(
        state, artifact_key="job_plan", content=job_plan,
        worker_class="worker_mid", node_name="plan_agent",
        artifact_type="job_plan", source_ids=state.get("source_ids"),
    )
    _node_done("plan_agent", state,
               mode=job_plan.get("outline_mode", "?"),
               course_areas=job_plan.get("course_areas", []))
    return {"job_plan": job_plan}


# ─────────────────────────────────────────────────────────────────────────────
# 2. research_agent — multistep tool-calling loop
# ─────────────────────────────────────────────────────────────────────────────

# Cluster count scales with corpus size: a 2-page essay does not contain ten
# distinct testable doctrines, and asking for a fixed 6-10 made the model shred
# one §2 analysis into "Exclusionary Conduct", "Monopoly Maintenance" and
# "Consumer Preference" as separate sections. ~1 cluster per 10 chunks, clamped.
CHUNKS_PER_CLUSTER = int(os.getenv("ATTACK_CHUNKS_PER_CLUSTER", "10"))
MIN_CLUSTERS = 3
MAX_CLUSTERS = 10


def _target_cluster_count(n_chunks: int) -> int:
    """Cluster target for a corpus of n_chunks (0 = unknown → mid-range)."""
    if n_chunks <= 0:
        return 6
    return max(MIN_CLUSTERS, min(MAX_CLUSTERS, round(n_chunks / CHUNKS_PER_CLUSTER)))


def _research_system(n_chunks: int) -> str:
    """Build the research system prompt, scaled to the corpus actually in scope."""
    target = _target_cluster_count(n_chunks)
    corpus_note = (
        f"This corpus holds ~{n_chunks} chunks total."
        if n_chunks > 0 else "Corpus size unknown."
    )
    return (
        "You are a legal research agent building the evidence base for a T-14 "
        "attack outline. You have retrieval tools over the student's source "
        "documents. Work in bounded steps:\n"
        "  1. Orient: the CORPUS SURVEY — every source, its outline/key concepts, "
        "and a section-by-section map — is ALREADY PROVIDED in the first "
        "message. Read it and pick your doctrines from it. Do NOT spend turns "
        "re-fetching it with list_sources / get_doc_outline / get_doc_metadata / "
        "find_docs_about / find_sections_about; that data is already in front "
        "of you. Calling any of them wastes a whole turn.\n"
        "  2. Retrieve (START HERE on turn 1): hybrid_search + search_passages per "
        "doctrine (batch MANY tool calls in a single turn — they run in parallel). "
        "Drill into sections with get_section / get_neighbors where a hit looks "
        "central; the section map gives you exact section_paths to target.\n"
        "  3. Cluster: group the doctrines into outline sections and capture, for "
        "each, the black-letter rule plus the chunk_ids that prove it. You do NOT "
        "extract a card taxonomy — the section writers derive elements, "
        "exceptions, defenses and traps from the evidence themselves.\n\n"
        "RULES:\n"
        "• Your turn budget is for RETRIEVAL. Turn 1 should already be a large "
        "batch of search calls, not discovery.\n"
        "• Prefer FEW turns with MANY parallel tool calls over many small turns.\n"
        "• Only claim rules/holdings you actually retrieved — cite chunk_ids.\n"
        "• STOP CRITERION (concrete — check it every turn): once every major "
        "doctrine you identified in your survey has a black-letter rule plus "
        "supporting chunk_ids, STOP calling tools and emit the final dossier "
        "immediately, even if turns remain. Do not keep researching doctrines "
        "you've already covered just because turns are left.\n"
        f"• CLUSTER COUNT: {corpus_note} Produce about {target} clusters "
        f"(hard ceiling {MAX_CLUSTERS}). A short source does NOT contain ten "
        "distinct doctrines — if you are tempted to split one analysis into "
        "several clusters (e.g. separating 'exclusionary conduct' from "
        "'monopoly maintenance' from 'consumer preference' within one §2 "
        "claim), MERGE them. Few well-scoped clusters beat many overlapping "
        "ones.\n"
        "• EXAM RELEVANCE: cluster only doctrines a law professor would actually "
        "TEST. Prefer claims, defenses, elements, and tests. Drop pure practitioner "
        "mechanics (filing/notice/fee-award/administrative procedure) unless the "
        "course is plainly about them — they are not exam material.\n"
        "• SOURCE TYPE MATTERS: if a source is a law-review article, critique, "
        "op-ed or advocacy brief, its conclusions are ARGUMENTS, not law. Record "
        "the neutral black-letter rule as the rule, and mark the source's position "
        "in `contested_positions`. Never promote one commentator's litigation "
        "position into the rule statement.\n\n"
        "FINAL ANSWER — return ONLY this JSON object (no prose):\n"
        "{\n"
        '  "source_profiles": [{"source_id", "course_area", "document_type", "summary"}],\n'
        '  "clusters": [{\n'
        '     "cluster_id": snake_case string,\n'
        '     "label": display label (prefix Claim:/Defense:/Doctrine:/Remedy:/Procedural:),\n'
        '     "course_area": string,\n'
        '     "priority": 1|2|3,\n'
        '     "rule_statement": the neutral black-letter rule, ONE sentence (<40 words),\n'
        '     "doctrine_summary": ONE sentence,\n'
        '     "contested_positions": [short strings — positions argued by advocacy '
        'sources, or points later authority may have overtaken; [] if none],\n'
        '     "evidence_chunk_ids": up to 12 chunk_ids most central to this '
        "cluster — or every relevant chunk if the corpus holds fewer than that. "
        "This is the ONLY evidence selector the section writer gets, so include "
        "each chunk that supports the rule; never pad with ids you did not "
        'actually retrieve.,\n'
        '     "if_then_edges": [{"to_cluster", "condition"}]\n'
        "  }]\n"
        "}\n"
        "Cover EVERY major testable doctrine in the sources. Keep the dossier terse "
        "— it is an internal hand-off, not prose for a reader."
        + _word_budget_line(DOSSIER_TARGET_WORDS, int(DOSSIER_TARGET_WORDS * 1.5),
                            kind="dossier")
    )


def _harvest_evidence(tool_name: str, result_str: str, store: Dict[str, Dict]) -> None:
    """
    Capture chunk payloads returned by retrieval tools into the evidence store
    so generators can ground without re-retrieving. Best-effort — never raises.
    """
    if tool_name not in {
        "search_passages", "hybrid_search", "get_section",
        "get_neighbors", "find_concept_across_docs", "get_parents",
    }:
        return
    try:
        data = json.loads(result_str)
    except (TypeError, ValueError):
        return
    items = data if isinstance(data, list) else data.get("results", []) if isinstance(data, dict) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or item.get("id") or "")
        content = item.get("content") or item.get("text") or ""
        if chunk_id and content:
            store.setdefault(chunk_id, {
                "chunk_id": chunk_id,
                "content": content,
                "source_id": str(item.get("source_id", "")),
                "page": item.get("page_number") or item.get("page"),
            })


async def _exec_tool_call(
    tool_map: Dict[str, Any], tc: Dict[str, Any]
) -> Tuple[ToolMessage, str]:
    """Execute one tool call; errors come back as tool output, never raise."""
    name = tc.get("name", "")
    args = tc.get("args") or {}
    tool = tool_map.get(name)
    if tool is None:
        result = f"ERROR: unknown tool '{name}'"
    else:
        try:
            result = await tool.ainvoke(args)
        except Exception as exc:
            result = f"ERROR: {type(exc).__name__}: {exc}"
    result_str = result if isinstance(result, str) else json.dumps(result, default=str)
    return ToolMessage(
        content=result_str[:TOOL_RESULT_CHAR_CAP],
        tool_call_id=tc.get("id", ""),
        name=name,
    ), result_str


async def _prefetch_corpus_survey(state: Dict, tools: List[Any]) -> Tuple[str, int]:
    """
    Deterministically fetch the corpus survey so research_agent can skip
    discovery entirely and spend every turn on actual retrieval.

    Three layers, all plain indexed DB reads — no LLM, no embeddings, no
    vector search:
      1. corpus   — list_sources: filenames, chunk counts, doc summaries
      2. document — get_doc_outline per source: TOC + doc concepts
      3. section  — the section map. Prefers document_sections (2-4 sentence
                    LLM summaries from hierarchical ingest), and FALLS BACK to
                    distinct section_paths straight out of document_vector_store.

    The fallback is not an edge case, it is the normal path: ingest dispatches
    build_section_summaries fire-and-forget (upload_tasks.py) at the same moment
    the doc goes COMPLETE, and try_fire_pending_note fires ~0.8s later — so
    document_sections is reliably EMPTY at survey time for speculative uploads.
    Section paths themselves are written with the chunks, so they are always
    available; we lose the summaries but keep the structural map.

    Why any of this: the model was burning 2 of its 5 turns rediscovering this
    via list_sources / get_doc_outline / find_sections_about — data that needs
    no intelligence to fetch and whose primary keys we already hold. It also
    lands in the FIRST message, so it stays a stable, cacheable prefix instead
    of growing the transcript turn by turn.

    Returns (survey_text, n_chunks). n_chunks scales the cluster-count target;
    0 means "unknown". Never raises: any layer that fails degrades to a note,
    and the model still has the tools to fetch it the slow way.
    """
    source_ids: List[str] = list(state.get("source_ids") or [])
    parts: List[str] = []
    n_chunks = 0
    tool_by_name = {t.name: t for t in tools}

    # ── Layer 1: corpus ───────────────────────────────────────────────────
    list_tool = tool_by_name.get("list_sources")
    if list_tool is not None:
        try:
            parts.append(f"## Sources in this project\n{await list_tool.ainvoke({})}")
        except Exception as exc:
            _node_warn("research_agent", state, f"prefetch list_sources failed: {exc}")

    # ── Layer 2: per-document outlines (concurrent) ───────────────────────
    outline_tool = tool_by_name.get("get_doc_outline")
    if outline_tool is not None and source_ids:
        targets = source_ids[:PREFETCH_MAX_OUTLINES]
        results = await asyncio.gather(
            *(outline_tool.ainvoke({"source_id": sid}) for sid in targets),
            return_exceptions=True,
        )
        outlines = [
            f"### {sid}\n{res}"
            for sid, res in zip(targets, results)
            if not isinstance(res, Exception)
        ]
        if outlines:
            parts.append("## Document outlines (TOC + key concepts)\n" + "\n".join(outlines))

    # ── Layer 3: section map (+ corpus size, same round trip) ─────────────
    if source_ids:
        try:
            from tasks.database import get_global_async_db_pool, init_async_pools

            await init_async_pools()
            pool = get_global_async_db_pool()
            async with pool.acquire() as conn:
                n_chunks = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM document_vector_store
                    WHERE project_id = $1 AND source_id = ANY($2::uuid[])
                    """,
                    state["project_id"], source_ids,
                ) or 0

                rows = await conn.fetch(
                    """
                    SELECT source_id, section_path, section_summary
                    FROM document_sections
                    WHERE project_id = $1
                      AND source_id = ANY($2::uuid[])
                    ORDER BY source_id, start_chunk_idx
                    LIMIT $3
                    """,
                    state["project_id"], source_ids, PREFETCH_MAX_SECTIONS,
                )

                # Fallback: summaries are usually still being built when the
                # note fires, but section_path ships with every chunk.
                if not rows:
                    rows = await conn.fetch(
                        """
                        SELECT DISTINCT ON (source_id, section_path)
                               source_id, section_path,
                               NULL::text AS section_summary
                        FROM document_vector_store
                        WHERE project_id = $1
                          AND source_id = ANY($2::uuid[])
                          AND section_path IS NOT NULL
                          AND section_path <> ''
                        ORDER BY source_id, section_path, chunk_index
                        LIMIT $3
                        """,
                        state["project_id"], source_ids, PREFETCH_MAX_SECTIONS,
                    )
                    if rows:
                        logger.info(
                            "  📚 [research_agent] document_sections empty — "
                            "derived %d section paths from chunks instead", len(rows)
                        )

            if rows:
                lines = []
                for r in rows:
                    line = f"[{str(r['source_id'])[:8]}] {r['section_path']}"
                    summary = (r["section_summary"] or "")[:PREFETCH_SECTION_SUMMARY_CHARS]
                    if summary:
                        line += f"\n    {summary}"
                    lines.append(line)
                parts.append("## Section map\n" + "\n".join(lines))
                logger.info("  📚 [research_agent] prefetched %d section entries", len(rows))
        except Exception as exc:
            _node_warn("research_agent", state, f"prefetch section map failed: {exc}")

    if not parts:
        return (
            "(Corpus survey pre-fetch unavailable — use list_sources / "
            "get_doc_outline / find_sections_about to orient yourself first.)"
        ), n_chunks

    return (
        "=== CORPUS SURVEY (pre-fetched for you) ===\n"
        + "\n\n".join(parts)
        + "\n=== END CORPUS SURVEY ==="
    ), n_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Shared tool-calling infrastructure — used by BOTH research_agent (below) and
# block_generator (further down), not just section 2.
# ─────────────────────────────────────────────────────────────────────────────

async def _build_tool_model(worker_class: str, provider: Optional[str] = None):
    """
    Resolve a tool-bindable LangChain chat model via the tiered
    WORKER_MODEL_MAP framework (_fetch_worker_model), instead of hardcoding a
    provider's client. Escalating a node to a flagship model becomes a
    provider/env-var change, not a code change.

    Returns (base_model, semaphore). base_model is UNBOUND — callers call
    base_model.bind_tools(tools) themselves (see _run_bounded_tool_loop), so
    the same base_model instance also serves as the "no tools" model for a
    forced-final turn when the turn budget runs out.

    Only deepseek and anthropic are wired for tool-calling here; any other
    resolved provider raises rather than silently mis-binding tools.
    """
    from .worker_config import _fetch_worker_model

    resolved_provider, model_name, thinking = _fetch_worker_model(worker_class, provider)

    # HARD_OUTPUT_TOKEN_CAP sits deliberately ABOVE the soft budget stated in
    # the system prompts: the prompt does the budgeting, this is only a runaway
    # guard. Setting it near the target is what causes mid-sentence clipping.
    if resolved_provider == "deepseek":
        from utils.llm_clients.deepseek_client import DeepSeekClient
        client = DeepSeekClient(
            model_name=model_name, temperature=0.3,
            max_output_tokens=HARD_OUTPUT_TOKEN_CAP, thinking=thinking,
        )
        base_model = client._client
        sem = _get_deepseek_semaphore()
    elif resolved_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from utils.llm_clients.anthropic_rate_limits import get_llm_semaphore
        # Built directly rather than via LLMFactory.get_langchain_model() —
        # that helper hardcodes max_tokens=4096, too tight for a multi-cluster
        # JSON dossier or a full attack-outline section.
        base_model = ChatAnthropic(
            model=model_name, max_tokens=HARD_OUTPUT_TOKEN_CAP, temperature=0.3
        )
        sem = await get_llm_semaphore()
    else:
        raise NotImplementedError(
            f"_build_tool_model: provider '{resolved_provider}' is not wired "
            "for tool-calling here (only deepseek/anthropic are supported)."
        )

    logger.info("  🤖 [_build_tool_model] worker_class=%s provider=%s model=%s thinking=%s",
                worker_class, resolved_provider, model_name, thinking)
    return base_model, sem


async def _run_bounded_tool_loop(
    base_model: Any,
    tools: List[Any],
    messages: List[Any],
    max_turns: int,
    sem: Any,
    node_name: str,
    state: Dict,
    evidence_store: Optional[Dict[str, Dict]] = None,
    countdown_from_turn: int = 3,
    forced_stop_prompt: str = "STOP calling tools. Answer NOW using only what you've gathered.",
) -> str:
    """
    Shared bounded ReAct loop used by both research_agent and block_generator.

    `max_turns` is the RETRIEVAL budget. The final answer is emitted in an
    additional reserved turn on top of it, via the UNBOUND base_model (no
    tools attached, so it cannot emit further tool calls).

    The model is told about that reserved turn in advance — a countdown while
    the budget runs down, then an explicit "last retrieval turn" notice — so
    the emission is a planned hand-off rather than an interrupt mid-research.
    If the model finishes early (a turn with no tool calls), that response IS
    the answer and the reserved turn is skipped entirely.

    Returns the final answer's raw text; callers decide how to parse it
    (JSON dossier vs. markdown section).
    """
    tool_map = {t.name: t for t in tools}
    bound_model = base_model.bind_tools(tools)

    for turn in range(1, max_turns + 1):
        # Turn-budget ladder, only once the countdown window opens. Gating the
        # whole ladder on countdown_from_turn keeps short budgets quiet early:
        # block_generator runs a 2-turn budget whose common case is emitting on
        # turn 1, and a "one turn remains" nudge there would invite tool calls
        # it would not otherwise make.
        if turn >= countdown_from_turn:
            if turn == max_turns:
                messages.append(HumanMessage(content=(
                    f"(⚠ FINAL RETRIEVAL TURN — turn {turn} of {max_turns}. Make any "
                    "last tool calls now. Your NEXT response must be the final "
                    "answer itself, with no tool calls.)"
                )))
            elif turn == max_turns - 1:
                messages.append(HumanMessage(content=(
                    f"(Turn {turn} of {max_turns} — one retrieval turn remains after "
                    "this one, then you must emit the final answer. Gather anything "
                    "still missing now.)"
                )))
            else:
                remaining = max_turns - turn + 1
                messages.append(HumanMessage(content=(
                    f"({remaining} of {max_turns} retrieval turns remain — wrap up and "
                    "answer as soon as you have sufficient evidence.)"
                )))
        async with sem:
            try:
                resp: AIMessage = await bound_model.ainvoke(messages)
            except Exception as exc:
                # One retry for transient/429 blips, then bail to fallback.
                _node_warn(node_name, state, f"turn {turn} LLM error ({exc}) — retrying once")
                await asyncio.sleep(5)
                resp = await bound_model.ainvoke(messages)
        messages.append(resp)

        tool_calls = getattr(resp, "tool_calls", None) or []
        if not tool_calls:
            return resp.content

        logger.info("  🔧 [%s] turn %d/%d → %d tool call(s): %s",
                    node_name, turn, max_turns, len(tool_calls),
                    [tc.get("name") for tc in tool_calls][:8])
        results = await asyncio.gather(
            *(_exec_tool_call(tool_map, tc) for tc in tool_calls)
        )
        for (tool_msg, full_result), tc in zip(results, tool_calls):
            messages.append(tool_msg)
            if evidence_store is not None:
                _harvest_evidence(tc.get("name", ""), full_result, evidence_store)

    # Reserved emission turn. The model was warned this was coming, so this is
    # a planned hand-off, not an interrupt. Unbound model = cannot call tools.
    logger.info("  📝 [%s] reserved emission turn (retrieval budget %d/%d used)",
                node_name, max_turns, max_turns)
    messages.append(HumanMessage(content=forced_stop_prompt))
    async with sem:
        final = await base_model.ainvoke(messages)  # unbound — cannot call tools
    return final.content


async def research_agent(state: AgentState) -> Dict:
    """
    The all-in-one multistep research agent.

    A hand-rolled ReAct loop (via _run_bounded_tool_loop) rather than
    langgraph.prebuilt.create_react_agent so we keep: per-turn telemetry, a
    hard turn budget, parallel tool execution, evidence harvesting, and the
    repo's tiered WORKER_MODEL_MAP model selection.

    Runs in PARALLEL with plan_agent (see graph.py's sync_barrier) — it does
    NOT depend on job_plan; its own turn-1/2 survey stands in for that.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import ATTACK_RESEARCH_TOOLS

    logger.info("🔀 [%s] PARALLEL fire → research_agent (alongside plan_agent)",
                (state.get("job_id") or "")[:8] or "no-job")
    _node_start("research_agent", state, max_turns=MAX_RESEARCH_TURNS)

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=ATTACK_RESEARCH_TOOLS,
    )

    # "orchestrator" tier is deliberate: this is the one multistep reasoning
    # stage. Provider resolved via WORKER_MODEL_MAP — ATTACK_RESEARCH_PROVIDER
    # overrides just this node; unset, it falls through to ATTACK_AGENT_PROVIDER
    # (default deepseek, the same model the clean baseline run used).
    base_model, sem = await _build_tool_model(
        "orchestrator", provider=os.getenv("ATTACK_RESEARCH_PROVIDER")
    )

    # Deterministic corpus survey — hands the model the orientation data it used
    # to spend 2 of 5 turns rediscovering, so turn 1 starts on retrieval. The
    # chunk count comes back with it and scales the cluster-count target.
    survey, n_chunks = await _prefetch_corpus_survey(state, tools)
    target_clusters = _target_cluster_count(n_chunks)
    logger.info(
        "  📚 [research_agent] corpus survey pre-fetched (%d chars) "
        "n_chunks=%d → target_clusters=%d",
        len(survey), n_chunks, target_clusters,
    )

    messages: List[Any] = [
        SystemMessage(content=_research_system(n_chunks)),
        HumanMessage(content=(
            f"{survey}\n\n"
            f"Source ids in scope: {state.get('source_ids', [])}\n"
            f"User request: {state['request']}\n\n"
            "You already have the corpus survey above — do not re-fetch it. "
            "Begin RETRIEVAL immediately: issue a large batch of parallel "
            "search_passages / hybrid_search calls for the doctrines you can "
            "already see in the survey."
        )),
    ]

    evidence_store: Dict[str, Dict] = {}
    raw_answer = await _run_bounded_tool_loop(
        base_model, tools, messages, MAX_RESEARCH_TURNS, sem,
        "research_agent", state, evidence_store=evidence_store,
        countdown_from_turn=3,
        forced_stop_prompt=(
            "Retrieval is complete. Emit the final dossier JSON now, using the "
            "evidence gathered. Return ONLY the JSON object."
        ),
    )
    try:
        dossier: Optional[Dict[str, Any]] = _parse_json(raw_answer)
    except Exception as exc:
        _node_warn("research_agent", state, f"dossier JSON parse failed: {exc}")
        dossier = None

    if not isinstance(dossier, dict):
        dossier = {"source_profiles": [], "clusters": []}

    clusters = [c for c in (dossier.get("clusters") or []) if isinstance(c, dict)]
    dossier["clusters"] = clusters

    _dbg_artifact("research_agent", dossier)
    await _try_save_artifact(
        state, artifact_key="research_dossier",
        content={"dossier": dossier, "evidence_chunks": len(evidence_store)},
        worker_class="orchestrator", node_name="research_agent",
        artifact_type="research_dossier", source_ids=state.get("source_ids"),
    )
    _node_done("research_agent", state,
               n_clusters=len(clusters),
               n_evidence_chunks=len(evidence_store),
               labels=[c.get("label", "?")[:30] for c in clusters[:6]])
    return {"research_dossier": dossier, "evidence_store": evidence_store}


# ─────────────────────────────────────────────────────────────────────────────
# sync_barrier — join point for the parallel plan_agent / research_agent branches
# ─────────────────────────────────────────────────────────────────────────────

async def sync_barrier(state: AgentState) -> Dict:
    """
    Join node. plan_agent and research_agent run in parallel from START
    (graph.py) since research_agent doesn't depend on job_plan — but
    block_generator needs BOTH job_plan and the research dossier. A node with
    two incoming static edges waits for both predecessors before running
    (standard LangGraph fan-in), so routing the research_to_generators
    conditional edge from here — instead of directly from research_agent —
    guarantees job_plan is always present in block_generator's Send payload.

    Logs proof the join actually waited for both branches (not just that it
    ran) — job_plan present and non-empty is the tell; if it were ever False
    here, the parallel-entry edges wouldn't be firing as designed.
    """
    job = (state.get("job_id") or "")[:8] or "no-job"
    has_plan = bool(state.get("job_plan"))
    n_clusters = len((state.get("research_dossier") or {}).get("clusters") or [])
    logger.info(
        "🔗 [%s] sync_barrier — joined plan_agent + research_agent  "
        "job_plan=%s research_clusters=%d",
        job, "present" if has_plan else "MISSING (join failed?)", n_clusters,
    )
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 3. block_generator — [Send×N] one per cluster
# ─────────────────────────────────────────────────────────────────────────────

def research_to_generators(state: AgentState):
    """Fan-out: one block_generator per dossier cluster; skip to formatter if none."""
    clusters = (state.get("research_dossier") or {}).get("clusters") or []
    if not clusters:
        logger.warning("research_to_generators: no clusters — routing straight to final_formatter")
        return "final_formatter"
    logger.info("attack_outline(compact) x%d node fan out for block_generator", len(clusters))
    return [Send("block_generator", {"cluster": c, **state}) for c in clusters]


async def block_generator(state: Dict) -> Dict:
    """
    One-stop section builder for a single cluster.
    Drafts the full T-14 attack block AND self-grounds it against the
    evidence chunks in its prompt — compressing builder + doctrine-graph
    if/then logic + the critic's grounding pass.

    Has bounded tool access (ATTACK_BLOCK_MAX_TURNS turns): a free local grep
    over the full research corpus, then a live DB fallback to reach primary
    source, for claims the prompt's evidence doesn't already cover.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import ATTACK_VERIFIER_TOOLS
    from langchain_core.tools import tool as _tool_decorator

    cluster: Dict[str, Any] = state["cluster"]
    job_plan: Dict[str, Any] = state.get("job_plan") or {}
    evidence_store: Dict[str, Dict] = state.get("evidence_store") or {}
    label = cluster.get("label", cluster.get("cluster_id", "?"))

    _node_start("block_generator", state, cluster=label[:40],
                n_chunk_ids=len(cluster.get("evidence_chunk_ids") or []))

    # Evidence selection. The dossier no longer carries artifact cards (their
    # chunk_ids used to be a secondary source here), so evidence_chunk_ids is
    # now the ONLY selector — hence the prompt asks research_agent for 10-12.
    chunk_ids: List[str] = list(cluster.get("evidence_chunk_ids") or [])
    seen = set()
    evidence: List[Dict] = []
    for cid in chunk_ids:
        if cid in seen or cid not in evidence_store:
            continue
        seen.add(cid)
        evidence.append(evidence_store[cid])
        if len(evidence) >= GENERATOR_MAX_CHUNKS:
            break

    evidence_text = "\n\n".join(
        f"[{e['chunk_id']}] (source {e.get('source_id', '?')[:8]}, p.{e.get('page', '?')})\n"
        f"{e['content'][:GENERATOR_CHUNK_CHAR_CAP]}"
        for e in evidence
    ) or "(no evidence chunks captured — use grep_research_corpus to find support)"

    # Local, zero-cost grep over the FULL research corpus (not just this
    # cluster's cited chunks) — closure-defined so it needs no ToolContext and
    # makes no DB round trip.
    @_tool_decorator
    def grep_research_corpus(keyword: str) -> str:
        """Case-insensitive substring search over the research corpus already
        harvested by research_agent (every chunk retrieved for this outline,
        not just the ones cited by this cluster). Use this FIRST — it's free
        and instant — before reaching for a live database tool. Returns up to
        10 matching chunk excerpts with their chunk_id."""
        kw = keyword.lower()
        matches = [
            {"chunk_id": cid, "excerpt": e["content"][:400]}
            for cid, e in evidence_store.items()
            if kw in e["content"].lower()
        ][:10]
        return json.dumps(matches) if matches else json.dumps({"result": "no matches"})

    verifier_tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=[t for t in ATTACK_VERIFIER_TOOLS
                    if t in ("find_supporting_evidence", "get_citations_for")],
    )
    tools = [grep_research_corpus] + verifier_tools

    job_plan_ctx = (
        f"Outline mode: {job_plan.get('outline_mode', 'single_course')}\n"
        f"Course areas: {job_plan.get('course_areas', [])}\n"
        f"Target format: {job_plan.get('target_format', 'T-14 comprehensive attack outline')}\n\n"
    ) if job_plan else ""

    system = (
        "You are a T-14 law student writing ONE section of an ATTACK OUTLINE — "
        "the condensed checklist you scan during a timed exam, not a study "
        "guide and not a case brief. Every line must be something you would "
        "actually use with 40 minutes on the clock.\n\n"
        "Produce EXACTLY this markdown skeleton — no extra headings, no prose "
        "sections, nothing outside it:\n\n"
        "## {label}\n"
        "**RULE:** one sentence of black-letter law (<40 words).\n"
        "**TRIGGERS:** 3-5 bullet fragments — the fact patterns that raise this "
        "issue. Fragments, not sentences.\n"
        "**ELEMENTS / STEPS:** numbered list. One line per element. Format each "
        "as `Element — key question` and, where it decides the outcome, add a "
        "nested `IF … → THEN …` line. No argument paragraphs.\n"
        "**EXCEPTIONS & DEFENSES:** bullet fragments, one per exception or "
        "defense. Name it; do not argue it.\n"
        "**TRAPS:** 2-4 bullets — the specific errors students make here.\n"
        "**IF/THEN → NEXT:** one line per cross-reference to another section.\n\n"
        "STYLE RULES (these are what make it an attack outline):\n"
        "• Fragments over sentences. Cut every article and filler word you can.\n"
        "• NO 'For / Against / Rebuttal' blocks. NO model answer paragraph. NO "
        "block quotes. NO restating the rule in multiple places.\n"
        "• A student must be able to read the whole section in ~30 seconds.\n"
        "• Omit any heading that has no real content — do not pad.\n\n"
        "NEUTRALITY: state the black-letter rule as the RULE. If a source is a "
        "law-review article, critique, op-ed or brief, its conclusions are one "
        "side's ARGUMENT — put those under EXCEPTIONS & DEFENSES and attribute "
        "them ('critics argue…'), never as the rule. If the cluster lists "
        "contested_positions, treat them that way.\n\n"
        "GROUNDING: every rule, element and case must be supported by the "
        "evidence excerpts provided. If something you want to state isn't "
        "covered, call grep_research_corpus first (free, instant); fall back to "
        "find_supporting_evidence / get_citations_for only if that comes up "
        "empty — most sections need no tool calls at all. Cite support inline "
        "as [chunk_id] immediately after the proposition it supports (these are "
        "stripped before the student sees them, so they cost you no length). "
        "Do not invent authority.\n\n"
        "Return ONLY the markdown section, starting with '## ' — no other text."
        + _word_budget_line(BLOCK_TARGET_WORDS, BLOCK_MAX_WORDS)
    )
    prompt = (
        f"{job_plan_ctx}"
        f"Cluster:\n{json.dumps(cluster, indent=2)}\n\n"
        f"Evidence excerpts:\n{evidence_text}"
    )

    # orchestrator tier is deliberate: the reasoning stage where drafting +
    # self-grounding happen together. Provider resolved via WORKER_MODEL_MAP.
    base_model, sem = await _build_tool_model("orchestrator")
    messages: List[Any] = [SystemMessage(content=system), HumanMessage(content=prompt)]
    markdown = await _run_bounded_tool_loop(
        base_model, tools, messages, ATTACK_BLOCK_MAX_TURNS, sem,
        "block_generator", state,
        countdown_from_turn=ATTACK_BLOCK_MAX_TURNS,
        forced_stop_prompt=(
            "STOP calling tools. Write the final markdown section NOW using "
            "only the evidence already gathered."
        ),
    )
    markdown = (markdown or "").strip()
    if not markdown.startswith("##"):
        markdown = f"## {label}\n\n{markdown}"

    block = {
        "cluster_id": cluster.get("cluster_id", ""),
        "label": label,
        "course_area": cluster.get("course_area", ""),
        "priority": cluster.get("priority", 2),
        "markdown": markdown,
    }
    await _try_save_artifact(
        state, artifact_key=f"compact_block:{block['cluster_id']}",
        content=block, worker_class="orchestrator", node_name="block_generator",
        artifact_type="attack_block", source_ids=state.get("source_ids"),
    )
    _node_done("block_generator", state, cluster=label[:40], chars=len(markdown))
    return {"compact_blocks": [block]}


block_generator.default_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 4. final_formatter — deterministic, zero LLM
# ─────────────────────────────────────────────────────────────────────────────

_MIN_SECTION_CHARS = 200

# ── Chunk-citation stripping ──────────────────────────────────────────────────
# block_generator cites evidence inline as [<chunk uuid>] so its self-grounding
# pass is checkable. Those are internal retrieval IDs — meaningless to a student
# and visually wrecking — so they are stripped here, at the presentation
# boundary. The RAW markdown (citations intact) stays in state["compact_blocks"]
# and in the ledger artifact, so grounding remains auditable after the fact.
_UUID_PAT = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
# Matches [uuid], [uuid; uuid], [uuid, uuid] and runs of adjacent brackets.
# Leading whitespace is [ \t] only — never \s — so a citation at the start of a
# line cannot swallow the preceding newline and weld two lines together.
_CHUNK_CITE_RE = re.compile(
    r"[ \t]*\[\s*" + _UUID_PAT + r"(?:\s*[;,]\s*" + _UUID_PAT + r")*\s*\]"
)


def _strip_chunk_citations(markdown: str) -> Tuple[str, int]:
    """Remove inline [chunk-uuid] citations and tidy the punctuation they leave
    behind. Returns (cleaned_markdown, n_citations_removed)."""
    if not markdown:
        return markdown, 0
    n = len(_CHUNK_CITE_RE.findall(markdown))
    cleaned = _CHUNK_CITE_RE.sub("", markdown)
    cleaned = re.sub(r"[ \t]+([.,;:!?)])", r"\1", cleaned)  # " ." -> "."
    cleaned = re.sub(r"\(\s*\)", "", cleaned)               # empty parens
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)            # collapse space runs
    cleaned = re.sub(r"[ \t]+$", "", cleaned, flags=re.M)   # trailing space
    return cleaned, n


async def final_formatter(state: AgentState) -> Dict:
    """Deterministic assembly: order, stitch, TOC, blank-section checks."""
    job_plan = state.get("job_plan") or {}
    dossier = state.get("research_dossier") or {}
    blocks: List[Dict] = list(state.get("compact_blocks") or [])

    _node_start("final_formatter", state, n_blocks=len(blocks))

    # Order: course_area, then priority, then label — stable and deterministic.
    blocks.sort(key=lambda b: (b.get("course_area", ""), b.get("priority", 2), b.get("label", "")))

    course_areas = job_plan.get("course_areas") or sorted(
        {b.get("course_area", "") for b in blocks if b.get("course_area")}
    )
    title_suffix = " — ".join(course_areas[:3]) if course_areas else "Attack Outline"

    # Strip internal chunk-uuid citations BEFORE the size check, so the floor is
    # measured against what the student actually sees rather than against
    # citation noise. `clean_markdown` is used for rendering only; b["markdown"]
    # keeps its citations for the ledger/audit trail.
    citations_removed = 0
    for b in blocks:
        cleaned, n = _strip_chunk_citations(b.get("markdown", ""))
        b["clean_markdown"] = cleaned
        citations_removed += n
    if citations_removed:
        logger.info("  🧹 [final_formatter] stripped %d inline chunk citation(s)",
                    citations_removed)

    kept, dropped = [], []
    for b in blocks:
        if len(b.get("clean_markdown", "").strip()) >= _MIN_SECTION_CHARS:
            kept.append(b)
        else:
            dropped.append(b.get("label", "?"))
    if dropped:
        _node_warn("final_formatter", state, f"dropped {len(dropped)} near-empty section(s): {dropped}")

    lines: List[str] = [f"# Attack Outline: {title_suffix}", ""]
    if kept:
        lines.append("## Contents")
        lines += [f"{i}. {b['label']}" for i, b in enumerate(kept, 1)]
        lines.append("")
        current_area = None
        for b in kept:
            area = b.get("course_area", "")
            if area and area != current_area and len(course_areas) > 1:
                lines += [f"# {area}", ""]
                current_area = area
            lines += [b["clean_markdown"].strip(), "", "---", ""]
    else:
        # Total generation failure — surface what research found instead of
        # returning an empty note.
        lines.append("*(Outline generation produced no sections — research summary below.)*")
        for c in dossier.get("clusters") or []:
            summary, _ = _strip_chunk_citations(
                c.get("rule_statement") or c.get("doctrine_summary", "")
            )
            lines += [f"## {c.get('label', '?')}", summary, ""]

    lines.append(
        f"<!-- attack-outline compact pipeline | {len(kept)} sections | "
        f"{len((dossier.get('clusters') or []))} clusters researched -->"
    )
    final_md = "\n".join(lines)

    await _try_save_artifact(
        state, artifact_key="final_output",
        content={"markdown_length": len(final_md), "sections": len(kept)},
        worker_class="tool_only", node_name="final_formatter",
        artifact_type="final_output", source_ids=state.get("source_ids"),
    )
    _node_done("final_formatter", state, sections=len(kept), total_chars=len(final_md))
    # assembled_outline kept for streaming-consumer compatibility.
    return {"final_output": final_md, "assembled_outline": final_md}
