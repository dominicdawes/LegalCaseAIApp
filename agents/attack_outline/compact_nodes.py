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
# SOFT target is stated in the system prompt so the model self-regulates and
# finishes its structure; the HARD cap sits well above it purely as a runaway
# guard. Keeping the hard cap above the soft target is what prevents a
# mid-sentence clip (the whole point of budgeting by prompt, not by max_tokens).
SOFT_OUTPUT_TOKEN_TARGET = int(os.getenv("ATTACK_SOFT_OUTPUT_TOKENS", "10000"))
HARD_OUTPUT_TOKEN_CAP = int(os.getenv("ATTACK_HARD_OUTPUT_TOKENS", "14000"))

_SOFT_BUDGET_LINE = (
    f"\n\n**IMPORTANT**: keep your total response under ~{SOFT_OUTPUT_TOKEN_TARGET:,} "
    "tokens. This is a soft budget — prefer tightening wording over dropping "
    "required structure, and always finish the element you started rather than "
    "stopping mid-way."
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

ARTIFACT_CARD_TYPES = (
    "rule_card, element_card, exception_card, issue_trigger_card, "
    "defense_card, case_card, policy_card, remedy_card, exam_trap_card"
)


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

_RESEARCH_SYSTEM = (
    "You are a legal research agent building the evidence base for a T-14 "
    "attack outline. You have retrieval tools over the student's source "
    "documents. Work in bounded steps:\n"
    "  1. Orient: the CORPUS SURVEY — every source, its outline/key concepts, "
    "and a section-by-section summary map — is ALREADY PROVIDED in the first "
    "message. Read it and pick your doctrines from it. Do NOT spend turns "
    "re-fetching it with list_sources / get_doc_outline / find_docs_about / "
    "find_sections_about; that data is already in front of you.\n"
    "  2. Retrieve (START HERE on turn 1): hybrid_search + search_passages per "
    "doctrine (batch MANY tool calls in a single turn — they run in parallel). "
    "Drill into sections with get_section / get_neighbors where a hit looks "
    "central; the section map gives you exact section_paths to target.\n"
    "  3. Extract & cluster: identify every legal artifact present "
    f"({ARTIFACT_CARD_TYPES}) and group doctrines into outline clusters.\n\n"
    "RULES:\n"
    "• Your turn budget is for RETRIEVAL. Turn 1 should already be a large "
    "batch of search calls, not discovery.\n"
    "• Prefer FEW turns with MANY parallel tool calls over many small turns.\n"
    "• Only claim rules/holdings you actually retrieved — cite chunk_ids.\n"
    "• STOP CRITERION (concrete — check it every turn): once every major "
    "doctrine you identified in your survey has at least 1-2 grounded "
    "artifacts, STOP calling tools and emit the final dossier immediately, "
    "even if turns remain. Do not keep researching doctrines you've already "
    "covered just because turns are left.\n"
    "• Target 6-10 clusters total; do NOT exceed 12. If tempted to split one "
    "doctrine into multiple clusters, prefer merging — fewer, well-scoped "
    "clusters beat many overlapping ones.\n\n"
    "FINAL ANSWER — return ONLY this JSON object (no prose):\n"
    "{\n"
    '  "source_profiles": [{"source_id", "course_area", "document_type", "summary"}],\n'
    '  "clusters": [{\n'
    '     "cluster_id": snake_case string,\n'
    '     "label": display label (prefix Claim:/Defense:/Doctrine:/Remedy:/Procedural:),\n'
    '     "course_area": string,\n'
    '     "priority": 1|2|3,\n'
    '     "doctrine_summary": 2-3 sentences,\n'
    '     "artifacts": [{"artifact_type": one of the card types, "text": the '
    'card content, "elements": [..], "exceptions": [..], "chunk_ids": [..]}],\n'
    '     "evidence_chunk_ids": chunk_ids most central to this cluster,\n'
    '     "if_then_edges": [{"to_cluster", "condition"}]\n'
    "  }]\n"
    "}\n"
    "Keep every artifact text under 120 words. Cover EVERY major doctrine in "
    "the sources — completeness beats depth here; the outline writers deepen "
    "each cluster later."
    + _SOFT_BUDGET_LINE
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


async def _prefetch_corpus_survey(state: Dict, tools: List[Any]) -> str:
    """
    Deterministically fetch the corpus survey so research_agent can skip
    discovery entirely and spend every turn on actual retrieval.

    Three layers, all plain indexed DB reads — no LLM, no embeddings, no
    vector search:
      1. corpus   — list_sources: filenames, chunk counts, doc summaries
      2. document — get_doc_outline per source: TOC + doc concepts
      3. section  — document_sections: every section_path plus the 2-4 sentence
                    summary generated during hierarchical ingest
                    (tasks/hierarchical_ingest_tasks.py::build_section_summaries)

    Why: the model was burning 2 of its 5 turns rediscovering exactly this via
    list_sources / get_doc_outline / find_sections_about — data that requires
    no intelligence to fetch and that we already have primary keys for. It also
    lands in the FIRST message, so it stays a stable, cacheable prefix instead
    of growing the transcript turn by turn.

    Never raises: any layer that fails degrades to a note, and the model still
    has the tools to fetch it the slow way.
    """
    source_ids: List[str] = list(state.get("source_ids") or [])
    parts: List[str] = []
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

    # ── Layer 3: hierarchical section summaries ───────────────────────────
    if source_ids:
        try:
            from tasks.database import get_global_async_db_pool, init_async_pools

            await init_async_pools()
            pool = get_global_async_db_pool()
            async with pool.acquire() as conn:
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
            if rows:
                lines = [
                    f"[{str(r['source_id'])[:8]}] {r['section_path']}\n"
                    f"    {(r['section_summary'] or '')[:PREFETCH_SECTION_SUMMARY_CHARS]}"
                    for r in rows
                ]
                parts.append(
                    "## Section map (hierarchical summaries built at ingest)\n"
                    + "\n".join(lines)
                )
                logger.info("  📚 [research_agent] prefetched %d section summaries", len(rows))
        except Exception as exc:
            _node_warn("research_agent", state, f"prefetch section summaries failed: {exc}")

    if not parts:
        return (
            "(Corpus survey pre-fetch unavailable — use list_sources / "
            "get_doc_outline / find_sections_about to orient yourself first.)"
        )

    return (
        "=== CORPUS SURVEY (pre-fetched for you) ===\n"
        + "\n\n".join(parts)
        + "\n=== END CORPUS SURVEY ==="
    )


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

    # Deterministic corpus survey (R1) — hands the model the orientation data
    # it used to spend 2 of 5 turns rediscovering, so turn 1 starts on retrieval.
    survey = await _prefetch_corpus_survey(state, tools)
    logger.info("  📚 [research_agent] corpus survey pre-fetched (%d chars)", len(survey))

    messages: List[Any] = [
        SystemMessage(content=_RESEARCH_SYSTEM),
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
                n_artifacts=len(cluster.get("artifacts") or []))

    # Evidence: prefer the cluster's own chunk ids, then artifact-cited ids.
    chunk_ids: List[str] = list(cluster.get("evidence_chunk_ids") or [])
    for art in cluster.get("artifacts") or []:
        chunk_ids.extend(art.get("chunk_ids") or [])
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
    ) or "(no evidence chunks captured — rely strictly on the artifact cards)"

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
        "You are a T-14 law student writing ONE section of an attack outline. "
        "Produce the complete markdown section for this doctrine cluster:\n\n"
        "## {label}\n"
        "**Big exam takeaway** — 1-2 sentences.\n"
        "**Exam-ready rule statement** — one complete sentence.\n"
        "**Issue triggers** — fact-pattern signals.\n"
        "**Attack steps** — numbered; each step: the element/test, key facts "
        "for and against, and IF/THEN logic lines.\n"
        "**Exceptions & limits**\n"
        "**Defenses & counterarguments**\n"
        "**Remedies** (when applicable)\n"
        "**Exam traps**\n"
        "**One-paragraph model application**\n"
        "**If/then transitions** — one line per related cluster edge.\n\n"
        "Write this section consistent with the overall outline's course "
        "structure (given below).\n\n"
        "GROUNDING (critical): every rule, element, holding, and case you state "
        "must be supported by the artifact cards or the evidence excerpts "
        "provided. If a claim you want to make ISN'T covered by what's given, "
        "call grep_research_corpus first (free, instant) and only fall back to "
        "find_supporting_evidence/get_citations_for (live database) if that "
        "comes up empty — most sections need no tool calls at all. After "
        "drafting, RE-CHECK each claim against the evidence; delete or soften "
        "anything unsupported. Cite chunk ids inline like [chunk_id] after "
        "grounded rules. Do not invent authority.\n\n"
        "Once satisfied, return ONLY the markdown section, starting with the "
        "'## ' heading — no other text, no further tool calls."
        + _SOFT_BUDGET_LINE
    )
    prompt = (
        f"{job_plan_ctx}"
        f"Cluster: {json.dumps({k: v for k, v in cluster.items() if k != 'artifacts'}, indent=2)}\n\n"
        f"Artifact cards:\n{json.dumps(cluster.get('artifacts') or [], indent=2)}\n\n"
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

    kept, dropped = [], []
    for b in blocks:
        if len(b.get("markdown", "").strip()) >= _MIN_SECTION_CHARS:
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
            lines += [b["markdown"].strip(), "", "---", ""]
    else:
        # Total generation failure — surface what research found instead of
        # returning an empty note.
        lines.append("*(Outline generation produced no sections — research summary below.)*")
        for c in dossier.get("clusters") or []:
            lines += [f"## {c.get('label', '?')}", c.get("doctrine_summary", ""), ""]

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
