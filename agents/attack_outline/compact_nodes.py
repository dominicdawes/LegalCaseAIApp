# agents/attack_outline/compact_nodes.py
"""
Compact 4-stage attack-outline pipeline ("the haircut").

Replaces the 15-node graph's five serial LLM stages with one multistep
tool-calling research agent, cutting LLM roundtrips from ~9 serial stages to:

  1. plan_agent        — one worker_mid call → master job plan
  2. research_agent    — ONE agent, thinking=True, native tool-calling ReAct
                         loop (bounded turns, parallel tool execution).
                         Compresses: source_profiler + corpus_topic_mapper +
                         retrieval_planner + legal_artifact_extractor +
                         concept_clusterer. Emits a research dossier:
                         clusters + per-cluster legal artifact cards +
                         evidence chunk references. Every retrieval tool
                         result is harvested into state["evidence_store"] so
                         downstream generators never re-retrieve.
  3. block_generator   — [Send×N] one per cluster, thinking=True. One-stop
                         section builder (compresses attack_block_builder +
                         doctrine_graph_builder's if/then logic + the critic's
                         grounding pass): drafts the full T-14 block and
                         self-checks every rule claim against the evidence
                         chunks included in its prompt.
  4. final_formatter   — pure Python. Deterministic assembly, blank-section
                         checks, TOC. Zero LLM calls.

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

# Max model turns in the research loop. Each turn may issue MANY tool calls
# (executed in parallel), so 8 turns is a lot of retrieval ground.
MAX_RESEARCH_TURNS = int(os.getenv("ATTACK_RESEARCH_MAX_TURNS", "8"))

# Per-tool-result cap injected back into the conversation. Full chunk payloads
# are harvested separately into evidence_store, so the transcript stays lean.
TOOL_RESULT_CHAR_CAP = 6000

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
    "  1. Survey: get_doc_outline / find_docs_about to map what each source covers.\n"
    "  2. Retrieve: hybrid_search + search_passages per doctrine (batch MANY tool "
    "calls in a single turn — they run in parallel). Drill into sections with "
    "get_section / get_neighbors where a hit looks central.\n"
    "  3. Extract & cluster: identify every legal artifact present "
    f"({ARTIFACT_CARD_TYPES}) and group doctrines into outline clusters.\n\n"
    "RULES:\n"
    "• Prefer FEW turns with MANY parallel tool calls over many small turns.\n"
    "• Only claim rules/holdings you actually retrieved — cite chunk_ids.\n"
    "• When you have enough evidence for every major doctrine, STOP calling "
    "tools and emit the final dossier.\n\n"
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


async def research_agent(state: AgentState) -> Dict:
    """
    The all-in-one multistep research agent (thinking=True).

    A hand-rolled ReAct loop rather than langgraph.prebuilt.create_react_agent
    so we keep: per-turn telemetry, a hard turn budget, parallel tool
    execution, evidence harvesting, and the repo's DeepSeek semaphore.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import ATTACK_RESEARCH_TOOLS
    from utils.llm_clients.deepseek_client import DeepSeekClient

    job_plan = state.get("job_plan") or {}
    _node_start("research_agent", state,
                priority_doctrines=(job_plan.get("priority_doctrines") or [])[:5])

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=ATTACK_RESEARCH_TOOLS,
    )
    tool_map = {t.name: t for t in tools}

    # thinking=True is deliberate: this is the one multistep reasoning stage.
    client = DeepSeekClient(
        model_name="deepseek-v4-pro",
        temperature=0.3,
        max_output_tokens=8000,
        thinking=True,
    )
    model = client._client.bind_tools(tools)
    logger.info("  🤖 [research_agent] deepseek-v4-pro thinking=True tools=%d max_turns=%d",
                len(tools), MAX_RESEARCH_TURNS)

    messages: List[Any] = [
        SystemMessage(content=_RESEARCH_SYSTEM),
        HumanMessage(content=(
            f"Job plan:\n{json.dumps(job_plan, indent=2)}\n\n"
            f"Source ids in scope: {state.get('source_ids', [])}\n"
            f"User request: {state['request']}\n\n"
            "Begin your research. Batch parallel tool calls aggressively."
        )),
    ]

    evidence_store: Dict[str, Dict] = {}
    dossier: Optional[Dict[str, Any]] = None
    sem = _get_deepseek_semaphore()

    for turn in range(1, MAX_RESEARCH_TURNS + 1):
        async with sem:
            try:
                resp: AIMessage = await model.ainvoke(messages)
            except Exception as exc:
                # One retry for transient/429 blips, then bail to fallback.
                _node_warn("research_agent", state, f"turn {turn} LLM error ({exc}) — retrying once")
                await asyncio.sleep(5)
                resp = await model.ainvoke(messages)
        messages.append(resp)

        tool_calls = getattr(resp, "tool_calls", None) or []
        if not tool_calls:
            # Model is done researching — this should be the dossier.
            try:
                dossier = _parse_json(resp.content)
            except Exception as exc:
                _node_warn("research_agent", state,
                           f"final answer JSON parse failed on turn {turn}: {exc}")
                dossier = None
            break

        logger.info("  🔧 [research_agent] turn %d/%d → %d tool call(s): %s",
                    turn, MAX_RESEARCH_TURNS, len(tool_calls),
                    [tc.get("name") for tc in tool_calls][:8])
        results = await asyncio.gather(
            *(_exec_tool_call(tool_map, tc) for tc in tool_calls)
        )
        for (tool_msg, full_result), tc in zip(results, tool_calls):
            messages.append(tool_msg)
            _harvest_evidence(tc.get("name", ""), full_result, evidence_store)

    if dossier is None and MAX_RESEARCH_TURNS > 0:
        # Turn budget exhausted mid-research (or final parse failed):
        # force the dossier from what has been gathered so far.
        _node_warn("research_agent", state, "forcing final dossier emission")
        messages.append(HumanMessage(content=(
            "STOP researching. Emit the final dossier JSON NOW using only the "
            "evidence already gathered. Return ONLY the JSON object."
        )))
        async with sem:
            final = await client._client.ainvoke(messages)  # unbound — no more tools
        try:
            dossier = _parse_json(final.content)
        except Exception as exc:
            _node_warn("research_agent", state, f"forced dossier parse failed: {exc}")
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
    One-stop section builder for a single cluster (thinking=True).
    Drafts the full T-14 attack block AND self-grounds it against the evidence
    chunks in its prompt — compressing builder + graph-edges + critic passes.
    """
    cluster: Dict[str, Any] = state["cluster"]
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
        "GROUNDING (critical): every rule, element, holding, and case you state "
        "must be supported by the artifact cards or the evidence excerpts "
        "provided. After drafting, RE-CHECK each claim against the evidence; "
        "delete or soften anything unsupported. Cite chunk ids inline like "
        "[chunk_id] after grounded rules. Do not invent authority.\n\n"
        "Return ONLY the markdown section, starting with the '## ' heading."
    )
    prompt = (
        f"Cluster: {json.dumps({k: v for k, v in cluster.items() if k != 'artifacts'}, indent=2)}\n\n"
        f"Artifact cards:\n{json.dumps(cluster.get('artifacts') or [], indent=2)}\n\n"
        f"Evidence excerpts:\n{evidence_text}"
    )

    # orchestrator tier = deepseek-v4-pro thinking=True — the deliberate
    # reasoning stage where drafting + self-grounding happen together.
    markdown = await _llm("orchestrator", prompt, system=system,
                          max_tokens=10000, _node="block_generator")
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
