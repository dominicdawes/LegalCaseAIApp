# agents/case_brief/nodes.py
"""
All nodes for the case-brief LangGraph agent.

Node index (in execution order):
  1.  head_orchestrator            — orchestrator; plans brief mode, scope, budget
  2.  source_profiler              — worker_mid; per-source profile (parallel Send)
  3.  corpus_orientation_synthesizer — worker_mid; identifies primary case, source roles
  4.  retrieval_planner            — orchestrator; artifact-specific retrieval probes
  5.  planned_retriever            — tool_only; executes + reranks per target (parallel Send)
  6.  evidence_card_builder        — worker_low; converts chunks → evidence cards (parallel Send)
  7.  legal_artifact_extractor     — worker_mid; extracts typed artifact per extractor_type (parallel ×4)
  8.  doctrinal_synthesizer        — worker_mid; resolves holding, rule, limits, exam triggers
  9.  brief_drafter                — orchestrator; creates drafting manifest
  10. section_writer               — worker_mid; writes one section per section_type (parallel ×10)
  11. section_grounder             — worker_mid; verifies each section against evidence cards (parallel)
  12. section_reviser              — worker_mid; revises all failed sections in one pass (sequential)
  13. brief_assembler              — worker_low; merges final sections in manifest order
  14. global_coherence_editor      — worker_mid; checks consistency, harmonises terminology
  15. critic                       — orchestrator; scores brief, issues revision targets
  16. brief_revision_agent         — worker_mid; targeted revision of flagged sections, re-assembles
  17. final_formatter              — worker_low; final polish and export formatting

Fan-out routing helpers (not nodes):
  head_orchestrator_to_profiler, retrieval_planner_to_retriever,
  retriever_to_card_builder, card_builder_to_extractors,
  drafter_to_writers, writers_to_grounders, should_revise
"""

import asyncio
import json
import logging
import re
import uuid as _uuid_mod
from typing import Any, Dict, List, Optional

from langgraph.types import Send

from .state import (
    AgentState,
    BriefCritique,
    CorpusOrientation,
    DoctrinalSynthesis,
    EvidenceCard,
    ExtractedArtifact,
    GroundingReport,
    RetrievalBundle,
    RetrievalPlan,
    SectionDraft,
    SourceProfile,
)
from .worker_config import _fetch_worker_model, model_costs

logger = logging.getLogger(__name__)

MAX_REVISIONS = 2

SECTION_TYPES = [
    "case_identity",
    "procedural_posture",
    "facts",
    "issue_holding",
    "rule",
    "reasoning",
    "dissent",
    "pedagogy",
    "exam_translation",
    "cold_call",
]

RETRIEVAL_TARGETS = [
    "case_identity",
    "procedural_posture",
    "material_facts",
    "issue",
    "holding",
    "rule",
    "reasoning",
    "dissent",
    "pedagogy",
]


# ── Shared helpers ─────────────────────────────────────────────────────────────

async def _llm(
    worker_class: str,
    prompt: str,
    system: str = "",
    max_tokens: int = 2048,
    provider: Optional[str] = None,
) -> str:
    from utils.llm_clients.llm_factory import LLMFactory
    _provider, model_name = _fetch_worker_model(worker_class, provider)
    client = LLMFactory.get_client_for(
        _provider, model_name,
        temperature=0.7, streaming=False, max_output_tokens=max_tokens,
    )
    if hasattr(client, "achat"):
        return await client.achat(prompt, system_prompt=system or None)
    chunks: List[str] = []
    async for chunk in client.stream_chat(prompt, system_prompt=system or None):
        chunks.append(chunk)
    return "".join(chunks)


async def _try_save_artifact(
    state: Dict,
    artifact_key: str,
    content: Dict[str, Any],
    worker_class: str,
    node_name: str,
    artifact_type: str,
    source_ids: Optional[List[str]] = None,
) -> None:
    """Non-fatal ledger write — never raises; skipped if job_id absent."""
    job_id_str = (state.get("job_id") or "").strip()
    if not job_id_str:
        return
    try:
        from agents.ledger import AgentLedgerService
        ledger = AgentLedgerService()
        await ledger.save_artifact(
            job_id=_uuid_mod.UUID(job_id_str),
            artifact_key=artifact_key,
            content=content,
            worker_class=worker_class,
            node_name=node_name,
            artifact_type=artifact_type,
            source_ids=[_uuid_mod.UUID(s) for s in (source_ids or []) if s],
        )
    except Exception as exc:
        logger.warning("Ledger save '%s' failed (non-fatal): %s", artifact_key, exc)


def _parse_json(raw: str) -> Any:
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return json.loads(text.strip())


def _add_budget(state: AgentState, model_name: str, in_tok: int, out_tok: int) -> Dict:
    in_cost, out_cost = model_costs(model_name)
    delta = in_tok * in_cost + out_tok * out_cost
    budget = dict(state.get("budget") or {})
    budget["input_tokens"] = budget.get("input_tokens", 0) + in_tok
    budget["output_tokens"] = budget.get("output_tokens", 0) + out_tok
    budget["cost_usd"] = round(budget.get("cost_usd", 0.0) + delta, 6)
    return budget


def _cards_for_roles(cards: List[EvidenceCard], roles: List[str]) -> List[EvidenceCard]:
    return [c for c in cards if c.get("role") in roles]


def _cards_context(cards: List[EvidenceCard], max_cards: int = 15) -> str:
    return "\n\n---\n\n".join(
        f"[card_id:{c['card_id']} role:{c['role']} source:{c['source_id']} p.{c['page_range']}]\n{c['quote']}"
        for c in cards[:max_cards]
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. head_orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def head_orchestrator(state: AgentState) -> Dict:
    """
    Owns the full case-brief job. Reads the source list, validates IDs,
    determines brief_mode, and produces a job_plan for downstream nodes.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import BRIEF_PLANNER_TOOLS

    tools = make_tools(
        state["project_id"],
        source_ids=state["source_ids"],
        use_voyage=state.get("use_voyage", False),
        tool_names=BRIEF_PLANNER_TOOLS,
    )
    list_sources_tool = next(t for t in tools if t.name == "list_sources")
    sources_json = await list_sources_tool.ainvoke({})

    system = (
        "You are the workflow controller for a law-school case brief generation graph. "
        "Determine the case brief mode, source scope, output length, model budget, "
        "fanout plan, and revision policy. Do not write the brief.\n\n"
        "Return JSON with keys:\n"
        "  job_type: 'case_brief'\n"
        "  brief_mode: 'single_case' | 'multi_case' | 'casebook_excerpt' | 'doctrine_packet' | 'mixed_source'\n"
        "  source_ids: list of source UUIDs in scope\n"
        "  target_length: 'short' | 'standard' | 'long'\n"
        "  output_format: 'markdown'\n"
        "  stages: list of pipeline stage names\n"
        "  retrieval_depth: 'shallow' | 'standard' | 'deep'\n"
        "  revision_policy: 'strict' | 'standard' | 'permissive'\n"
        "Return only JSON."
    )

    raw = await _llm(
        "orchestrator",
        f"Sources available:\n{sources_json}\n\nUser request: {state['request']}",
        system=system,
        max_tokens=512,
    )
    try:
        job_plan = _parse_json(raw)
    except Exception:
        job_plan = {
            "job_type": "case_brief",
            "brief_mode": "single_case",
            "source_ids": state["source_ids"],
            "target_length": "standard",
            "output_format": "markdown",
            "stages": ["profiling", "retrieval", "extraction", "synthesis", "drafting", "grounding", "assembly", "critique"],
            "retrieval_depth": "standard",
            "revision_policy": "standard",
        }

    await _try_save_artifact(
        state, "job_plan", job_plan, "orchestrator", "head_orchestrator", "job_plan",
        source_ids=state.get("source_ids"),
    )
    return {"job_plan": job_plan}


head_orchestrator.default_worker_class = "orchestrator"


def head_orchestrator_to_profiler(state: AgentState) -> List[Send]:
    """Fan-out: one source_profiler per source document."""
    return [
        Send("source_profiler", {"source_id": sid, **state})
        for sid in state["source_ids"]
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. source_profiler  (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def source_profiler(state: Dict) -> Dict:
    """
    Profile a single source document: type, case name, court, year,
    has_dissent, summary, and section map.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import BRIEF_PROFILER_TOOLS

    source_id = state["source_id"]
    project_id = state["project_id"]

    tools = make_tools(
        project_id,
        source_ids=[source_id],
        use_voyage=state.get("use_voyage", False),
        tool_names=BRIEF_PROFILER_TOOLS,
    )
    outline_tool = next(t for t in tools if t.name == "get_doc_outline")
    outline_json = await outline_tool.ainvoke({"source_id": source_id})
    outline = json.loads(outline_json)

    sections_brief = json.dumps(outline.get("toc", [])[:20], indent=2)
    doc_summary = outline.get("doc_summary") or ""
    concepts = outline.get("doc_concepts", [])[:15]

    system = (
        "You are a law professor profiling a legal source document for case briefing. "
        "Return JSON with exactly these keys:\n"
        "  doc_type_guess: 'full_case_opinion' | 'casebook_excerpt' | 'lecture_notes' | 'secondary' | 'other'\n"
        "  case_name_guess: string (e.g. 'Palsgraf v. Long Island R.R. Co.' or '' if unknown)\n"
        "  court_guess: string (e.g. 'N.Y. Court of Appeals' or '')\n"
        "  year_guess: string (e.g. '1928' or '')\n"
        "  has_dissent_guess: boolean\n"
        "  document_summary: string (2-3 sentences)\n"
        "  section_map: array of {section_id, heading, summary} for the 12 most important sections\n"
        "  confidence: float 0.0-1.0\n"
        "Return only JSON."
    )

    prompt = (
        f"Document summary: {doc_summary}\n\n"
        f"Key concepts: {json.dumps(concepts)}\n\n"
        f"Table of contents:\n{sections_brief}\n\n"
        f"Profile this document for case-brief purposes."
    )

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=1200)
    try:
        data = _parse_json(raw)
    except Exception:
        data = {}

    profile: SourceProfile = {
        "source_id": source_id,
        "doc_type_guess": data.get("doc_type_guess", "other"),
        "case_name_guess": data.get("case_name_guess", ""),
        "court_guess": data.get("court_guess", ""),
        "year_guess": data.get("year_guess", ""),
        "has_dissent_guess": bool(data.get("has_dissent_guess", False)),
        "document_summary": data.get("document_summary", doc_summary[:500]),
        "section_map": data.get("section_map", [])[:15],
        "confidence": float(data.get("confidence", 0.5)),
    }

    await _try_save_artifact(
        state, f"source_profile:{source_id}", profile,
        "worker_mid", "source_profiler", "source_profile", source_ids=[source_id],
    )
    return {"source_profiles": [profile]}


source_profiler.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 3. corpus_orientation_synthesizer
# ─────────────────────────────────────────────────────────────────────────────

async def corpus_orientation_synthesizer(state: AgentState) -> Dict:
    """
    Merges all source profiles into a single understanding of the packet.
    Identifies primary case, supporting materials, source roles, and brief scope.
    """
    profiles = state.get("source_profiles") or []
    job_plan = state.get("job_plan") or {}

    profiles_brief = [
        {
            "source_id": p["source_id"],
            "doc_type_guess": p["doc_type_guess"],
            "case_name_guess": p["case_name_guess"],
            "court_guess": p["court_guess"],
            "year_guess": p["year_guess"],
            "has_dissent_guess": p["has_dissent_guess"],
            "summary": p["document_summary"][:200],
        }
        for p in profiles
    ]

    system = (
        "You are a corpus orientation agent. Given profiles of multiple source documents, "
        "decide what the student is actually trying to brief and how the sources relate.\n\n"
        "Return JSON with:\n"
        "  primary_case_source_id: UUID of the main case opinion source\n"
        "  supporting_source_ids: list of UUIDs for supporting materials\n"
        "  case_brief_scope: 'single_case' | 'multi_case' | 'casebook_excerpt' | 'doctrine_packet' | 'mixed_source'\n"
        "  source_roles: [{source_id, role}] where role is one of:\n"
        "    primary_opinion | casebook_excerpt | lecture_notes | secondary | duplicate | irrelevant\n"
        "  primary_case_name: string (best guess at full case name)\n"
        "  briefing_notes: 1-2 sentence explanation of the briefing scope\n"
        "Return only JSON."
    )

    prompt = (
        f"Source profiles:\n{json.dumps(profiles_brief, indent=2)}\n\n"
        f"Brief mode from plan: {job_plan.get('brief_mode', 'single_case')}\n"
        f"User request: {state['request']}\n\n"
        f"Identify the primary case and how the sources relate."
    )

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=768)
    try:
        data = _parse_json(raw)
    except Exception:
        data = {}

    # Fallback: pick the highest-confidence full_case_opinion or the first source
    primary_id = data.get("primary_case_source_id", "")
    if not primary_id and profiles:
        opinions = [p for p in profiles if p["doc_type_guess"] == "full_case_opinion"]
        primary_id = (opinions[0] if opinions else profiles[0])["source_id"]

    orientation: CorpusOrientation = {
        "primary_case_source_id": primary_id,
        "supporting_source_ids": [
            s for s in (data.get("supporting_source_ids") or []) if s != primary_id
        ],
        "case_brief_scope": data.get("case_brief_scope", "single_case"),
        "source_roles": data.get("source_roles", [
            {"source_id": p["source_id"], "role": p["doc_type_guess"]} for p in profiles
        ]),
        "primary_case_name": data.get("primary_case_name", ""),
        "briefing_notes": data.get("briefing_notes", ""),
    }

    await _try_save_artifact(
        state, "corpus_orientation", dict(orientation),
        "worker_mid", "corpus_orientation_synthesizer", "corpus_orientation",
        source_ids=state.get("source_ids"),
    )
    return {"corpus_orientation": orientation}


corpus_orientation_synthesizer.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 4. retrieval_planner
# ─────────────────────────────────────────────────────────────────────────────

async def retrieval_planner(state: AgentState) -> Dict:
    """
    Creates artifact-specific retrieval probes for every brief section.
    Generates BM25 queries, vector queries, regex patterns, and section filters
    per target: case_identity, posture, facts, issue, holding, rule, reasoning, dissent, pedagogy.
    """
    profiles = state.get("source_profiles") or []
    orientation = state.get("corpus_orientation") or {}
    job_plan = state.get("job_plan") or {}

    primary_id = orientation.get("primary_case_source_id", "")
    primary_profile = next((p for p in profiles if p["source_id"] == primary_id), profiles[0] if profiles else {})

    section_index: Dict[str, List[str]] = {}
    for p in profiles:
        section_index[p["source_id"]] = [s.get("section_id", "") for s in p.get("section_map", [])]

    case_name = orientation.get("primary_case_name") or primary_profile.get("case_name_guess", "")

    system = (
        "You are a legal RAG retrieval planner for case briefing. "
        "Generate targeted retrieval plans for each case brief artifact. "
        "Avoid generic semantic queries — prefer case-name anchored BM25, "
        "court-specific terms, and citation regex.\n\n"
        "Return a JSON array. Each item:\n"
        "  target: one of [case_identity, procedural_posture, material_facts, issue, "
        "holding, rule, reasoning, dissent, pedagogy]\n"
        "  methods: list from [bm25, vector, section_summary, regex, metadata]\n"
        "  queries: 2-4 query strings (BM25-style keyword dense or semantic)\n"
        "  regex_patterns: 0-2 regex strings for mandatory terms\n"
        "  section_filters: list of section_ids to prioritise\n"
        "  source_filters: list of source UUIDs to restrict to\n"
        "  max_chunks: int (8-20)\n"
        "  priority: 'high' | 'medium' | 'low'\n"
        "Produce a plan for ALL targets. Return only the JSON array."
    )

    prompt = (
        f"Case being briefed: {case_name}\n"
        f"Primary source: {primary_id}\n"
        f"All source sections: {json.dumps(section_index, indent=2)}\n\n"
        f"Retrieval depth: {job_plan.get('retrieval_depth', 'standard')}\n"
        f"User request: {state['request']}\n\n"
        f"Generate retrieval plans for all brief artifacts."
    )

    raw = await _llm("orchestrator", prompt, system=system, max_tokens=3000)
    try:
        plans: List[RetrievalPlan] = _parse_json(raw)
        if not isinstance(plans, list):
            plans = []
    except Exception:
        plans = [
            {
                "target": target,
                "methods": ["bm25", "vector"],
                "queries": [f"{case_name} {target.replace('_', ' ')}", f"{target.replace('_', ' ')} legal analysis"],
                "regex_patterns": [],
                "section_filters": [],
                "source_filters": [primary_id] if primary_id else state["source_ids"],
                "max_chunks": 12,
                "priority": "high" if target in ("holding", "rule", "material_facts") else "medium",
            }
            for target in RETRIEVAL_TARGETS
        ]

    await _try_save_artifact(
        state, "retrieval_plans", {"plans": plans},
        "orchestrator", "retrieval_planner", "retrieval_plan",
        source_ids=state.get("source_ids"),
    )
    return {"retrieval_plans": plans}


retrieval_planner.default_worker_class = "orchestrator"


def retrieval_planner_to_retriever(state: AgentState) -> List[Send]:
    """Fan-out: one planned_retriever per retrieval plan."""
    return [
        Send("planned_retriever", {"plan": p, **state})
        for p in (state.get("retrieval_plans") or [])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 5. planned_retriever  (parallel leaf — tool_only with inline reranking)
# ─────────────────────────────────────────────────────────────────────────────

async def planned_retriever(state: Dict) -> Dict:
    """
    Execute one retrieval plan: BM25, vector, and section-based retrieval.
    Inline reranker: score-sort + dedup by chunk_id. Returns one RetrievalBundle.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import BRIEF_RETRIEVER_TOOLS

    plan: RetrievalPlan = state["plan"]
    project_id = state["project_id"]
    source_ids = plan.get("source_filters") or state.get("source_ids") or []

    tools = make_tools(
        project_id,
        source_ids=source_ids,
        use_voyage=state.get("use_voyage", False),
        tool_names=BRIEF_RETRIEVER_TOOLS,
    )
    hybrid_tool   = next((t for t in tools if t.name == "hybrid_search"), None)
    section_tool  = next((t for t in tools if t.name == "find_sections_about"), None)
    passages_tool = next((t for t in tools if t.name == "search_passages"), None)

    seen_ids: set = set()
    all_chunks: List[Dict[str, Any]] = []

    for q in (plan.get("queries") or [])[:5]:
        if not q:
            continue
        try:
            if hybrid_tool:
                raw = await hybrid_tool.ainvoke({"query": q, "k": plan.get("max_chunks", 15)})
            elif passages_tool:
                raw = await passages_tool.ainvoke({"query": q, "k": plan.get("max_chunks", 15)})
            else:
                continue
            chunks = json.loads(raw)
            for c in chunks:
                cid = c.get("id") or c.get("chunk_id") or ""
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(c)
        except Exception as exc:
            logger.debug("planned_retriever target='%s' query='%s' failed: %s", plan.get("target"), q[:60], exc)

    for section_filter in (plan.get("section_filters") or [])[:4]:
        try:
            if section_tool:
                target_label = plan.get("target", "").replace("_", " ")
                raw = await section_tool.ainvoke({"query": target_label, "k": 8})
                chunks = json.loads(raw)
                for c in chunks:
                    cid = c.get("id") or c.get("chunk_id") or ""
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        all_chunks.append(c)
        except Exception:
            pass

    def _score(c: Dict) -> float:
        return float(c.get("score") or c.get("similarity") or 0.0)

    ranked = sorted(all_chunks, key=_score, reverse=True)[:plan.get("max_chunks", 20)]

    # Use primary_case_source_id as source_id for the bundle
    orientation = state.get("corpus_orientation") or {}
    bundle_source = orientation.get("primary_case_source_id") or (source_ids[0] if source_ids else "")

    bundle: RetrievalBundle = {
        "target": plan.get("target", "unknown"),
        "source_id": bundle_source,
        "chunks": ranked,
    }
    return {"retrieval_bundles": [bundle]}


planned_retriever.default_worker_class = "tool_only"


def retriever_to_card_builder(state: AgentState) -> List[Send]:
    """Fan-out: one evidence_card_builder per retrieval bundle."""
    return [
        Send("evidence_card_builder", {"bundle": b, **state})
        for b in (state.get("retrieval_bundles") or [])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 6. evidence_card_builder  (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def evidence_card_builder(state: Dict) -> Dict:
    """
    Convert a retrieval bundle into grounded evidence cards.
    Labels each card with its role: holding, facts, rule, reasoning,
    dissent, posture, issue, citation, pedagogy, exam_trigger.
    """
    bundle: RetrievalBundle = state["bundle"]
    target = bundle.get("target", "unknown")
    chunks = bundle.get("chunks") or []

    if not chunks:
        return {"evidence_cards": []}

    context = "\n\n---\n\n".join(
        f"[chunk_id:{c.get('id', c.get('chunk_id', '?'))} "
        f"source:{c.get('source_id', '?')} p.{c.get('page_number', '?')}]\n"
        f"{c.get('content', '')}"
        for c in chunks[:20]
    )

    system = (
        "You are an evidence card builder for a law-school case brief. "
        "Convert retrieved chunks into concise, grounded evidence cards. "
        "Preserve full source traceability.\n\n"
        "Return a JSON array. Each item:\n"
        "  card_id: short unique string (e.g. 'card_001')\n"
        "  role: one of [holding, facts, rule, reasoning, dissent, posture, issue, citation, pedagogy, exam_trigger]\n"
        "  source_id: source UUID from the chunk header\n"
        "  chunk_id: chunk UUID from the chunk header\n"
        "  page_range: page number(s) string\n"
        "  quote: verbatim 1-3 sentence quote or tight paraphrase\n"
        "  confidence: float 0.0-1.0\n"
        "Extract ALL evidence cards present. Return only the JSON array."
    )

    prompt = (
        f"Retrieval target: {target}\n\n"
        f"Chunks:\n{context}\n\n"
        f"Build evidence cards."
    )

    raw = await _llm("worker_low", prompt, system=system, max_tokens=2500)
    try:
        cards_raw = _parse_json(raw)
        if not isinstance(cards_raw, list):
            cards_raw = []
    except Exception:
        cards_raw = []

    # Role mapping based on retrieval target when card role is ambiguous
    TARGET_ROLE_MAP = {
        "case_identity": "citation",
        "procedural_posture": "posture",
        "material_facts": "facts",
        "issue": "issue",
        "holding": "holding",
        "rule": "rule",
        "reasoning": "reasoning",
        "dissent": "dissent",
        "pedagogy": "pedagogy",
    }
    default_role = TARGET_ROLE_MAP.get(target, "facts")

    cards: List[EvidenceCard] = []
    for i, raw_card in enumerate(cards_raw):
        if not isinstance(raw_card, dict):
            continue
        cards.append({
            "card_id": raw_card.get("card_id") or f"card_{target}_{i:03d}",
            "role": raw_card.get("role", default_role),
            "source_id": raw_card.get("source_id", bundle.get("source_id", "")),
            "chunk_id": raw_card.get("chunk_id", ""),
            "page_range": str(raw_card.get("page_range", "")),
            "quote": raw_card.get("quote", "")[:600],
            "confidence": float(raw_card.get("confidence", 0.6)),
        })

    await _try_save_artifact(
        state, f"evidence_cards:{target}", {"target": target, "cards": cards},
        "worker_low", "evidence_card_builder", "evidence_cards",
        source_ids=state.get("source_ids"),
    )
    return {"evidence_cards": cards}


evidence_card_builder.default_worker_class = "worker_low"


def card_builder_to_extractors(state: AgentState) -> List[Send]:
    """Fan-out: spawn all 4 legal artifact extractors in parallel."""
    extractor_types = ["facts_posture", "issue_holding", "rule_reasoning", "dissent"]
    return [
        Send("legal_artifact_extractor", {"extractor_type": et, **state})
        for et in extractor_types
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 7. legal_artifact_extractor  (parallel ×4, dispatched by extractor_type)
# ─────────────────────────────────────────────────────────────────────────────

async def legal_artifact_extractor(state: Dict) -> Dict:
    """
    Extract one typed legal artifact from evidence cards.
    Dispatches internally based on state['extractor_type']:
      facts_posture   → procedural posture + material facts
      issue_holding   → issue statement + narrow holding + disposition + winner
      rule_reasoning  → black-letter rule + test + elements + reasoning types
      dissent         → dissent/concurrence summary (returns empty if none)
    """
    extractor_type: str = state.get("extractor_type", "facts_posture")
    all_cards: List[EvidenceCard] = state.get("evidence_cards") or []

    ROLE_MAPS = {
        "facts_posture":  ["posture", "facts"],
        "issue_holding":  ["issue", "holding", "facts"],
        "rule_reasoning": ["rule", "reasoning", "holding", "citation"],
        "dissent":        ["dissent"],
    }
    relevant_cards = _cards_for_roles(all_cards, ROLE_MAPS.get(extractor_type, []))
    context = _cards_context(relevant_cards, max_cards=18)

    SYSTEM_PROMPTS = {
        "facts_posture": (
            "You are a case brief facts and posture specialist. "
            "Separate procedural posture from merits. Separate material facts "
            "from background facts.\n\n"
            "Return JSON with:\n"
            "  procedural_posture: {lower_court, current_stage, standard_or_frame, disposition_below, supporting_card_ids}\n"
            "  material_facts: [{fact, why_material, supporting_card_ids}] (max 10)\n"
            "  background_facts: [str] (max 5)\n"
            "  uncertainties: [str]\n"
            "Return only JSON."
        ),
        "issue_holding": (
            "You are an issue and holding specialist. "
            "Frame the issue as a legal question tied to material facts. "
            "State the holding narrowly and distinguish it from the disposition.\n\n"
            "Return JSON with:\n"
            "  issue: 'Whether ...' string\n"
            "  holding_narrow: string (answers the issue yes/no where possible)\n"
            "  disposition: 'affirmed' | 'reversed' | 'remanded' | 'vacated' | 'modified' | 'other'\n"
            "  winner: 'plaintiff' | 'defendant' | 'appellant' | 'appellee' | 'unclear'\n"
            "  confidence: float 0.0-1.0\n"
            "  supporting_card_ids: [str]\n"
            "Return only JSON."
        ),
        "rule_reasoning": (
            "You are a doctrine extraction agent. Separate black-letter rule from "
            "application, reasoning, dicta, policy, and precedent.\n\n"
            "Return JSON with:\n"
            "  rule: {black_letter, test (list of steps), elements (list), "
            "exceptions (list), limitations (list), supporting_card_ids}\n"
            "  reasoning: [{type, text}] where type is one of:\n"
            "    application | precedent | policy | textual | institutional | fairness | administrability\n"
            "Return only JSON."
        ),
        "dissent": (
            "You are a dissent and concurrence specialist. "
            "Identify separate opinions and explain their doctrinal disagreement. "
            "If no dissent or concurrence exists, return empty fields (has_dissent: false).\n\n"
            "Return JSON with:\n"
            "  has_dissent: bool\n"
            "  has_concurrence: bool\n"
            "  dissent_summary: str\n"
            "  concurrence_summary: str\n"
            "  alternative_rule: str\n"
            "  key_disagreement: str\n"
            "  supporting_card_ids: [str]\n"
            "Return only JSON."
        ),
    }

    system = SYSTEM_PROMPTS.get(extractor_type, SYSTEM_PROMPTS["facts_posture"])

    if not relevant_cards:
        # No evidence cards for this extractor type — return empty artifact
        empty_contents = {
            "facts_posture":  {"procedural_posture": {}, "material_facts": [], "background_facts": [], "uncertainties": []},
            "issue_holding":  {"issue": "", "holding_narrow": "", "disposition": "", "winner": "", "confidence": 0.0, "supporting_card_ids": []},
            "rule_reasoning": {"rule": {}, "reasoning": []},
            "dissent":        {"has_dissent": False, "has_concurrence": False, "dissent_summary": "", "concurrence_summary": "", "alternative_rule": "", "key_disagreement": "", "supporting_card_ids": []},
        }
        artifact: ExtractedArtifact = {
            "artifact_type": extractor_type,
            "content": empty_contents.get(extractor_type, {}),
            "supporting_card_ids": [],
            "confidence": 0.0,
        }
        return {"extracted_artifacts": [artifact]}

    prompt = (
        f"Evidence cards (role-filtered for {extractor_type}):\n\n{context}\n\n"
        f"Extract the {extractor_type.replace('_', ' ')} artifact."
    )

    # Rule/reasoning extractor gets a stronger model for hard cases
    wclass = "worker_mid"
    raw = await _llm(wclass, prompt, system=system, max_tokens=2500)
    try:
        content = _parse_json(raw)
    except Exception:
        content = {}

    card_ids = [c["card_id"] for c in relevant_cards]
    artifact = {
        "artifact_type": extractor_type,
        "content": content,
        "supporting_card_ids": card_ids[:20],
        "confidence": float(content.get("confidence", 0.7)) if extractor_type == "issue_holding" else 0.7,
    }

    await _try_save_artifact(
        state, f"extracted_artifact:{extractor_type}", artifact,
        wclass, "legal_artifact_extractor", f"artifact_{extractor_type}",
        source_ids=state.get("source_ids"),
    )
    return {"extracted_artifacts": [artifact]}


legal_artifact_extractor.default_worker_class = "worker_mid"
legal_artifact_extractor.escalation_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 8. doctrinal_synthesizer
# ─────────────────────────────────────────────────────────────────────────────

async def doctrinal_synthesizer(state: AgentState) -> Dict:
    """
    Merge all 4 extracted artifacts into a unified doctrinal understanding.
    Resolves narrow holding vs broad rule, identifies limits, exam triggers,
    cold-call traps, and common misreadings.
    """
    artifacts = state.get("extracted_artifacts") or []
    orientation = state.get("corpus_orientation") or {}

    by_type = {a["artifact_type"]: a["content"] for a in artifacts}
    facts_data    = by_type.get("facts_posture", {})
    ih_data       = by_type.get("issue_holding", {})
    rule_data     = by_type.get("rule_reasoning", {})
    dissent_data  = by_type.get("dissent", {})

    facts_posture_summary = {
        "current_stage": (facts_data.get("procedural_posture") or {}).get("current_stage", ""),
        "material_facts": [f.get("fact", "") for f in (facts_data.get("material_facts") or [])[:5]],
    }
    rule_summary = {
        "black_letter": (rule_data.get("rule") or {}).get("black_letter", ""),
        "elements": (rule_data.get("rule") or {}).get("elements", [])[:6],
    }

    system = (
        "You are a senior law tutor synthesising a case brief. Resolve conflicts, "
        "distinguish narrow holding from broad rule, identify limits, and explain "
        "why the case matters.\n\n"
        "Return JSON with:\n"
        "  doctrinal_role: 'introduces' | 'refines' | 'limits' | 'overrules' | 'applies' | 'distinguishes'\n"
        "  narrow_holding: string (the specific answer to the case's issue)\n"
        "  broad_rule: string (the general rule students should remember)\n"
        "  rule_limits: [str] (situations where the rule does not apply)\n"
        "  exam_triggers: [str] (fact patterns that call for this case)\n"
        "  cold_call_traps: [str] (common mistakes professors catch students on)\n"
        "  common_misreadings: [str] (overbroad applications to avoid)\n"
        "  related_doctrines: [str] (adjacent concepts students should connect)\n"
        "  pedagogical_note: string (1-2 sentences on why this case was assigned)\n"
        "Return only JSON."
    )

    prompt = (
        f"Case: {orientation.get('primary_case_name', 'Unknown')}\n\n"
        f"Issue: {ih_data.get('issue', '')}\n"
        f"Narrow holding: {ih_data.get('holding_narrow', '')}\n"
        f"Disposition: {ih_data.get('disposition', '')}\n"
        f"Winner: {ih_data.get('winner', '')}\n\n"
        f"Black-letter rule: {rule_summary.get('black_letter', '')}\n"
        f"Elements: {json.dumps(rule_summary.get('elements', []))}\n\n"
        f"Procedural stage: {facts_posture_summary.get('current_stage', '')}\n"
        f"Material facts: {json.dumps(facts_posture_summary.get('material_facts', []))}\n\n"
        f"Has dissent: {dissent_data.get('has_dissent', False)}\n"
        f"Dissent summary: {dissent_data.get('dissent_summary', '')[:300]}\n\n"
        f"Synthesise the doctrinal significance."
    )

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=1500)
    try:
        data = _parse_json(raw)
    except Exception:
        data = {}

    synthesis: DoctrinalSynthesis = {
        "doctrinal_role":    data.get("doctrinal_role", "introduces"),
        "narrow_holding":    data.get("narrow_holding", ih_data.get("holding_narrow", "")),
        "broad_rule":        data.get("broad_rule", (rule_data.get("rule") or {}).get("black_letter", "")),
        "rule_limits":       data.get("rule_limits", [])[:6],
        "exam_triggers":     data.get("exam_triggers", [])[:6],
        "cold_call_traps":   data.get("cold_call_traps", [])[:5],
        "common_misreadings":data.get("common_misreadings", [])[:4],
        "related_doctrines": data.get("related_doctrines", [])[:6],
        "pedagogical_note":  data.get("pedagogical_note", ""),
    }

    await _try_save_artifact(
        state, "doctrinal_synthesis", dict(synthesis),
        "worker_mid", "doctrinal_synthesizer", "doctrinal_synthesis",
        source_ids=state.get("source_ids"),
    )
    return {"doctrinal_synthesis": synthesis}


doctrinal_synthesizer.default_worker_class = "worker_mid"
doctrinal_synthesizer.escalation_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 9. brief_drafter
# ─────────────────────────────────────────────────────────────────────────────

async def brief_drafter(state: AgentState) -> Dict:
    """
    Create the drafting manifest: which sections to include, target word counts,
    and any section-specific instructions. Does not write the brief itself.
    """
    job_plan   = state.get("job_plan") or {}
    synthesis  = state.get("doctrinal_synthesis") or {}
    artifacts  = state.get("extracted_artifacts") or []
    orientation = state.get("corpus_orientation") or {}

    by_type    = {a["artifact_type"]: a["content"] for a in artifacts}
    dissent_data = by_type.get("dissent", {})
    has_dissent  = bool(dissent_data.get("has_dissent", False))
    target_length = job_plan.get("target_length", "standard")

    WORD_BUDGETS = {
        "short":    {"case_identity": 60,  "procedural_posture": 80, "facts": 120, "issue_holding": 100, "rule": 150, "reasoning": 200, "dissent": 80, "pedagogy": 80, "exam_translation": 120, "cold_call": 60},
        "standard": {"case_identity": 80,  "procedural_posture": 120,"facts": 200, "issue_holding": 150, "rule": 250, "reasoning": 350, "dissent": 120,"pedagogy": 120,"exam_translation": 200, "cold_call": 100},
        "long":     {"case_identity": 100, "procedural_posture": 200,"facts": 350, "issue_holding": 200, "rule": 400, "reasoning": 600, "dissent": 200,"pedagogy": 200,"exam_translation": 350, "cold_call": 150},
    }
    budgets = WORD_BUDGETS.get(target_length, WORD_BUDGETS["standard"])

    section_plan = []
    for stype in SECTION_TYPES:
        if stype == "dissent" and not has_dissent:
            section_plan.append({"section_id": stype, "include": False, "target_words": 0, "notes": "No separate opinion found in sources."})
        else:
            section_plan.append({"section_id": stype, "include": True, "target_words": budgets.get(stype, 100), "notes": ""})

    manifest = {
        "case_name": orientation.get("primary_case_name", ""),
        "target_length": target_length,
        "section_plan": section_plan,
        "has_dissent": has_dissent,
        "doctrinal_role": synthesis.get("doctrinal_role", ""),
        "output_format": job_plan.get("output_format", "markdown"),
    }

    await _try_save_artifact(
        state, "drafting_manifest", manifest,
        "orchestrator", "brief_drafter", "drafting_manifest",
        source_ids=state.get("source_ids"),
    )
    return {"drafting_manifest": manifest}


brief_drafter.default_worker_class = "orchestrator"


def drafter_to_writers(state: AgentState) -> List[Send]:
    """Fan-out: one section_writer per section type in the drafting manifest."""
    manifest = state.get("drafting_manifest") or {}
    section_plan = manifest.get("section_plan") or [{"section_id": st, "include": True} for st in SECTION_TYPES]
    return [
        Send("section_writer", {"section_type": sp["section_id"], **state})
        for sp in section_plan
        if sp.get("include", True)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 10. section_writer  (parallel ×N, dispatched by section_type)
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_SYSTEMS = {
    "case_identity": (
        "Write the case identity section using only supplied metadata and evidence. "
        "Do not infer missing court, year, citation, or parties.\n"
        "Include: full case name, court, year, citation (if available), parties, source type.\n"
        "Return JSON: {section_id, title, draft_text, claims:[{claim,supporting_card_ids}], word_count, warnings}"
    ),
    "procedural_posture": (
        "Explain how the case reached this court and what procedural frame controls analysis. "
        "Distinguish lower-court result, current stage, standard/frame, and final disposition.\n"
        "Return JSON: {section_id, title, draft_text, procedural_stage, standard_or_frame, claims, word_count, warnings}"
    ),
    "facts": (
        "Write only legally material facts. Explain why each fact matters. "
        "Avoid narrative bloat. Separate material from background facts.\n"
        "Return JSON: {section_id, title, draft_text, material_facts:[{fact,why_material,supporting_card_ids}], omitted_background_facts, word_count, claims, warnings}"
    ),
    "issue_holding": (
        "Write the issue and holding so they mirror each other. "
        "The issue must be a legal question tied to facts. "
        "The holding must answer yes/no where possible and remain narrow.\n"
        "Return JSON: {section_id, title, issue, holding, disposition, draft_text, claims, word_count}"
    ),
    "rule": (
        "Write the governing rule in reusable form. "
        "Separate: general rule, operational test, elements, exceptions, and limitations.\n"
        "Return JSON: {section_id, title, black_letter_rule, test, elements, exceptions, limitations, draft_text, claims, word_count}"
    ),
    "reasoning": (
        "Explain how the court moved from facts to rule to holding. "
        "Classify reasoning by type: application, precedent, policy, textual, institutional, fairness, administrability.\n"
        "Return JSON: {section_id, title, reasoning_outline:[str], draft_text, word_count, warnings}"
    ),
    "dissent": (
        "Write separate-opinion analysis only if the evidence supports it. "
        "Do not invent a dissent or concurrence. If none, return has_section:false with empty draft_text.\n"
        "Return JSON: {section_id, title, has_section, draft_text, majority_vs_dissent:{majority_rule,dissent_rule,core_disagreement}, word_count, warnings}"
    ),
    "pedagogy": (
        "Explain why this case was likely assigned and what doctrinal move students should learn. "
        "Reference casebook context or lecture notes where available.\n"
        "Return JSON: {section_id, title, why_assigned, doctrinal_role, common_misreadings, draft_text, word_count}"
    ),
    "exam_translation": (
        "Translate the case into exam triggers, attack-outline use, cold-call angles, and limits. "
        "Do not overstate the doctrine.\n"
        "Return JSON: {section_id, title, use_this_case_when, do_not_overread_as, exam_hypo_triggers, cold_call_questions, draft_text, word_count}"
    ),
    "cold_call": (
        "Produce the cold-call survival guide for this case. "
        "List the 3-5 questions a professor is most likely to ask, with model answers.\n"
        "Return JSON: {section_id, title, questions:[{question,model_answer,why_asked}], draft_text, word_count}"
    ),
}

_SECTION_CARD_ROLES = {
    "case_identity":      ["citation", "posture"],
    "procedural_posture": ["posture", "holding"],
    "facts":              ["facts", "posture"],
    "issue_holding":      ["issue", "holding", "facts"],
    "rule":               ["rule", "holding", "citation"],
    "reasoning":          ["reasoning", "rule", "holding", "facts"],
    "dissent":            ["dissent"],
    "pedagogy":           ["pedagogy", "exam_trigger", "reasoning"],
    "exam_translation":   ["exam_trigger", "rule", "holding", "pedagogy"],
    "cold_call":          ["rule", "holding", "facts", "reasoning", "exam_trigger"],
}


async def section_writer(state: Dict) -> Dict:
    """
    Write one case-brief section, dispatched by state['section_type'].
    Uses relevant evidence cards and extracted artifacts for grounding.
    """
    section_type: str = state.get("section_type", "facts")
    artifacts = state.get("extracted_artifacts") or []
    all_cards: List[EvidenceCard] = state.get("evidence_cards") or []
    synthesis = state.get("doctrinal_synthesis") or {}
    manifest  = state.get("drafting_manifest") or {}
    orientation = state.get("corpus_orientation") or {}

    # Get relevant cards for this section
    relevant_roles = _SECTION_CARD_ROLES.get(section_type, ["facts"])
    relevant_cards = _cards_for_roles(all_cards, relevant_roles)
    cards_ctx = _cards_context(relevant_cards, max_cards=15)

    # Build artifact context for this section
    by_type = {a["artifact_type"]: a["content"] for a in artifacts}
    artifact_ctx_parts: List[str] = []
    if section_type in ("facts", "procedural_posture"):
        fp = by_type.get("facts_posture", {})
        if fp:
            artifact_ctx_parts.append(f"Extracted facts/posture:\n{json.dumps(fp, indent=2)[:1500]}")
    if section_type in ("issue_holding",):
        ih = by_type.get("issue_holding", {})
        if ih:
            artifact_ctx_parts.append(f"Extracted issue/holding:\n{json.dumps(ih, indent=2)}")
    if section_type in ("rule", "reasoning"):
        rr = by_type.get("rule_reasoning", {})
        if rr:
            artifact_ctx_parts.append(f"Extracted rule/reasoning:\n{json.dumps(rr, indent=2)[:1500]}")
    if section_type == "dissent":
        ds = by_type.get("dissent", {})
        if ds:
            artifact_ctx_parts.append(f"Extracted dissent:\n{json.dumps(ds, indent=2)}")
    if section_type in ("pedagogy", "exam_translation", "cold_call"):
        artifact_ctx_parts.append(f"Doctrinal synthesis:\n{json.dumps(synthesis, indent=2)[:1200]}")

    artifact_ctx = "\n\n".join(artifact_ctx_parts)

    # Word budget from manifest
    section_plans = {sp["section_id"]: sp for sp in (manifest.get("section_plan") or [])}
    target_words = (section_plans.get(section_type) or {}).get("target_words", 150)

    system = _SECTION_SYSTEMS.get(section_type, _SECTION_SYSTEMS["facts"])
    case_name = orientation.get("primary_case_name", "")

    prompt = (
        f"Case: {case_name}\n"
        f"Section to write: {section_type}\n"
        f"Target word count: ~{target_words} words\n\n"
        + (f"Extracted artifacts:\n{artifact_ctx}\n\n" if artifact_ctx else "")
        + f"Evidence cards:\n{cards_ctx}\n\n"
        f"Write the '{section_type}' section."
    )

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=1500)
    try:
        data = _parse_json(raw)
    except Exception:
        data = {}

    draft_text = data.get("draft_text", raw[:800])
    word_count = data.get("word_count", len(draft_text.split()))

    # Build generic claims list from data
    claims = data.get("claims", [])
    if not claims:
        # For sections that embed claims differently, synthesise them
        if section_type == "facts" and data.get("material_facts"):
            claims = [{"claim": f.get("fact", ""), "supporting_card_ids": f.get("supporting_card_ids", [])} for f in data["material_facts"][:6]]
        elif section_type == "issue_holding":
            claims = [
                {"claim": data.get("issue", ""), "supporting_card_ids": []},
                {"claim": data.get("holding", ""), "supporting_card_ids": []},
            ]

    section_draft: SectionDraft = {
        "section_id": section_type,
        "title": data.get("title", section_type.replace("_", " ").title()),
        "draft_text": draft_text,
        "claims": claims,
        "word_count": word_count,
        "warnings": data.get("warnings", []),
    }

    await _try_save_artifact(
        state, f"section_draft:{section_type}", dict(section_draft),
        "worker_mid", "section_writer", "section_draft",
        source_ids=state.get("source_ids"),
    )
    return {"raw_sections": [section_draft]}


section_writer.default_worker_class = "worker_mid"
section_writer.escalation_worker_class = "orchestrator"


def writers_to_grounders(state: AgentState) -> List[Send]:
    """Fan-out: one section_grounder per drafted section."""
    return [
        Send("section_grounder", {"section": s, **state})
        for s in (state.get("raw_sections") or [])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 11. section_grounder  (parallel per section)
# ─────────────────────────────────────────────────────────────────────────────

async def section_grounder(state: Dict) -> Dict:
    """
    Verify a drafted section against evidence cards.
    Flags unsupported, overbroad, and under-specified claims.
    Uses verify_claim tool when available; falls back to LLM audit.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import BRIEF_VERIFIER_TOOLS

    section: SectionDraft = state["section"]
    section_id = section["section_id"]
    all_cards: List[EvidenceCard] = state.get("evidence_cards") or []

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=BRIEF_VERIFIER_TOOLS,
    )
    verify_tool = next((t for t in tools if t.name == "verify_claim"), None)

    claims = section.get("claims") or []
    draft_text = section.get("draft_text", "")

    supported: List[str] = []
    unsupported: List[str] = []
    weak: List[str] = []
    missing: List[str] = []

    # Verify top claims against source tools
    for claim_obj in claims[:5]:
        claim_text = claim_obj.get("claim", "") if isinstance(claim_obj, dict) else str(claim_obj)
        if not claim_text:
            continue
        if verify_tool:
            try:
                result_json = await verify_tool.ainvoke({"claim": claim_text[:400], "k": 8})
                result = json.loads(result_json)
                verdict = result.get("verdict", "insufficient")
                if verdict == "supported":
                    supported.append(claim_text[:150])
                elif verdict == "contradicted":
                    unsupported.append(claim_text[:150])
                else:
                    weak.append(claim_text[:150])
            except Exception:
                weak.append(claim_text[:150])
        else:
            # LLM grounding audit when verify_claim unavailable
            relevant_cards = _cards_for_roles(all_cards, _SECTION_CARD_ROLES.get(section_id, ["facts"]))
            if not relevant_cards:
                weak.append(claim_text[:150])
                continue
            cards_ctx = _cards_context(relevant_cards, max_cards=8)
            audit_system = (
                "You are a grounding auditor. Determine if the claim is:\n"
                "  supported — directly backed by evidence\n"
                "  weak — implied but not explicit\n"
                "  unsupported — no supporting evidence\n"
                "Return JSON: {verdict: 'supported'|'weak'|'unsupported', reason: str}"
            )
            audit_raw = await _llm(
                "worker_low",
                f"Claim: {claim_text}\n\nEvidence cards:\n{cards_ctx}",
                system=audit_system, max_tokens=200,
            )
            try:
                audit = _parse_json(audit_raw)
                v = audit.get("verdict", "weak")
                (supported if v == "supported" else unsupported if v == "unsupported" else weak).append(claim_text[:150])
            except Exception:
                weak.append(claim_text[:150])

    # Check for obvious overbroad patterns
    overbroad: List[str] = []
    OVERBROAD_PATTERNS = [
        r'\balways\b', r'\bnever\b', r'\beveryone\b', r'\ball cases\b', r'\bany (case|situation)\b',
    ]
    for pat in OVERBROAD_PATTERNS:
        if re.search(pat, draft_text, re.IGNORECASE):
            overbroad.append(f"Possible overbroad language matching '{pat}' in section text")
            break

    grounding_pass = not unsupported and len(weak) <= len(supported)
    required_fixes = []
    if unsupported:
        required_fixes += [f"Remove or ground unsupported claim: '{c[:80]}'" for c in unsupported[:3]]
    if overbroad:
        required_fixes += overbroad[:2]
    if not supported and not weak:
        missing.append("Section has no verifiable claims against available evidence cards")
        grounding_pass = False

    report: GroundingReport = {
        "section_id":        section_id,
        "grounding_pass":    grounding_pass,
        "unsupported_claims": unsupported,
        "overbroad_claims":  overbroad,
        "missing_evidence":  missing,
        "required_fixes":    required_fixes,
    }

    await _try_save_artifact(
        state, f"grounding:{section_id}", dict(report),
        "worker_mid", "section_grounder", "grounding_report",
        source_ids=state.get("source_ids"),
    )
    return {"grounding_reports": [report]}


section_grounder.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 12. section_reviser  (sequential — one pass for all failed sections)
# ─────────────────────────────────────────────────────────────────────────────

async def section_reviser(state: AgentState) -> Dict:
    """
    Revise all sections that failed grounding in one sequential pass.
    Passing sections are carried forward unchanged.
    Produces final_sections: the authoritative merged section list.
    """
    raw_sections   = state.get("raw_sections") or []
    grounding_reports = state.get("grounding_reports") or []
    evidence_cards = state.get("evidence_cards") or []
    artifacts      = state.get("extracted_artifacts") or []
    orientation    = state.get("corpus_orientation") or {}

    # Index reports by section_id
    report_by_id = {r["section_id"]: r for r in grounding_reports}
    failed_ids   = {r["section_id"] for r in grounding_reports if not r["grounding_pass"]}

    final: List[SectionDraft] = []

    for section in raw_sections:
        sid = section["section_id"]
        if sid not in failed_ids:
            final.append(section)
            continue

        report = report_by_id.get(sid, {})
        required_fixes = report.get("required_fixes") or []
        relevant_cards = _cards_for_roles(evidence_cards, _SECTION_CARD_ROLES.get(sid, ["facts"]))
        cards_ctx = _cards_context(relevant_cards, max_cards=10)

        system = (
            "You are a legal section reviser. Revise ONLY what the grounding report flags. "
            "Do not add unsupported claims. Do not rewrite clean sections.\n\n"
            "Return JSON: {section_id, title, draft_text, claims:[{claim,supporting_card_ids}], "
            "word_count, warnings, changes_made:[str]}"
        )

        prompt = (
            f"Section to revise (section_id: {sid}):\n"
            f"Original draft:\n{section['draft_text']}\n\n"
            f"Required fixes:\n" + "\n".join(f"- {f}" for f in required_fixes) + "\n\n"
            f"Available evidence:\n{cards_ctx}\n\n"
            f"Revise to address the grounding failures."
        )

        raw = await _llm("worker_mid", prompt, system=system, max_tokens=1200)
        try:
            data = _parse_json(raw)
        except Exception:
            data = {}

        revised: SectionDraft = {
            "section_id": sid,
            "title":      data.get("title", section["title"]),
            "draft_text": data.get("draft_text", section["draft_text"]),
            "claims":     data.get("claims", section["claims"]),
            "word_count": data.get("word_count", section["word_count"]),
            "warnings":   section["warnings"] + ["[revised by section_reviser]"],
        }

        await _try_save_artifact(
            state, f"section_revised:{sid}", dict(revised),
            "worker_mid", "section_reviser", "revised_section",
            source_ids=state.get("source_ids"),
        )
        final.append(revised)

    return {"final_sections": final}


section_reviser.default_worker_class = "worker_mid"
section_reviser.escalation_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 13. brief_assembler
# ─────────────────────────────────────────────────────────────────────────────

def _assemble_markdown(sections: List[SectionDraft], manifest: Dict) -> str:
    """Order and render sections into a structured markdown brief."""
    order = [sp["section_id"] for sp in (manifest.get("section_plan") or []) if sp.get("include", True)]
    by_id = {s["section_id"]: s for s in sections}

    SECTION_TITLES = {
        "case_identity":      "Case Identification",
        "procedural_posture": "Procedural Posture",
        "facts":              "Material Facts",
        "issue_holding":      "Issue and Holding",
        "rule":               "Rule",
        "reasoning":          "Reasoning",
        "dissent":            "Dissent / Concurrence",
        "pedagogy":           "Why This Case Matters",
        "exam_translation":   "Exam Translation",
        "cold_call":          "Cold Call Survival Guide",
    }

    toc_lines: List[str] = []
    section_lines: List[str] = []

    for sid in order:
        section = by_id.get(sid)
        if not section or not section.get("draft_text"):
            continue
        title = section.get("title") or SECTION_TITLES.get(sid, sid.replace("_", " ").title())
        anchor = sid.replace("_", "-")
        toc_lines.append(f"- [{title}](#{anchor})")
        section_lines.append(f"## {title} {{#{anchor}}}\n\n{section['draft_text']}")

    header = f"# Case Brief: {manifest.get('case_name', 'Unknown Case')}\n\n"
    toc    = "**Contents**\n\n" + "\n".join(toc_lines) + "\n\n---\n\n" if toc_lines else ""
    body   = "\n\n---\n\n".join(section_lines)
    return header + toc + body


async def brief_assembler(state: AgentState) -> Dict:
    """
    Merge verified/revised sections into a coherent markdown brief.
    Uses final_sections if available, falls back to raw_sections.
    """
    sections  = state.get("final_sections") or state.get("raw_sections") or []
    manifest  = state.get("drafting_manifest") or {}

    assembled = _assemble_markdown(sections, manifest)

    await _try_save_artifact(
        state, "assembled_brief", {"markdown": assembled, "n_sections": len(sections)},
        "tool_only", "brief_assembler", "assembled_brief",
        source_ids=state.get("source_ids"),
    )
    return {"assembled_brief": assembled}


brief_assembler.default_worker_class = "tool_only"


# ─────────────────────────────────────────────────────────────────────────────
# 14. global_coherence_editor
# ─────────────────────────────────────────────────────────────────────────────

async def global_coherence_editor(state: AgentState) -> Dict:
    """
    Check the assembled brief for consistency and flow.
    Harmonises terminology, ensures issue/holding/rule alignment,
    removes repeated explanations. Does not add unsupported doctrine.
    """
    assembled = state.get("assembled_brief") or ""
    if not assembled:
        return {"coherence_edit": {"edited_markdown": "", "contradictions_found": [], "repetition_removed": [], "terminology_fixes": [], "remaining_warnings": []}}

    excerpt = assembled[:5000]

    system = (
        "You are a global coherence editor reviewing a law-school case brief. "
        "Check for: contradictions, repetition, inconsistent terminology, "
        "and mismatched issue/holding/rule statements. Do NOT add unsupported doctrine.\n\n"
        "Return JSON with:\n"
        "  coherence_pass: bool\n"
        "  edited_markdown: string (the edited brief — may be same as input if clean)\n"
        "  contradictions_found: [str]\n"
        "  repetition_removed: [str]\n"
        "  terminology_fixes: [str]\n"
        "  remaining_warnings: [str]\n"
        "Return only JSON."
    )

    prompt = (
        f"Case brief to review:\n\n{excerpt}\n\n"
        f"Edit for coherence and consistency."
    )

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=4500)
    try:
        data = _parse_json(raw)
    except Exception:
        data = {"coherence_pass": True, "edited_markdown": assembled}

    edit_result = {
        "coherence_pass":      bool(data.get("coherence_pass", True)),
        "edited_markdown":     data.get("edited_markdown") or assembled,
        "contradictions_found":data.get("contradictions_found", []),
        "repetition_removed":  data.get("repetition_removed", []),
        "terminology_fixes":   data.get("terminology_fixes", []),
        "remaining_warnings":  data.get("remaining_warnings", []),
    }

    await _try_save_artifact(
        state, "coherence_edit", {k: v for k, v in edit_result.items() if k != "edited_markdown"},
        "worker_mid", "global_coherence_editor", "coherence_edit",
        source_ids=state.get("source_ids"),
    )
    return {"coherence_edit": edit_result}


global_coherence_editor.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 15. critic
# ─────────────────────────────────────────────────────────────────────────────

async def critic(state: AgentState) -> Dict:
    """
    Perform final legal and pedagogical critique.
    Scores accuracy, briefing quality, exam usefulness, and citation discipline.
    Issues section revision targets if the brief would not survive a cold call.
    """
    coherence_edit = state.get("coherence_edit") or {}
    brief_text = coherence_edit.get("edited_markdown") or state.get("assembled_brief") or ""
    grounding_reports = state.get("grounding_reports") or []

    failing_sections = [r["section_id"] for r in grounding_reports if not r["grounding_pass"]]
    excerpt = brief_text[:4500]

    system = (
        "You are a demanding law professor. Determine whether the brief would survive "
        "a cold call and help on an exam.\n\n"
        "Score each dimension 0.0-10.0:\n"
        "  accuracy: factual and legal correctness\n"
        "  briefing_quality: issue precision, holding narrowness, rule usability\n"
        "  exam_usefulness: exam-trigger clarity, do-not-overread guidance\n"
        "  citation_discipline: source grounding and citation traceability\n\n"
        "Then:\n"
        "  quality_pass: true if all scores ≥ 6.5 and no failed grounding sections\n"
        "  critique: list of specific issues found\n"
        "  revise: true if quality_pass is false\n"
        "  sections_to_revise: list of section_ids most needing improvement\n"
        "  revision_instructions: 2-4 sentences of targeted guidance\n\n"
        "Return JSON only."
    )

    prompt = (
        f"Case brief excerpt:\n{excerpt}\n\n"
        f"Sections with grounding failures: {failing_sections}\n"
        f"Coherence issues: {coherence_edit.get('contradictions_found', [])}\n\n"
        f"Critique this brief as a demanding law professor."
    )

    raw = await _llm("orchestrator", prompt, system=system, max_tokens=1200)
    try:
        data = _parse_json(raw)
    except Exception:
        data = {}

    scores = {
        "accuracy":           float(data.get("scores", {}).get("accuracy", 7.0) if isinstance(data.get("scores"), dict) else data.get("accuracy", 7.0)),
        "briefing_quality":   float(data.get("scores", {}).get("briefing_quality", 7.0) if isinstance(data.get("scores"), dict) else data.get("briefing_quality", 7.0)),
        "exam_usefulness":    float(data.get("scores", {}).get("exam_usefulness", 7.0) if isinstance(data.get("scores"), dict) else data.get("exam_usefulness", 7.0)),
        "citation_discipline":float(data.get("scores", {}).get("citation_discipline", 7.0) if isinstance(data.get("scores"), dict) else data.get("citation_discipline", 7.0)),
    }

    sections_to_revise = list(data.get("sections_to_revise") or [])
    for sid in failing_sections:
        if sid not in sections_to_revise:
            sections_to_revise.append(sid)

    quality_pass = bool(data.get("quality_pass", all(v >= 6.5 for v in scores.values()) and not failing_sections))
    revise = bool(data.get("revise", not quality_pass))

    critique_result: BriefCritique = {
        "quality_pass":         quality_pass,
        "scores":               scores,
        "critique":             data.get("critique", []),
        "revise":               revise,
        "sections_to_revise":   sections_to_revise[:6],
        "revision_instructions":data.get("revision_instructions", ""),
    }

    await _try_save_artifact(
        state, f"critique:{state.get('revision_count', 0)}", dict(critique_result),
        "orchestrator", "critic", "brief_critique",
        source_ids=state.get("source_ids"),
    )
    return {"critique": critique_result}


critic.default_worker_class = "orchestrator"


def should_revise(state: AgentState) -> str:
    """Route to brief_revision_agent if revision needed and under limit."""
    count = state.get("revision_count") or 0
    if count >= MAX_REVISIONS:
        return "final_formatter"
    critique = state.get("critique")
    if critique and critique.get("revise") and critique.get("sections_to_revise"):
        return "brief_revision_agent"
    return "final_formatter"


# ─────────────────────────────────────────────────────────────────────────────
# 16. brief_revision_agent
# ─────────────────────────────────────────────────────────────────────────────

async def brief_revision_agent(state: AgentState) -> Dict:
    """
    Apply targeted revisions to sections flagged by the critic.
    Re-assembles the brief and updates assembled_brief for re-evaluation.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import BRIEF_RETRIEVER_TOOLS

    critique  = state.get("critique") or {}
    targets   = critique.get("sections_to_revise") or []
    instructions = critique.get("revision_instructions") or ""
    final_sections = list(state.get("final_sections") or state.get("raw_sections") or [])
    manifest  = state.get("drafting_manifest") or {}
    evidence_cards = state.get("evidence_cards") or []

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=BRIEF_RETRIEVER_TOOLS,
    )
    hybrid_tool = next((t for t in tools if t.name == "hybrid_search"), None)

    section_map: Dict[str, SectionDraft] = {s["section_id"]: s for s in final_sections}

    for target_id in targets[:4]:
        section = section_map.get(target_id)
        if not section:
            continue

        # Re-retrieve fresh evidence
        fresh_ctx = ""
        if hybrid_tool:
            try:
                q = f"{target_id.replace('_', ' ')} {manifest.get('case_name', '')} case brief"
                raw_chunks = await hybrid_tool.ainvoke({"query": q, "k": 12})
                chunks = json.loads(raw_chunks)
                fresh_ctx = "\n\n".join(c.get("content", "") for c in chunks[:8])
            except Exception:
                pass

        # Also use existing evidence cards
        relevant_cards = _cards_for_roles(evidence_cards, _SECTION_CARD_ROLES.get(target_id, ["facts"]))
        cards_ctx = _cards_context(relevant_cards, max_cards=8)

        system = (
            "You are a law professor revising a weak brief section. "
            "Revise ONLY what the critique flags. Do not add unsupported claims.\n\n"
            "Return JSON: {section_id, title, draft_text, claims:[{claim,supporting_card_ids}], "
            "word_count, warnings}"
        )

        prompt = (
            f"Critique instructions: {instructions}\n\n"
            f"Section to revise (id: {target_id}):\n{section['draft_text']}\n\n"
            f"Fresh evidence:\n{fresh_ctx[:1500]}\n\n"
            f"Evidence cards:\n{cards_ctx}\n\n"
            f"Revise to address the critique."
        )

        raw = await _llm("worker_mid", prompt, system=system, max_tokens=1200)
        try:
            data = _parse_json(raw)
        except Exception:
            data = {}

        revised: SectionDraft = {
            "section_id": target_id,
            "title":      data.get("title", section["title"]),
            "draft_text": data.get("draft_text", section["draft_text"]),
            "claims":     data.get("claims", section["claims"]),
            "word_count": data.get("word_count", section["word_count"]),
            "warnings":   section.get("warnings", []) + ["[revised by brief_revision_agent]"],
        }
        section_map[target_id] = revised

    updated_sections = [section_map.get(s["section_id"], s) for s in final_sections]
    reassembled = _assemble_markdown(updated_sections, manifest)

    return {
        "final_sections":  updated_sections,
        "assembled_brief": reassembled,
        "revision_count":  (state.get("revision_count") or 0) + 1,
    }


brief_revision_agent.default_worker_class    = "worker_mid"
brief_revision_agent.escalation_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 17. final_formatter
# ─────────────────────────────────────────────────────────────────────────────

async def final_formatter(state: AgentState) -> Dict:
    """
    Format the approved brief for student use.
    Applies headings, TOC, source references, length stats, and clean markdown.
    Does NOT perform new legal reasoning.
    """
    coherence_edit = state.get("coherence_edit") or {}
    brief_text = coherence_edit.get("edited_markdown") or state.get("assembled_brief") or ""
    manifest   = state.get("drafting_manifest") or {}
    budget     = state.get("budget") or {}

    if not brief_text:
        return {"final_output": ""}

    system = (
        "You are the final formatting agent for a law-school case brief. "
        "Do not perform new legal reasoning. Format verified content into clean markdown.\n\n"
        "Rules:\n"
        "  - Ensure consistent heading hierarchy (# for title, ## for sections, ### for subsections)\n"
        "  - Bold the rule statement and holding in each section\n"
        "  - Issue is formatted 'Whether ...' on its own line\n"
        "  - Holding answers the issue directly on the next line\n"
        "  - Exam triggers in a bulleted list under exam_translation\n"
        "  - Cold-call questions numbered\n"
        "  - Do not include raw JSON, code fences, or template artifacts\n"
        "  - Preserve all existing content; clean formatting only\n"
        "Return only the final Markdown — no explanation."
    )

    prompt = (
        f"Case brief to format:\n\n{brief_text[:7000]}\n\n"
        f"Format for student use."
    )

    formatted = await _llm("worker_low", prompt, system=system, max_tokens=7000)

    final_sections = state.get("final_sections") or state.get("raw_sections") or []
    total_words    = sum(s.get("word_count", 0) for s in final_sections)
    est_pages      = max(1, round(total_words / 250))

    budget_note = (
        f"\n\n<!-- case-brief-agent | "
        f"tokens: {budget.get('input_tokens', 0)} in / {budget.get('output_tokens', 0)} out | "
        f"est. ${budget.get('cost_usd', 0):.4f} | "
        f"words: {total_words} (~{est_pages}p) | "
        f"revisions: {state.get('revision_count', 0)} -->"
    )
    final = formatted + budget_note

    await _try_save_artifact(
        state, "final_output", {"markdown": final, "word_count": total_words},
        "worker_low", "final_formatter", "final_output",
        source_ids=state.get("source_ids"),
    )
    return {"final_output": final}


final_formatter.default_worker_class = "worker_low"
