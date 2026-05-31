# agents/case_brief/nodes.py
"""
All nodes for the case-brief LangGraph agent.

Node index (in execution order):
  1.  head_orchestrator            — orchestrator; plans brief mode, scope, budget
  2.  source_profiler              — worker_mid; per-source profile (parallel Send)
  3.  retrieval_planner            — orchestrator; inlines corpus orientation, generates retrieval probes
  4.  planned_retriever            — tool_only; executes + reranks per target (parallel Send)
  5.  evidence_card_builder        — worker_low; converts chunks → evidence cards (parallel Send)
  6.  legal_artifact_extractor     — worker_mid; extracts typed artifact per extractor_type (parallel ×4)
  7.  doctrinal_synthesizer        — worker_mid; resolves holding, rule, limits, exam triggers
  8.  brief_drafter                — orchestrator; creates drafting manifest
  9.  section_writer               — worker_mid; writes one section per section_type (parallel ×10)
  10. section_grounder             — worker_mid; verifies each section against evidence cards (parallel)
  11. brief_assembler              — worker_low; merges sections in manifest order (raw_sections fallback)
  12. critic                       — orchestrator; scores brief, issues revision targets
  13. brief_revision_agent         — worker_mid; targeted revision of flagged sections, re-assembles
  14. final_formatter              — worker_mid; synthesis into 16-section standard template

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
from utils.llm_clients.anthropic_rate_limits import get_llm_semaphore

logger = logging.getLogger(__name__)

MAX_REVISIONS = 3

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


# ── Rate-limit guard ──────────────────────────────────────────────────────────
_LLM_RATE_LIMIT_RETRIES = 3
_LLM_RATE_LIMIT_DELAY   = 60   # seconds (linear: 60, 120, 180 s)
_LLM_CALL_TIMEOUT       = 180  # seconds before a hung LLM call is aborted

_DEEPSEEK_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _get_deepseek_semaphore() -> asyncio.Semaphore:
    global _DEEPSEEK_SEMAPHORE
    if _DEEPSEEK_SEMAPHORE is None:
        _DEEPSEEK_SEMAPHORE = asyncio.Semaphore(10)
    return _DEEPSEEK_SEMAPHORE


# ── Debug helpers ─────────────────────────────────────────────────────────────

def _node_start(name: str, state: Dict, **extras: Any) -> None:
    job = (state.get("job_id") or "")[:8] or "no-job"
    parts = " ".join(f"{k}={v}" for k, v in extras.items())
    logger.info("▶ [%s] %s  %s", job, name, parts)


def _node_done(name: str, state: Dict, **extras: Any) -> None:
    job = (state.get("job_id") or "")[:8] or "no-job"
    parts = " ".join(f"{k}={v}" for k, v in extras.items())
    logger.info("✓ [%s] %s  %s", job, name, parts)


def _node_warn(name: str, state: Dict, msg: str) -> None:
    job = (state.get("job_id") or "")[:8] or "no-job"
    logger.warning("⚠ [%s] %s  %s", job, name, msg)


def _llm_call(name: str, worker_class: str, model_name: str, max_tokens: int) -> None:
    logger.info("  🤖 [%s] LLM %s (%s) max_tokens=%d", name, worker_class, model_name, max_tokens)


# ── Shared helpers ─────────────────────────────────────────────────────────────

async def _llm(
    worker_class: str,
    prompt: str,
    system: str = "",
    max_tokens: int = 2048,
    provider: Optional[str] = None,
    _node: str = "",
) -> str:
    """Call the LLM with rate-limit protection and DeepSeek thinking-mode support."""
    from utils.llm_clients.llm_factory import LLMFactory
    _provider, model_name, thinking = _fetch_worker_model(worker_class, provider)
    if _node:
        _llm_call(_node, worker_class, model_name, max_tokens)

    # Auto-append token budget soft limit to every system prompt that doesn't already have one
    if system and "IMPORTANT" not in system:
        system = system + f"\n\n**IMPORTANT**: keep your response under {max_tokens} tokens."

    client_kwargs: Dict[str, Any] = {}
    if thinking is not None:
        client_kwargs["thinking"] = thinking

    client = LLMFactory.get_client_for(
        _provider, model_name,
        temperature=0.7, streaming=False, max_output_tokens=max_tokens,
        **client_kwargs,
    )

    if _provider == "deepseek":
        sem = _get_deepseek_semaphore()
    else:
        sem = await get_llm_semaphore()

    async with sem:
        for attempt in range(_LLM_RATE_LIMIT_RETRIES + 1):
            try:
                if hasattr(client, "achat"):
                    return await asyncio.wait_for(
                        client.achat(prompt, system_prompt=system or None),
                        timeout=_LLM_CALL_TIMEOUT,
                    )
                async def _stream() -> str:
                    chunks: List[str] = []
                    async for chunk in client.stream_chat(prompt, system_prompt=system or None):
                        chunks.append(chunk)
                    return "".join(chunks)
                return await asyncio.wait_for(_stream(), timeout=_LLM_CALL_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning(
                    "⏱️ [%s] LLM call timed out after %ds (attempt %d/%d)",
                    _node or worker_class, _LLM_CALL_TIMEOUT,
                    attempt + 1, _LLM_RATE_LIMIT_RETRIES + 1,
                )
                raise
            except Exception as exc:
                err = str(exc)
                is_rate_limit = (
                    "429" in err
                    or "rate_limit" in err.lower()
                    or "rate limit" in err.lower()
                )
                if is_rate_limit and attempt < _LLM_RATE_LIMIT_RETRIES:
                    wait = _LLM_RATE_LIMIT_DELAY * (attempt + 1)
                    logger.warning(
                        "🚦 [%s] Rate limit hit (429) — waiting %ds before retry %d/%d",
                        _node or worker_class, wait, attempt + 1, _LLM_RATE_LIMIT_RETRIES,
                    )
                    await asyncio.sleep(wait)
                    continue
                # Retry transient network drops (httpx.ReadError, connection resets)
                # that occur mid-stream on long DeepSeek calls.
                is_network_error = (
                    getattr(exc.__class__, "__module__", "").startswith("httpx")
                    or any(
                        kw in type(exc).__name__
                        for kw in ("ReadError", "ConnectError", "RemoteProtocol", "ConnectionReset")
                    )
                )
                if is_network_error and attempt < _LLM_RATE_LIMIT_RETRIES:
                    wait = 5 * (attempt + 1)
                    logger.warning(
                        "🌐 [%s] Network error (%s) — waiting %ds before retry %d/%d",
                        _node or worker_class, type(exc).__name__, wait,
                        attempt + 1, _LLM_RATE_LIMIT_RETRIES,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise


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
    """Strip markdown code fences then parse JSON. Raises json.JSONDecodeError on failure.
    Automatically emits a 😵‍💫 warning when the failure looks like output truncation.
    """
    import inspect
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as exc:
        frame = inspect.currentframe()
        try:
            caller = frame.f_back.f_code.co_name if frame and frame.f_back else "?"
        finally:
            del frame
        _check_token_limit(exc, caller, {})
        raise


def _check_token_limit(exc: Exception, node_name: str, state: Dict) -> None:
    """
    Detect when a JSON parse failure was caused by output truncation (max_tokens hit).
    'Unterminated string' / 'Unexpected end' are the json module's tell-tale messages
    when the LLM response was cut off mid-JSON.
    Logs a distinct 😵‍💫 line so token-limit failures are instantly recognisable in logs.
    """
    err = str(exc).lower()
    if "unterminated string" in err or "unexpected end" in err or "end of data" in err:
        job = (state.get("job_id") or "")[:8] or "no-job"
        logger.warning("😵‍💫 [%s] %s  Output-token-limit error — response truncated mid-JSON; "
                       "increase max_tokens for this node", job, node_name)


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

    _node_start("head_orchestrator", state,
                n_sources=len(state.get("source_ids", [])))

    system = (
        "You are the T-14 workflow controller for a law-school case brief generation graph.\n\n"
        "PLANNING REQUIREMENTS:\n"
        "1. Determine brief_mode based on source types: single_case = one primary opinion; "
        "   multi_case = multiple opinions for doctrine comparison; "
        "   casebook_excerpt = edited excerpt with professor notes; "
        "   doctrine_packet = multiple sources on a single doctrine.\n"
        "2. Set target_length based on brief complexity: "
        "   'short' = < 5 sources or simple rule; "
        "   'standard' = most cases; "
        "   'long' = multi-issue cases, policy-heavy, or with significant dissents.\n"
        "3. Set retrieval_depth = 'deep' if the source has a dissent, policy debate, "
        "   or circuit split — these require extra retrieval probes.\n"
        "4. Set revision_policy = 'strict' if the user asked for exam-ready or "
        "   cold-call quality; 'standard' otherwise.\n"
        "Do not write the brief. Return ONLY JSON with keys:\n"
        "  job_type: 'case_brief'\n"
        "  brief_mode: 'single_case' | 'multi_case' | 'casebook_excerpt' | 'doctrine_packet' | 'mixed_source'\n"
        "  source_ids: list of source UUIDs in scope\n"
        "  target_length: 'short' | 'standard' | 'long'\n"
        "  output_format: 'markdown'\n"
        "  stages: list of pipeline stage names\n"
        "  retrieval_depth: 'shallow' | 'standard' | 'deep'\n"
        "  revision_policy: 'strict' | 'standard' | 'permissive'"
    )

    raw = await _llm(
        "orchestrator",
        f"Sources available:\n{sources_json}\n\nUser request: {state['request']}",
        system=system,
        max_tokens=640,
        _node="head_orchestrator",
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
    _node_done("head_orchestrator", state,
               brief_mode=job_plan.get("brief_mode", "?"),
               target_length=job_plan.get("target_length", "?"),
               retrieval_depth=job_plan.get("retrieval_depth", "?"))
    return {"job_plan": job_plan}


head_orchestrator.default_worker_class = "orchestrator"


def head_orchestrator_to_profiler(state: AgentState) -> List[Send]:
    """Fan-out: one source_profiler per source document."""
    source_ids = state["source_ids"]
    logger.info("case_brief x%d node fan out for source_profiler", len(source_ids))
    return [
        Send("source_profiler", {"source_id": sid, **state})
        for sid in source_ids
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
# 3. retrieval_planner
# ─────────────────────────────────────────────────────────────────────────────

async def retrieval_planner(state: AgentState) -> Dict:
    """
    Inlines corpus orientation synthesis, then generates artifact-specific retrieval
    probes for every brief section. Absorbs corpus_orientation_synthesizer to eliminate
    a serial LLM round-trip between source_profiler fan-out and retrieval planning.
    """
    profiles = state.get("source_profiles") or []
    job_plan = state.get("job_plan") or {}

    # ── Inline corpus orientation (was corpus_orientation_synthesizer) ──────────
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

    orient_system = (
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

    orient_prompt = (
        f"Source profiles:\n{json.dumps(profiles_brief, indent=2)}\n\n"
        f"Brief mode from plan: {job_plan.get('brief_mode', 'single_case')}\n"
        f"User request: {state['request']}\n\n"
        f"Identify the primary case and how the sources relate."
    )

    orient_raw = await _llm("worker_mid", orient_prompt, system=orient_system, max_tokens=768)
    try:
        orient_data = _parse_json(orient_raw)
    except Exception:
        orient_data = {}

    primary_id = orient_data.get("primary_case_source_id", "")
    if not primary_id and profiles:
        opinions = [p for p in profiles if p["doc_type_guess"] == "full_case_opinion"]
        primary_id = (opinions[0] if opinions else profiles[0])["source_id"]

    orientation: CorpusOrientation = {
        "primary_case_source_id": primary_id,
        "supporting_source_ids": [
            s for s in (orient_data.get("supporting_source_ids") or []) if s != primary_id
        ],
        "case_brief_scope": orient_data.get("case_brief_scope", "single_case"),
        "source_roles": orient_data.get("source_roles", [
            {"source_id": p["source_id"], "role": p["doc_type_guess"]} for p in profiles
        ]),
        "primary_case_name": orient_data.get("primary_case_name", ""),
        "briefing_notes": orient_data.get("briefing_notes", ""),
    }
    # ─────────────────────────────────────────────────────────────────────────

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
        "Produce a plan for ALL targets. "
        "**IMPORTANT**: keep the total JSON response under 4 000 tokens "
        "(~440 tokens per target). Use terse keyword phrases — not prose sentences — "
        "for queries and patterns. Return only the JSON array."
    )

    prompt = (
        f"Case being briefed: {case_name}\n"
        f"Primary source: {primary_id}\n"
        f"All source sections: {json.dumps(section_index, indent=2)}\n\n"
        f"Retrieval depth: {job_plan.get('retrieval_depth', 'standard')}\n"
        f"User request: {state['request']}\n\n"
        f"Generate retrieval plans for all brief artifacts."
    )

    raw = await _llm("orchestrator", prompt, system=system, max_tokens=5000)
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
    return {"retrieval_plans": plans, "corpus_orientation": orientation}


retrieval_planner.default_worker_class = "orchestrator"


def retrieval_planner_to_retriever(state: AgentState) -> List[Send]:
    """Fan-out: one planned_retriever per retrieval plan."""
    plans = state.get("retrieval_plans") or []
    logger.info("case_brief x%d node fan out for planned_retriever", len(plans))
    return [
        Send("planned_retriever", {"plan": p, **state})
        for p in plans
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
    bundles = state.get("retrieval_bundles") or []
    logger.info("case_brief x%d node fan out for evidence_card_builder", len(bundles))
    return [
        Send("evidence_card_builder", {"bundle": b, **state})
        for b in bundles
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

    max_tokens = 3000
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
        f"Extract ALL evidence cards present. Return only the JSON array. "
        f"**IMPORTANT**: keep your total response under {max_tokens} tokens — "
        f"use tight quotes (1-2 sentences max) and omit any explanation outside the JSON array."
    )

    prompt = (
        f"Retrieval target: {target}\n\n"
        f"Chunks:\n{context}\n\n"
        f"Build evidence cards."
    )

    raw = await _llm("worker_low", prompt, system=system, max_tokens=max_tokens)
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
    logger.info("case_brief x%d node fan out for legal_artifact_extractor", len(extractor_types))
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
            "You are a T-14 law professor specialising in procedural posture and "
            "material fact extraction.\n\n"
            "PROCEDURAL POSTURE REQUIREMENTS:\n"
            "- lower_court: name the specific lower court and what it held.\n"
            "- current_stage: e.g., 'appeal to N.Y. Court of Appeals from Appellate Division'.\n"
            "- standard_or_frame: the standard of review or procedural frame that controls "
            "  analysis — e.g., 'de novo review of legal issues', 'Rule 12(b)(6) motion'.\n"
            "- disposition_below: what the lower court actually did (granted/denied/reversed).\n\n"
            "MATERIAL FACTS REQUIREMENTS:\n"
            "- A fact is material if removing it would change the legal outcome.\n"
            "- For each material fact: state the fact precisely, explain WHY it is material "
            "  (which element it establishes or defeats), and cite the supporting card_ids.\n"
            "- Background facts (narrative colour without legal significance) go in "
            "  background_facts — max 5 items.\n"
            "- List uncertainties where the record is ambiguous.\n"
            "Return ONLY JSON:\n"
            "  procedural_posture: {lower_court, current_stage, standard_or_frame, "
            "disposition_below, supporting_card_ids}\n"
            "  material_facts: [{fact, why_material, supporting_card_ids}] (max 10)\n"
            "  background_facts: [str] (max 5)\n"
            "  uncertainties: [str]"
        ),
        "issue_holding": (
            "You are a T-14 law professor specialising in issue formulation and holding "
            "extraction.\n\n"
            "ISSUE REQUIREMENTS:\n"
            "- Format: 'Whether [legal standard] applies when [specific facts].' — "
            "  NOT 'Whether the defendant was negligent' (too broad).\n"
            "- The issue must be tied to the specific facts of this case, not the general doctrine.\n"
            "- If there are multiple issues (rare), list the primary one.\n\n"
            "HOLDING REQUIREMENTS:\n"
            "- holding_narrow: answers the issue yes/no where possible, "
            "  states the specific rule applied, and limits it to these facts.\n"
            "- Do NOT conflate the holding with the disposition. Holding = legal rule. "
            "  Disposition = what the court ordered (affirmed/reversed).\n"
            "- confidence: 0.9+ if directly quoted; 0.7 if closely paraphrased; "
            "  < 0.7 if uncertain.\n"
            "Return ONLY JSON:\n"
            "  issue: 'Whether ...' string\n"
            "  holding_narrow: string (answers the issue, states the rule, limits to these facts)\n"
            "  disposition: 'affirmed' | 'reversed' | 'remanded' | 'vacated' | 'modified' | 'other'\n"
            "  winner: 'plaintiff' | 'defendant' | 'appellant' | 'appellee' | 'unclear'\n"
            "  confidence: float 0.0-1.0\n"
            "  supporting_card_ids: [str]"
        ),
        "rule_reasoning": (
            "You are a T-14 doctrine extraction specialist.\n\n"
            "RULE EXTRACTION REQUIREMENTS:\n"
            "- black_letter: the operative legal rule in reusable standalone form — "
            "  state it without reference to this case by name.\n"
            "- test: the multi-part test or balancing factors, each as a separate string.\n"
            "- elements: the required elements a party must prove, each as a separate string.\n"
            "- exceptions: factual or legal conditions that take a case outside the rule.\n"
            "- limitations: scope restrictions — when the rule does NOT apply.\n\n"
            "REASONING EXTRACTION REQUIREMENTS:\n"
            "- Classify each reasoning step by type:\n"
            "  application = applying the rule to facts; "
            "  precedent = relying on prior cases; "
            "  policy = invoking policy goals; "
            "  textual = interpreting statutory/constitutional text; "
            "  institutional = deferring to another body; "
            "  fairness = invoking equitable considerations; "
            "  administrability = choosing rules that are easier to apply.\n"
            "- Each reasoning entry must have a specific text (not a summary).\n"
            "Return ONLY JSON:\n"
            "  rule: {black_letter, test: [str], elements: [str], "
            "exceptions: [str], limitations: [str], supporting_card_ids: [str]}\n"
            "  reasoning: [{type, text}]"
        ),
        "dissent": (
            "You are a T-14 law professor specialising in separate opinion analysis.\n\n"
            "DISSENT EXTRACTION REQUIREMENTS:\n"
            "- If has_dissent is false, return empty fields — do NOT invent a dissent.\n"
            "- dissent_summary (2-3 sentences): what the dissent argues; "
            "  specifically which element of the majority's analysis it rejects.\n"
            "- alternative_rule: the rule the dissent would apply instead — "
            "  state it in the same form as the majority's rule for easy comparison.\n"
            "- key_disagreement: the single most important point of doctrinal disagreement "
            "  (e.g., 'majority defines duty by geographic proximity; "
            "  dissent defines it by foreseeability alone').\n"
            "- The dissent is a HIGH-YIELD exam and cold-call source — extract it precisely.\n"
            "Return ONLY JSON:\n"
            "  has_dissent: bool\n"
            "  has_concurrence: bool\n"
            "  dissent_summary: str\n"
            "  concurrence_summary: str\n"
            "  alternative_rule: str\n"
            "  key_disagreement: str\n"
            "  supporting_card_ids: [str]"
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

    _node_start("doctrinal_synthesizer", state,
                case=orientation.get("primary_case_name", "?")[:30])

    system = (
        "You are a T-14 law tutor synthesising a case brief into a complete doctrinal "
        "understanding.\n\n"
        "SYNTHESIS REQUIREMENTS:\n"
        "- doctrinal_role: how this case fits in the doctrine arc — "
        "'introduces' (first case on this rule), 'refines' (narrows/clarifies), "
        "'limits' (carves out exception), 'overrules' (changes prior rule), "
        "'applies' (routine application), 'distinguishes' (draws a line).\n"
        "- narrow_holding: the specific answer to the issue in these facts — "
        "NOT a general rule statement.\n"
        "- broad_rule: the general rule students should take away — "
        "stated without reference to the case name.\n"
        "- rule_limits: factual or doctrinal situations where the rule does NOT apply "
        "— these become the basis for wrong MCQ distractors and cold-call traps.\n"
        "- exam_triggers: specific fact patterns that should trigger citation of this case — "
        "'use this case when [fact pattern]'.\n"
        "- cold_call_traps: the 3-5 mistakes professors most often catch students on — "
        "e.g., over-reading the holding, ignoring the procedural posture, "
        "confusing dicta with the rule.\n"
        "- common_misreadings: overbroad statements of the rule that are wrong — "
        "state the misreading, then the correct formulation.\n"
        "- related_doctrines: adjacent rules or cases students should connect to this one.\n"
        "- pedagogical_note: 1-2 sentences on WHY this case was assigned and what "
        "doctrinal move the professor wants students to learn.\n"
        "Return ONLY JSON. No extra text."
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

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=2000,
                     _node="doctrinal_synthesizer")
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
    _node_done("doctrinal_synthesizer", state,
               doctrinal_role=synthesis.get("doctrinal_role", "?"),
               n_triggers=len(synthesis.get("exam_triggers", [])),
               n_traps=len(synthesis.get("cold_call_traps", [])))
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
    sends = [
        Send("section_writer", {"section_type": sp["section_id"], **state})
        for sp in section_plan
        if sp.get("include", True)
    ]
    logger.info("case_brief x%d node fan out for section_writer", len(sends))
    return sends


# ─────────────────────────────────────────────────────────────────────────────
# 10. section_writer  (parallel ×N, dispatched by section_type)
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_SYSTEMS = {
    "case_identity": (
        "You are a T-14 law professor writing the case identity section of a case brief.\n\n"
        "STANDARDS:\n"
        "- Use ONLY supplied metadata and evidence cards — do not infer missing fields.\n"
        "- Include: full case name (both parties), court, year decided, citation (if available), "
        "  parties' roles (plaintiff/defendant/appellant/appellee), and source document type.\n"
        "- If citation is not in the evidence, omit it — do not fabricate.\n"
        "- draft_text: 2-4 sentences, clean and precise.\n"
        "Return JSON: {section_id, title, draft_text, claims:[{claim,supporting_card_ids}], "
        "word_count, warnings:[str]}"
    ),
    "procedural_posture": (
        "You are a T-14 law professor writing the procedural posture section.\n\n"
        "STANDARDS:\n"
        "- State how the case reached this court: lower court name + what it held + "
        "  who appealed + current stage.\n"
        "- Identify the standard/frame that controls analysis "
        "  (e.g., 'de novo review of legal issues', 'Rule 12(b)(6) pleading standard', "
        "  'abuse of discretion').\n"
        "- Note the final disposition (affirmed, reversed, remanded).\n"
        "- The procedural frame is critical on exams — students who confuse the frame "
        "  misapply the rule. Make it explicit.\n"
        "Return JSON: {section_id, title, draft_text, procedural_stage, "
        "standard_or_frame, claims, word_count, warnings:[str]}"
    ),
    "facts": (
        "You are a T-14 law professor writing the material facts section.\n\n"
        "STANDARDS:\n"
        "- ONLY include facts the court actually relied on — removing a material fact "
        "  should change the legal outcome.\n"
        "- For each fact: state it precisely and explain WHY it matters "
        "(which element it establishes, or which rule it triggers).\n"
        "- Omit narrative background that does not affect the outcome — "
        "  list omitted background facts briefly in omitted_background_facts.\n"
        "- Do NOT characterise facts as good or bad — state them neutrally.\n"
        "- draft_text: present tense, tight prose. Each sentence carries legal weight.\n"
        "Return JSON: {section_id, title, draft_text, "
        "material_facts:[{fact,why_material,supporting_card_ids}], "
        "omitted_background_facts, word_count, claims, warnings:[str]}"
    ),
    "issue_holding": (
        "You are a T-14 law professor writing the issue and holding section.\n\n"
        "STANDARDS:\n"
        "- Issue format: 'Whether [legal standard] applies when [specific facts from this case].' "
        "— NOT a generic question about the doctrine.\n"
        "- Issue and holding must mirror each other: the holding answers the issue "
        "yes or no (where possible) and states the operative legal rule.\n"
        "- holding_narrow: stays narrow to these facts — do not over-generalise to "
        "create a rule that does not exist in the opinion.\n"
        "- Distinguish holding from disposition: holding = rule of decision; "
        "disposition = what the court ordered (affirmed/reversed).\n"
        "Return JSON: {section_id, title, issue, holding, disposition, "
        "draft_text, claims, word_count}"
    ),
    "rule": (
        "You are a T-14 law professor writing the governing rule section.\n\n"
        "STANDARDS:\n"
        "- black_letter_rule: standalone restatement usable in a future case "
        "without referring back to this case by name.\n"
        "- test: the multi-factor test or elements, each on a separate line.\n"
        "- exceptions: specific factual conditions that take a case outside the rule.\n"
        "- limitations: scope restrictions — when and to whom the rule does not apply.\n"
        "- The rule section is the most exam-tested section — be precise and reusable.\n"
        "- Do NOT include reasoning or policy in the rule section; those go in 'reasoning'.\n"
        "Return JSON: {section_id, title, black_letter_rule, test, elements, "
        "exceptions, limitations, draft_text, claims, word_count}"
    ),
    "reasoning": (
        "You are a T-14 law professor writing the reasoning section.\n\n"
        "STANDARDS:\n"
        "- Trace the logical path from facts → rule → holding.\n"
        "- Classify each reasoning step: "
        "application (rule → facts), precedent (prior cases), policy (goals served), "
        "textual (statutory/constitutional language), institutional (deference), "
        "fairness (equitable), administrability (workable rules).\n"
        "- Identify the PRIMARY reasoning type — most courts use 1-2 dominant types.\n"
        "- Do NOT summarise the facts again — assume the reader has read the facts section.\n"
        "- If the court's reasoning is weak or questionable, note it briefly "
        "  (this is what professors challenge on cold calls).\n"
        "Return JSON: {section_id, title, reasoning_outline:[str], "
        "draft_text, word_count, warnings:[str]}"
    ),
    "dissent": (
        "You are a T-14 law professor writing the dissent section.\n\n"
        "STANDARDS:\n"
        "- Write this section ONLY if the evidence cards contain a dissent or concurrence. "
        "  If none exists, return has_section: false with empty draft_text.\n"
        "- majority_rule: state the majority's operative rule in one sentence.\n"
        "- dissent_rule: state the dissent's alternative rule in one sentence "
        "  (in the same structure as the majority rule for easy comparison).\n"
        "- core_disagreement: one sentence on WHERE the majority and dissent part ways "
        "  (e.g., 'Majority defines duty by geographic proximity; dissent "
        "  by foreseeability alone').\n"
        "- The dissent is HIGH-YIELD for exams and cold calls — extract it carefully.\n"
        "Return JSON: {section_id, title, has_section, draft_text, "
        "majority_vs_dissent:{majority_rule,dissent_rule,core_disagreement}, "
        "word_count, warnings:[str]}"
    ),
    "pedagogy": (
        "You are a T-14 law professor writing the pedagogical note section.\n\n"
        "STANDARDS:\n"
        "- Answer WHY this case was assigned: what doctrinal move does the professor "
        "  want students to learn from it?\n"
        "- Identify the doctrinal_role: introduces / refines / limits / "
        "overrules / applies / distinguishes.\n"
        "- List common_misreadings: the overbroad or under-read versions of the rule "
        "  students typically fall into — these are what professors test on exams.\n"
        "- Reference casebook context or lecture note context where available in the evidence.\n"
        "Return JSON: {section_id, title, why_assigned, doctrinal_role, "
        "common_misreadings, draft_text, word_count}"
    ),
    "exam_translation": (
        "You are a T-14 law professor writing the exam translation section.\n\n"
        "STANDARDS:\n"
        "- use_this_case_when: specific fact triggers (at least 3) — "
        "'Use this case when the facts show [X]'.\n"
        "- do_not_overread_as: the overbroad statement students write on exams "
        "  (at least 2 examples) — 'Do NOT write that the rule applies whenever [Y]; "
        "  it is limited to [Z]'.\n"
        "- exam_hypo_triggers: concrete hypothetical fact patterns that should "
        "  trigger this case (at least 3).\n"
        "- cold_call_questions: the 2-3 questions a professor would ask based on "
        "  this case.\n"
        "- Do NOT overstate the doctrine — specificity beats breadth.\n"
        "Return JSON: {section_id, title, use_this_case_when, do_not_overread_as, "
        "exam_hypo_triggers, cold_call_questions, draft_text, word_count}"
    ),
    "cold_call": (
        "You are a T-14 law professor writing the cold-call survival guide section.\n\n"
        "STANDARDS:\n"
        "- List 3-5 questions a professor is most likely to ask about this case.\n"
        "- For each question:\n"
        "  question: the exact professor-voice question (e.g., 'What was the holding?', "
        "  'Why does it matter that plaintiff was outside the zone of danger?', "
        "  'What would the dissent say about this hypothetical?')\n"
        "  model_answer: a 2-4 sentence model answer that would survive a cold call "
        "  (Because/Unless/But/Therefore structure where appropriate).\n"
        "  why_asked: one sentence on WHY this question is a professor favourite — "
        "  what misconception or doctrinal nuance it tests.\n"
        "Return JSON: {section_id, title, questions:[{question,model_answer,why_asked}], "
        "draft_text, word_count}"
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
    sections = state.get("raw_sections") or []
    logger.info("case_brief x%d node fan out for section_grounder", len(sections))
    return [
        Send("section_grounder", {"section": s, **state})
        for s in sections
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
# 11. brief_assembler
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
# 12. critic
# ─────────────────────────────────────────────────────────────────────────────

async def critic(state: AgentState) -> Dict:
    """
    Perform final legal and pedagogical critique.
    Scores accuracy, briefing quality, exam usefulness, and citation discipline.
    Issues section revision targets if the brief would not survive a cold call.
    """
    brief_text = state.get("assembled_brief") or ""
    grounding_reports = state.get("grounding_reports") or []

    failing_sections = [r["section_id"] for r in grounding_reports if not r["grounding_pass"]]
    excerpt = brief_text[:4500]

    _node_start("critic", state,
                n_sections=len(state.get("final_sections") or state.get("raw_sections") or []),
                n_failing=len(failing_sections))

    system = (
        "You are a T-14 law professor performing a demanding critique of a student "
        "case brief to determine if it would survive a cold call and help on an exam.\n\n"
        "SCORING CRITERIA (0.0–10.0 each):\n"
        "  accuracy: Are all factual and legal claims correct? "
        "(0 = materially wrong; 10 = flawless)\n"
        "  briefing_quality: Is the issue tied to specific facts? Is the holding narrow? "
        "Is the rule reusable standalone? (0 = fails all; 10 = all excellent)\n"
        "  exam_usefulness: Are exam triggers specific? Is do-not-overread guidance present? "
        "Would a student know when to use this case? (0 = useless; 10 = exam-ready)\n"
        "  citation_discipline: Is every claim traced to source evidence? "
        "Are there unsupported assertions? (0 = hallucinated; 10 = fully grounded)\n\n"
        "PASS THRESHOLD: all scores >= 6.5 AND no failed grounding sections.\n"
        "CRITIQUE: list specific issues — name the section_id and the problem. "
        "Generic praise or criticism ('the brief is good overall') fails the critique standard.\n"
        "REVISION INSTRUCTIONS: 2-4 targeted sentences on what to fix and how.\n\n"
        "Return ONLY JSON:\n"
        "  quality_pass: bool\n"
        "  scores: {accuracy, briefing_quality, exam_usefulness, citation_discipline}\n"
        "  critique: [str] (specific per-section issues)\n"
        "  revise: bool\n"
        "  sections_to_revise: [str] (section_ids)\n"
        "  revision_instructions: str"
    )

    prompt = (
        f"Case brief excerpt:\n{excerpt}\n\n"
        f"Sections with grounding failures: {failing_sections}\n\n"
        f"Critique this brief."
    )

    raw = await _llm("orchestrator", prompt, system=system, max_tokens=1500,
                     _node="critic")
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
    _node_done("critic", state,
               quality_pass=critique_result["quality_pass"],
               revise=critique_result["revise"],
               n_targets=len(critique_result["sections_to_revise"]))
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
            "word_count, warnings:[str]}"
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

    # Clear grounding reports for revised sections — their old failures are stale.
    # The next critic pass evaluates the revised text on its merits without
    # being shown failures that no longer apply to the rewritten content.
    revised_ids = set(targets[:4])
    surviving_reports = [
        r for r in (state.get("grounding_reports") or [])
        if r["section_id"] not in revised_ids
    ]

    return {
        "final_sections":    updated_sections,
        "assembled_brief":   reassembled,
        "revision_count":    (state.get("revision_count") or 0) + 1,
        "grounding_reports": surviving_reports,
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
    brief_text = state.get("assembled_brief") or ""
    manifest   = state.get("drafting_manifest") or {}
    budget     = state.get("budget") or {}

    if not brief_text:
        return {"final_output": ""}

    _node_start("final_formatter", state,
                n_sections=len(state.get("final_sections") or []),
                revision_count=state.get("revision_count", 0))

    system = (
        "You are the final synthesis and formatting agent for a T-14 law-school case brief.\n"
        "Your job: reorganise the assembled draft into the STANDARD TEMPLATE below, "
        "synthesising any missing derived sections from the content already present. "
        "You may distil, reorder, and clarify — but do NOT invent facts or legal "
        "conclusions that are absent from the draft.\n\n"

        "FORMATTING RULES:\n"
        "  - Heading hierarchy: # for case title, ## for Roman-numeral sections, "
        "### for subsections.\n"
        "  - **Bold** every black-letter rule statement and every holding.\n"
        "  - Issues: 'Whether ...' form, one per numbered line.\n"
        "  - Holdings: bold, one sentence answering each issue.\n"
        "  - Rule elements: bulleted list, one element per bullet.\n"
        "  - Exam triggers: bulleted list.\n"
        "  - Cold-call: numbered Qs with model answer in blockquote (> **Model Answer:** ...).\n"
        "  - ⚠️ Do-not-overread warnings: prefix with '⚠️ '.\n"
        "  - Omit optional sections marked [omit if absent] when the draft contains no "
        "relevant content — do not hallucinate placeholder text.\n"
        "  - Do NOT include raw JSON, code fences, or template artifacts.\n"
        "Return ONLY the final Markdown document. No explanation, no preamble.\n\n"

        "STANDARD TEMPLATE (output sections in this exact order):\n\n"

        "# [Case Name]\n"
        "**Citation:** [full citation]  \n"
        "**Court:** [court]  **Decided:** [year]  \n"
        "**Disposition:** [outcome]  **Opinion:** [authoring justice]\n\n"
        "---\n\n"

        "## I. One-Sentence Rule\n"
        "[Single sentence: subject + may/cannot/must + condition. "
        "Captures the holding and its doctrinal significance in one line — "
        "the most quotable statement of what this case stands for.]\n\n"
        "---\n\n"

        "## II. Procedural Posture\n"
        "[Numbered steps: how the dispute moved from origin → trial → appeals → "
        "this court. Include what each court held and why.]\n\n"
        "---\n\n"

        "## III. Facts\n"
        "### A. Parties\n"
        "[Petitioner: ... | Respondent: ...]\n"
        "### B. Background & Context\n"
        "[Operative facts — who, what, where, scale of the enterprise or conduct]\n"
        "### C. The Dispute\n"
        "[Specific conduct at issue and why it led to litigation]\n\n"
        "---\n\n"

        "## IV. Statutory / Constitutional Framework  [omit if absent]\n"
        "[If a statute: identify the Act, list the key sections and their operative "
        "language (§ number + what it says). "
        "If a constitutional case: quote the clause at issue. "
        "Omit entirely if the case does not turn on a specific text.]\n\n"
        "---\n\n"

        "## V. Issues Presented\n"
        "1. Whether [issue 1]\n"
        "2. Whether [issue 2 — omit if only one issue]\n\n"
        "---\n\n"

        "## VI. Holdings\n"
        "**1.** [Direct answer to Issue 1]\n"
        "**2.** [Direct answer to Issue 2 — omit if only one issue]\n\n"
        "---\n\n"

        "## VII. Rule & Legal Test\n"
        "**[Black-letter rule — bold full sentence]**\n\n"
        "Elements / test:\n"
        "- [element or factor 1]\n"
        "- [element or factor 2]\n"
        "...\n\n"
        "**Limiting Principle:** [what the rule expressly does NOT cover]\n\n"
        "---\n\n"

        "## VIII. Reasoning\n"
        "[Court's analytical steps in logical sequence. "
        "Use ### subsections for multi-step reasoning "
        "(e.g., ### A. The Framing Move, ### B. The Key Distinction).]\n\n"
        "---\n\n"

        "## IX. Arguments & Court's Answers  [omit if absent]\n"
        "[The losing party's main arguments and the Court's specific responses. "
        "Format each as: **Argument:** [X] → **Court:** [Y]. "
        "Omit if the draft contains no adversarial framing.]\n\n"
        "---\n\n"

        "## X. Black Letter Law\n"
        "[Bulleted doctrine extracted from the case, grouped by category.]\n"
        "**[Category — e.g., Commerce Clause]:**\n"
        "- [rule]\n"
        "- [rule]\n\n"
        "---\n\n"

        "## XI. Doctrinal Significance\n"
        "[One focused paragraph: what jurisprudential shift this case marks, "
        "what earlier cases it modifies or distinguishes, "
        "where it sits in the doctrinal timeline.]\n\n"
        "---\n\n"

        "## XII. Dissent  [omit if absent]\n"
        "[Summary of dissent: who dissented, on what grounds, "
        "why it matters for understanding the majority. "
        "Omit if no dissent.]\n\n"
        "---\n\n"

        "## XIII. Pedagogy\n"
        "[What this case teaches, what concept it illustrates, "
        "how it fits the course arc, what a student should take away.]\n\n"
        "---\n\n"

        "## XIV. Exam Translation\n"
        "**Exam Triggers:**\n"
        "- [fact pattern that should make you think of this case]\n"
        "- [another trigger]\n\n"
        "⚠️ **Do-Not-Overread:** [what the holding expressly does NOT say — "
        "the most common exam mistake]\n\n"
        "---\n\n"

        "## XV. If/Then Case Map\n"
        "- **If** [strong analogy fact], **then** this case applies because [reason]\n"
        "- **If** [distinguishing fact], **then** this case does **not** apply because [reason]\n"
        "- **If** [ambiguous fact], **then** argue both sides using [factors]\n\n"
        "---\n\n"

        "## XVI. Cold Call Q&A\n"
        "**1.** [Question]\n"
        "> **Model Answer:** [answer]\n\n"
        "**2.** [Question]\n"
        "> **Model Answer:** [answer]\n\n"
        "**3.** [Question — include at least one that asks how this case differs from "
        "a related/prior case]\n"
        "> **Model Answer:** [answer]\n"
    )

    prompt = (
        f"Case brief draft to reorganise into the standard template:\n\n"
        f"{brief_text[:18000]}\n\n"
        f"Reorganise into the standard template. Synthesise all sections. "
        f"Omit optional sections only if the draft truly contains no relevant content."
    )

    formatted = await _llm("worker_mid", prompt, system=system, max_tokens=12000,
                           _node="final_formatter")

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
        "worker_mid", "final_formatter", "final_output",
        source_ids=state.get("source_ids"),
    )
    _node_done("final_formatter", state,
               total_words=total_words, est_pages=est_pages)
    return {"final_output": final}


final_formatter.default_worker_class = "worker_mid"
