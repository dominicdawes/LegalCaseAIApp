# agents/attack_outline/nodes.py
"""
All nodes for the attack-outline LangGraph agent.

Node index (in execution order):
  1.  head_orchestrator         — orchestrate/plan; validates sources, sets job plan
  2.  source_profiler           — worker_mid; per-doc profile (parallel Send)
  3.  corpus_topic_mapper       — worker_mid; cross-doc topic map
  4.  retrieval_planner         — orchestrator; targeted retrieval intents per topic
  5.  planned_retriever         — tool_only; executes + reranks retrieval per concept (parallel Send)
  6.  legal_artifact_extractor  — worker_mid; extracts typed legal artifacts per bundle (parallel Send)
  7.  artifact_normalizer       — worker_mid; deduplicates and normalises artifacts
  8.  concept_clusterer         — worker_mid; groups artifacts into doctrine modules
  9.  doctrine_graph_builder    — orchestrator; builds if/then doctrine graph
  10. attack_block_builder      — worker_mid; builds one attack block per doctrine (parallel Send)
  11. attack_outline_assembler  — worker_mid; orders + assembles final outline markdown
  12. grounding_verifier        — worker_mid; verifies claims per block (parallel Send)
  13. attack_outline_critic     — orchestrator; scores outline quality, issues revision targets
  14. revision_agent            — worker_mid; revises flagged blocks, re-assembles
  15. final_compressor_formatter— worker_low; final polish and student-ready formatting

Fan-out routing helpers (not nodes):
  head_orchestrator_to_profiler, retrieval_planner_to_retriever,
  retriever_to_extractor, doctrine_to_block_builder, assembler_to_verifier,
  should_revise
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from langgraph.types import Send

from .state import (
    AgentState,
    AttackBlock,
    AttackStep,
    ConceptCluster,
    ConceptRetrievalPlan,
    CritiqueResult,
    DoctrineEdge,
    DoctrineGraph,
    DoctrineNode,
    LegalArtifact,
    NormalizedArtifact,
    RankedBundle,
    SourceProfile,
    TopicEntry,
    VerificationReport,
)
from .worker_config import _fetch_worker_model, model_costs

logger = logging.getLogger(__name__)

MAX_REVISIONS = 2


# ── Shared helpers ────────────────────────────────────────────────────────────

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
        from uuid import UUID as _UUID
        from agents.ledger import AgentLedgerService
        ledger = AgentLedgerService()
        await ledger.save_artifact(
            job_id=_UUID(job_id_str),
            artifact_key=artifact_key,
            content=content,
            worker_class=worker_class,
            node_name=node_name,
            artifact_type=artifact_type,
            source_ids=[_UUID(s) for s in (source_ids or []) if s],
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


# ─────────────────────────────────────────────────────────────────────────────
# 1. head_orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def head_orchestrator(state: AgentState) -> Dict:
    """
    Owns the full attack-outline job.
    Reads the source list, validates IDs, determines outline mode, and produces
    a job plan that downstream nodes can consult for constraints and priorities.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import ATTACK_PLANNER_TOOLS

    tools = make_tools(
        state["project_id"],
        source_ids=state["source_ids"],
        use_voyage=state.get("use_voyage", False),
        tool_names=ATTACK_PLANNER_TOOLS,
    )
    list_sources_tool = next(t for t in tools if t.name == "list_sources")
    sources_json = await list_sources_tool.ainvoke({})

    system = (
        "You are the orchestrator for a T-14 law-school attack-outline generator. "
        "Given the available source documents, create a concise job plan for building "
        "a source-grounded attack outline. Identify: (1) how many distinct course areas "
        "are covered, (2) the likely depth of doctrine coverage, (3) whether the outline "
        "should be single-course or multi-course, (4) any retrieval constraints. "
        "Return JSON with keys: job_type, source_ids, outline_mode "
        "(single_course|multi_course), course_areas (list), target_format, stages (list), "
        "retrieval_depth (shallow|standard|deep)."
    )

    raw = await _llm(
        "orchestrator",
        f"Sources available:\n{sources_json}\n\nUser request: {state['request']}",
        system=system,
        max_tokens=768,
    )
    try:
        job_plan = _parse_json(raw)
    except Exception:
        job_plan = {
            "job_type": "attack_outline",
            "source_ids": state["source_ids"],
            "outline_mode": "single_course",
            "course_areas": [],
            "target_format": "concise decision-tree checklist",
            "stages": ["profiling", "retrieval", "extraction", "doctrine_graph", "assembly", "critique"],
            "retrieval_depth": "standard",
        }

    await _try_save_artifact(
        state,
        artifact_key="job_plan",
        content=job_plan,
        worker_class="orchestrator",
        node_name="head_orchestrator",
        artifact_type="job_plan",
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
    Profile a single source document.
    Identifies course area, document type, likely exam doctrines, and section map.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import ATTACK_PROFILER_TOOLS

    source_id = state["source_id"]
    project_id = state["project_id"]

    tools = make_tools(
        project_id,
        source_ids=[source_id],
        use_voyage=state.get("use_voyage", False),
        tool_names=ATTACK_PROFILER_TOOLS,
    )
    outline_tool = next(t for t in tools if t.name == "get_doc_outline")
    outline_json = await outline_tool.ainvoke({"source_id": source_id})
    outline = json.loads(outline_json)

    sections_brief = json.dumps(outline.get("toc", [])[:20], indent=2)
    concepts = outline.get("doc_concepts", [])[:15]
    doc_summary = outline.get("doc_summary") or ""

    system = (
        "You are a law professor analysing a legal source document. "
        "Return JSON with exactly these keys:\n"
        "  course_area: string (e.g. 'Civil Procedure', 'Torts', 'Contracts')\n"
        "  document_type: string (lecture_notes | casebook | outline | statute | other)\n"
        "  document_summary: string (2-3 sentences)\n"
        "  likely_exam_doctrines: array of {doctrine, sections (list of section_ids), "
        "priority (high|medium|low), reason}\n"
        "  section_map: array of {section_id, heading, summary} for the 10 most important sections\n"
        "Return only JSON — no other text."
    )

    prompt = (
        f"Source document summary: {doc_summary}\n\n"
        f"Key concepts extracted: {json.dumps(concepts)}\n\n"
        f"Table of contents (sections):\n{sections_brief}\n\n"
        f"Analyse this document for exam-outline purposes."
    )

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=1200)
    try:
        data = _parse_json(raw)
    except Exception:
        data = {}

    profile: SourceProfile = {
        "source_id": source_id,
        "course_area": data.get("course_area", "Unknown"),
        "document_type": data.get("document_type", "other"),
        "document_summary": data.get("document_summary", doc_summary[:500]),
        "likely_exam_doctrines": data.get("likely_exam_doctrines", [])[:12],
        "section_map": data.get("section_map", [])[:15],
    }

    await _try_save_artifact(
        state,
        artifact_key=f"source_profile:{source_id}",
        content=profile,
        worker_class="worker_mid",
        node_name="source_profiler",
        artifact_type="source_profile",
        source_ids=[source_id],
    )
    return {"source_profiles": [profile]}


source_profiler.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 3. corpus_topic_mapper
# ─────────────────────────────────────────────────────────────────────────────

async def corpus_topic_mapper(state: AgentState) -> Dict:
    """
    Combine all source profiles into a global topic map.
    Clusters doctrines across documents, detects overlap and gaps,
    identifies source-specific emphasis.
    """
    profiles = state.get("source_profiles") or []
    job_plan = state.get("job_plan") or {}

    profiles_brief = [
        {
            "source_id": p["source_id"],
            "course_area": p["course_area"],
            "document_type": p["document_type"],
            "doctrines": [d["doctrine"] for d in p.get("likely_exam_doctrines", [])[:8]],
        }
        for p in profiles
    ]

    system = (
        "You are a law professor mapping legal doctrine coverage across multiple documents. "
        "Return a JSON array of topic objects, each with:\n"
        "  topic_id: short_snake_case string\n"
        "  label: human-readable doctrine/topic name\n"
        "  source_ids: list of source UUIDs that cover this topic\n"
        "  sections: list of section_ids hinted from source profiles\n"
        "  priority: 1 (must include) | 2 (should include) | 3 (if space permits)\n\n"
        "Order by priority. Include all major doctrines; merge near-duplicates. "
        "Return only the JSON array — no other text."
    )

    prompt = (
        f"Source profiles:\n{json.dumps(profiles_brief, indent=2)}\n\n"
        f"Outline mode: {job_plan.get('outline_mode', 'single_course')}\n"
        f"User request: {state['request']}\n\n"
        f"Map all exam-relevant topics across these documents."
    )

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=2048)
    try:
        topic_map: List[TopicEntry] = _parse_json(raw)
    except Exception:
        # Fallback: one topic per doctrine per profile
        topic_map = []
        for p in profiles:
            for d in p.get("likely_exam_doctrines", []):
                tid = re.sub(r'\W+', '_', d["doctrine"].lower())[:40]
                topic_map.append({
                    "topic_id": tid,
                    "label": d["doctrine"],
                    "source_ids": [p["source_id"]],
                    "sections": d.get("sections", []),
                    "priority": 1 if d.get("priority") == "high" else 2,
                })

    await _try_save_artifact(
        state,
        artifact_key="topic_map",
        content={"topics": topic_map},
        worker_class="worker_mid",
        node_name="corpus_topic_mapper",
        artifact_type="topic_map",
        source_ids=state.get("source_ids"),
    )
    return {"topic_map": topic_map}


corpus_topic_mapper.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 4. retrieval_planner
# ─────────────────────────────────────────────────────────────────────────────

async def retrieval_planner(state: AgentState) -> Dict:
    """
    Convert the topic map into precise, multi-intent retrieval plans.
    For each topic generates targeted BM25 queries, vector queries, regex patterns,
    and section filters covering: rules, elements, exceptions, issue triggers,
    cases, policy, and exam traps.
    """
    topic_map = state.get("topic_map") or []
    source_profiles = state.get("source_profiles") or []

    # Build a section-index for fast lookup
    section_index: Dict[str, List[str]] = {}
    for p in source_profiles:
        for s in p.get("section_map", []):
            sid = s.get("section_id", "")
            section_index.setdefault(p["source_id"], []).append(sid)

    system = (
        "You are a law-school exam retrieval expert. For each topic, generate a "
        "precise retrieval plan covering every intent needed for an attack outline.\n\n"
        "For each topic return a JSON object with:\n"
        "  concept_id: the topic_id\n"
        "  concept_label: the label\n"
        "  source_ids: list of source UUIDs\n"
        "  retrieval_intents: array of intent objects, each with:\n"
        "    intent: one of [black_letter_rule, elements_test, exceptions, "
        "issue_triggers, cases, policy, exam_traps, defenses, remedies]\n"
        "    bm25_queries: 2-3 short keyword-dense queries\n"
        "    vector_queries: 1-2 semantic queries as full sentences\n"
        "    regex_patterns: 0-3 regex patterns for mandatory terms\n"
        "  section_filters: list of section_ids to prioritise\n\n"
        "Produce plans for ALL topics. Return a JSON array — no other text."
    )

    priority_topics = [t for t in topic_map if t.get("priority", 2) <= 2][:20]
    prompt = (
        f"Topics to plan retrieval for:\n{json.dumps(priority_topics, indent=2)}\n\n"
        f"Available sections by source:\n{json.dumps(section_index, indent=2)}\n\n"
        f"User request: {state['request']}"
    )

    raw = await _llm("orchestrator", prompt, system=system, max_tokens=4096)
    try:
        plans: List[ConceptRetrievalPlan] = _parse_json(raw)
    except Exception:
        # Fallback: minimal plan per topic
        plans = [
            {
                "concept_id": t["topic_id"],
                "concept_label": t["label"],
                "source_ids": t.get("source_ids", state["source_ids"]),
                "retrieval_intents": [
                    {
                        "intent": "black_letter_rule",
                        "bm25_queries": [t["label"] + " rule elements test"],
                        "vector_queries": ["Legal rule and elements for " + t["label"]],
                        "regex_patterns": [],
                    }
                ],
                "section_filters": t.get("sections", []),
            }
            for t in priority_topics
        ]

    await _try_save_artifact(
        state,
        artifact_key="retrieval_plans",
        content={"plans": plans},
        worker_class="orchestrator",
        node_name="retrieval_planner",
        artifact_type="retrieval_plan",
        source_ids=state.get("source_ids"),
    )
    return {"retrieval_plans": plans}


retrieval_planner.default_worker_class = "orchestrator"


def retrieval_planner_to_retriever(state: AgentState) -> List[Send]:
    """Fan-out: one planned_retriever per concept retrieval plan."""
    return [
        Send("planned_retriever", {"plan": p, **state})
        for p in (state.get("retrieval_plans") or [])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 5. planned_retriever  (parallel leaf — tool_only with inline reranking)
# ─────────────────────────────────────────────────────────────────────────────

async def planned_retriever(state: Dict) -> Dict:
    """
    Execute all retrieval intents for one concept plan, then rerank and deduplicate.
    Combines BM25, vector, and section-based retrieval across all intents.
    Inline candidate_reranker: score-based sort + dedup by chunk_id.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import ATTACK_RETRIEVER_TOOLS

    plan: ConceptRetrievalPlan = state["plan"]
    project_id = state["project_id"]
    source_ids = plan.get("source_ids") or state.get("source_ids") or []

    tools = make_tools(
        project_id,
        source_ids=source_ids,
        use_voyage=state.get("use_voyage", False),
        tool_names=ATTACK_RETRIEVER_TOOLS,
    )
    hybrid_tool   = next((t for t in tools if t.name == "hybrid_search"), None)
    section_tool  = next((t for t in tools if t.name == "find_sections_about"), None)
    passages_tool = next((t for t in tools if t.name == "search_passages"), None)

    seen_ids: set = set()
    all_chunks: List[Dict[str, Any]] = []

    intents = plan.get("retrieval_intents") or []
    for intent in intents[:6]:  # cap intents per concept for cost control
        queries = intent.get("bm25_queries", [])[:2] + intent.get("vector_queries", [])[:2]
        for q in queries:
            if not q:
                continue
            try:
                if hybrid_tool:
                    raw = await hybrid_tool.ainvoke({"query": q, "k": 20})
                elif passages_tool:
                    raw = await passages_tool.ainvoke({"query": q, "k": 20})
                else:
                    continue
                chunks = json.loads(raw)
                for c in chunks:
                    cid = c.get("id") or c.get("chunk_id") or ""
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        all_chunks.append(c)
            except Exception as exc:
                logger.debug("planned_retriever: query '%s' failed: %s", q[:60], exc)

    # Section-based drilldown for section_filters
    for section_filter in (plan.get("section_filters") or [])[:5]:
        try:
            if section_tool:
                raw = await section_tool.ainvoke({"query": plan["concept_label"], "k": 8})
                chunks = json.loads(raw)
                for c in chunks:
                    cid = c.get("id") or c.get("chunk_id") or ""
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        all_chunks.append(c)
        except Exception:
            pass

    # Inline rerank: sort by score desc, keep top 30
    def _score(c: Dict) -> float:
        return float(c.get("score") or c.get("similarity") or 0.0)

    ranked = sorted(all_chunks, key=_score, reverse=True)[:30]

    bundle: RankedBundle = {
        "concept_id":    plan["concept_id"],
        "concept_label": plan["concept_label"],
        "chunks":        ranked,
    }
    return {"retrieval_bundles": [bundle]}


planned_retriever.default_worker_class = "tool_only"


def retriever_to_extractor(state: AgentState) -> List[Send]:
    """Fan-out: one legal_artifact_extractor per retrieval bundle."""
    return [
        Send("legal_artifact_extractor", {"bundle": b, **state})
        for b in (state.get("retrieval_bundles") or [])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 6. legal_artifact_extractor  (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def legal_artifact_extractor(state: Dict) -> Dict:
    """
    Convert ranked chunk bundles into typed legal artifacts.
    Extracts: rule_card, element_card, exception_card, issue_trigger_card,
    defense_card, case_card, policy_card, remedy_card, exam_trap_card.
    """
    bundle: RankedBundle = state["bundle"]
    concept_id = bundle["concept_id"]
    concept_label = bundle["concept_label"]

    context = "\n\n---\n\n".join(
        f"[chunk_id:{c.get('id', c.get('chunk_id', '?'))} "
        f"source:{c.get('source_id', '?')} p.{c.get('page_number', '?')}]\n"
        f"{c.get('content', '')}"
        for c in bundle["chunks"][:20]
    )

    system = (
        "You are a law professor extracting legal artifacts for an attack outline. "
        "Extract ALL of these artifact types present in the context:\n"
        "  rule_card:         the black-letter rule statement\n"
        "  element_card:      each required element of a claim or test\n"
        "  exception_card:    exceptions, carve-outs, and limiting doctrines\n"
        "  issue_trigger_card: the fact pattern signal that raises this issue\n"
        "  defense_card:      defenses and counterarguments available\n"
        "  case_card:         key cases with their holdings\n"
        "  policy_card:       policy rationales behind the rule\n"
        "  remedy_card:       available remedies and their requirements\n"
        "  exam_trap_card:    common student errors and traps for this doctrine\n\n"
        "Return a JSON array. Each item:\n"
        "  artifact_type: one of the types above\n"
        "  concept_id: the concept UUID/label\n"
        "  text: the full artifact text (source-grounded)\n"
        "  elements: list of sub-elements if applicable (else [])\n"
        "  exceptions: list of exceptions if applicable (else [])\n"
        "  source_refs: [{source_id, chunk_id, page}] for each supporting chunk\n"
        "  confidence: 0.0-1.0\n\n"
        "Extract only what is explicitly supported by the provided context. "
        "Do NOT invent or paraphrase without support. Return only the JSON array."
    )

    prompt = (
        f"Concept: {concept_label} (id: {concept_id})\n\n"
        f"Legal context:\n{context}\n\n"
        f"Extract all legal artifacts present."
    )

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=3000)
    try:
        artifacts_raw = _parse_json(raw)
        if not isinstance(artifacts_raw, list):
            artifacts_raw = []
    except Exception:
        artifacts_raw = []

    artifacts: List[LegalArtifact] = []
    for a in artifacts_raw:
        if not isinstance(a, dict):
            continue
        artifacts.append({
            "artifact_type": a.get("artifact_type", "rule_card"),
            "concept_id":    concept_id,
            "text":          a.get("text", ""),
            "elements":      a.get("elements", []),
            "exceptions":    a.get("exceptions", []),
            "source_refs":   a.get("source_refs", []),
            "confidence":    float(a.get("confidence", 0.5)),
        })

    await _try_save_artifact(
        state,
        artifact_key=f"raw_artifacts:{concept_id}",
        content={"concept_id": concept_id, "artifacts": artifacts},
        worker_class="worker_mid",
        node_name="legal_artifact_extractor",
        artifact_type="raw_artifacts",
        source_ids=state.get("source_ids"),
    )
    return {"raw_artifacts": artifacts}


legal_artifact_extractor.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 7. artifact_normalizer
# ─────────────────────────────────────────────────────────────────────────────

async def artifact_normalizer(state: AgentState) -> Dict:
    """
    Deduplicate and normalise all extracted artifacts.
    Merges equivalent rules, resolves naming conflicts, removes unsupported
    artifacts, and flags meaningful disagreements across sources.
    """
    raw_artifacts = state.get("raw_artifacts") or []
    if not raw_artifacts:
        return {"normalized_artifacts": []}

    # Group artifacts by concept_id
    by_concept: Dict[str, List[LegalArtifact]] = {}
    for a in raw_artifacts:
        by_concept.setdefault(a["concept_id"], []).append(a)

    # Prepare a compact summary for the LLM (avoid token overflow)
    concept_summaries = {}
    for cid, arts in by_concept.items():
        rules = [a["text"][:300] for a in arts if a["artifact_type"] == "rule_card"][:3]
        elements = list({el for a in arts for el in a.get("elements", [])})[:10]
        exceptions = list({ex for a in arts for ex in a.get("exceptions", [])})[:6]
        source_refs = [ref for a in arts for ref in a.get("source_refs", [])][:5]
        concept_summaries[cid] = {
            "canonical_name": arts[0]["concept_id"],
            "rules": rules,
            "elements": elements,
            "exceptions": exceptions,
            "source_refs": source_refs,
            "n_artifacts": len(arts),
        }

    system = (
        "You are a legal editor normalising and deduplicating a law-school outline artifact set. "
        "For each concept, merge duplicate rules (preserving meaningful disagreement as conflicts), "
        "canonicalise element names, remove unsupported claims, and produce a cleaned summary.\n\n"
        "Return a JSON array. Each item:\n"
        "  concept_id: the concept identifier\n"
        "  canonical_name: the authoritative doctrine name\n"
        "  rules: list of merged, cleaned rule statements (unique)\n"
        "  elements: list of required elements (unique, canonical names)\n"
        "  exceptions: list of exception/carve-out statements (unique)\n"
        "  conflicts: list of cross-source disagreements (if any)\n"
        "  source_refs: [{source_id, chunk_id, page}] for the strongest supporting chunks\n\n"
        "Return only the JSON array."
    )

    prompt = (
        f"Concept artifact summaries to normalise:\n"
        f"{json.dumps(concept_summaries, indent=2)}\n\n"
        f"Normalise, deduplicate, and clean."
    )

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=4096)
    try:
        normalised: List[NormalizedArtifact] = _parse_json(raw)
        if not isinstance(normalised, list):
            normalised = []
    except Exception:
        # Fallback: build minimal normalised artifacts directly from grouped data
        normalised = [
            {
                "concept_id":     cid,
                "canonical_name": data["canonical_name"],
                "rules":          data["rules"],
                "elements":       data["elements"],
                "exceptions":     data["exceptions"],
                "conflicts":      [],
                "source_refs":    data["source_refs"],
            }
            for cid, data in concept_summaries.items()
        ]

    await _try_save_artifact(
        state,
        artifact_key="normalized_artifacts",
        content={"artifacts": normalised},
        worker_class="worker_mid",
        node_name="artifact_normalizer",
        artifact_type="normalized_artifacts",
        source_ids=state.get("source_ids"),
    )
    return {"normalized_artifacts": normalised}


artifact_normalizer.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 8. concept_clusterer
# ─────────────────────────────────────────────────────────────────────────────

async def concept_clusterer(state: AgentState) -> Dict:
    """
    Group normalised artifacts into coherent doctrine modules suitable
    for an attack outline.  Each cluster becomes one attack block.
    """
    normalised = state.get("normalized_artifacts") or []
    topic_map = state.get("topic_map") or []

    concepts_brief = [
        {
            "concept_id":     n["concept_id"],
            "canonical_name": n["canonical_name"],
            "n_rules":        len(n.get("rules", [])),
            "n_elements":     len(n.get("elements", [])),
        }
        for n in normalised
    ]
    topics_brief = [
        {"topic_id": t["topic_id"], "label": t["label"], "priority": t.get("priority", 2)}
        for t in topic_map[:20]
    ]

    system = (
        "You are a law professor organising doctrine into exam attack blocks. "
        "Group the provided concepts into coherent cluster objects, one cluster per "
        "standalone attack block. Sub-elements of the same doctrine should be in the same cluster.\n\n"
        "Return a JSON array. Each item:\n"
        "  cluster_id: short_snake_case\n"
        "  label: human-readable doctrine name (e.g. 'Negligence — Duty of Care')\n"
        "  artifact_ids: list of concept_ids in this cluster\n"
        "  parent_topic: topic_id this cluster belongs to\n"
        "  priority: 1 (high) | 2 (medium) | 3 (low)\n\n"
        "Return only the JSON array."
    )

    prompt = (
        f"Available topics:\n{json.dumps(topics_brief, indent=2)}\n\n"
        f"Concepts to cluster:\n{json.dumps(concepts_brief, indent=2)}\n\n"
        f"User request: {state['request']}"
    )

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=2048)
    try:
        clusters: List[ConceptCluster] = _parse_json(raw)
        if not isinstance(clusters, list):
            clusters = []
    except Exception:
        clusters = [
            {
                "cluster_id":   n["concept_id"],
                "label":        n["canonical_name"],
                "artifact_ids": [n["concept_id"]],
                "parent_topic": "",
                "priority":     2,
            }
            for n in normalised
        ]

    await _try_save_artifact(
        state,
        artifact_key="concept_clusters",
        content={"clusters": clusters},
        worker_class="worker_mid",
        node_name="concept_clusterer",
        artifact_type="concept_clusters",
        source_ids=state.get("source_ids"),
    )
    return {"concept_clusters": clusters}


concept_clusterer.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 9. doctrine_graph_builder
# ─────────────────────────────────────────────────────────────────────────────

async def doctrine_graph_builder(state: AgentState) -> Dict:
    """
    Build the if/then doctrine graph showing the order of analysis and
    conditional transitions between doctrine blocks.
    """
    clusters = state.get("concept_clusters") or []

    cluster_info = [
        {
            "cluster_id": c["cluster_id"],
            "label":      c["label"],
            "priority":   c.get("priority", 2),
            "parent_topic": c.get("parent_topic", ""),
        }
        for c in clusters
    ]

    system = (
        "You are a law professor building a doctrine analysis graph for an attack outline. "
        "Create a graph that shows:\n"
        "  - The order a student should analyse doctrines (nodes)\n"
        "  - Conditional transitions: when to proceed from one doctrine to another (edges)\n"
        "  - Fallback paths and alternative theories\n\n"
        "Return JSON with:\n"
        "  nodes: [{id, label}] — one node per doctrine cluster\n"
        "  edges: [{from_node, to_node, condition, transition_text}]\n"
        "    condition: the if/then trigger for this transition\n"
        "    transition_text: one sentence a student reads at this branch point\n\n"
        "Order nodes by typical exam analysis sequence. Return only JSON."
    )

    prompt = (
        f"Doctrine clusters to connect into a graph:\n"
        f"{json.dumps(cluster_info, indent=2)}\n\n"
        f"Build the if/then doctrine analysis graph."
    )

    raw = await _llm("orchestrator", prompt, system=system, max_tokens=2048)
    try:
        graph_data = _parse_json(raw)
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
    except Exception:
        nodes = [{"id": c["cluster_id"], "label": c["label"]} for c in clusters]
        edges = []

    doctrine_graph: DoctrineGraph = {"nodes": nodes, "edges": edges}

    await _try_save_artifact(
        state,
        artifact_key="doctrine_graph",
        content={"nodes": nodes, "edges": edges},
        worker_class="orchestrator",
        node_name="doctrine_graph_builder",
        artifact_type="doctrine_graph",
        source_ids=state.get("source_ids"),
    )
    return {"doctrine_graph": doctrine_graph}


doctrine_graph_builder.default_worker_class = "orchestrator"


def doctrine_to_block_builder(state: AgentState) -> List[Send]:
    """Fan-out: one attack_block_builder per concept cluster."""
    clusters = state.get("concept_clusters") or []
    return [
        Send("attack_block_builder", {"cluster": c, **state})
        for c in clusters
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 10. attack_block_builder  (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def attack_block_builder(state: Dict) -> Dict:
    """
    Build one concise attack-outline block for a doctrine cluster.
    Produces: trigger, rule, elements checklist, exceptions, arguments both sides,
    exam traps, and conclusion pattern — all source-grounded.
    """
    cluster: ConceptCluster = state["cluster"]
    doctrine_graph: DoctrineGraph = state.get("doctrine_graph") or {"nodes": [], "edges": []}
    normalised = state.get("normalized_artifacts") or []
    raw_artifacts = state.get("raw_artifacts") or []

    # Gather artifacts for this cluster's concept_ids
    cluster_cids = set(cluster.get("artifact_ids", [cluster["cluster_id"]]))
    relevant_normalised = [n for n in normalised if n["concept_id"] in cluster_cids]
    relevant_raw = [a for a in raw_artifacts if a["concept_id"] in cluster_cids]

    # Build source-grounded context from raw artifacts
    rules_text     = "\n".join(n.get("rules", []) for n in relevant_normalised for _ in [None])
    elements_text  = json.dumps(list({el for n in relevant_normalised for el in n.get("elements", [])}))
    exceptions_text = json.dumps(list({ex for n in relevant_normalised for ex in n.get("exceptions", [])}))
    issue_triggers = [a["text"] for a in relevant_raw if a["artifact_type"] == "issue_trigger_card"][:3]
    defenses       = [a["text"] for a in relevant_raw if a["artifact_type"] == "defense_card"][:3]
    exam_traps_src = [a["text"] for a in relevant_raw if a["artifact_type"] == "exam_trap_card"][:4]
    source_refs    = list({ref["chunk_id"] for a in relevant_raw for ref in a.get("source_refs", []) if ref.get("chunk_id")})[:10]

    # Find graph context for this node (predecessors + successors)
    graph_edges = doctrine_graph.get("edges", [])
    incoming_conditions = [e["condition"] for e in graph_edges if e.get("to_node") == cluster["cluster_id"]][:2]
    outgoing_conditions = [e["transition_text"] for e in graph_edges if e.get("from_node") == cluster["cluster_id"]][:2]

    system = (
        "You are a law professor building an exam attack block. "
        "The block must be concise, rule-heavy, and decision-tree-like — NOT a summary. "
        "Every element must be source-grounded.\n\n"
        "Return JSON with:\n"
        "  concept_id: string\n"
        "  title: doctrine name\n"
        "  trigger: one sentence — the fact-pattern signal that raises this issue\n"
        "  attack_steps: array of {step (int), label, rule (black-letter), "
        "ask (list of checklist questions), arguments_for (list), "
        "arguments_against (list), source_refs (list of chunk_ids)}\n"
        "  exceptions_or_limits: list of exception/limitation statements\n"
        "  exam_traps: list of common errors to avoid\n"
        "  source_refs: list of chunk_ids supporting this block overall\n"
        "  revised: false\n\n"
        "Keep each attack_step focused. Argue BOTH sides in arguments_for/against. "
        "Return only JSON."
    )

    prompt = (
        f"Doctrine: {cluster['label']} (cluster_id: {cluster['cluster_id']})\n\n"
        f"Black-letter rules:\n{rules_text[:1500]}\n\n"
        f"Elements: {elements_text}\n\n"
        f"Exceptions: {exceptions_text}\n\n"
        f"Issue triggers: {json.dumps(issue_triggers)}\n\n"
        f"Defenses available: {json.dumps(defenses)}\n\n"
        f"Exam traps from sources: {json.dumps(exam_traps_src)}\n\n"
        f"Preceding doctrine conditions: {json.dumps(incoming_conditions)}\n"
        f"Following doctrine transitions: {json.dumps(outgoing_conditions)}\n\n"
        f"Build the attack block."
    )

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=2500)
    try:
        data = _parse_json(raw)
    except Exception:
        data = {}

    block: AttackBlock = {
        "concept_id":         cluster["cluster_id"],
        "title":              data.get("title", cluster["label"]),
        "trigger":            data.get("trigger", f"Analyse {cluster['label']} when..."),
        "attack_steps":       data.get("attack_steps", []),
        "exceptions_or_limits": data.get("exceptions_or_limits", []),
        "exam_traps":         data.get("exam_traps", exam_traps_src),
        "source_refs":        data.get("source_refs", source_refs),
        "revised":            False,
    }

    await _try_save_artifact(
        state,
        artifact_key=f"attack_block:{cluster['cluster_id']}",
        content=dict(block),
        worker_class="worker_mid",
        node_name="attack_block_builder",
        artifact_type="attack_block",
        source_ids=state.get("source_ids"),
    )
    return {"raw_blocks": [block]}


attack_block_builder.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 11. attack_outline_assembler
# ─────────────────────────────────────────────────────────────────────────────

def _topological_order(
    blocks: List[AttackBlock],
    graph: DoctrineGraph,
) -> List[AttackBlock]:
    """
    Sort attack blocks by doctrine graph topology using a simple BFS from roots.
    Falls back to priority-based ordering if the graph has no edges.
    """
    if not graph.get("edges"):
        return blocks  # no topology, use arrival order

    node_order: Dict[str, int] = {}
    adj: Dict[str, List[str]] = {}
    for e in graph.get("edges", []):
        adj.setdefault(e.get("from_node", ""), []).append(e.get("to_node", ""))

    all_ids = {b["concept_id"] for b in blocks}
    in_degree = {nid: 0 for nid in all_ids}
    for nid, children in adj.items():
        for child in children:
            if child in in_degree:
                in_degree[child] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    idx = 0
    while queue:
        nid = queue.pop(0)
        node_order[nid] = idx
        idx += 1
        for child in adj.get(nid, []):
            in_degree[child] = in_degree.get(child, 1) - 1
            if in_degree[child] == 0:
                queue.append(child)

    return sorted(blocks, key=lambda b: node_order.get(b["concept_id"], 999))


async def attack_outline_assembler(state: AgentState) -> Dict:
    """
    Assemble all attack blocks into a coherent exam-usable outline.
    Uses the doctrine graph to determine section order.
    Produces the assembled markdown and sets attack_blocks (working list).
    """
    raw_blocks = state.get("raw_blocks") or []
    doctrine_graph = state.get("doctrine_graph") or {"nodes": [], "edges": []}

    if not raw_blocks:
        return {"assembled_outline": "", "attack_blocks": []}

    ordered_blocks = _topological_order(raw_blocks, doctrine_graph)

    # Build doctrine graph transition map for inter-block arrows
    edge_map: Dict[str, str] = {
        e.get("from_node", ""): e.get("transition_text", "")
        for e in doctrine_graph.get("edges", [])
    }

    sections: List[str] = []
    for block in ordered_blocks:
        cid = block["concept_id"]
        title = block.get("title", cid)
        trigger = block.get("trigger", "")
        steps = block.get("attack_steps", [])
        exceptions = block.get("exceptions_or_limits", [])
        traps = block.get("exam_traps", [])
        transition = edge_map.get(cid, "")

        # Build step checklist
        step_lines: List[str] = []
        for s in steps:
            step_num = s.get("step", "?")
            label = s.get("label", "")
            rule = s.get("rule", "")
            asks = s.get("ask", [])
            args_for = s.get("arguments_for", [])
            args_against = s.get("arguments_against", [])

            step_lines.append(f"  **{step_num}. {label}**")
            if rule:
                step_lines.append(f"  > *Rule:* {rule}")
            for q in asks:
                step_lines.append(f"  - [ ] {q}")
            if args_for:
                step_lines.append(f"  - *For:* " + "; ".join(args_for[:2]))
            if args_against:
                step_lines.append(f"  - *Against:* " + "; ".join(args_against[:2]))

        exceptions_md = ""
        if exceptions:
            exceptions_md = "\n**Exceptions / Limits:**\n" + "\n".join(f"- {e}" for e in exceptions)

        traps_md = ""
        if traps:
            traps_md = "\n**Exam Traps:**\n" + "\n".join(f"- ⚠️ {t}" for t in traps)

        transition_md = f"\n*→ {transition}*" if transition else ""

        block_md = (
            f"## {title}\n\n"
            f"**Trigger:** {trigger}\n\n"
            + "\n".join(step_lines)
            + exceptions_md
            + traps_md
            + transition_md
        )
        sections.append(block_md)

    assembled = "\n\n---\n\n".join(sections)

    await _try_save_artifact(
        state,
        artifact_key="assembled_outline",
        content={"outline": assembled, "n_blocks": len(ordered_blocks)},
        worker_class="tool_only",
        node_name="attack_outline_assembler",
        artifact_type="assembled_outline",
        source_ids=state.get("source_ids"),
    )
    return {
        "assembled_outline": assembled,
        "attack_blocks": ordered_blocks,  # working list for revision + verifier
    }


attack_outline_assembler.default_worker_class = "tool_only"


def assembler_to_verifier(state: AgentState) -> List[Send]:
    """Fan-out: one grounding_verifier per attack block."""
    blocks = state.get("attack_blocks") or []
    return [
        Send("grounding_verifier", {"block": b, **state})
        for b in blocks
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 12. grounding_verifier  (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def grounding_verifier(state: Dict) -> Dict:
    """
    Verify that each rule, element, and exception in an attack block
    is supported by cited source chunks.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import ATTACK_VERIFIER_TOOLS

    block: AttackBlock = state["block"]
    project_id = state["project_id"]

    tools = make_tools(
        project_id,
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=ATTACK_VERIFIER_TOOLS,
    )
    verify_tool = next((t for t in tools if t.name == "verify_claim"), None)
    cite_tool   = next((t for t in tools if t.name == "get_citations_for"), None)

    # Extract claims: rules + elements from attack steps
    claims_to_verify: List[str] = []
    for step in (block.get("attack_steps") or [])[:4]:
        if step.get("rule"):
            claims_to_verify.append(step["rule"][:400])

    supported: List[str] = []
    unsupported: List[str] = []
    weak: List[str] = []
    missing: List[str] = []

    for claim in claims_to_verify[:5]:
        if verify_tool:
            try:
                result_json = await verify_tool.ainvoke({"claim": claim, "k": 8})
                result = json.loads(result_json)
                verdict = result.get("verdict", "insufficient")
                if verdict == "supported":
                    supported.append(claim[:120])
                elif verdict == "contradicted":
                    unsupported.append(claim[:120])
                else:
                    weak.append(claim[:120])
            except Exception:
                weak.append(claim[:120])
        else:
            # No verify tool available — mark as weak
            weak.append(claim[:120])

    # Fetch citations for chunk_ids referenced by this block
    block_refs = (block.get("source_refs") or [])[:10]
    _UUID_RE = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I)
    chunk_ids = [m.group() for r in block_refs if (m := _UUID_RE.search(str(r)))]

    if chunk_ids and cite_tool:
        try:
            cites_json = await cite_tool.ainvoke({"chunk_ids": chunk_ids[:8]})
            citations = json.loads(cites_json)
            if not citations:
                missing.append("Block references chunks that returned no citations")
        except Exception:
            pass

    # Overall verdict
    if unsupported:
        overall = "fail"
    elif weak and not supported:
        overall = "warn"
    elif not claims_to_verify:
        overall = "warn"
    else:
        overall = "pass"

    report: VerificationReport = {
        "block_id":          block["concept_id"],
        "supported_claims":  supported,
        "unsupported_claims": unsupported,
        "weak_claims":       weak,
        "missing_citations": missing,
        "overall_verdict":   overall,
    }

    await _try_save_artifact(
        state,
        artifact_key=f"verification:{block['concept_id']}",
        content=dict(report),
        worker_class="worker_mid",
        node_name="grounding_verifier",
        artifact_type="verification_report",
        source_ids=state.get("source_ids"),
    )
    return {"verification_reports": [report]}


grounding_verifier.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 13. attack_outline_critic
# ─────────────────────────────────────────────────────────────────────────────

async def attack_outline_critic(state: AgentState) -> Dict:
    """
    Evaluate whether the assembled outline is a functional exam attack outline.
    Scores rule density, checklist structure, issue triggers, exception coverage,
    counterarguments, concision, and exam usability.
    Issues revision targets for failing blocks.
    """
    assembled = state.get("assembled_outline") or ""
    verification_reports = state.get("verification_reports") or []
    attack_blocks = state.get("attack_blocks") or []

    # Summarise verification verdicts
    verdict_summary = {r["block_id"]: r["overall_verdict"] for r in verification_reports}
    failed_blocks = [bid for bid, v in verdict_summary.items() if v == "fail"]

    # Abbreviate outline for critic (avoid token overflow)
    outline_excerpt = assembled[:4000] if len(assembled) > 4000 else assembled

    system = (
        "You are a senior law professor and bar-exam coach evaluating an attack outline. "
        "Score the outline on these dimensions (0-10 each):\n"
        "  rule_density: how many black-letter rules appear per block\n"
        "  checklist_structure: quality of step-by-step checklist format\n"
        "  issue_triggers: how clear and specific the issue-spotting triggers are\n"
        "  exception_coverage: how thoroughly exceptions and carve-outs are covered\n"
        "  counterargument_coverage: quality of both-sides argument structure\n"
        "  concision: is the outline tight and skimmable, not verbose\n\n"
        "Then:\n"
        "  must_revise: true if overall_score < 6.5 or any block has grounding 'fail'\n"
        "  revision_targets: list of concept_ids most needing improvement\n"
        "  revision_instructions: 2-4 sentences of targeted guidance\n\n"
        "Return JSON with: rule_density_score, checklist_structure_score, "
        "issue_trigger_score, exception_coverage_score, counterargument_score, "
        "concision_score, overall_score, must_revise, revision_targets, "
        "revision_instructions. Return only JSON."
    )

    prompt = (
        f"Attack outline excerpt:\n{outline_excerpt}\n\n"
        f"Grounding verdicts by block: {json.dumps(verdict_summary)}\n"
        f"Blocks with failed grounding: {failed_blocks}\n\n"
        f"Evaluate this attack outline."
    )

    raw = await _llm("orchestrator", prompt, system=system, max_tokens=1024)
    try:
        data = _parse_json(raw)
    except Exception:
        data = {}

    # Ensure failed grounding blocks are always in revision_targets
    revision_targets = list(data.get("revision_targets") or [])
    for bid in failed_blocks:
        if bid not in revision_targets:
            revision_targets.append(bid)

    overall_score = float(data.get("overall_score") or 7.0)
    must_revise = bool(data.get("must_revise", overall_score < 6.5 or bool(failed_blocks)))

    critique: CritiqueResult = {
        "rule_density_score":      float(data.get("rule_density_score", 7.0)),
        "checklist_structure_score": float(data.get("checklist_structure_score", 7.0)),
        "issue_trigger_score":     float(data.get("issue_trigger_score", 7.0)),
        "exception_coverage_score": float(data.get("exception_coverage_score", 7.0)),
        "counterargument_score":   float(data.get("counterargument_score", 7.0)),
        "concision_score":         float(data.get("concision_score", 7.0)),
        "overall_score":           overall_score,
        "must_revise":             must_revise,
        "revision_targets":        revision_targets[:6],
        "revision_instructions":   data.get("revision_instructions", ""),
    }

    await _try_save_artifact(
        state,
        artifact_key=f"critique:{state.get('revision_count', 0)}",
        content=dict(critique),
        worker_class="orchestrator",
        node_name="attack_outline_critic",
        artifact_type="critique",
        source_ids=state.get("source_ids"),
    )
    return {"critique": critique}


attack_outline_critic.default_worker_class = "orchestrator"


def should_revise(state: AgentState) -> str:
    """
    Route to revision_agent if must_revise and under revision limit.
    Otherwise route to final_compressor_formatter.
    """
    count = state.get("revision_count") or 0
    if count >= MAX_REVISIONS:
        return "final_compressor_formatter"
    critique = state.get("critique")
    if critique and critique.get("must_revise") and critique.get("revision_targets"):
        return "revision_agent"
    return "final_compressor_formatter"


# ─────────────────────────────────────────────────────────────────────────────
# 14. revision_agent
# ─────────────────────────────────────────────────────────────────────────────

async def revision_agent(state: AgentState) -> Dict:
    """
    Apply targeted revisions to flagged attack blocks.
    Retrieves fresh evidence for each failing block, rewrites the block,
    then re-assembles the outline so attack_outline_critic can re-evaluate.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import ATTACK_RETRIEVER_TOOLS

    critique = state.get("critique") or {}
    targets = critique.get("revision_targets") or []
    instructions = critique.get("revision_instructions") or ""
    attack_blocks = list(state.get("attack_blocks") or [])
    doctrine_graph = state.get("doctrine_graph") or {"nodes": [], "edges": []}

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=ATTACK_RETRIEVER_TOOLS,
    )
    hybrid_tool = next((t for t in tools if t.name == "hybrid_search"), None)

    revised_map: Dict[str, AttackBlock] = {b["concept_id"]: b for b in attack_blocks}

    for target_id in targets[:4]:  # cap revisions per pass
        block = revised_map.get(target_id)
        if not block:
            continue

        # Re-retrieve fresh evidence for this block
        context = ""
        if hybrid_tool:
            try:
                q = f"{block['title']} rule elements exceptions checklist"
                raw_chunks = await hybrid_tool.ainvoke({"query": q, "k": 15})
                chunks = json.loads(raw_chunks)
                context = "\n\n".join(c.get("content", "") for c in chunks[:10])
            except Exception:
                context = ""

        system = (
            "You are a law professor revising a weak attack block. "
            "Revise ONLY what the critique flags. Preserve source-grounded claims. "
            "Make the block more concise, rule-heavy, and decision-tree-like.\n\n"
            "Return JSON in the same AttackBlock format:\n"
            "  concept_id, title, trigger, attack_steps, "
            "exceptions_or_limits, exam_traps, source_refs, revised (set to true)"
        )

        prompt = (
            f"Critique instructions: {instructions}\n\n"
            f"Block to revise:\n{json.dumps(dict(block), indent=2)}\n\n"
            f"Fresh supporting evidence:\n{context[:2000]}\n\n"
            f"Revise this block to address the critique."
        )

        raw = await _llm("worker_mid", prompt, system=system, max_tokens=2000)
        try:
            revised_data = _parse_json(raw)
            revised_block: AttackBlock = {
                "concept_id":         block["concept_id"],
                "title":              revised_data.get("title", block["title"]),
                "trigger":            revised_data.get("trigger", block["trigger"]),
                "attack_steps":       revised_data.get("attack_steps", block["attack_steps"]),
                "exceptions_or_limits": revised_data.get("exceptions_or_limits", block["exceptions_or_limits"]),
                "exam_traps":         revised_data.get("exam_traps", block["exam_traps"]),
                "source_refs":        revised_data.get("source_refs", block["source_refs"]),
                "revised":            True,
            }
            revised_map[target_id] = revised_block
        except Exception:
            # Fallback: mark the original as revised without content change
            updated = dict(block)
            updated["revised"] = True
            revised_map[target_id] = updated  # type: ignore[assignment]

    # Reconstruct the working block list preserving order
    updated_blocks = [revised_map.get(b["concept_id"], b) for b in attack_blocks]

    # Re-assemble the outline with revised blocks
    ordered = _topological_order(updated_blocks, doctrine_graph)
    edge_map = {
        e.get("from_node", ""): e.get("transition_text", "")
        for e in doctrine_graph.get("edges", [])
    }

    sections: List[str] = []
    for block in ordered:
        cid = block["concept_id"]
        title = block.get("title", cid)
        trigger = block.get("trigger", "")
        steps = block.get("attack_steps", [])
        exceptions = block.get("exceptions_or_limits", [])
        traps = block.get("exam_traps", [])
        transition = edge_map.get(cid, "")

        step_lines: List[str] = []
        for s in steps:
            step_num = s.get("step", "?")
            label = s.get("label", "")
            rule = s.get("rule", "")
            asks = s.get("ask", [])
            args_for = s.get("arguments_for", [])
            args_against = s.get("arguments_against", [])

            step_lines.append(f"  **{step_num}. {label}**")
            if rule:
                step_lines.append(f"  > *Rule:* {rule}")
            for q in asks:
                step_lines.append(f"  - [ ] {q}")
            if args_for:
                step_lines.append(f"  - *For:* " + "; ".join(args_for[:2]))
            if args_against:
                step_lines.append(f"  - *Against:* " + "; ".join(args_against[:2]))

        exceptions_md = ("\n**Exceptions / Limits:**\n" + "\n".join(f"- {e}" for e in exceptions)) if exceptions else ""
        traps_md = ("\n**Exam Traps:**\n" + "\n".join(f"- ⚠️ {t}" for t in traps)) if traps else ""
        transition_md = f"\n*→ {transition}*" if transition else ""

        block_md = (
            f"## {title}\n\n"
            f"**Trigger:** {trigger}\n\n"
            + "\n".join(step_lines)
            + exceptions_md
            + traps_md
            + transition_md
        )
        sections.append(block_md)

    revised_outline = "\n\n---\n\n".join(sections)

    return {
        "attack_blocks":    updated_blocks,
        "assembled_outline": revised_outline,
        "revision_count":   (state.get("revision_count") or 0) + 1,
    }


revision_agent.default_worker_class    = "worker_mid"
revision_agent.escalation_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 15. final_compressor_formatter
# ─────────────────────────────────────────────────────────────────────────────

async def final_compressor_formatter(state: AgentState) -> Dict:
    """
    Final polish of the approved outline.
    Formats for student use: concise, skimmable, exam-ready.
    Does NOT add new legal substance.
    """
    assembled = state.get("assembled_outline") or ""
    attack_blocks = state.get("attack_blocks") or []
    budget = state.get("budget") or {}

    if not assembled:
        return {"final_output": ""}

    system = (
        "You are a law-exam prep editor. Format the attack outline for student use. "
        "Rules:\n"
        "  - Do NOT add new legal substance — only polish formatting\n"
        "  - Tighten wording; remove redundancy\n"
        "  - Ensure consistent heading hierarchy (## for doctrine, ### for sub-doctrine)\n"
        "  - Checklist items use '- [ ]' format\n"
        "  - Rule statements are bold or blockquoted\n"
        "  - Transition arrows use → symbol\n"
        "  - Keep exam traps in ⚠️ callouts\n"
        "  - Add a brief Table of Contents at the top linking to each ## section\n"
        "  - Do not include raw JSON, code fences, or template artefacts\n"
        "Return only the final Markdown — no explanation."
    )

    prompt = (
        f"Attack outline to format:\n\n{assembled[:6000]}\n\n"
        f"Format for exam-day student use."
    )

    formatted = await _llm("worker_low", prompt, system=system, max_tokens=6000)

    # Append budget comment
    budget_note = (
        f"\n\n<!-- tokens: {budget.get('input_tokens', 0)} in / "
        f"{budget.get('output_tokens', 0)} out | "
        f"est. ${budget.get('cost_usd', 0):.4f} | "
        f"blocks: {len(attack_blocks)} | "
        f"revisions: {state.get('revision_count', 0)} -->"
    )
    final = formatted + budget_note

    await _try_save_artifact(
        state,
        artifact_key="final_output",
        content={"markdown": final, "n_blocks": len(attack_blocks)},
        worker_class="worker_low",
        node_name="final_compressor_formatter",
        artifact_type="final_output",
        source_ids=state.get("source_ids"),
    )
    return {"final_output": final}


final_compressor_formatter.default_worker_class = "worker_low"
