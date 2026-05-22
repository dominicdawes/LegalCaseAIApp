# agents/exam_questions/nodes.py
"""
All 12 nodes for the exam-questions LangGraph agent.

Node responsibilities:
  1.  Planner            — read source outlines, write retrieval strategy
  2.  SourceProfiler     — per-doc profile (parallel Send from Planner)
  3.  ConceptSynthesizer — cross-doc throughlines after all profiles merge
  4.  IssueClusterer     — cluster profiles + synthesis into exam issues
  5.  Retriever          — per-issue evidence bundle (parallel Send)
  6.  QuestionDrafter    — per-issue fact-pattern (parallel Send, orchestrator)
  7.  AnswerKeyBuilder   — per-question answer key (parallel Send, orchestrator)
  8.  Grounder           — per-question claim verification (parallel Send, worker_low)
  9.  Critic             — whole-exam critique (worker_low)
  10. Reviser            — targeted revision of failing questions (conditional, worker_mid)
  11. FinalDrafter       — editorial polish and IRAC enforcement (orchestrator)
  12. Assembler          — deterministic final Markdown assembly
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from langgraph.types import Send

from .state import (
    AgentState,
    DraftQuestion,
    IssueCluster,
    RetrievalBundle,
    SourceProfile,
    VerifiedQuestion,
)
from .worker_config import _fetch_worker_model, model_costs

logger = logging.getLogger(__name__)

# ── IRAC style guide injected into FinalDrafter ──────────────────────────────
_IRAC_STYLE_GUIDE = """
IRAC Structure per issue:
  Issue:       One sentence identifying the precise legal question.
  Rule:        State the applicable rule/standard; cite source materials by name.
  Application: Apply the rule to the facts; argue BOTH sides; address counterarguments.
  Conclusion:  Clear, practical outcome. One sentence.

Style constraints:
  - Plain legal English; define any technical term on first use.
  - Consistent party names throughout (pick one label per party and keep it).
  - One major IRAC block per paragraph; blank line between blocks.
  - Fact patterns: multi-character narrative, chronological, no explicit legal labels.
  - Answer keys: full IRAC per sub-issue; all raised defences addressed.
  - Markdown: ## Question N / **Call of the Question** / ### Answer Key — Question N
  - No duplicate sub-issues across questions in the same exam.
"""


# ── LLM call via LLMFactory ───────────────────────────────────────────────────

async def _llm(
    worker_class: str,
    prompt: str,
    system: str = "",
    max_tokens: int = 2048,
    provider: Optional[str] = None,
) -> str:
    """
    Route an LLM call through LLMFactory using the worker-class abstraction.
    Falls back to draining stream_chat if the client has no achat() method.
    """
    from utils.llm_clients.llm_factory import LLMFactory
    _provider, model_name = _fetch_worker_model(worker_class, provider)
    client = LLMFactory.get_client_for(
        _provider, model_name,
        temperature=0.7, streaming=False, max_output_tokens=max_tokens,
    )
    if hasattr(client, "achat"):
        return await client.achat(prompt, system_prompt=system or None)
    # Fallback: collect tokens from the async streaming generator
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
    """
    Non-fatal ledger write — never raises.
    Nodes call this after producing output; a ledger failure must never crash a node.
    Silently skips if job_id is not present in state (e.g. during local testing).
    """
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


def _add_budget(state: AgentState, model_name: str, in_tok: int, out_tok: int) -> Dict:
    in_cost, out_cost = model_costs(model_name)
    delta = in_tok * in_cost + out_tok * out_cost
    budget = dict(state.get("budget") or {})
    budget["input_tokens"] = budget.get("input_tokens", 0) + in_tok
    budget["output_tokens"] = budget.get("output_tokens", 0) + out_tok
    budget["cost_usd"] = round(budget.get("cost_usd", 0.0) + delta, 6)
    return budget


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


# ─────────────────────────────────────────────────────────────────────────────
# 1. Planner
# ─────────────────────────────────────────────────────────────────────────────

async def planner(state: AgentState) -> Dict:
    """Read the list of source documents and write a concise retrieval strategy."""
    from agents.tools.base import make_tools
    from agents.tools.registry import PLANNER_TOOLS

    tools = make_tools(
        state["project_id"],
        source_ids=state["source_ids"],
        use_voyage=state.get("use_voyage", False),
        tool_names=PLANNER_TOOLS,
    )
    list_sources_tool = next(t for t in tools if t.name == "list_sources")
    sources_json = await list_sources_tool.ainvoke({})

    system = (
        "You are a law professor designing an exam. "
        "Analyse the available documents and write a 3-5 sentence strategy "
        "for generating {n} exam questions. Focus on identifying the richest "
        "legal issues across the sources."
    ).format(n=state["n_questions"])

    plan = await _llm(
        "worker_mid",
        f"Documents available:\n{sources_json}\n\nUser request: {state['request']}",
        system=system,
        max_tokens=512,
    )
    return {"plan": plan}


planner.default_worker_class = "worker_mid"


def planner_to_profiler(state: AgentState) -> List[Send]:
    """Fan-out: one SourceProfiler invocation per source document."""
    source_ids = state["source_ids"]
    logger.info("exam_questions x%d node fan out for source_profiler", len(source_ids))
    return [
        Send("source_profiler", {"source_id": sid, **state})
        for sid in source_ids
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. SourceProfiler (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def source_profiler(state: Dict) -> Dict:
    """Build a profile for a single source document: concepts, toc, summary."""
    from agents.tools.base import make_tools
    from agents.tools.registry import PROFILER_TOOLS

    source_id = state["source_id"]
    project_id = state["project_id"]

    tools = make_tools(
        project_id,
        source_ids=[source_id],
        use_voyage=state.get("use_voyage", False),
        tool_names=PROFILER_TOOLS,
    )
    outline_tool = next(t for t in tools if t.name == "get_doc_outline")
    outline_json = await outline_tool.ainvoke({"source_id": source_id})
    outline = json.loads(outline_json)

    profile: SourceProfile = {
        "source_id": source_id,
        "filename": outline.get("filename", ""),
        "doc_summary": outline.get("doc_summary") or "",
        "key_concepts": outline.get("doc_concepts", [])[:20],
        "toc": outline.get("toc", [])[:30],
    }
    await _try_save_artifact(
        state,
        artifact_key=f"source_profile:{source_id}",
        content=profile,
        worker_class="tool_only",
        node_name="source_profiler",
        artifact_type="source_profile",
        source_ids=[source_id],
    )
    return {"source_profiles": [profile]}


source_profiler.default_worker_class = "tool_only"


# ─────────────────────────────────────────────────────────────────────────────
# 3. ConceptSynthesizer — cross-doc throughlines
# ─────────────────────────────────────────────────────────────────────────────

async def concept_synthesizer(state: AgentState) -> Dict:
    """
    Identify conceptual throughlines that span multiple source documents.

    Runs after all SourceProfiler branches merge. The synthesis JSON is stored
    in state["concept_synthesis"] and passed as extra context to IssueClusterer,
    helping it spot cross-cutting exam themes the per-doc profiler cannot see.
    """
    profiles = state.get("source_profiles") or []
    if len(profiles) <= 1:
        # Nothing to synthesise across a single document
        return {"concept_synthesis": ""}

    from agents.tools.base import make_tools
    from agents.tools.registry import SYNTHESIZER_TOOLS
    from collections import Counter

    tools = make_tools(
        state["project_id"],
        source_ids=state["source_ids"],
        use_voyage=state.get("use_voyage", False),
        tool_names=SYNTHESIZER_TOOLS,
    )
    crossdoc_tool = next(t for t in tools if t.name == "find_concept_across_docs")

    # Identify concepts shared across two or more documents
    all_concepts: List[str] = []
    for p in profiles:
        all_concepts.extend(p.get("key_concepts", [])[:6])
    counts = Counter(all_concepts)
    shared = [c for c, n in counts.most_common(8) if n > 1]
    if not shared:
        # No exact overlap — take the most common unique concepts as candidates
        shared = [c for c, _ in counts.most_common(6)]

    # Pull cross-doc evidence for the top shared concepts
    cross_results: List[Dict[str, Any]] = []
    for concept in shared[:5]:
        try:
            result_json = await crossdoc_tool.ainvoke({"concept": concept, "k_per_doc": 2})
            cross_results.append({"concept": concept, "hits": json.loads(result_json)})
        except Exception:
            pass

    profiles_brief = [
        {
            "source_id": p["source_id"],
            "filename":  p["filename"],
            "concepts":  p.get("key_concepts", [])[:8],
        }
        for p in profiles
    ]

    prompt = (
        f"You are a law professor identifying conceptual throughlines across "
        f"{len(profiles)} legal documents.\n\n"
        f"Document summaries:\n{json.dumps(profiles_brief, indent=2)}\n\n"
        f"Cross-document concept evidence:\n{json.dumps(cross_results, indent=2)}\n\n"
        f"Identify 3-5 conceptual throughlines: legal concepts, doctrines, or themes "
        f"that span multiple documents and would make strong exam question material.\n\n"
        f"For each throughline return:\n"
        f'  "concept":    the concept or doctrine name\n'
        f'  "summary":    2-3 sentences on how the documents relate to or develop it\n'
        f'  "source_ids": list of document UUIDs where this concept appears\n\n'
        f"Return a JSON array only. No extra text."
    )

    raw = await _llm("worker_mid", prompt, max_tokens=1024)
    try:
        throughlines = _parse_json(raw)
    except Exception:
        throughlines = []

    synthesis = json.dumps({"throughlines": throughlines, "shared_concepts": shared})
    await _try_save_artifact(
        state,
        artifact_key="concept_synthesis",
        content={"throughlines": throughlines, "shared_concepts": shared},
        worker_class="worker_mid",
        node_name="concept_synthesizer",
        artifact_type="concept_synthesis",
        source_ids=state.get("source_ids"),
    )
    return {"concept_synthesis": synthesis}


concept_synthesizer.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 4. IssueClusterer
# ─────────────────────────────────────────────────────────────────────────────

async def issue_clusterer(state: AgentState) -> Dict:
    """Given all source profiles and the concept synthesis, cluster N exam-worthy issues."""
    profiles = state.get("source_profiles") or []
    n = state["n_questions"]
    synthesis = state.get("concept_synthesis", "")

    profiles_text = json.dumps(profiles, indent=2)
    synthesis_section = (
        f"\nCross-document concept synthesis:\n{synthesis}\n"
        if synthesis else ""
    )

    prompt = (
        f"You are a law professor. Given the source document profiles below "
        f"and the exam request, identify exactly {n} high-quality legal issues "
        f"suitable for exam fact-patterns.\n\n"
        f"Prioritise issues that appear in the cross-document synthesis where available.\n\n"
        f"For each issue output a JSON object with:\n"
        f'  "issue_label":   short label (e.g. "Negligence — proximate cause")\n'
        f'  "source_ids":    list of source UUIDs where material exists\n'
        f'  "section_hints": list of section_path hints to retrieve\n'
        f'  "priority":      1 (high) | 2 (medium) | 3 (low)\n\n'
        f"Return a JSON array of exactly {n} objects. No extra text.\n\n"
        f"Source profiles:\n{profiles_text}"
        f"{synthesis_section}\n"
        f"User request: {state['request']}\n"
        f"Plan: {state.get('plan', '')}"
    )

    raw = await _llm("worker_mid", prompt, max_tokens=1024)
    try:
        clusters: List[IssueCluster] = _parse_json(raw)
    except Exception:
        clusters = [
            {
                "issue_label":   f"Legal issues from document {i+1}",
                "source_ids":    [p["source_id"]],
                "section_hints": [],
                "priority":      2,
            }
            for i, p in enumerate((profiles or [])[:n])
        ]
    return {"chosen_issues": clusters[:n]}


issue_clusterer.default_worker_class = "worker_mid"


def clusterer_to_retriever(state: AgentState) -> List[Send]:
    """Fan-out: one Retriever per issue cluster."""
    issues = state.get("chosen_issues") or []
    logger.info("exam_questions x%d node fan out for retriever", len(issues))
    return [
        Send("retriever", {"issue": issue, **state})
        for issue in issues
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Retriever (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def retriever(state: Dict) -> Dict:
    """Retrieve evidence passages for a single issue cluster."""
    from agents.tools.base import make_tools
    from agents.tools.registry import RETRIEVER_TOOLS

    issue: IssueCluster = state["issue"]
    project_id = state["project_id"]
    source_ids = issue.get("source_ids") or state.get("source_ids") or []

    tools = make_tools(
        project_id,
        source_ids=source_ids,
        use_voyage=state.get("use_voyage", False),
        tool_names=RETRIEVER_TOOLS,
    )
    hybrid_tool = next(t for t in tools if t.name == "hybrid_search")
    tables_tool = next(t for t in tools if t.name == "find_tables_about")

    chunks_json = await hybrid_tool.ainvoke({"query": issue["issue_label"], "k": 25})
    tables_json = await tables_tool.ainvoke({"query": issue["issue_label"], "k": 3})

    chunks = json.loads(chunks_json)
    tables = json.loads(tables_json)

    bundle: RetrievalBundle = {
        "issue_label": issue["issue_label"],
        "chunks":      chunks[:20],
        "table_ids":   [t["chunk_id"] for t in tables],
    }
    return {"retrieval_bundles": [bundle]}


retriever.default_worker_class = "tool_only"


def retriever_to_drafter(state: AgentState) -> List[Send]:
    """Fan-out: one QuestionDrafter per retrieval bundle."""
    bundles = state.get("retrieval_bundles") or []
    logger.info("exam_questions x%d node fan out for question_drafter", len(bundles))
    return [
        Send("question_drafter", {"bundle": b, "bundle_index": i, **state})
        for i, b in enumerate(bundles)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 6. QuestionDrafter (parallel leaf, orchestrator model)
# ─────────────────────────────────────────────────────────────────────────────

async def question_drafter(state: Dict) -> Dict:
    """Draft a fact-pattern hypothetical for one issue bundle."""
    bundle: RetrievalBundle = state["bundle"]
    idx: int = state["bundle_index"]

    context = "\n\n---\n\n".join(
        f"[chunk_id:{c.get('id','?')} p.{c.get('page_number','?')}]\n{c.get('content','')}"
        for c in bundle["chunks"][:15]
    )

    system = (
        "You are an expert law professor. Draft a single, realistic fact-pattern "
        "hypothetical exam question for the issue provided. The question must:\n"
        "- Be a multi-character narrative with chronological events\n"
        "- Embed the legal issue naturally without naming it explicitly\n"
        "- End with a clear 'Call of the Question' in bold\n"
        "- Be based ONLY on the provided legal context\n"
        "Return JSON with keys: fact_pattern (str), call_of_question (str), "
        "chunk_ids_used (list of chunk_id UUID strings from the [chunk_id:...] markers above — "
        "copy only the UUID, not the page number)"
    )

    prompt = (
        f"Issue: {bundle['issue_label']}\n\n"
        f"Legal context:\n{context}\n\n"
        f"Draft the fact-pattern hypothetical."
    )

    raw = await _llm("orchestrator", prompt, system=system, max_tokens=1500)
    try:
        data = _parse_json(raw)
    except Exception:
        data = {
            "fact_pattern":     raw,
            "call_of_question": "Discuss the rights and liabilities of all parties.",
            "chunk_ids_used":   [],
        }

    draft: DraftQuestion = {
        "question_index":  idx,
        "issue_label":     bundle["issue_label"],
        "fact_pattern":    data.get("fact_pattern", raw),
        "call_of_question": data.get("call_of_question", ""),
        "answer_key":      "",
        "chunk_ids_used":  data.get("chunk_ids_used", [c.get("id", "") for c in bundle["chunks"][:10]]),
    }
    return {"draft_questions": [draft]}


question_drafter.default_worker_class = "orchestrator"


def drafter_to_answerkey(state: AgentState) -> List[Send]:
    """Fan-out: one AnswerKeyBuilder per draft question."""
    drafts = state.get("draft_questions") or []
    bundles = state.get("retrieval_bundles") or []
    bundle_map = {b["issue_label"]: b for b in bundles}
    logger.info("exam_questions x%d node fan out for answer_key_builder", len(drafts))
    return [
        Send("answer_key_builder", {
            "draft":  d,
            "bundle": bundle_map.get(d["issue_label"], {"chunks": [], "table_ids": []}),
            **state,
        })
        for d in drafts
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 7. AnswerKeyBuilder (parallel leaf, orchestrator model)
# ─────────────────────────────────────────────────────────────────────────────

async def answer_key_builder(state: Dict) -> Dict:
    """Generate a full IRAC answer key for a drafted question."""
    draft: DraftQuestion = state["draft"]
    bundle: RetrievalBundle = state["bundle"]

    context = "\n\n---\n\n".join(
        c.get("content", "") for c in bundle["chunks"][:15]
    )

    system = (
        "You are an expert law professor writing the model answer key for an exam question. "
        "Produce a detailed IRAC analysis covering all major issues in the fact pattern. "
        "For each issue: state the Issue, Rule (from the provided materials), Application "
        "(both sides), and Conclusion. Include any defences raised."
    )

    prompt = (
        f"Fact Pattern:\n{draft['fact_pattern']}\n\n"
        f"Call: {draft['call_of_question']}\n\n"
        f"Legal context:\n{context}\n\n"
        f"Write the detailed Answer Key & Analysis."
    )

    answer_key = await _llm("orchestrator", prompt, system=system, max_tokens=2500)

    updated_draft = dict(draft)
    updated_draft["answer_key"] = answer_key
    return {"draft_questions": [updated_draft]}


answer_key_builder.default_worker_class = "orchestrator"


def answerkey_to_grounder(state: AgentState) -> List[Send]:
    """Fan-out: one Grounder per drafted question."""
    drafts = state.get("draft_questions") or []
    logger.info("exam_questions x%d node fan out for grounder", len(drafts))
    return [
        Send("grounder", {"draft": d, **state})
        for d in drafts
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 8. Grounder (parallel leaf, worker_low / escalation: worker_mid)
# ─────────────────────────────────────────────────────────────────────────────

async def grounder(state: Dict) -> Dict:
    """Extract 3-5 factual claims from the answer key and verify each against the docs."""
    from agents.tools.base import make_tools
    from agents.tools.registry import VERIFIER_TOOLS

    draft: DraftQuestion = state["draft"]

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=VERIFIER_TOOLS,
    )
    verify_tool = next(t for t in tools if t.name == "verify_claim")
    cite_tool   = next(t for t in tools if t.name == "get_citations_for")

    extract_prompt = (
        f"Extract 3-5 specific legal claims from the answer key below as a JSON array of strings.\n\n"
        f"Answer key:\n{draft['answer_key'][:3000]}\n\n"
        "Return only the JSON array, no extra text."
    )
    claims_raw = await _llm("worker_low", extract_prompt, max_tokens=512)
    try:
        claims = _parse_json(claims_raw)
        if not isinstance(claims, list):
            claims = [claims_raw]
    except Exception:
        claims = [draft["answer_key"][:300]]

    verdicts = []
    for claim in claims[:5]:
        result_json = await verify_tool.ainvoke({"claim": claim, "k": 8})
        verdicts.append(json.loads(result_json))

    statuses = [v.get("verdict", "insufficient") for v in verdicts]
    if all(s == "supported" for s in statuses):
        overall = "pass"
        notes   = "All claims supported."
    elif any(s == "contradicted" for s in statuses):
        overall = "fail"
        notes   = "One or more claims contradicted by source material."
    else:
        overall = "warn"
        notes   = "Some claims have insufficient evidence."

    _UUID_RE = re.compile(
        r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I
    )
    raw_chunk_ids = draft.get("chunk_ids_used") or []
    chunk_ids = [m.group() for raw in raw_chunk_ids if (m := _UUID_RE.search(str(raw)))]
    if chunk_ids:
        cites_json = await cite_tool.ainvoke({"chunk_ids": chunk_ids[:10]})
        citations  = json.loads(cites_json)
    else:
        citations = []

    vq: VerifiedQuestion = {
        "question_index":    draft["question_index"],
        "fact_pattern":      draft["fact_pattern"],
        "call_of_question":  draft["call_of_question"],
        "answer_key":        draft["answer_key"],
        "grounding_verdict": overall,
        "grounding_notes":   notes,
        "citations":         citations,
        "revised":           False,
    }
    await _try_save_artifact(
        state,
        artifact_key=f"verification:{draft['question_index']}",
        content=dict(vq),
        worker_class="worker_low",
        node_name="grounder",
        artifact_type="verification_result",
        source_ids=state.get("source_ids"),
    )
    return {"verified_questions": [vq]}


grounder.default_worker_class    = "worker_low"
grounder.escalation_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Critic (worker_low / escalation: worker_mid)
# ─────────────────────────────────────────────────────────────────────────────

async def critic(state: AgentState) -> Dict:
    """Holistic critique of the full draft exam. Flags questions that need revision."""
    questions = state.get("verified_questions") or []
    n = state["n_questions"]

    questions_text = "\n\n---\n\n".join(
        f"Q{q['question_index']+1} [{q['grounding_verdict'].upper()}]:\n"
        f"{q['fact_pattern'][:500]}\n{q['call_of_question']}"
        for q in questions
    )

    prompt = (
        f"You are a senior law professor reviewing {n} exam questions. "
        f"For each question, identify if it:\n"
        f"  - Has a clear, non-ambiguous call of the question\n"
        f"  - Is appropriately complex (not too simple)\n"
        f"  - Has a grounding_verdict of 'fail' (must be revised)\n\n"
        f"Return a JSON array of objects with:\n"
        f'  "question_index": int\n'
        f'  "needs_revision": bool\n'
        f'  "critique":       one-sentence note\n\n'
        f"Questions:\n{questions_text}\n\nReturn only JSON array, no extra text."
    )

    raw = await _llm("worker_low", prompt, max_tokens=1024)
    try:
        critiques = _parse_json(raw)
    except Exception:
        critiques = []

    revision_map = {c["question_index"]: c for c in critiques if isinstance(c, dict)}
    updated = []
    for q in questions:
        crit = revision_map.get(q["question_index"], {})
        if crit.get("needs_revision") or q["grounding_verdict"] == "fail":
            updated_q = dict(q)
            updated_q["grounding_notes"] = (
                q["grounding_notes"] + " | Critic: " + crit.get("critique", "")
            ).strip(" | ")
            updated.append(updated_q)
        else:
            updated.append(q)

    return {"verified_questions": updated}


critic.default_worker_class    = "worker_low"
critic.escalation_worker_class = "worker_mid"


def should_revise(state: AgentState) -> str:
    """
    Route to Reviser if any question failed grounding and we haven't hit the
    max revision count (2). Otherwise route to FinalDrafter.
    """
    count = state.get("revision_count") or 0
    if count >= 2:
        return "final_drafter"
    questions = state.get("verified_questions") or []
    if any(q["grounding_verdict"] == "fail" for q in questions):
        return "reviser"
    return "final_drafter"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Reviser (conditional, worker_mid / escalation: orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

async def reviser(state: AgentState) -> Dict:
    """Revise only the questions that failed grounding. Max 2 passes."""
    questions  = state.get("verified_questions") or []
    project_id = state["project_id"]
    use_voyage = state.get("use_voyage", False)

    revised = list(questions)
    for i, q in enumerate(revised):
        if q["grounding_verdict"] != "fail":
            continue

        from agents.tools.base import make_tools
        from agents.tools.registry import RETRIEVER_TOOLS

        tools = make_tools(
            project_id,
            source_ids=state.get("source_ids", []),
            use_voyage=use_voyage,
            tool_names=RETRIEVER_TOOLS,
        )
        search_tool = next(t for t in tools if t.name == "hybrid_search")
        evidence_json = await search_tool.ainvoke({"query": q["fact_pattern"][:200], "k": 15})
        evidence = json.loads(evidence_json)
        context  = "\n\n".join(c.get("content", "") for c in evidence[:10])

        prompt = (
            f"The following exam question failed grounding verification.\n"
            f"Critique: {q['grounding_notes']}\n\n"
            f"Original fact pattern:\n{q['fact_pattern']}\n\n"
            f"Supporting evidence from documents:\n{context}\n\n"
            f"Revise the fact pattern so all claims are supported by the evidence. "
            f"Keep the same legal issue and call of the question. "
            f"Return only the revised fact_pattern text."
        )

        revised_fp = await _llm("worker_mid", prompt, max_tokens=1200)
        updated_q  = dict(q)
        updated_q["fact_pattern"]      = revised_fp
        updated_q["grounding_verdict"] = "warn"
        updated_q["revised"]           = True
        revised[i] = updated_q

    return {
        "verified_questions": revised,
        "revision_count": (state.get("revision_count") or 0) + 1,
    }


reviser.default_worker_class    = "worker_mid"
reviser.escalation_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 11. FinalDrafter — editorial polish + IRAC enforcement (orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

async def final_drafter(state: AgentState) -> Dict:
    """
    Final editorial pass: polish each question individually in parallel.

    Processing one question per LLM call avoids output-token limits that occur
    when all questions are batched into a single request. Each call receives the
    full IRAC style guide and the gold-standard format from exam-questions-prompt.yaml.
    On JSON parse failure for an individual question the original is kept unchanged.
    """
    questions = sorted(
        state.get("verified_questions") or [],
        key=lambda q: q["question_index"],
    )
    if not questions:
        return {}

    system = (
        "You are a senior legal examinations editor. Your task is to polish one "
        "law exam question and its answer key as a final editorial review.\n\n"
        "Output format requirements:\n"
        "  Fact pattern — realistic multi-character narrative, chronological, "
        "no explicit legal labels embedded in the text.\n"
        "  Call of the Question — clear, open-ended prompt that tells the student "
        "what to analyse (e.g. 'Discuss all potential tort claims...').\n"
        "  Answer key — IRAC structure per sub-issue:\n"
        "    Issue:       one sentence identifying the precise legal question\n"
        "    Rule:        applicable rule/standard; cite source materials by name\n"
        "    Application: apply rule to facts; argue BOTH sides; address counterarguments\n"
        "    Conclusion:  clear, practical outcome — one sentence\n\n"
        "Additional constraints:\n"
        "  - Plain legal English; define technical terms on first use\n"
        "  - Consistent party names throughout (pick one label per party and keep it)\n"
        "  - One major IRAC block per paragraph; blank line between blocks\n"
        "  - No raw JSON, code fences, or formatting artefacts in the fact pattern\n"
        "  - Markdown headers: ## Question N / **Call of the Question** / "
        "### Answer Key — Question N\n\n"
        f"{_IRAC_STYLE_GUIDE}"
    )

    async def _polish_one(q: Dict) -> Dict:
        prompt = (
            f"Polish the following exam question and answer key.\n\n"
            f"Fact pattern:\n{q['fact_pattern']}\n\n"
            f"Call of the question:\n{q['call_of_question']}\n\n"
            f"Answer key:\n{q['answer_key']}\n\n"
            f"Editorial instructions:\n"
            f"  - Ensure the fact pattern reads as smooth, realistic narrative prose — "
            f"remove any JSON fragments, code fences, or template artefacts\n"
            f"  - Enforce consistent party names and legal terminology\n"
            f"  - Verify full IRAC structure in the answer key; fill any missing elements\n"
            f"  - Tighten wording without altering legal substance\n\n"
            f"Return JSON with exactly these three keys and no other text:\n"
            f'  "fact_pattern"     — polished fact pattern (str)\n'
            f'  "call_of_question" — polished call of the question (str)\n'
            f'  "answer_key"       — polished answer key (str)\n'
        )
        raw = await _llm("orchestrator", prompt, system=system, max_tokens=3000)
        try:
            patch = _parse_json(raw)
            updated = dict(q)
            updated["fact_pattern"]     = patch.get("fact_pattern",     q["fact_pattern"])
            updated["call_of_question"] = patch.get("call_of_question", q["call_of_question"])
            updated["answer_key"]       = patch.get("answer_key",       q["answer_key"])
            return updated
        except Exception as exc:
            logger.warning(
                "final_drafter: Q%d JSON parse failed (%s) — keeping original",
                q["question_index"], exc,
            )
            return q

    polished = list(await asyncio.gather(*[_polish_one(q) for q in questions]))

    await _try_save_artifact(
        state,
        artifact_key="polished_questions",
        content={"questions": [dict(q) for q in polished]},
        worker_class="orchestrator",
        node_name="final_drafter",
        artifact_type="polished_questions",
        source_ids=state.get("source_ids"),
    )
    return {"verified_questions": polished}


final_drafter.default_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 12. Assembler — deterministic Markdown formatter
# ─────────────────────────────────────────────────────────────────────────────

async def assembler(state: AgentState) -> Dict:
    """
    Deterministic assembly of the final Markdown exam document.
    No LLM call — questions arrive fully polished from FinalDrafter.
    """
    questions = sorted(
        state.get("verified_questions") or [],
        key=lambda q: q["question_index"],
    )

    sections    = []
    answer_keys = []

    for i, q in enumerate(questions, 1):
        sections.append(
            f"## Question {i}\n\n"
            f"{q['fact_pattern']}\n\n"
            f"**{q['call_of_question']}**"
        )
        cites    = q.get("citations") or []
        cite_str = ""
        if cites:
            cite_str = "\n\n*Sources: " + "; ".join(
                c.get("citation_string", "") for c in cites[:5]
            ) + "*"

        answer_keys.append(
            f"### Answer Key — Question {i}\n\n"
            f"{q['answer_key']}{cite_str}"
        )

    exam_body      = "\n\n---\n\n".join(sections)
    answer_section = "\n\n---\n\n".join(answer_keys)

    budget     = state.get("budget") or {}
    budget_note = (
        f"\n\n<!-- tokens: {budget.get('input_tokens',0)} in / "
        f"{budget.get('output_tokens',0)} out | "
        f"est. ${budget.get('cost_usd',0):.4f} -->"
    )

    final = (
        f"{exam_body}\n\n"
        f"---\n\n"
        f"# Answer Key & Analysis\n\n"
        f"{answer_section}"
        f"{budget_note}"
    )
    await _try_save_artifact(
        state,
        artifact_key="final_output",
        content={"markdown": final, "n_questions": len(questions)},
        worker_class="tool_only",
        node_name="assembler",
        artifact_type="final_output",
    )
    return {"final_output": final}


assembler.default_worker_class = "tool_only"


# ─────────────────────────────────────────────────────────────────────────────
# 13. FinalDrafter → ExamCardWriter fan-out
# ─────────────────────────────────────────────────────────────────────────────

def final_drafter_to_writer(state: AgentState) -> List[Send]:
    """
    Map step: fan out one ExamCardWriter per polished VerifiedQuestion.

    Each branch independently coerces and persists one exam_questions row
    + one exam_answers row, running in parallel across the LangGraph
    worker pool.  Results reduce back into `persisted_question_ids` before
    Assembler runs.
    """
    questions = state.get("verified_questions") or []
    logger.info("exam_questions x%d node fan out for exam_card_writer", len(questions))
    return [
        Send("exam_card_writer", {"question": q, **state})
        for q in questions
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 14. ExamCardWriter (parallel leaf, tool_only — deterministic DB write)
# ─────────────────────────────────────────────────────────────────────────────

async def exam_card_writer(state: Dict) -> Dict:
    """
    Persist one exam question + answer pair to the database.

    Runs in parallel via Send fan-out from FinalDrafter (one coroutine per
    VerifiedQuestion).  Uses ExamProcessor to coerce/validate the question
    dict before writing, mirroring the pattern used by QuizProcessor in the
    quiz pipeline.

    Writes:
      - one row into public.exam_questions  (exam_id → notes.id)
      - one row into public.exam_answers    (exam_id → notes.id,
                                             question_id → exam_questions.id)

    Returns {"persisted_question_ids": [<uuid>]} which LangGraph appends
    into the shared list-typed state field.
    """
    import uuid as _uuid
    from datetime import datetime, timezone
    from tasks.database import get_db_connection
    from utils.note_processing.exam_processor import ExamProcessor

    q_raw: Dict = state["question"]
    exam_id: str = (state.get("job_id") or "").strip()
    user_id: str = (state.get("user_id") or "").strip()

    if not exam_id:
        logger.warning("exam_card_writer: job_id missing — skipping DB write")
        return {"persisted_question_ids": []}

    processor = ExamProcessor()
    coerced = processor.coerce_one(q_raw, fallback_index=q_raw.get("question_index", 0))
    if coerced is None:
        logger.warning("exam_card_writer: coerce_one returned None for Q%s", q_raw.get("question_index"))
        return {"persisted_question_ids": []}

    question_db_id = str(_uuid.uuid4())
    answer_db_id   = str(_uuid.uuid4())
    now            = datetime.now(timezone.utc)

    try:
        async with get_db_connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO exam_questions (
                        id, exam_id, user_id, question_index, issue_label,
                        fact_pattern, call_of_question, grounding_verdict,
                        revised, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                    question_db_id,
                    exam_id,
                    user_id or None,
                    coerced["question_index"],
                    coerced["issue_label"],
                    coerced["fact_pattern"],
                    coerced["call_of_question"],
                    coerced["grounding_verdict"],
                    coerced["revised"],
                    now,
                )
                import json as _json
                await conn.execute(
                    """
                    INSERT INTO exam_answers (
                        id, exam_id, question_id, answer_key, citations, created_at
                    ) VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                    """,
                    answer_db_id,
                    exam_id,
                    question_db_id,
                    coerced["answer_key"],
                    _json.dumps(coerced["citations"]),
                    now,
                )
        logger.info("exam_card_writer: persisted Q%d → %s", coerced["question_index"], question_db_id[:8])
        return {"persisted_question_ids": [question_db_id]}

    except Exception as exc:
        logger.error("exam_card_writer: DB write failed for Q%d: %s", coerced["question_index"], exc)
        await _try_save_artifact(
            state,
            artifact_key=f"exam_card_write_error:{coerced['question_index']}",
            content={"error": str(exc), "question_index": coerced["question_index"]},
            worker_class="tool_only",
            node_name="exam_card_writer",
            artifact_type="error",
        )
        return {"persisted_question_ids": []}


exam_card_writer.default_worker_class = "tool_only"
