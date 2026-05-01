# agents/exam_questions/nodes.py
"""
All 10 nodes for the exam-questions LangGraph agent.

Node responsibilities:
  1. Planner          — read source outlines, write retrieval strategy
  2. SourceProfiler   — per-doc profile (parallel Send from Planner)
  3. IssueClusterer   — cluster profiles into exam issues
  4. Retriever        — per-issue evidence bundle (parallel Send)
  5. QuestionDrafter  — per-issue fact-pattern (parallel Send, flagship)
  6. AnswerKeyBuilder — per-question answer key (parallel Send, flagship)
  7. Grounder         — per-question claim verification (parallel Send, cheap)
  8. Critic           — whole-exam critique (cheap)
  9. Reviser          — targeted revision of failing questions (conditional)
  10. Assembler        — deterministic final Markdown assembly
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List

from langgraph.types import Send

from .state import (
    AgentState,
    DraftQuestion,
    IssueCluster,
    RetrievalBundle,
    SourceProfile,
    VerifiedQuestion,
)

logger = logging.getLogger(__name__)

_HAIKU = "claude-haiku-4-5-20251001"
_SONNET = "claude-sonnet-4-6"
_OPUS = "claude-opus-4-7"

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")


async def _llm(model: str, prompt: str, system: str = "", max_tokens: int = 2048) -> str:
    import httpx

    messages = [{"role": "user", "content": prompt}]
    body: Dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        body["system"] = system

    async with httpx.AsyncClient(timeout=120.0) as http:
        resp = await http.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()


def _add_budget(state: AgentState, model: str, in_tok: int, out_tok: int) -> Dict:
    costs = {
        _HAIKU: (0.25e-6, 1.25e-6),
        _SONNET: (3e-6, 15e-6),
        _OPUS: (15e-6, 75e-6),
    }
    in_cost, out_cost = costs.get(model, (3e-6, 15e-6))
    delta = in_tok * in_cost + out_tok * out_cost
    budget = dict(state.get("budget") or {})
    budget["input_tokens"] = budget.get("input_tokens", 0) + in_tok
    budget["output_tokens"] = budget.get("output_tokens", 0) + out_tok
    budget["cost_usd"] = round(budget.get("cost_usd", 0.0) + delta, 6)
    return budget


# ─────────────────────────────────────────────────────────────────────────────
# 1. Planner
# ─────────────────────────────────────────────────────────────────────────────

async def planner(state: AgentState) -> Dict:
    """
    Read the list of source documents and write a concise retrieval strategy.
    Emits a Send per source_id to SourceProfiler.
    """
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
        _SONNET,
        f"Documents available:\n{sources_json}\n\nUser request: {state['request']}",
        system=system,
        max_tokens=512,
    )

    return {"plan": plan}


def planner_to_profiler(state: AgentState) -> List[Send]:
    """Fan-out: one SourceProfiler invocation per source document."""
    return [
        Send("source_profiler", {"source_id": sid, **state})
        for sid in state["source_ids"]
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. SourceProfiler (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def source_profiler(state: Dict) -> Dict:
    """
    Build a profile for a single source document: concepts, toc, summary.
    Receives an augmented state dict injected by Send.
    """
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

    # Merge into shared list (LangGraph reduces parallel branch outputs)
    return {"source_profiles": [profile]}


# ─────────────────────────────────────────────────────────────────────────────
# 3. IssueClusterer
# ─────────────────────────────────────────────────────────────────────────────

async def issue_clusterer(state: AgentState) -> Dict:
    """
    Given all source profiles, identify and cluster N exam-worthy legal issues.
    """
    profiles = state.get("source_profiles") or []
    n = state["n_questions"]

    profiles_text = json.dumps(profiles, indent=2)
    prompt = (
        f"You are a law professor. Given the source document profiles below "
        f"and the exam request, identify exactly {n} high-quality legal issues "
        f"suitable for exam fact-patterns.\n\n"
        f"For each issue output a JSON object with:\n"
        f'  "issue_label": short label (e.g. "Negligence — proximate cause")\n'
        f'  "source_ids": list of source UUIDs where material exists\n'
        f'  "section_hints": list of section_path hints to retrieve\n'
        f'  "priority": 1 (high) | 2 (medium) | 3 (low)\n\n'
        f"Return a JSON array of exactly {n} objects. No extra text.\n\n"
        f"Source profiles:\n{profiles_text}\n\n"
        f"User request: {state['request']}\n"
        f"Plan: {state.get('plan', '')}"
    )

    raw = await _llm(_SONNET, prompt, max_tokens=1024)

    try:
        clusters: List[IssueCluster] = json.loads(raw)
    except Exception:
        # Fallback: create one generic cluster per source
        clusters = [
            {
                "issue_label": f"Legal issues from document {i+1}",
                "source_ids": [p["source_id"]],
                "section_hints": [],
                "priority": 2,
            }
            for i, p in enumerate((profiles or [])[:n])
        ]

    return {"chosen_issues": clusters[:n]}


def clusterer_to_retriever(state: AgentState) -> List[Send]:
    """Fan-out: one Retriever per issue cluster."""
    return [
        Send("retriever", {"issue": issue, **state})
        for issue in (state.get("chosen_issues") or [])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Retriever (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def retriever(state: Dict) -> Dict:
    """
    Retrieve evidence passages for a single issue cluster.
    """
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
        "chunks": chunks[:20],
        "table_ids": [t["chunk_id"] for t in tables],
    }
    return {"retrieval_bundles": [bundle]}


def retriever_to_drafter(state: AgentState) -> List[Send]:
    """Fan-out: one QuestionDrafter per retrieval bundle."""
    bundles = state.get("retrieval_bundles") or []
    return [
        Send("question_drafter", {"bundle": b, "bundle_index": i, **state})
        for i, b in enumerate(bundles)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 5. QuestionDrafter (parallel leaf, flagship model)
# ─────────────────────────────────────────────────────────────────────────────

async def question_drafter(state: Dict) -> Dict:
    """
    Draft a fact-pattern hypothetical for one issue bundle.
    """
    bundle: RetrievalBundle = state["bundle"]
    idx: int = state["bundle_index"]

    context = "\n\n---\n\n".join(
        f"[{c.get('source_id','?')} p.{c.get('page_number','?')}]\n{c.get('content','')}"
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
        "chunk_ids_used (list of chunk id strings)"
    )

    prompt = (
        f"Issue: {bundle['issue_label']}\n\n"
        f"Legal context:\n{context}\n\n"
        f"Draft the fact-pattern hypothetical."
    )

    raw = await _llm(_OPUS, prompt, system=system, max_tokens=1500)

    try:
        data = json.loads(raw)
    except Exception:
        data = {"fact_pattern": raw, "call_of_question": "Discuss the rights and liabilities of all parties.", "chunk_ids_used": []}

    draft: DraftQuestion = {
        "question_index": idx,
        "issue_label": bundle["issue_label"],
        "fact_pattern": data.get("fact_pattern", raw),
        "call_of_question": data.get("call_of_question", ""),
        "answer_key": "",
        "chunk_ids_used": data.get("chunk_ids_used", [c.get("id","") for c in bundle["chunks"][:10]]),
    }
    return {"draft_questions": [draft]}


def drafter_to_answerkey(state: AgentState) -> List[Send]:
    """Fan-out: one AnswerKeyBuilder per draft question."""
    drafts = state.get("draft_questions") or []
    bundles = state.get("retrieval_bundles") or []
    bundle_map = {b["issue_label"]: b for b in bundles}
    return [
        Send("answer_key_builder", {
            "draft": d,
            "bundle": bundle_map.get(d["issue_label"], {"chunks": [], "table_ids": []}),
            **state,
        })
        for d in drafts
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 6. AnswerKeyBuilder (parallel leaf, flagship model)
# ─────────────────────────────────────────────────────────────────────────────

async def answer_key_builder(state: Dict) -> Dict:
    """
    Generate a full IRAC answer key for a drafted question.
    """
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

    answer_key = await _llm(_OPUS, prompt, system=system, max_tokens=2500)

    updated_draft = dict(draft)
    updated_draft["answer_key"] = answer_key
    return {"draft_questions": [updated_draft]}


def answerkey_to_grounder(state: AgentState) -> List[Send]:
    """Fan-out: one Grounder per drafted question."""
    return [
        Send("grounder", {"draft": d, **state})
        for d in (state.get("draft_questions") or [])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Grounder (parallel leaf, cheap model)
# ─────────────────────────────────────────────────────────────────────────────

async def grounder(state: Dict) -> Dict:
    """
    Extract 3-5 factual claims from the answer key and verify each against the docs.
    """
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
    cite_tool = next(t for t in tools if t.name == "get_citations_for")

    # Extract claims to verify
    extract_prompt = (
        f"Extract 3-5 specific legal claims from the answer key below as a JSON array of strings.\n\n"
        f"Answer key:\n{draft['answer_key'][:3000]}\n\n"
        "Return only the JSON array, no extra text."
    )
    claims_raw = await _llm(_HAIKU, extract_prompt, max_tokens=512)
    try:
        claims = json.loads(claims_raw)
        if not isinstance(claims, list):
            claims = [claims_raw]
    except Exception:
        claims = [draft["answer_key"][:300]]

    # Verify each claim
    verdicts = []
    for claim in claims[:5]:
        result_json = await verify_tool.ainvoke({"claim": claim, "k": 8})
        verdicts.append(json.loads(result_json))

    # Determine overall grounding verdict
    statuses = [v.get("verdict", "insufficient") for v in verdicts]
    if all(s == "supported" for s in statuses):
        overall = "pass"
        notes = "All claims supported."
    elif any(s == "contradicted" for s in statuses):
        overall = "fail"
        notes = "One or more claims contradicted by source material."
    else:
        overall = "warn"
        notes = "Some claims have insufficient evidence."

    # Build citations
    chunk_ids = draft.get("chunk_ids_used") or []
    if chunk_ids:
        cites_json = await cite_tool.ainvoke({"chunk_ids": chunk_ids[:10]})
        citations = json.loads(cites_json)
    else:
        citations = []

    vq: VerifiedQuestion = {
        "question_index": draft["question_index"],
        "fact_pattern": draft["fact_pattern"],
        "call_of_question": draft["call_of_question"],
        "answer_key": draft["answer_key"],
        "grounding_verdict": overall,
        "grounding_notes": notes,
        "citations": citations,
        "revised": False,
    }
    return {"verified_questions": [vq]}


# ─────────────────────────────────────────────────────────────────────────────
# 8. Critic
# ─────────────────────────────────────────────────────────────────────────────

async def critic(state: AgentState) -> Dict:
    """
    Holistic critique of the full draft exam. Flags questions that need revision.
    """
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
        f'  "critique": one-sentence note\n\n'
        f"Questions:\n{questions_text}\n\nReturn only JSON array, no extra text."
    )

    raw = await _llm(_HAIKU, prompt, max_tokens=1024)
    try:
        critiques = json.loads(raw)
    except Exception:
        critiques = []

    # Mark questions for revision
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


def should_revise(state: AgentState) -> str:
    """
    Conditional edge: route to Reviser if any question failed and
    we haven't hit the max revision count (2).
    """
    count = state.get("revision_count") or 0
    if count >= 2:
        return "assembler"
    questions = state.get("verified_questions") or []
    if any(q["grounding_verdict"] == "fail" for q in questions):
        return "reviser"
    return "assembler"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Reviser (conditional)
# ─────────────────────────────────────────────────────────────────────────────

async def reviser(state: AgentState) -> Dict:
    """
    Revise only the questions that failed grounding.  Max 2 passes.
    """
    questions = state.get("verified_questions") or []
    project_id = state["project_id"]
    use_voyage = state.get("use_voyage", False)

    revised = list(questions)
    for i, q in enumerate(revised):
        if q["grounding_verdict"] != "fail":
            continue

        # Pull fresh evidence
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

        context = "\n\n".join(c.get("content", "") for c in evidence[:10])

        prompt = (
            f"The following exam question failed grounding verification.\n"
            f"Critique: {q['grounding_notes']}\n\n"
            f"Original fact pattern:\n{q['fact_pattern']}\n\n"
            f"Supporting evidence from documents:\n{context}\n\n"
            f"Revise the fact pattern so all claims are supported by the evidence. "
            f"Keep the same legal issue and call of the question. "
            f"Return only the revised fact_pattern text."
        )

        revised_fp = await _llm(_OPUS, prompt, max_tokens=1200)
        updated_q = dict(q)
        updated_q["fact_pattern"] = revised_fp
        updated_q["grounding_verdict"] = "warn"
        updated_q["revised"] = True
        revised[i] = updated_q

    return {
        "verified_questions": revised,
        "revision_count": (state.get("revision_count") or 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10. Assembler
# ─────────────────────────────────────────────────────────────────────────────

async def assembler(state: AgentState) -> Dict:
    """
    Deterministic assembly of the final Markdown exam document.
    No LLM call — just formatting.
    """
    questions = sorted(
        state.get("verified_questions") or [],
        key=lambda q: q["question_index"],
    )

    sections = []
    answer_keys = []

    for i, q in enumerate(questions, 1):
        sections.append(
            f"## Question {i}\n\n"
            f"{q['fact_pattern']}\n\n"
            f"**{q['call_of_question']}**"
        )
        cites = q.get("citations") or []
        cite_str = ""
        if cites:
            cite_str = "\n\n*Sources: " + "; ".join(
                c.get("citation_string", "") for c in cites[:5]
            ) + "*"

        answer_keys.append(
            f"### Answer Key — Question {i}\n\n"
            f"{q['answer_key']}{cite_str}"
        )

    exam_body = "\n\n---\n\n".join(sections)
    answer_section = "\n\n---\n\n".join(answer_keys)

    budget = state.get("budget") or {}
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

    return {"final_output": final}
