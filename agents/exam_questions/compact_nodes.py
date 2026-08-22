# agents/exam_questions/compact_nodes.py
"""
Compact 4-stage exam-questions pipeline.

Replaces the 13-node graph's ~49 LLM round-trips (24 `_llm` + ~25 `verify_claim`)
with four stages:

  1. plan_agent        — one worker_mid call → difficulty AND archetype mix.
                         Runs in PARALLEL with research_agent.
  2. research_agent    — ONE bounded tool-calling loop. Absorbs source_profiler
                         + concept_synthesizer + issue_clusterer + retriever.
                         Emits exactly `n_questions` question specs plus an
                         evidence store.
     sync_barrier      — no-op join (also replaces the legacy
                         `grounder_dispatcher`, which existed purely to stop an
                         N² grounder fan-out).
  3. question_generator — [Send × n_questions]. Absorbs question_drafter +
                         answer_key_builder + grounder + critic + reviser +
                         final_drafter (6 nodes → 1 call). Per-item fan-out
                         because each carries a full model answer.
  4. final_formatter   — pure Python. DB write + markdown. Zero LLM.

Three latent bugs in the legacy path are fixed here:
  • `assembler` was defined but never wired into the graph, so `final_output`
    was NEVER produced (the dispatcher read `""` every run). The compact
    formatter emits real markdown.
  • `initial_state` never seeded the `operator.add` fields (unlike every other
    agent) — fixed in graph.py.
  • Re-running a note APPENDED duplicate rows (uuid4 ids, no unique
    constraint, `exam_id = note_id`). The formatter now DELETEs the note's
    existing rows inside the same transaction, so regeneration replaces.

NEW product specs (neither existed in the legacy prompts):
  • model answers target 1,000-2,000 words (2-4 pages) — the old code had a
    max_tokens=3000 ceiling and no floor;
  • a required pedagogical archetype mix — every question must go beyond
    factual recall.

Legacy 13-node pipeline remains in nodes.py; graph.py selects via
EXAM_QUESTIONS_COMPACT (default true).
"""

import json
import logging
import os
import re
import uuid as _uuid_mod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from agents.common.compact_core import (
    build_tool_model,
    prefetch_corpus_survey,
    run_bounded_tool_loop,
    strip_chunk_citations,
    word_budget_line,
)

from .nodes import (
    _llm,
    _parse_json,
    _try_save_artifact,
    _node_start,
    _node_done,
    _node_warn,
)
from .state import AgentState

logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────

MAX_RESEARCH_TURNS = int(os.getenv("EXAM_RESEARCH_MAX_TURNS", "5"))
QUESTION_GEN_MAX_TURNS = int(os.getenv("EXAM_GEN_MAX_TURNS", "2"))
HARD_OUTPUT_TOKENS = int(os.getenv("EXAM_HARD_OUTPUT_TOKENS", "14000"))

DOSSIER_TARGET_WORDS = int(os.getenv("EXAM_DOSSIER_TARGET_WORDS", "700"))
# 2-4 pages of model answer ≈ 1,000-2,000 words; the fact pattern adds ~200.
ANSWER_TARGET_WORDS = int(os.getenv("EXAM_ANSWER_TARGET_WORDS", "1500"))
ANSWER_MAX_WORDS = int(os.getenv("EXAM_ANSWER_MAX_WORDS", "2200"))

GENERATOR_MAX_CHUNKS = 12
GENERATOR_CHUNK_CHAR_CAP = 1400

# ── Pedagogical archetypes (REQUIRED — no pure-recall archetype exists) ───────
# Every exam question must go beyond factual recall. Research assigns one
# archetype per spec and must satisfy the coverage floors below.
EXAM_ARCHETYPES = [
    "application",              # apply the rule to a fresh fact pattern
    "fact_change_hypothetical", # change one operative fact; does the result flip?
    "rule_boundary",            # where does the rule stop applying?
    "counterargument",          # best argument for the losing side
    "domain_shift",             # same doctrine, different factual domain
    "far_transfer_analogy",     # novel domain requiring analogical reasoning
    "policy",                   # the policy values the rule serves
]

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

# Markdown contract, ported from the legacy `_IRAC_STYLE_GUIDE`.
_MD_QUESTION_HEADING = "## Question {n}"
_MD_CALL_HEADING = "**Call of the Question**"
_MD_ANSWER_HEADING = "### Answer Key — Question {n}"


def _fetch_worker_model_local():
    from .worker_config import _fetch_worker_model
    return _fetch_worker_model


# ─────────────────────────────────────────────────────────────────────────────
# 1. plan_agent
# ─────────────────────────────────────────────────────────────────────────────

async def plan_agent(state: AgentState) -> Dict:
    """
    Exam strategy — one fast call. Absorbs the legacy `planner`, upgraded from
    free text to a structured plan (the old node's difficulty guidance never
    reached the drafter in any machine-readable form).
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import PLANNER_TOOLS

    logger.info("🔀 [%s] PARALLEL fire → plan_agent (alongside research_agent)",
                (state.get("job_id") or "")[:8] or "no-job")
    n_questions = int(state.get("n_questions") or 5)
    _node_start("plan_agent", state,
                n_sources=len(state.get("source_ids") or []), n_questions=n_questions)

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=PLANNER_TOOLS,
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
        "You are a T-14 law professor planning an issue-spotter exam. Given the "
        "available sources, produce the exam strategy — do NOT write questions.\n\n"
        "1. Identify the richest legal issues across the sources — prioritise "
        "issues with multi-party conflicts, competing defences, or factual "
        "ambiguity that forces legal analysis.\n"
        "2. Note any documents with dissenting opinions, policy debates, or "
        "multi-factor tests — these generate the best exam material.\n"
        "3. Recommend a difficulty distribution and an ARCHETYPE MIX.\n\n"
        f"Available archetypes (there is no pure-recall archetype — every "
        f"question must go beyond factual recall): {EXAM_ARCHETYPES}\n\n"
        "Return JSON with keys: job_type, course_context, "
        "priority_doctrines (list), "
        "difficulty_mix (e.g. {'straightforward_application': 2, 'nuanced_analysis': 3}), "
        "archetype_mix (map archetype → count, summing to the question count), "
        "notes (one sentence)."
    )
    raw = await _llm(
        "worker_mid",
        f"Sources available:\n{sources_json}\n\n"
        f"User request: {state['request']}\n"
        f"Questions to produce: {n_questions}",
        system=system, max_tokens=800, _node="plan_agent",
    )
    try:
        job_plan = _parse_json(raw)
    except Exception as exc:
        _node_warn("plan_agent", state, f"JSON parse failed ({exc}) — minimal plan")
        job_plan = {
            "job_type": "exam_questions",
            "course_context": "",
            "priority_doctrines": [],
            "difficulty_mix": {},
            "archetype_mix": {},
            "notes": "",
        }

    _node_done("plan_agent", state,
               doctrines=job_plan.get("priority_doctrines", [])[:4])
    return {"job_plan": job_plan}


# ─────────────────────────────────────────────────────────────────────────────
# 2. research_agent
# ─────────────────────────────────────────────────────────────────────────────

def _research_system(n_questions: int) -> str:
    """Folds in source_profiler + concept_synthesizer + issue_clusterer + retriever."""
    return (
        "You are a legal research agent selecting and grounding the issues for "
        "a T-14 issue-spotter exam. Work in bounded steps:\n"
        "  1. Orient: the CORPUS SURVEY is ALREADY PROVIDED in the first "
        "message. Do NOT re-fetch it with list_sources / get_doc_outline / "
        "get_doc_metadata / find_docs_about / find_sections_about — that wastes "
        "a whole turn.\n"
        "  2. Retrieve (START HERE on turn 1): batch MANY parallel searches "
        "across the candidate issues.\n"
        "  3. Select and ground exactly the issues the exam needs.\n\n"
        "ISSUE SELECTION — this is the highest-value judgement you make:\n"
        "• Select issues that generate genuine legal analysis — NOT trivial "
        "recall.\n"
        "• Prefer issues with: multi-party liability, competing doctrines, "
        "factual ambiguity that activates the rule, or dissent/minority-rule "
        "tension.\n"
        "• Look for CROSS-DOCUMENT throughlines — e.g. tension between the "
        "majority rule in one source and the minority rule in another creates a "
        "natural call of the question. Record these in `cross_doc_tension`.\n"
        "• NO duplicate issues — do not select two issues testing the same "
        "element of the same rule.\n"
        "• issue_label format: 'Doctrine — specific sub-issue' "
        "(e.g. 'Negligence — duty to foreseeable plaintiff').\n\n"
        f"ARCHETYPE ASSIGNMENT — assign each spec one of: {EXAM_ARCHETYPES}\n"
        "Coverage floors per 5 questions: at least 1 fact_change_hypothetical, "
        "at least 1 of rule_boundary/counterargument, and at least 1 of "
        "far_transfer_analogy/domain_shift. There is no recall archetype — "
        "every question must test reasoning, application or transfer.\n\n"
        "RULES:\n"
        "• Your turn budget is for RETRIEVAL. Turn 1 should already be a large "
        "batch of searches, not discovery.\n"
        "• Only select issues you actually grounded — every spec carries "
        "chunk_ids you retrieved.\n"
        f"• STOP CRITERION: once you have {n_questions} distinct, grounded "
        "issue specs, STOP calling tools and emit the dossier, even if turns "
        "remain.\n\n"
        f"FINAL ANSWER — return ONLY this JSON object with EXACTLY {n_questions} "
        "specs (no prose):\n"
        "{\n"
        '  "cross_doc_throughlines": [{"concept", "sources", "exam_angle"}],\n'
        '  "specs": [{\n'
        '     "question_index": 1-based int,\n'
        '     "issue_label": "Doctrine — specific sub-issue",\n'
        '     "archetype": one of the archetypes above,\n'
        '     "difficulty": "straightforward"|"nuanced"|"hard",\n'
        '     "source_ids": [uuid],\n'
        '     "governing_rule": one sentence — the rule the question tests,\n'
        '     "defences_in_play": [str],\n'
        '     "cross_doc_tension": str or "",\n'
        '     "evidence_chunk_ids": [10-12 chunk_ids — or every relevant chunk\n'
        "        if the corpus holds fewer. This is the ONLY evidence selector\n"
        "        the question writer gets; never pad with ids you did not\n"
        "        actually retrieve]\n"
        "  }]\n"
        "}"
        + word_budget_line(DOSSIER_TARGET_WORDS,
                           int(DOSSIER_TARGET_WORDS * 1.5), kind="dossier")
    )


async def research_agent(state: AgentState) -> Dict:
    from agents.tools.base import make_tools
    from agents.tools.registry import EXAM_RESEARCH_TOOLS

    logger.info("🔀 [%s] PARALLEL fire → research_agent (alongside plan_agent)",
                (state.get("job_id") or "")[:8] or "no-job")
    n_questions = int(state.get("n_questions") or 5)
    _node_start("research_agent", state,
                max_turns=MAX_RESEARCH_TURNS, n_questions=n_questions)

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=EXAM_RESEARCH_TOOLS,
    )
    base_model, sem, rebuild = await build_tool_model(
        _fetch_worker_model_local(), "orchestrator",
        provider=os.getenv("EXAM_RESEARCH_PROVIDER"),
        hard_cap=HARD_OUTPUT_TOKENS,
    )

    survey, n_chunks = await prefetch_corpus_survey(state, tools)
    logger.info("  📚 [research_agent] corpus survey pre-fetched (%d chars) n_chunks=%d",
                len(survey), n_chunks)

    messages: List[Any] = [
        SystemMessage(content=_research_system(n_questions)),
        HumanMessage(content=(
            f"{survey}\n\n"
            f"Source ids in scope: {state.get('source_ids', [])}\n"
            f"User request: {state['request']}\n"
            f"Questions required: {n_questions}\n\n"
            "You already have the corpus survey — do not re-fetch it. Begin "
            "RETRIEVAL immediately with a large batch of parallel searches "
            "across the candidate issues."
        )),
    ]

    evidence_store: Dict[str, Dict] = {}
    raw = await run_bounded_tool_loop(
        base_model, tools, messages, MAX_RESEARCH_TURNS, sem,
        "research_agent", state, evidence_store=evidence_store,
        countdown_from_turn=3, rebuild=rebuild,
        forced_stop_prompt=(
            f"Retrieval is complete. Emit the dossier JSON now with exactly "
            f"{n_questions} specs, using the evidence gathered. Return ONLY the "
            "JSON object."
        ),
    )
    try:
        dossier: Optional[Dict[str, Any]] = _parse_json(raw)
    except Exception as exc:
        _node_warn("research_agent", state, f"dossier JSON parse failed: {exc}")
        dossier = None
    if not isinstance(dossier, dict):
        dossier = {}

    specs = [s for s in (dossier.get("specs") or []) if isinstance(s, dict)][:n_questions]
    for i, s in enumerate(specs, start=1):
        s["question_index"] = i          # renumber contiguously
        if s.get("archetype") not in EXAM_ARCHETYPES:
            s["archetype"] = "application"
    dossier["specs"] = specs
    dossier.setdefault("cross_doc_throughlines", [])

    if len(specs) < n_questions:
        _node_warn("research_agent", state,
                   f"only {len(specs)} of {n_questions} specs produced — "
                   "exam will be short")

    await _try_save_artifact(
        state, "exam_dossier",
        {"dossier": dossier, "evidence_chunks": len(evidence_store)},
        "orchestrator", "research_agent", "exam_dossier",
        source_ids=state.get("source_ids"),
    )
    _node_done("research_agent", state,
               n_specs=len(specs), n_evidence_chunks=len(evidence_store),
               archetypes=[s.get("archetype") for s in specs])
    return {"exam_dossier": dossier, "evidence_store": evidence_store}


# ─────────────────────────────────────────────────────────────────────────────
# sync_barrier
# ─────────────────────────────────────────────────────────────────────────────

async def sync_barrier(state: AgentState) -> Dict:
    """
    Join for the parallel plan/research branches. Also does the job of the
    legacy `grounder_dispatcher`, which existed purely as a barrier to stop the
    grounder fan-out from re-evaluating per parallel task.
    """
    job = (state.get("job_id") or "")[:8] or "no-job"
    dossier = state.get("exam_dossier") or {}
    logger.info(
        "🔗 [%s] sync_barrier — joined plan_agent + research_agent  "
        "job_plan=%s n_specs=%d",
        job, "present" if state.get("job_plan") else "MISSING (join failed?)",
        len(dossier.get("specs") or []),
    )
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 3. question_generator — [Send × n_questions]
# ─────────────────────────────────────────────────────────────────────────────

def dossier_to_generators(state: AgentState):
    """Per-item fan-out — each question carries a full 2-4 page model answer."""
    specs = (state.get("exam_dossier") or {}).get("specs") or []
    if not specs:
        logger.warning("dossier_to_generators: no specs — routing to final_formatter")
        return "final_formatter"
    logger.info("exam_questions(compact) x%d node fan out for question_generator",
                len(specs))
    return [Send("question_generator", {"spec": s, **state}) for s in specs]


async def question_generator(state: Dict) -> Dict:
    """
    Emit ONE complete exam question: fact pattern + call + full IRAC model
    answer + citations, self-checked. Absorbs question_drafter +
    answer_key_builder + grounder + critic + reviser + final_drafter.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import VERIFIER_TOOLS
    from langchain_core.tools import tool as _tool_decorator

    spec: Dict[str, Any] = state["spec"]
    dossier: Dict[str, Any] = state.get("exam_dossier") or {}
    job_plan: Dict[str, Any] = state.get("job_plan") or {}
    evidence_store: Dict[str, Dict] = state.get("evidence_store") or {}
    q_index = spec.get("question_index", 1)

    _node_start("question_generator", state,
                q=q_index, issue=str(spec.get("issue_label", ""))[:40],
                archetype=spec.get("archetype"))

    evidence: List[Dict] = []
    for cid in (spec.get("evidence_chunk_ids") or [])[:GENERATOR_MAX_CHUNKS]:
        e = evidence_store.get(str(cid))
        if e:
            evidence.append(e)
    evidence_text = "\n\n".join(
        f"[chunk_id:{e['chunk_id']} p.{e.get('page', '?')}]\n"
        f"{e['content'][:GENERATOR_CHUNK_CHAR_CAP]}"
        for e in evidence
    ) or "(no evidence indexed — use grep_research_corpus to find support)"

    @_tool_decorator
    def grep_research_corpus(keyword: str) -> str:
        """Case-insensitive substring search over the research corpus already
        harvested for this exam. Free and instant — use this BEFORE any live
        database tool. Returns up to 10 matching excerpts with chunk_ids."""
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
        tool_names=[t for t in VERIFIER_TOOLS
                    if t in ("verify_claim", "find_supporting_evidence",
                             "get_citations_for")],
    )
    tools = [grep_research_corpus] + verifier_tools

    system = (
        "You are a T-14 law professor writing ONE exam question and its model "
        "answer. This output is final — no editor runs after you.\n\n"
        "FACT PATTERN STANDARDS:\n"
        "- Multi-character narrative (2-5 named parties) with chronological "
        "events.\n"
        "- Embed the legal issue NATURALLY — do NOT name the doctrine "
        "explicitly (never write 'this is a negligence case'). The student must "
        "identify the issue from the facts.\n"
        "- Include enough legally operative facts to resolve every sub-issue in "
        "the answer key — missing facts create unanswerable questions.\n"
        "- Add at least one fact that activates a defence or complication, and "
        "one fact that creates genuine ambiguity requiring analysis (not a fact "
        "with an obvious result).\n"
        "- Base the facts ONLY on the provided legal context — do not import "
        "facts from other doctrines or real-world cases.\n"
        "- 150-250 words.\n\n"
        "CALL OF THE QUESTION STANDARDS:\n"
        "- Open-ended: tell the student what to analyse WITHOUT giving away the "
        "issue.\n"
        "- Do NOT ask 'Was there negligence?' — that names the issue. Prefer "
        "'Discuss all claims and defences.'\n\n"
        "ARCHETYPE — this question must be a "
        f"**{spec.get('archetype', 'application')}** question:\n"
        "  application → apply the rule to a fresh fact pattern\n"
        "  fact_change_hypothetical → the facts turn on one changed operative "
        "fact vs. the source case; the analysis must hinge on it\n"
        "  rule_boundary → the facts sit at the edge of the rule's reach\n"
        "  counterargument → the facts make the losing side's argument genuinely "
        "strong\n"
        "  domain_shift → same doctrine, a factual domain the sources did not "
        "cover\n"
        "  far_transfer_analogy → a novel domain requiring analogical reasoning "
        "from the source rule\n"
        "  policy → the facts force the student to argue the policy values "
        "behind the rule\n\n"
        "MODEL ANSWER (IRAC per sub-issue):\n"
        "- Write a full IRAC analysis for EVERY major issue in the fact pattern.\n"
        "- Issue: one sentence framing the precise legal question — "
        "'Whether [party] is liable for [claim] because [key fact].'\n"
        "- Rule: state the applicable rule from the source materials. Name the "
        "doctrine and, if available, the case or statute. Include ALL elements "
        "the rule requires.\n"
        "- Application: argue BOTH sides — the best argument for "
        "plaintiff/prosecution AND the best for defendant. Address every "
        "affirmative defence raised. Do NOT reach a conclusion until you have "
        "analysed both sides.\n"
        "- Conclusion: one sentence stating the most likely result and why.\n\n"
        "COMPLETENESS:\n"
        "- Cover ALL major issues raised by the facts — missing a sub-issue is a "
        "model-answer failure.\n"
        "- Include the primary claim, all defences, and any counterclaims or "
        "cross-claims the facts suggest.\n"
        "- Do NOT import doctrine unsupported by the provided legal context.\n\n"
        "SELF-CHECK before you answer (there is no critic downstream):\n"
        "  1. call_clarity — is the call open-ended and free of the issue name?\n"
        "  2. fact_completeness — does the fact pattern contain every fact the "
        "answer key relies on?\n"
        "  3. appropriate_complexity — does this require genuine analysis, or is "
        "the answer obvious without reading the facts?\n"
        "  4. grounding — is every rule you state supported by the evidence?\n"
        "Fix anything that fails before emitting.\n\n"
        "GROUNDING: if something you want to state is not in the evidence, call "
        "grep_research_corpus first (free, instant); use find_supporting_evidence "
        "/ get_citations_for to reach the source, and verify_claim when unsure a "
        "legal assertion holds. Most questions need no tool calls. Never invent "
        "authority.\n\n"
        "Return ONLY this JSON object:\n"
        "{\n"
        '  "issue_label": str,\n'
        '  "fact_pattern": str (150-250 words),\n'
        '  "call_of_question": str,\n'
        '  "answer_key": str (full IRAC, plain text, blank line between blocks),\n'
        '  "chunk_ids_used": [chunk_id UUID strings copied from the\n'
        "     [chunk_id:...] markers — the UUID only, not the page number],\n"
        '  "grounding_verdict": "pass"|"warn"|"fail" (your honest self-assessment)\n'
        "}\n"
        f"The answer_key must be a THOROUGH 2-4 page analysis: aim for "
        f"~{ANSWER_TARGET_WORDS} words, never exceeding {ANSWER_MAX_WORDS}. "
        "Depth of both-sides reasoning is the point — do not summarise."
    )

    prompt = (
        f"SPEC: {json.dumps(spec, indent=2)}\n\n"
        f"Course context: {job_plan.get('course_context', '')}\n"
        f"Cross-document throughlines: "
        f"{json.dumps(dossier.get('cross_doc_throughlines') or [])}\n\n"
        f"LEGAL CONTEXT:\n{evidence_text}"
    )

    base_model, sem, rebuild = await build_tool_model(
        _fetch_worker_model_local(), "orchestrator", hard_cap=HARD_OUTPUT_TOKENS,
    )
    messages: List[Any] = [SystemMessage(content=system), HumanMessage(content=prompt)]
    raw = await run_bounded_tool_loop(
        base_model, tools, messages, QUESTION_GEN_MAX_TURNS, sem,
        "question_generator", state,
        countdown_from_turn=QUESTION_GEN_MAX_TURNS, rebuild=rebuild,
        forced_stop_prompt=(
            "STOP calling tools. Emit the question JSON NOW using only the "
            "evidence already gathered."
        ),
    )
    try:
        data = _parse_json(raw)
    except Exception as exc:
        _node_warn("question_generator", state, f"Q{q_index} JSON parse failed: {exc}")
        data = {}
    if not isinstance(data, dict):
        data = {}

    verdict = data.get("grounding_verdict", "warn")
    if verdict not in ("pass", "warn", "fail"):
        verdict = "warn"   # DB CHECK constraint allows only these three

    chunk_ids = [c for c in (data.get("chunk_ids_used") or [])
                 if isinstance(c, str) and _UUID_RE.fullmatch(c.strip())]

    question = {
        "question_index": q_index,
        "issue_label": data.get("issue_label") or spec.get("issue_label", "General Legal Issue"),
        "archetype": spec.get("archetype", "application"),
        "fact_pattern": data.get("fact_pattern", ""),
        "call_of_question": data.get("call_of_question", ""),
        "answer_key": data.get("answer_key", ""),
        "chunk_ids_used": chunk_ids[:10],
        "grounding_verdict": verdict,
        "revised": False,
    }

    await _try_save_artifact(
        state, f"exam_question:{q_index}", question, "orchestrator",
        "question_generator", "exam_question", source_ids=state.get("source_ids"),
    )
    _node_done("question_generator", state, q=q_index,
               answer_words=len(question["answer_key"].split()), verdict=verdict)
    return {"generated_questions": [question]}


question_generator.default_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 4. final_formatter — deterministic, zero LLM
# ─────────────────────────────────────────────────────────────────────────────

async def final_formatter(state: AgentState) -> Dict:
    """
    DB persistence + markdown. Makes NO LLM call.

    Two fixes vs legacy: the `assembler` that would have produced `final_output`
    was never wired into the graph (so `final_output` was always ""), and
    re-running a note appended duplicate rows. Regeneration now REPLACES: the
    note's existing rows are deleted inside the same transaction.
    """
    from agents.tools.base import make_tools
    from utils.note_processing.exam_processor import ExamProcessor

    questions: List[Dict] = list(state.get("generated_questions") or [])
    job_id = state.get("job_id") or ""
    user_id = state.get("user_id") or ""

    _node_start("final_formatter", state, n_questions=len(questions))
    questions.sort(key=lambda q: q.get("question_index", 99))

    # Strip internal chunk-uuid citations before anything user-facing or DB-bound.
    citations_removed = 0
    for q in questions:
        for key in ("fact_pattern", "call_of_question", "answer_key"):
            cleaned, n = strip_chunk_citations(q.get(key, ""))
            q[key] = cleaned
            citations_removed += n
    if citations_removed:
        logger.info("  🧹 [final_formatter] stripped %d inline chunk citation(s)",
                    citations_removed)

    # Build the exam_answers.citations payload from each question's
    # chunk_ids_used. This is the job the legacy `grounder` did; it is a plain
    # DB lookup (no LLM), so this node stays zero-LLM.
    try:
        cite_tools = make_tools(
            state["project_id"],
            source_ids=state.get("source_ids", []),
            use_voyage=state.get("use_voyage", False),
            tool_names=["get_citations_for"],
        )
        cite_tool = next((t for t in cite_tools if t.name == "get_citations_for"), None)
        if cite_tool is not None:
            for q in questions:
                ids = q.get("chunk_ids_used") or []
                if not ids:
                    q["citations"] = []
                    continue
                try:
                    q["citations"] = json.loads(
                        await cite_tool.ainvoke({"chunk_ids": ids[:10]})
                    )
                except Exception:
                    q["citations"] = []
    except Exception as exc:
        _node_warn("final_formatter", state, f"citation lookup failed: {exc}")

    processor = ExamProcessor()
    persisted_ids: List[str] = []
    if not job_id:
        _node_warn("final_formatter", state,
                   "job_id empty — skipping DB writes (ledger disabled)")
    else:
        try:
            from tasks.database import get_db_connection

            now = datetime.now(timezone.utc)
            async with get_db_connection() as conn:
                async with conn.transaction():
                    # Replace-on-regenerate. exam_id == notes.id and ids are
                    # random uuid4 with no unique constraint, so without this a
                    # re-run silently doubles the question set.
                    await conn.execute(
                        "DELETE FROM exam_answers WHERE exam_id = $1", job_id
                    )
                    deleted = await conn.execute(
                        "DELETE FROM exam_questions WHERE exam_id = $1", job_id
                    )
                    logger.info("  ♻️ [final_formatter] cleared prior rows for "
                                "this note (%s)", deleted)

                    for q in questions:
                        coerced = processor.coerce_one(
                            q, fallback_index=q.get("question_index", 0)
                        )
                        if not coerced:
                            _node_warn("final_formatter", state,
                                       f"Q{q.get('question_index')} rejected by "
                                       "ExamProcessor (empty fact_pattern or "
                                       "answer_key)")
                            continue
                        q_id = str(_uuid_mod.uuid4())
                        await conn.execute(
                            """
                            INSERT INTO exam_questions (
                                id, exam_id, user_id, question_index, issue_label,
                                fact_pattern, call_of_question, grounding_verdict,
                                revised, created_at
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                            """,
                            q_id, job_id, user_id or None,
                            coerced["question_index"], coerced["issue_label"],
                            coerced["fact_pattern"], coerced["call_of_question"],
                            coerced["grounding_verdict"], coerced.get("revised", False),
                            now,
                        )
                        citations = json.dumps(coerced.get("citations") or [])
                        await conn.execute(
                            """
                            INSERT INTO exam_answers (
                                id, exam_id, question_id, answer_key, citations, created_at
                            ) VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                            """,
                            str(_uuid_mod.uuid4()), job_id, q_id,
                            coerced["answer_key"], citations, now,
                        )
                        persisted_ids.append(q_id)
        except Exception as exc:
            logger.error("final_formatter DB write failed: %s", exc)
            await _try_save_artifact(
                state, "exam_card_write_error", {"error": str(exc)},
                "tool_only", "final_formatter", "error",
                source_ids=state.get("source_ids"),
            )

    # ── Markdown (the legacy `assembler` never actually ran) ──────────────
    lines: List[str] = ["# Exam Questions", ""]
    for q in questions:
        n = q.get("question_index", 0)
        lines += [
            _MD_QUESTION_HEADING.format(n=n),
            f"*{q.get('issue_label', '')}*",
            "",
            q.get("fact_pattern", ""),
            "",
            _MD_CALL_HEADING,
            "",
            q.get("call_of_question", ""),
            "",
            _MD_ANSWER_HEADING.format(n=n),
            "",
            q.get("answer_key", ""),
            "",
            "---",
            "",
        ]
    total_words = sum(len((q.get("answer_key") or "").split()) for q in questions)
    lines.append(
        f"<!-- exam-questions compact | {len(questions)} questions | "
        f"{len(persisted_ids)} persisted | answer words: {total_words} -->"
    )
    markdown = "\n".join(lines)

    await _try_save_artifact(
        state, "final_output",
        {"markdown_length": len(markdown), "persisted": len(persisted_ids)},
        "tool_only", "final_formatter", "final_output",
        source_ids=state.get("source_ids"),
    )
    _node_done("final_formatter", state,
               questions=len(questions), persisted=len(persisted_ids),
               answer_words=total_words)
    return {"persisted_question_ids": persisted_ids, "final_output": markdown}
