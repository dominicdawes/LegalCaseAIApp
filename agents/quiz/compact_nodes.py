# agents/quiz/compact_nodes.py
"""
Compact 4-stage quiz pipeline.

Replaces the 8-node graph's ~45 LLM round-trips with four stages:

  1. plan_agent        — count math (deterministic) + one worker_mid call for
                         allocation strategy. Runs in PARALLEL with research.
                         (The legacy head_orchestrator made an LLM call whose
                         result was never assigned to anything.)
  2. research_agent    — ONE bounded tool-calling loop. Absorbs source_profiler
                         (×N) + case_rule_extractor (×≤15 orchestrator calls —
                         the single biggest cut) + cross_doc_concepts_synthesis
                         + quiz_blueprint_planner. Emits exactly num_questions
                         specs plus traps/confusables and an evidence store.
     sync_barrier      — no-op join.
  3. question_batch_generator — [Send × ceil(num_questions/batch_size)].
                         Absorbs question_drafter + false_trap_red_herring_
                         generator + question_evaluator + reviser + grounder.
                         Batch fan-out amortises the shared context that every
                         Send must re-carry.
  4. final_formatter   — pure Python. Merge, DB write, notes UPDATE. Zero LLM.

NEW product specs:
  • every answer choice < 400 chars (no such limit existed);
  • ≤10% recall / ≥90% reasoning-application-analysis — EXCEPT when the user
    explicitly selects quiz_mode="recall", where their choice wins.

Regenerate semantics changed: quiz rows use random uuid4 with a no-op
ON CONFLICT, so re-running a note used to APPEND duplicates. The formatter now
deletes this note's prior rows inside the same transaction.

Legacy pipeline remains in nodes.py; graph.py selects via QUIZ_COMPACT
(default true).
"""

import json
import logging
import math
import os
import random
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

from .constants import DISTRACTOR_TYPES, MC_QUESTION_TYPES, QUIZ_MODES
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

MAX_RESEARCH_TURNS = int(os.getenv("QUIZ_RESEARCH_MAX_TURNS", "5"))
BATCH_GEN_MAX_TURNS = int(os.getenv("QUIZ_GEN_MAX_TURNS", "2"))
HARD_OUTPUT_TOKENS = int(os.getenv("QUIZ_HARD_OUTPUT_TOKENS", "14000"))
DOSSIER_TARGET_WORDS = int(os.getenv("QUIZ_DOSSIER_TARGET_WORDS", "800"))

# Hard product spec: each answer choice must stay under this.
ANSWER_CHOICE_MAX_CHARS = int(os.getenv("QUIZ_ANSWER_CHOICE_MAX_CHARS", "400"))

GENERATOR_MAX_CHUNKS = 14
GENERATOR_CHUNK_CHAR_CAP = 1200

CHOICE_LETTERS = ["A", "B", "C", "D"]

# Recall-flavoured question types — capped at 10% unless the user asked for
# quiz_mode="recall" explicitly.
_RECALL_TYPES = {
    "PROCEDURAL_POSTURE", "CASE_FACTS", "LEGALLY_RELEVANT_FACTS",
    "ISSUE_PRECISION", "HOLDING_PRECISION", "RULE_EXTRACTION",
}


def _fetch_worker_model_local():
    from .worker_config import _fetch_worker_model
    return _fetch_worker_model


# ─────────────────────────────────────────────────────────────────────────────
# 1. plan_agent
# ─────────────────────────────────────────────────────────────────────────────

async def plan_agent(state: AgentState) -> Dict:
    """Count math + allocation strategy. Absorbs head_orchestrator."""
    from agents.tools.base import make_tools
    from agents.tools.registry import QUIZ_PLANNER_TOOLS

    logger.info("🔀 [%s] PARALLEL fire → plan_agent (alongside research_agent)",
                (state.get("job_id") or "")[:8] or "no-job")

    num_questions = max(1, state.get("num_questions") or 10)
    batch_size = min(max(state.get("batch_size") or 5, 1), 10)
    num_batches = math.ceil(num_questions / batch_size)
    quiz_mode = state.get("quiz_mode") or "mixed"
    if quiz_mode not in QUIZ_MODES:
        quiz_mode = "mixed"

    _node_start("plan_agent", state, num_questions=num_questions,
                batch_size=batch_size, num_batches=num_batches, mode=quiz_mode)

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=QUIZ_PLANNER_TOOLS,
    )
    list_tool = next((t for t in tools if t.name == "list_sources"), None)
    sources_json = "[]"
    if list_tool:
        try:
            sources_json = await list_tool.ainvoke({})
        except Exception as exc:
            _node_warn("plan_agent", state, f"list_sources failed: {exc}")

    system = (
        "You are a T-14 law professor designing a rigorous multiple-choice "
        "quiz. Plan the quiz — do NOT write questions.\n\n"
        "Select quiz_mode-appropriate emphasis: recall tests element "
        "definitions; application tests rule-to-fact fit; exam-style tests "
        "nuanced distinctions and competing doctrines. Flag any documents with "
        "dissents, circuit splits, or evolving standards — these are high-yield "
        "distractor sources.\n\n"
        "Return JSON with keys: job_type, course_context, "
        "priority_topics (list), high_yield_notes (one sentence on the best "
        "distractor sources in this corpus)."
    )
    raw = await _llm(
        "worker_mid",
        f"Sources available:\n{sources_json}\n\n"
        f"User request: {state['request']}\n"
        f"Questions: {num_questions} | mode: {quiz_mode} | "
        f"difficulty: {state.get('target_difficulty', 'application')}",
        system=system, max_tokens=512, _node="plan_agent",
    )
    try:
        job_plan = _parse_json(raw)
    except Exception:
        job_plan = {"job_type": "quiz", "course_context": "",
                    "priority_topics": [], "high_yield_notes": ""}

    _node_done("plan_agent", state, topics=job_plan.get("priority_topics", [])[:4])
    return {
        "job_plan": job_plan,
        "num_questions": num_questions,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "quiz_mode": quiz_mode,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. research_agent
# ─────────────────────────────────────────────────────────────────────────────

def _research_system(num_questions: int, quiz_mode: str, difficulty: str) -> str:
    """Absorbs source_profiler, case_rule_extractor, cross-doc synthesis, blueprint."""
    if quiz_mode == "recall":
        mix_rule = (
            "The user explicitly selected RECALL mode, so a recall-heavy mix is "
            "correct here: favour rule/element/holding recall types, with some "
            "application for variety."
        )
    else:
        mix_rule = (
            "PEDAGOGY MIX (hard requirement): AT MOST 10% of specs may be "
            "recall/holding-identification. AT LEAST 90% must test reasoning, "
            "application, or higher-order analysis — fact-to-rule application, "
            "mini-hypo outcomes, exception spotting, rule boundaries, "
            "counterarguments, compare/distinguish, policy, exam transfer. A "
            "student who merely memorised the cases should NOT score well."
        )
    return (
        "You are a legal research agent building the material for a T-14 "
        "multiple-choice quiz. Work in bounded steps:\n"
        "  1. Orient: the CORPUS SURVEY is ALREADY PROVIDED in the first "
        "message. Do NOT re-fetch it with list_sources / get_doc_outline / "
        "get_doc_metadata / find_docs_about / find_sections_about.\n"
        "  2. Retrieve (START HERE on turn 1): batch MANY parallel, "
        "case-name-anchored searches to pull each case's facts, rule, holding, "
        "dicta and dissent.\n"
        "  3. Analyse and blueprint: extract what distractors are built from, "
        "then allocate the question specs.\n\n"
        "WHAT TO EXTRACT PER CASE (keep it TERSE — the question writers grep "
        "the full corpus for anything they need beyond this):\n"
        "• legally_relevant_facts — AT MOST 4 facts that drove the outcome. "
        "**These feed distractor construction.**\n"
        "• rule — ONE sentence: the operative rule as a standalone statement, "
        "usable in a future case WITHOUT referring back to this case by name.\n"
        "• holding — ONE sentence, what was necessary to the result. (Do not "
        "list dicta here; note dicta-vs-holding confusions under common_traps "
        "instead, which is where distractors are built from.)\n"
        "• dissent — ONE sentence, the dissent's core position, if any. **A "
        "high-yield distractor source** (dissent logic presented as the "
        "majority's).\n\n"
        "CROSS-CUTTING ANALYSIS (drives the wrong answers):\n"
        "• confusable_concepts — pairs students routinely mix up, and WHY.\n"
        "• common_traps — overbroad readings, mis-stated rules, scope errors. "
        "**These map directly to distractor answer choices.**\n"
        "• high_yield_areas — most heavily tested doctrinal areas.\n\n"
        f"BLUEPRINT — allocate EXACTLY {num_questions} question specs.\n"
        f"{mix_rule}\n"
        "ALLOCATION RULES:\n"
        "• No more than 3 specs share the same question_type.\n"
        "• Every identified case must appear in at least 1 question.\n"
        "• Assign distractor_types matching that topic's common_traps (e.g. "
        "OVERBROAD_RULE where rule-scope confusion is the trap; "
        "DISSENT_AS_MAJORITY where a dissent exists).\n"
        f"• question_type must come from: {MC_QUESTION_TYPES}\n"
        f"• distractor_types must come from: {DISTRACTOR_TYPES}\n\n"
        "RULES:\n"
        "• Your turn budget is for RETRIEVAL. Turn 1 should already be a large "
        "batch of searches.\n"
        "• Only assert what you retrieved — every spec carries chunk_ids.\n"
        "• STOP CRITERION: once every case is extracted and you can allocate "
        f"{num_questions} distinct specs, STOP calling tools and emit.\n\n"
        f"FINAL ANSWER — return ONLY this JSON with EXACTLY {num_questions} specs:\n"
        "{\n"
        '  "case_extracts": [{"case_name", "rule" (1 sentence),\n'
        '      "holding" (1 sentence), "dissent" (1 sentence or ""),\n'
        '      "legally_relevant_facts": [at most 4 strings]}],\n'
        '  "common_traps": [str], "confusable_concepts": [str],\n'
        '  "specs": [{\n'
        '     "spec_index": 0-based int,\n'
        '     "question_type": one of the MC types,\n'
        '     "topic": str, "case_names": [str],\n'
        '     "difficulty": "recall"|"application"|"analysis",\n'
        '     "distractor_types": [exactly 3 tags for the wrong answers],\n'
        '     "evidence_chunk_ids": [8-12 chunk_ids — or every relevant chunk\n'
        "        if the corpus holds fewer; never pad with ids you did not\n"
        "        actually retrieve]\n"
        "  }]\n"
        "}"
        + word_budget_line(DOSSIER_TARGET_WORDS,
                           int(DOSSIER_TARGET_WORDS * 1.5), kind="dossier")
    )


async def research_agent(state: AgentState) -> Dict:
    from agents.tools.base import make_tools
    from agents.tools.registry import QUIZ_RESEARCH_TOOLS

    logger.info("🔀 [%s] PARALLEL fire → research_agent (alongside plan_agent)",
                (state.get("job_id") or "")[:8] or "no-job")

    num_questions = max(1, state.get("num_questions") or 10)
    quiz_mode = state.get("quiz_mode") or "mixed"
    difficulty = state.get("target_difficulty") or "application"
    _node_start("research_agent", state,
                max_turns=MAX_RESEARCH_TURNS, num_questions=num_questions)

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=QUIZ_RESEARCH_TOOLS,
    )
    base_model, sem, rebuild = await build_tool_model(
        _fetch_worker_model_local(), "orchestrator",
        provider=os.getenv("QUIZ_RESEARCH_PROVIDER"),
        hard_cap=HARD_OUTPUT_TOKENS,
    )

    survey, n_chunks = await prefetch_corpus_survey(state, tools)
    logger.info("  📚 [research_agent] corpus survey pre-fetched (%d chars) n_chunks=%d",
                len(survey), n_chunks)

    messages: List[Any] = [
        SystemMessage(content=_research_system(num_questions, quiz_mode, difficulty)),
        HumanMessage(content=(
            f"{survey}\n\n"
            f"Source ids in scope: {state.get('source_ids', [])}\n"
            f"User request: {state['request']}\n"
            f"Questions: {num_questions} | mode: {quiz_mode} | difficulty: {difficulty}\n\n"
            "You already have the corpus survey — do not re-fetch it. Begin "
            "RETRIEVAL immediately with a large batch of parallel searches."
        )),
    ]

    evidence_store: Dict[str, Dict] = {}
    raw = await run_bounded_tool_loop(
        base_model, tools, messages, MAX_RESEARCH_TURNS, sem,
        "research_agent", state, evidence_store=evidence_store,
        countdown_from_turn=3, rebuild=rebuild,
        forced_stop_prompt=(
            f"Retrieval is complete. Emit the blueprint JSON now with exactly "
            f"{num_questions} specs. Return ONLY the JSON object."
        ),
    )
    try:
        dossier: Optional[Dict[str, Any]] = _parse_json(raw)
    except Exception as exc:
        _node_warn("research_agent", state, f"dossier JSON parse failed: {exc}")
        dossier = None
    if not isinstance(dossier, dict):
        dossier = {}

    specs = [s for s in (dossier.get("specs") or []) if isinstance(s, dict)][:num_questions]
    for i, s in enumerate(specs):
        s["spec_index"] = i
        if s.get("question_type") not in MC_QUESTION_TYPES:
            s["question_type"] = "FACT_TO_RULE_APPLICATION"
        dts = [d for d in (s.get("distractor_types") or []) if d in DISTRACTOR_TYPES]
        while len(dts) < 3:
            dts.append("OVERBROAD_RULE")
        s["distractor_types"] = dts[:3]
    dossier["specs"] = specs
    dossier.setdefault("case_extracts", [])
    dossier.setdefault("common_traps", [])
    dossier.setdefault("confusable_concepts", [])

    # Pedagogy audit — advisory log, not a hard gate (the mix lives in prompts).
    if quiz_mode != "recall" and specs:
        n_recall = sum(1 for s in specs if s["question_type"] in _RECALL_TYPES)
        if n_recall > max(1, round(len(specs) * 0.10)):
            _node_warn("research_agent", state,
                       f"{n_recall}/{len(specs)} specs are recall-type — above "
                       "the 10% target for non-recall modes")

    await _try_save_artifact(
        state, "quiz_dossier",
        {"dossier": dossier, "evidence_chunks": len(evidence_store)},
        "orchestrator", "research_agent", "quiz_blueprint",
        source_ids=state.get("source_ids"),
    )
    _node_done("research_agent", state,
               n_specs=len(specs), n_evidence_chunks=len(evidence_store),
               n_traps=len(dossier["common_traps"]))
    return {"quiz_dossier": dossier, "evidence_store": evidence_store}


# ─────────────────────────────────────────────────────────────────────────────
# sync_barrier
# ─────────────────────────────────────────────────────────────────────────────

async def sync_barrier(state: AgentState) -> Dict:
    job = (state.get("job_id") or "")[:8] or "no-job"
    dossier = state.get("quiz_dossier") or {}
    logger.info(
        "🔗 [%s] sync_barrier — joined plan_agent + research_agent  "
        "job_plan=%s n_specs=%d",
        job, "present" if state.get("job_plan") else "MISSING (join failed?)",
        len(dossier.get("specs") or []),
    )
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 3. question_batch_generator — [Send × n_batches]
# ─────────────────────────────────────────────────────────────────────────────

def dossier_to_batches(state: AgentState):
    """Batch fan-out — amortises the shared trap/confusable context each Send
    must carry, versus one Send per question."""
    dossier = state.get("quiz_dossier") or {}
    specs = dossier.get("specs") or []
    if not specs:
        logger.warning("dossier_to_batches: no specs — routing to final_formatter")
        return "final_formatter"

    batch_size = min(max(state.get("batch_size") or 5, 1), 10)
    batches = [specs[i:i + batch_size] for i in range(0, len(specs), batch_size)]
    logger.info("quiz(compact) x%d node fan out for question_batch_generator "
                "(%d specs, batch_size=%d)", len(batches), len(specs), batch_size)
    return [
        Send("question_batch_generator", {"batch": {"batch_index": i, "specs": b}, **state})
        for i, b in enumerate(batches)
    ]


async def question_batch_generator(state: Dict) -> Dict:
    """
    Emit one batch of complete MCQs. Absorbs question_drafter +
    false_trap_red_herring_generator + question_evaluator + reviser + grounder.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import QUIZ_VERIFIER_TOOLS
    from langchain_core.tools import tool as _tool_decorator

    batch: Dict[str, Any] = state["batch"]
    dossier: Dict[str, Any] = state.get("quiz_dossier") or {}
    job_plan: Dict[str, Any] = state.get("job_plan") or {}
    evidence_store: Dict[str, Dict] = state.get("evidence_store") or {}
    specs: List[Dict] = batch["specs"]
    batch_idx = batch["batch_index"]

    _node_start("question_batch_generator", state,
                batch=batch_idx, n_specs=len(specs))

    chunk_ids: List[str] = []
    for s in specs:
        chunk_ids.extend(str(c) for c in (s.get("evidence_chunk_ids") or []))
    seen, evidence = set(), []
    for cid in chunk_ids:
        if cid in seen or cid not in evidence_store:
            continue
        seen.add(cid)
        evidence.append(evidence_store[cid])
        if len(evidence) >= GENERATOR_MAX_CHUNKS:
            break
    evidence_text = "\n\n".join(
        f"[{e['chunk_id']}] {e['content'][:GENERATOR_CHUNK_CHAR_CAP]}"
        for e in evidence
    ) or "(no evidence indexed — use grep_research_corpus)"

    @_tool_decorator
    def grep_research_corpus(keyword: str) -> str:
        """Case-insensitive substring search over the research corpus already
        harvested for this quiz. Free and instant — use BEFORE any live database
        tool. Returns up to 10 matching excerpts with chunk_ids."""
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
        tool_names=[t for t in QUIZ_VERIFIER_TOOLS if t == "verify_claim"],
    )
    tools = [grep_research_corpus] + verifier_tools

    system = (
        "You are a T-14 law professor writing multiple-choice quiz questions. "
        "This output is final — no editor runs after you.\n\n"
        "MCQ STEM STANDARDS:\n"
        "- Stems present a complete legal scenario or doctrinal question — never "
        "fill-in-the-blank, never 'which of the following' without context.\n"
        "- APPLICATION questions: a concrete fact pattern (2-5 sentences) ending "
        "in a specific legal question (e.g. 'Is D liable for negligence?').\n"
        "- RECALL questions: test the precise scope and limits of a rule, not "
        "just its name.\n"
        "- ANALYSIS questions: present two competing doctrines or arguments and "
        "ask which analysis is correct given the facts.\n"
        "- Stems must be self-contained. Maximum 120 words.\n\n"
        "CORRECT ANSWER STANDARDS:\n"
        "- Unambiguously correct — no 'best answer' hedging.\n"
        "- State the rule AND its application to the facts.\n"
        "- Never 'all of the above' or 'none of the above'.\n\n"
        "WRONG ANSWER (DISTRACTOR) STANDARDS:\n"
        "- Each wrong answer exploits a specific student misconception from the "
        "spec's distractor_types.\n"
        "- Distractors must be plausible to a student who partially understands "
        "the doctrine — if a zero-effort student can eliminate them, they fail.\n"
        "- Parallel in structure and length to the correct answer.\n\n"
        f"**LENGTH — HARD LIMIT: every answer choice must be UNDER "
        f"{ANSWER_CHOICE_MAX_CHARS} characters.** Tighten the wording rather "
        "than truncating; an over-length choice is a defect.\n\n"
        "FEEDBACK (one per choice):\n"
        "- Wrong answers: (a) name the misconception — 'A student who selects "
        "this likely believes [X]'; (b) explain the precise error — which "
        "element is wrong, which case is misapplied, how the rule scope was "
        "mis-stated; (c) correct it in one direct sentence.\n"
        "- Correct answer: (a) why it is right; (b) the case or doctrine it "
        "derives from; (c) any limiting condition or scope restriction (the "
        "'unless/but'). Use distractor_type 'correct_answer'.\n\n"
        "HINT: one sentence pointing to the key legal concept without naming "
        "the answer.\n\n"
        "SELF-CHECK before emitting (there is no evaluator downstream): every "
        "question must have exactly ONE defensible correct answer; no answer-key "
        "errors; no contradiction of the source material; no two questions in "
        "this batch testing the same point the same way. Fix anything that "
        "fails.\n\n"
        "GROUNDING: use only the evidence provided. If something is missing, "
        "call grep_research_corpus first (free, instant); use verify_claim at "
        "most once per batch when unsure a legal assertion holds. Most batches "
        "need no tool calls. Never invent authority. Cite support inline as "
        "[chunk_id] immediately after the proposition it supports, in the stem "
        "and in each feedback string — these are stripped before the student "
        "sees them, so they cost you no length.\n\n"
        "Return ONLY this JSON array — one object per spec, in spec order:\n"
        "[{\n"
        '  "spec_index": int (copy from the spec),\n'
        '  "question_type": str (copy from the spec),\n'
        '  "question_stem": str (≤120 words),\n'
        '  "hint": str,\n'
        '  "correct_answer": {"answer_text": str, "feedback": str},\n'
        '  "wrong_answers": [exactly 3 × {"answer_text", "distractor_type", "feedback"}],\n'
        '  "source_refs": [chunk_ids]\n'
        "}]"
    )

    plan_ctx = (
        f"COURSE CONTEXT: {job_plan.get('course_context', '')}\n"
        f"PRIORITY TOPICS: {json.dumps(job_plan.get('priority_topics') or [])}\n"
        f"HIGH-YIELD NOTES: {job_plan.get('high_yield_notes', '')}\n"
        f"QUIZ MODE: {state.get('quiz_mode', 'mixed')} | "
        f"DIFFICULTY: {state.get('target_difficulty', 'application')}\n\n"
    ) if job_plan else ""

    prompt = (
        f"{plan_ctx}"
        f"SPECS FOR THIS BATCH:\n{json.dumps(specs, indent=2)}\n\n"
        f"COMMON TRAPS (build distractors from these):\n"
        f"{json.dumps(dossier.get('common_traps') or [])}\n"
        f"CONFUSABLE CONCEPTS:\n{json.dumps(dossier.get('confusable_concepts') or [])}\n"
        f"CASE EXTRACTS:\n{json.dumps(dossier.get('case_extracts') or [], indent=2)[:4000]}\n\n"
        f"EVIDENCE:\n{evidence_text}"
    )

    base_model, sem, rebuild = await build_tool_model(
        _fetch_worker_model_local(), "orchestrator", hard_cap=HARD_OUTPUT_TOKENS,
    )
    messages: List[Any] = [SystemMessage(content=system), HumanMessage(content=prompt)]
    raw = await run_bounded_tool_loop(
        base_model, tools, messages, BATCH_GEN_MAX_TURNS, sem,
        "question_batch_generator", state,
        countdown_from_turn=BATCH_GEN_MAX_TURNS, rebuild=rebuild,
        forced_stop_prompt=(
            "STOP calling tools. Emit the questions JSON array NOW using only "
            "the evidence already gathered."
        ),
    )
    try:
        items = _parse_json(raw)
    except Exception as exc:
        _node_warn("question_batch_generator", state,
                   f"batch {batch_idx} JSON parse failed: {exc}")
        items = []
    if not isinstance(items, list):
        items = []

    spec_by_index = {s["spec_index"]: s for s in specs}
    drafts: List[Dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        spec = spec_by_index.get(it.get("spec_index"), specs[0] if specs else {})
        correct = it.get("correct_answer") or {}
        wrongs = [w for w in (it.get("wrong_answers") or []) if isinstance(w, dict)][:3]
        while len(wrongs) < 3:
            wrongs.append({
                "answer_text": "This answer misstates the governing rule.",
                "distractor_type": "OVERBROAD_RULE",
                "feedback": "This choice does not reflect the rule as stated in the source material.",
            })

        # Assemble A-D with the correct answer at a random position (matching
        # the legacy behaviour — the model must not learn a fixed slot).
        correct_pos = random.randint(0, 3)
        answers: List[Dict] = []
        wi = 0
        for pos in range(4):
            if pos == correct_pos:
                answers.append({
                    "answer_text": str(correct.get("answer_text", ""))[:ANSWER_CHOICE_MAX_CHARS],
                    "is_correct": True,
                    "distractor_type": "correct_answer",
                    "feedback": correct.get("feedback", ""),
                })
            else:
                w = wrongs[wi]; wi += 1
                dt = w.get("distractor_type")
                answers.append({
                    "answer_text": str(w.get("answer_text", ""))[:ANSWER_CHOICE_MAX_CHARS],
                    "is_correct": False,
                    "distractor_type": dt if dt in DISTRACTOR_TYPES else None,
                    "feedback": w.get("feedback", ""),
                })

        drafts.append({
            "spec_index": it.get("spec_index", spec.get("spec_index", 0)),
            "question_type": it.get("question_type") or spec.get("question_type", ""),
            "question_stem": it.get("question_stem", ""),
            "hint": it.get("hint", ""),
            "answers": answers,
            "source_refs": it.get("source_refs") or [],
        })

    _node_done("question_batch_generator", state,
               batch=batch_idx, n_questions=len(drafts))
    return {"generated_questions": drafts}


question_batch_generator.default_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 4. final_formatter — deterministic, zero LLM
# ─────────────────────────────────────────────────────────────────────────────

async def final_formatter(state: AgentState) -> Dict:
    """
    Merge batches, write the DB, update the note. NO LLM call.

    Replace-on-regenerate: quiz rows are random uuid4 with a no-op ON CONFLICT,
    so a re-run previously appended a duplicate question set. This deletes the
    note's prior rows inside the same transaction.
    """
    questions: List[Dict] = list(state.get("generated_questions") or [])
    job_id = state.get("job_id") or ""
    user_id = state.get("user_id") or ""
    num_requested = state.get("num_questions") or len(questions)

    _node_start("final_formatter", state, n_questions=len(questions))
    questions.sort(key=lambda q: q.get("spec_index", 99))

    # Strip internal chunk-uuid citations from everything user/DB-facing.
    citations_removed = 0
    over_length = 0
    for q in questions:
        cleaned, n = strip_chunk_citations(q.get("question_stem", ""))
        q["question_stem"] = cleaned
        citations_removed += n
        cleaned, n = strip_chunk_citations(q.get("hint", ""))
        q["hint"] = cleaned
        citations_removed += n
        for a in q.get("answers") or []:
            for key in ("answer_text", "feedback"):
                cleaned, n = strip_chunk_citations(a.get(key, ""))
                a[key] = cleaned
                citations_removed += n
            if len(a.get("answer_text", "")) >= ANSWER_CHOICE_MAX_CHARS:
                over_length += 1
    if citations_removed:
        logger.info("  🧹 [final_formatter] stripped %d inline chunk citation(s)",
                    citations_removed)
    if over_length:
        _node_warn("final_formatter", state,
                   f"{over_length} answer choice(s) hit the "
                   f"{ANSWER_CHOICE_MAX_CHARS}-char ceiling")

    persisted_ids: List[str] = []
    if not job_id:
        _node_warn("final_formatter", state, "job_id empty — skipping DB writes")
    else:
        try:
            from tasks.database import get_db_connection

            now = datetime.now(timezone.utc)
            async with get_db_connection() as conn:
                async with conn.transaction():
                    # Replace-on-regenerate (see docstring).
                    await conn.execute(
                        """
                        DELETE FROM quiz_answers
                        WHERE question_id IN (
                            SELECT id FROM quiz_questions WHERE quiz_id = $1
                        )
                        """,
                        job_id,
                    )
                    await conn.execute(
                        "DELETE FROM quiz_questions WHERE quiz_id = $1", job_id
                    )

                    for q in questions:
                        if not q.get("question_stem"):
                            continue
                        q_id = str(_uuid_mod.uuid4())
                        await conn.execute(
                            """
                            INSERT INTO quiz_questions
                              (id, quiz_id, user_id, question_text, question_type, hint, created_at)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (id) DO NOTHING
                            """,
                            q_id, job_id, user_id or None,
                            q["question_stem"], q.get("question_type", ""),
                            q.get("hint", ""), now,
                        )
                        for ans in q.get("answers") or []:
                            await conn.execute(
                                """
                                INSERT INTO quiz_answers
                                  (id, question_id, is_correct, feedback,
                                   answer_choice_text, distractor_type, created_at)
                                VALUES ($1, $2, $3, $4, $5, $6, $7)
                                ON CONFLICT (id) DO NOTHING
                                """,
                                str(_uuid_mod.uuid4()), q_id,
                                ans["is_correct"], ans.get("feedback", ""),
                                ans.get("answer_text", ""),
                                ans.get("distractor_type") or None, now,
                            )
                        persisted_ids.append(q_id)
        except Exception as exc:
            logger.error("final_formatter DB write failed: %s", exc)
            await _try_save_artifact(
                state, "quiz_write_error", {"error": str(exc)},
                "tool_only", "final_formatter", "error",
                source_ids=state.get("source_ids"),
            )

    # ── Markdown + type coverage ──────────────────────────────────────────
    type_coverage: Dict[str, int] = {}
    for q in questions:
        t = q.get("question_type", "")
        type_coverage[t] = type_coverage.get(t, 0) + 1

    lines: List[str] = [
        "# Quiz", "",
        f"**Questions:** {len(questions)} (requested {num_requested})  ",
        f"**Mode:** {state.get('quiz_mode', 'mixed')}  ",
        "", "---", "",
    ]
    for i, q in enumerate(questions, start=1):
        lines += [f"### {i}. {q.get('question_stem', '')}", ""]
        for letter, a in zip(CHOICE_LETTERS, q.get("answers") or []):
            mark = " ✅" if a.get("is_correct") else ""
            lines.append(f"- **{letter}.** {a.get('answer_text', '')}{mark}")
        if q.get("hint"):
            lines += ["", f"*Hint: {q['hint']}*"]
        lines += ["", "---", ""]
    lines.append(
        f"<!-- quiz compact | {len(questions)} questions | "
        f"{len(persisted_ids)} persisted | types: {len(type_coverage)} -->"
    )
    markdown = "\n".join(lines)

    # Notes UPDATE is owned by this agent (quiz's dispatcher does not update).
    if job_id:
        try:
            from tasks.database import get_db_connection

            metadata = {
                "quiz_mode": state.get("quiz_mode", "mixed"),
                "question_count": len(persisted_ids),
                "question_type_coverage": type_coverage,
            }
            async with get_db_connection() as conn:
                await conn.execute(
                    """
                    UPDATE notes
                    SET content_markdown     = $1,
                        metadata             = $2::jsonb,
                        num_questions        = $3,
                        note_progress_status = 'COMPLETE'
                    WHERE id = $4
                    """,
                    markdown, json.dumps(metadata), len(persisted_ids), job_id,
                )
        except Exception as exc:
            logger.error("final_formatter notes UPDATE failed: %s", exc)

    _node_done("final_formatter", state,
               questions=len(questions), persisted=len(persisted_ids))
    return {"persisted_question_ids": persisted_ids, "final_output": markdown}
