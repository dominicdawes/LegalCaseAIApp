# agents/cold_call/compact_nodes.py
"""
Compact 4-stage cold-call pipeline.

Replaces the 14-node graph's ~47 LLM round-trips (22 `_llm` + ~25 `verify_claim`)
with four stages:

  1. plan_agent        — one worker_mid call → coverage/difficulty calibration.
                         Runs in PARALLEL with research_agent.
  2. research_agent    — ONE bounded tool-calling loop. Absorbs source_profiler
                         + corpus_synthesizer + case_rule_extractor (×cases) +
                         doctrine_mapper + compare_distinguish_mapper. Emits a
                         LEAN case dossier; every retrieval result is harvested
                         into state["evidence_store"].
     sync_barrier      — no-op join so generators see plan AND dossier.
  3. sequence_generator — [Send × requested_sequence_count]. Absorbs
                         cold_call_seed_generator + seed_diversity_agent +
                         socratic_thread_builder + socratic_answer_agent +
                         grounder_agent. Each Send emits one complete
                         5-question (1A-1E) Socratic sequence with answers.
  4. final_formatter   — pure Python. Difficulty stamping, coverage map,
                         markdown, and the 3-table DB export. Zero LLM.

Key design change — DIVERSITY BY CONSTRUCTION:
the legacy pipeline generated 3× the needed seeds (SEEDS_PER_CASE_MULTIPLIER)
and then paid an LLM call to prune them for diversity, with a retry gate on top.
Here the fan-out assigns each sequence a distinct theme (round-robin over
SEED_THEMES) and case deterministically at Send time, which guarantees coverage,
removes ~4 LLM calls, and removes the retry path entirely.

Grounding: the legacy `grounder_agent` fired one `verify_claim` per answer —
25 serial gemini calls for a 5-sequence run. Generators now self-ground against
the harvested evidence with a budgeted `verify_claim` available as a tool.

Legacy 14-node pipeline remains in nodes.py; graph.py selects via
COLD_CALL_COMPACT (default true).
"""

import json
import logging
import os
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

from .constants import (
    QUESTION_TYPES_EARLY,
    QUESTION_TYPES_MIDDLE,
    QUESTION_TYPES_DEEP,
    QUESTION_TYPES_CONDITIONAL_MULTI_CASE_DEEP,
    QUESTION_TYPES_CONDITIONAL_DISSENT_MIDDLE,
    QUESTION_TYPES_CONDITIONAL_ADVANCED_DEEP,
    QUESTION_TYPES_CONDITIONAL_ADVANCED_MIDDLE,
    QUESTION_TYPES_OPTIONAL_MULTI_CASE,
    QUESTION_TYPES_OPTIONAL_UPPER_LEVEL,
    SEED_THEMES,
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

MAX_RESEARCH_TURNS = int(os.getenv("COLD_CALL_RESEARCH_MAX_TURNS", "5"))
SEQUENCE_GEN_MAX_TURNS = int(os.getenv("COLD_CALL_GEN_MAX_TURNS", "2"))
HARD_OUTPUT_TOKENS = int(os.getenv("COLD_CALL_HARD_OUTPUT_TOKENS", "14000"))

DOSSIER_TARGET_WORDS = int(os.getenv("COLD_CALL_DOSSIER_TARGET_WORDS", "900"))
# Each Q/A is "slightly larger than a flashcard": 5 questions + 5 four-part
# answers lands around 700 words.
SEQUENCE_TARGET_WORDS = int(os.getenv("COLD_CALL_SEQUENCE_TARGET_WORDS", "700"))

GENERATOR_MAX_CHUNKS = 12
GENERATOR_CHUNK_CHAR_CAP = 1400

# 1A-1E depth map (ported verbatim from nodes.py DEPTH_MAP).
QUESTION_LABELS = ["A", "B", "C", "D", "E"]
DEPTH_MAP = {"A": "early", "B": "early", "C": "middle", "D": "deep", "E": "deep"}

# Deterministic difficulty stamping — replaces the legacy critic's LLM pass.
_BASE_DIFFICULTY = {"early": "easy", "middle": "medium", "deep": "hard"}
_BUMP_TYPES = {
    "FACT_CHANGE_HYPO", "RULE_BOUNDARY", "COMPARE_DISTINGUISH", "POLICY_ANALYSIS",
}
_BUMP_NEXT = {"easy": "medium", "medium": "hard", "hard": "hard"}

# Coverage must-haves (ported from critic_coverage_agent).
_COVERAGE_MUST_HAVE = [
    "PROCEDURAL_POSTURE", "RULE_EXTRACTION", "FACT_CHANGE_HYPO",
    "COUNTERARGUMENT", "POLICY_ANALYSIS",
]


def select_question_type_bank(
    n_cases: int, any_dissent: bool, target_difficulty: str
) -> Dict[str, List[str]]:
    """
    Deterministic question-type bank — the legacy `question_type_bank_selector`
    node (which made no LLM call) ported to a pure function, conditional
    unlocks intact.
    """
    early = list(QUESTION_TYPES_EARLY)
    middle = list(QUESTION_TYPES_MIDDLE)
    deep = list(QUESTION_TYPES_DEEP)
    optional: List[str] = []

    if n_cases > 1:
        deep += QUESTION_TYPES_CONDITIONAL_MULTI_CASE_DEEP
        optional += QUESTION_TYPES_OPTIONAL_MULTI_CASE
    if any_dissent:
        middle += QUESTION_TYPES_CONDITIONAL_DISSENT_MIDDLE
    if target_difficulty == "advanced":
        deep += QUESTION_TYPES_CONDITIONAL_ADVANCED_DEEP
        middle += QUESTION_TYPES_CONDITIONAL_ADVANCED_MIDDLE
    if target_difficulty != "law_1l":
        optional += QUESTION_TYPES_OPTIONAL_UPPER_LEVEL

    return {"early": early, "middle": middle, "deep": deep, "optional": optional}


# ─────────────────────────────────────────────────────────────────────────────
# 1. plan_agent
# ─────────────────────────────────────────────────────────────────────────────

async def plan_agent(state: AgentState) -> Dict:
    """Course/coverage calibration — one fast call. Absorbs head_orchestrator."""
    from agents.tools.base import make_tools
    from agents.tools.registry import COLD_CALL_PLANNER_TOOLS

    logger.info("🔀 [%s] PARALLEL fire → plan_agent (alongside research_agent)",
                (state.get("job_id") or "")[:8] or "no-job")
    _node_start("plan_agent", state,
                n_sources=len(state.get("source_ids") or []),
                requested=state.get("requested_sequence_count", 3))

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=COLD_CALL_PLANNER_TOOLS,
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
        "You are the orchestrator for a T-14 law-school cold-call question "
        "pipeline. Given the available sources, plan the run — do NOT write "
        "questions.\n\n"
        "Return JSON with keys: job_type, course_context (the course/doctrinal "
        "area these sources belong to), coverage_mode "
        "(single_case|multi_case|doctrine_survey), expected_cases (list of case "
        "names you expect the research agent to find), retrieval_depth "
        "(shallow|standard|deep), notes (one sentence on anything unusual)."
    )
    raw = await _llm(
        "worker_mid",
        f"Sources available:\n{sources_json}\n\n"
        f"User request: {state['request']}\n"
        f"Target difficulty: {state.get('target_difficulty', 'day_one_t14')}",
        system=system, max_tokens=800, _node="plan_agent",
    )
    try:
        job_plan = _parse_json(raw)
    except Exception as exc:
        _node_warn("plan_agent", state, f"JSON parse failed ({exc}) — minimal plan")
        job_plan = {
            "job_type": "cold_call",
            "course_context": "",
            "coverage_mode": "single_case",
            "expected_cases": [],
            "retrieval_depth": "standard",
            "notes": "",
        }

    await _try_save_artifact(
        state, "job_plan", job_plan, "worker_mid", "plan_agent",
        "job_plan", source_ids=state.get("source_ids"),
    )
    _node_done("plan_agent", state,
               mode=job_plan.get("coverage_mode", "?"),
               course=str(job_plan.get("course_context", ""))[:40])
    return {"job_plan": job_plan}


# ─────────────────────────────────────────────────────────────────────────────
# 2. research_agent
# ─────────────────────────────────────────────────────────────────────────────

def _research_system(requested: int) -> str:
    """
    Folds in the intent of five deleted nodes: source_profiler,
    corpus_synthesizer, case_rule_extractor, doctrine_mapper,
    compare_distinguish_mapper.
    """
    return (
        "You are a legal research agent building the evidence base for T-14 "
        "cold-call questioning. Work in bounded steps:\n"
        "  1. Orient: the CORPUS SURVEY is ALREADY PROVIDED in the first "
        "message. Identify every distinct CASE in the corpus from it. Do NOT "
        "re-fetch it with list_sources / get_doc_outline / get_doc_metadata / "
        "find_docs_about / find_sections_about — that wastes a whole turn.\n"
        "  2. Retrieve (START HERE on turn 1): batch MANY parallel searches, "
        "case-name-anchored, to pull each case's facts, rule, reasoning, "
        "dissent and policy discussion.\n"
        "  3. Extract & relate: build one rule object per case, then map how "
        "the cases relate to each other.\n\n"
        "WHAT EACH CASE NEEDS (this is what questions get built from):\n"
        "• legally_relevant_facts: ONLY the facts the court's rule actually "
        "hinges on — not background narrative. **These become the "
        "FACT_CHANGE_HYPO targets**, so each must be one the outcome turns on.\n"
        "• rule.elements: ALL required elements, in the order courts apply "
        "them; plus exceptions and burdens.\n"
        "• reasoning: each step a DISTINCT analytical move the court made, not "
        "a summary.\n"
        "• dicta: statements about what the rule is NOT, or about future cases "
        "— flag them, they drive HOLDING_VS_DICTA questions.\n"
        "• dissent: the dissent's core objection, if one exists. If there is no "
        "separate opinion, set has_dissent=false and leave it empty — never "
        "invent a dissent.\n"
        "• policy_concerns: name the specific policy values at stake "
        "(efficiency, fairness, administrability, notice, …).\n\n"
        "Your extraction must be thorough enough to support 5-deep Socratic "
        "questioning — from direct fact comprehension through rule-boundary "
        "testing and policy analysis.\n\n"
        "CROSS-CASE (only when 2+ cases): give each pair a relationship "
        "(DEFINES | DISTINGUISHES | EXPANDS | LIMITS | EXCEPTION_TO | "
        "ANALOGOUS_TO | CONFLICTS_WITH) and, for the strongest pairs, a "
        "compare/distinguish question with the model distinction a strong "
        "student would draw.\n\n"
        "RULES:\n"
        "• Your turn budget is for RETRIEVAL. Turn 1 should already be a large "
        "batch of searches, not discovery.\n"
        "• Only assert what you actually retrieved — cite chunk_ids.\n"
        "• STOP CRITERION: once every case has facts + rule + reasoning + "
        "policy (and dissent where one exists), STOP calling tools and emit the "
        "dossier, even if turns remain.\n\n"
        "KEEP IT LEAN — the question writers do the depth work, you supply the "
        "raw material. Per case: ≤6 legally_relevant_facts, ≤6 elements, "
        "≤5 reasoning moves, ≤3 policy_concerns. At most "
        f"{max(2, requested)} compare/distinguish prompts total.\n\n"
        "FINAL ANSWER — return ONLY this JSON object (no prose):\n"
        "{\n"
        '  "cases": [{\n'
        '     "case_id": "case_001" (sequential, stable),\n'
        '     "case_name", "procedural_posture", "issue", "holding",\n'
        '     "legally_relevant_facts": [str],\n'
        '     "rule": {"rule_statement", "elements": [str], "exceptions": [str],\n'
        '              "burdens": str},\n'
        '     "reasoning": [str], "dicta": [str],\n'
        '     "has_dissent": true|false, "dissent": str,\n'
        '     "policy_concerns": [str],\n'
        '     "evidence_chunk_ids": [10-12 chunk_ids most central to this case —\n'
        "        or every relevant chunk if the corpus holds fewer]\n"
        "  }],\n"
        '  "doctrine_edges": [{"from_case_id", "to_case_id", "rel_type"}],\n'
        '  "compare_distinguish": [{"comparison_id", "case_a_id", "case_b_id",\n'
        '     "relationship", "question", "model_distinction"}]\n'
        "}\n"
        "Cover EVERY case in the corpus."
        + word_budget_line(DOSSIER_TARGET_WORDS,
                           int(DOSSIER_TARGET_WORDS * 1.5), kind="dossier")
    )


async def research_agent(state: AgentState) -> Dict:
    """One bounded tool loop replacing five legacy research/extraction nodes."""
    from agents.tools.base import make_tools
    from agents.tools.registry import COLD_CALL_RESEARCH_TOOLS

    logger.info("🔀 [%s] PARALLEL fire → research_agent (alongside plan_agent)",
                (state.get("job_id") or "")[:8] or "no-job")
    _node_start("research_agent", state, max_turns=MAX_RESEARCH_TURNS)

    requested = int(state.get("requested_sequence_count") or 3)
    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=COLD_CALL_RESEARCH_TOOLS,
    )
    base_model, sem, rebuild = await build_tool_model(
        _fetch_worker_model_local(), "orchestrator",
        provider=os.getenv("COLD_CALL_RESEARCH_PROVIDER"),
        hard_cap=HARD_OUTPUT_TOKENS,
    )

    survey, n_chunks = await prefetch_corpus_survey(state, tools)
    logger.info("  📚 [research_agent] corpus survey pre-fetched (%d chars) n_chunks=%d",
                len(survey), n_chunks)

    messages: List[Any] = [
        SystemMessage(content=_research_system(requested)),
        HumanMessage(content=(
            f"{survey}\n\n"
            f"Source ids in scope: {state.get('source_ids', [])}\n"
            f"User request: {state['request']}\n"
            f"Sequences to be built from this: {requested}\n\n"
            "You already have the corpus survey — do not re-fetch it. Begin "
            "RETRIEVAL immediately with a large batch of parallel, "
            "case-name-anchored searches."
        )),
    ]

    evidence_store: Dict[str, Dict] = {}
    raw = await run_bounded_tool_loop(
        base_model, tools, messages, MAX_RESEARCH_TURNS, sem,
        "research_agent", state, evidence_store=evidence_store,
        countdown_from_turn=3, rebuild=rebuild,
        forced_stop_prompt=(
            "Retrieval is complete. Emit the final case dossier JSON now using "
            "the evidence gathered. Return ONLY the JSON object."
        ),
    )
    try:
        dossier: Optional[Dict[str, Any]] = _parse_json(raw)
    except Exception as exc:
        _node_warn("research_agent", state, f"dossier JSON parse failed: {exc}")
        dossier = None
    if not isinstance(dossier, dict):
        dossier = {}

    cases = [c for c in (dossier.get("cases") or []) if isinstance(c, dict)]
    # Backfill stable ids so the fan-out key is never missing.
    for i, c in enumerate(cases, start=1):
        if not c.get("case_id"):
            c["case_id"] = f"case_{i:03d}"
    dossier["cases"] = cases
    dossier.setdefault("doctrine_edges", [])
    dossier.setdefault("compare_distinguish", [])

    await _try_save_artifact(
        state, "case_dossier",
        {"dossier": dossier, "evidence_chunks": len(evidence_store)},
        "orchestrator", "research_agent", "case_dossier",
        source_ids=state.get("source_ids"),
    )
    _node_done("research_agent", state,
               n_cases=len(cases),
               n_cmp=len(dossier["compare_distinguish"]),
               n_evidence_chunks=len(evidence_store),
               cases=[str(c.get("case_name", "?"))[:24] for c in cases[:4]])
    return {"case_dossier": dossier, "evidence_store": evidence_store}


def _fetch_worker_model_local():
    """Late import so worker_config stays the agent's own tier source."""
    from .worker_config import _fetch_worker_model
    return _fetch_worker_model


# ─────────────────────────────────────────────────────────────────────────────
# sync_barrier
# ─────────────────────────────────────────────────────────────────────────────

async def sync_barrier(state: AgentState) -> Dict:
    """
    Join for the parallel plan_agent / research_agent branches. Two incoming
    static edges = LangGraph fan-in, so generators always see both. Also keeps
    conditional edges off Send-parallel nodes.
    """
    job = (state.get("job_id") or "")[:8] or "no-job"
    dossier = state.get("case_dossier") or {}
    logger.info(
        "🔗 [%s] sync_barrier — joined plan_agent + research_agent  "
        "job_plan=%s n_cases=%d",
        job, "present" if state.get("job_plan") else "MISSING (join failed?)",
        len(dossier.get("cases") or []),
    )
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 3. sequence_generator — [Send × requested_sequence_count]
# ─────────────────────────────────────────────────────────────────────────────

def dossier_to_generators(state: AgentState):
    """
    Fan-out with DIVERSITY BY CONSTRUCTION.

    The legacy path generated 3× the needed seeds and paid an LLM call to prune
    them for diversity (plus a retry gate). Here each Send is assigned a
    distinct theme (round-robin over SEED_THEMES) and a case (round-robin over
    the dossier) deterministically, which guarantees coverage for free.

    sequence_index is assigned HERE, where the full ordered list exists —
    parallel workers cannot derive it from an accumulating state list.
    """
    dossier = state.get("case_dossier") or {}
    cases = dossier.get("cases") or []
    if not cases:
        logger.warning("dossier_to_generators: no cases — routing to final_formatter")
        return "final_formatter"

    requested = max(1, int(state.get("requested_sequence_count") or 3))
    difficulty = state.get("target_difficulty", "day_one_t14")
    any_dissent = any(bool(c.get("has_dissent")) for c in cases)
    type_bank = select_question_type_bank(len(cases), any_dissent, difficulty)
    cmp_prompts = dossier.get("compare_distinguish") or []

    sends = []
    for idx in range(1, requested + 1):
        case = cases[(idx - 1) % len(cases)]
        theme = SEED_THEMES[(idx - 1) % len(SEED_THEMES)]
        # A COMPARE_DISTINGUISH slot only makes sense with a real pair.
        cmp_prompt = None
        if theme == "COMPARE_DISTINGUISH":
            if cmp_prompts:
                cmp_prompt = cmp_prompts[(idx - 1) % len(cmp_prompts)]
            else:
                theme = "CORE_UNDERSTANDING"
        sends.append(Send("sequence_generator", {
            "assignment": {
                "sequence_index": idx,
                "theme": theme,
                "case_id": case.get("case_id"),
                "type_bank": type_bank,
                "cmp_prompt": cmp_prompt,
            },
            **state,
        }))

    logger.info("cold_call(compact) x%d node fan out for sequence_generator "
                "(themes=%s)", len(sends),
                [s.arg["assignment"]["theme"] for s in sends][:8])
    return sends


async def sequence_generator(state: Dict) -> Dict:
    """
    Emit ONE complete Socratic sequence: the 1A-1E question thread plus its
    five answers, self-grounded. Absorbs seed generation, thread building,
    answer writing, and per-answer grounding.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import COLD_CALL_VERIFIER_TOOLS
    from langchain_core.tools import tool as _tool_decorator

    assignment: Dict[str, Any] = state["assignment"]
    dossier: Dict[str, Any] = state.get("case_dossier") or {}
    evidence_store: Dict[str, Dict] = state.get("evidence_store") or {}
    seq_idx = assignment["sequence_index"]
    theme = assignment["theme"]

    case = next(
        (c for c in (dossier.get("cases") or [])
         if c.get("case_id") == assignment.get("case_id")),
        (dossier.get("cases") or [{}])[0],
    )
    case_id = case.get("case_id", "case_001")
    case_name = case.get("case_name", "")

    _node_start("sequence_generator", state,
                seq=seq_idx, theme=theme, case=case_name[:32])

    # Evidence for this case.
    evidence: List[Dict] = []
    for cid in (case.get("evidence_chunk_ids") or [])[:GENERATOR_MAX_CHUNKS]:
        e = evidence_store.get(str(cid))
        if e:
            evidence.append(e)
    evidence_text = "\n\n".join(
        f"[{e['chunk_id']}] (source {e.get('source_id', '?')[:8]}, p.{e.get('page', '?')})\n"
        f"{e['content'][:GENERATOR_CHUNK_CHAR_CAP]}"
        for e in evidence
    ) or "(no evidence indexed for this case — use grep_research_corpus)"

    @_tool_decorator
    def grep_research_corpus(keyword: str) -> str:
        """Case-insensitive substring search over the research corpus already
        harvested for this run. Free and instant — use this BEFORE any live
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
        tool_names=[t for t in COLD_CALL_VERIFIER_TOOLS
                    if t in ("verify_claim", "find_supporting_evidence")],
    )
    tools = [grep_research_corpus] + verifier_tools

    bank = assignment.get("type_bank") or {}
    cmp_prompt = assignment.get("cmp_prompt")
    cmp_ctx = ""
    if cmp_prompt:
        cmp_ctx = (
            f"\nCOMPARE/DISTINGUISH CONTEXT — build the thread around this:\n"
            f"  question: {cmp_prompt.get('question', '')}\n"
            f"  model distinction: {cmp_prompt.get('model_distinction', '')}\n"
        )

    system = (
        "You are a T-14 law professor building ONE Socratic cold-call sequence "
        "and its model answers. The sequence must escalate methodically — each "
        "question harder than the last, each building on what a strong student "
        "would have just said.\n\n"
        "**T-14 DEPTH STRUCTURE — 5 questions (A through E):**\n"
        "  A: Direct comprehension — 'What are the legally relevant facts?' "
        "(Ask for facts only; do not ask for a conclusion)\n"
        "  B: Reasoning chain — 'Why did the court reach that conclusion?' "
        "(The court's analytical steps, not just the holding)\n"
        "  C: Fact-change hypothetical — 'What if [specific key fact] had been "
        "different — would the result change?' (Name the exact fact changed)\n"
        "  D: Rule boundary — 'Where does the rule stop? Give me a case where "
        "it wouldn't apply.'\n"
        "  E: Counterargument or policy/exam application — 'What is the best "
        "argument for the losing party?' or 'What policy value does this rule "
        "serve?' or 'How would you argue this issue on an exam?'\n\n"
        "**T-14 QUESTION STANDARDS:**\n"
        "• Each question must be phrased as a professor would ask it aloud in "
        "class\n"
        "• Each question must reference specific case facts — never generic\n"
        "• Each question must logically follow from the prior (do not reset "
        "context)\n"
        "• expected_answer_shape: what an A student would say, 2-3 sentences\n\n"
        "**ANSWER FORMAT (Because / Unless / But / Therefore):**\n"
        "  model_answer: 'The [party] probably [wins/loses] on [issue]. "
        "Because [precise rule application to specific facts]. Unless [key "
        "exception or competing fact]. But [the most important limiting "
        "doctrine or counterweight]. Therefore [one-sentence conclusion with "
        "confidence level].'\n\n"
        "**T-14 ANSWER STANDARDS:**\n"
        "• model_answer: cite the specific rule elements and apply them to the "
        "named facts; never generic ('the rule applies here')\n"
        "• strong_answer: the tighter, more precise version an exceptional "
        "student gives — one or two sentences beyond the model answer\n"
        "• common_weak_answer: the surface answer an unprepared student gives "
        "— usually the right conclusion, missing the rule mechanics\n"
        "• professor_follow_up_trap: the NEXT question a professor asks after a "
        "strong answer, to probe deeper (e.g. 'What if the plaintiff had been "
        "warned?')\n"
        "• recovery_phrase: a rule-focused sentence a student can say if they "
        "blank — it must move the analysis forward, not stall ('I need a "
        "moment' fails)\n\n"
        f"THIS SEQUENCE'S ANGLE: {theme}. Open question A from that angle.\n"
        f"Question types available — early: {bank.get('early', [])}; "
        f"middle: {bank.get('middle', [])}; deep: {bank.get('deep', [])}.\n"
        "Tag each question with the type that fits it best.\n\n"
        "GROUNDING: every rule, holding and fact must come from the evidence "
        "provided. If something you want to use is not there, call "
        "grep_research_corpus first (free, instant); use verify_claim sparingly "
        "(at most twice) when unsure a legal assertion is supported. Most "
        "sequences need no tool calls. Never invent a holding, a citation, or a "
        "dissent. Cite support inline as [chunk_id] — these are stripped before "
        "the student sees them, so they cost you no length.\n\n"
        "Return ONLY this JSON object:\n"
        "{\n"
        '  "sequence_theme": str,\n'
        '  "questions": [{"question_id": "1A".."1E", "question_index": 1-5,\n'
        '     "question_type": str, "depth_position": "early|middle|deep",\n'
        '     "question_text": str, "target_skill": str,\n'
        '     "expected_answer_shape": str, "source_refs": [chunk_ids]}],\n'
        '  "answers": [{"question_id": "1A".."1E", "model_answer",\n'
        '     "strong_answer", "common_weak_answer",\n'
        '     "professor_follow_up_trap", "recovery_phrase"}],\n'
        '  "coverage_tags": [question_type strings used]\n'
        "}\n"
        "Exactly 5 questions and 5 answers, ids matched."
        + word_budget_line(SEQUENCE_TARGET_WORDS, int(SEQUENCE_TARGET_WORDS * 1.4))
    )

    prompt = (
        f"CASE: {json.dumps({k: v for k, v in case.items() if k != 'evidence_chunk_ids'}, indent=2)}\n"
        f"{cmp_ctx}\n"
        f"EVIDENCE:\n{evidence_text}"
    )

    base_model, sem, rebuild = await build_tool_model(
        _fetch_worker_model_local(), "orchestrator", hard_cap=HARD_OUTPUT_TOKENS,
    )
    messages: List[Any] = [SystemMessage(content=system), HumanMessage(content=prompt)]
    raw = await run_bounded_tool_loop(
        base_model, tools, messages, SEQUENCE_GEN_MAX_TURNS, sem,
        "sequence_generator", state,
        countdown_from_turn=SEQUENCE_GEN_MAX_TURNS, rebuild=rebuild,
        forced_stop_prompt=(
            "STOP calling tools. Emit the sequence JSON NOW using only the "
            "evidence already gathered."
        ),
    )
    try:
        data = _parse_json(raw)
    except Exception as exc:
        _node_warn("sequence_generator", state,
                   f"seq {seq_idx} JSON parse failed: {exc}")
        data = {}
    if not isinstance(data, dict):
        data = {}

    seq_id = f"seq_{case_id}_theme_{theme.lower()}_{seq_idx:03d}"
    questions: List[Dict] = []
    for i, q in enumerate((data.get("questions") or [])[:5]):
        if not isinstance(q, dict):
            continue
        label = QUESTION_LABELS[i] if i < len(QUESTION_LABELS) else str(i + 1)
        depth = q.get("depth_position") or DEPTH_MAP.get(label, "deep")
        qtype = q.get("question_type", "LEGALLY_RELEVANT_FACTS")
        # Deterministic difficulty — replaces the legacy critic's LLM pass.
        difficulty = _BASE_DIFFICULTY.get(depth, "medium")
        if qtype in _BUMP_TYPES:
            difficulty = _BUMP_NEXT[difficulty]
        questions.append({
            "question_id": q.get("question_id") or f"1{label}",
            "question_index": i + 1,
            "question_type": qtype,
            "depth_position": depth,
            "question_text": q.get("question_text", ""),
            "target_skill": q.get("target_skill", ""),
            "difficulty_label": difficulty,
            "expected_answer_shape": q.get("expected_answer_shape", ""),
            "source_refs": q.get("source_refs") or [],
            "metadata": {},
        })

    answers: List[Dict] = []
    for a in (data.get("answers") or [])[:5]:
        if not isinstance(a, dict):
            continue
        answers.append({
            "question_id": a.get("question_id", ""),
            "model_answer": a.get("model_answer", ""),
            "strong_answer": a.get("strong_answer", ""),
            "common_weak_answer": a.get("common_weak_answer", ""),
            "professor_follow_up_trap": a.get("professor_follow_up_trap", ""),
            "recovery_phrase": a.get("recovery_phrase", ""),
        })

    sequence = {
        "sequence_id": seq_id,
        "case_id": case_id,
        "case_name": case_name,
        "seed_id": f"theme_{theme.lower()}",
        "sequence_theme": data.get("sequence_theme") or theme,
        "sequence_index": seq_idx,
        "questions": questions,
        "coverage_tags": data.get("coverage_tags") or [q["question_type"] for q in questions],
        "source_refs": case.get("evidence_chunk_ids") or [],
        "metadata": {"theme": theme, "grounding": "self"},
        "answers": answers,
    }

    await _try_save_artifact(
        state, f"sequence:{seq_id}", sequence, "orchestrator",
        "sequence_generator", "question_sequence",
        source_ids=state.get("source_ids"),
    )
    _node_done("sequence_generator", state,
               seq=seq_idx, n_questions=len(questions), n_answers=len(answers))
    return {"generated_sequences": [sequence]}


sequence_generator.default_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 4. final_formatter — deterministic, zero LLM
# ─────────────────────────────────────────────────────────────────────────────

async def final_formatter(state: AgentState) -> Dict:
    """
    Coverage map, markdown, and the 3-table DB export. Makes NO LLM call —
    difficulty stamping happened deterministically in the generator, replacing
    the legacy critic_coverage_agent.
    """
    sequences: List[Dict] = list(state.get("generated_sequences") or [])
    note_id = state.get("note_id") or ""

    _node_start("final_formatter", state, n_sequences=len(sequences))
    sequences.sort(key=lambda s: s.get("sequence_index", 99))

    # Strip internal chunk-uuid citations from every user/DB-facing string —
    # the UI reads these tables directly, so they must be clean too.
    citations_removed = 0
    for seq in sequences:
        for q in seq.get("questions") or []:
            for key in ("question_text", "target_skill", "expected_answer_shape"):
                cleaned, n = strip_chunk_citations(q.get(key, ""))
                q[key] = cleaned
                citations_removed += n
        for a in seq.get("answers") or []:
            for key in ("model_answer", "strong_answer", "common_weak_answer",
                        "professor_follow_up_trap", "recovery_phrase"):
                cleaned, n = strip_chunk_citations(a.get(key, ""))
                a[key] = cleaned
                citations_removed += n
    if citations_removed:
        logger.info("  🧹 [final_formatter] stripped %d inline chunk citation(s)",
                    citations_removed)

    # ── Coverage map (ported from critic_coverage_agent, no LLM) ──────────
    by_type: Dict[str, int] = {}
    by_depth: Dict[str, int] = {}
    for seq in sequences:
        for q in seq.get("questions") or []:
            by_type[q["question_type"]] = by_type.get(q["question_type"], 0) + 1
            by_depth[q["depth_position"]] = by_depth.get(q["depth_position"], 0) + 1
    missing = [t for t in _COVERAGE_MUST_HAVE if t not in by_type]
    coverage_score = max(0.0, min(1.0, 1 - (len(missing) / 5) * 0.5))
    coverage_map = {
        "overall_coverage_score": coverage_score,
        "coverage_by_question_type": by_type,
        "coverage_by_depth": by_depth,
        "missing_or_weak_areas": missing,
        "difficulty_labels_added": True,
    }

    # ── DB export ─────────────────────────────────────────────────────────
    export_batch_id = str(_uuid_mod.uuid4())
    q_exported = 0
    try:
        from tasks.database import get_db_connection

        async with get_db_connection() as conn:
            for seq in sequences:
                seq_id = seq["sequence_id"]
                # Row IDs must be unique per run. seq_id is content-deterministic
                # and omits the run, so without this scoping a regenerated
                # cold-call yields identical ids and every ON CONFLICT DO NOTHING
                # silently no-ops — the note is written but no rows appear.
                db_seq_id = f"{seq_id}__{export_batch_id[:8]}"
                questions = seq.get("questions") or []
                try:
                    await conn.execute(
                        """
                        INSERT INTO cold_call_sequences (
                            id, run_id, note_id, user_id, case_id, case_name,
                            sequence_theme, sequence_index, difficulty_label,
                            question_count, coverage_tags, source_refs, metadata,
                            created_at
                        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,NOW())
                        ON CONFLICT (id) DO NOTHING
                        """,
                        db_seq_id, export_batch_id, note_id or None,
                        state.get("user_id") or None,
                        seq.get("case_id", ""), seq.get("case_name", ""),
                        seq.get("sequence_theme", ""), seq.get("sequence_index", 1),
                        (questions or [{}])[-1].get("difficulty_label", "medium"),
                        len(questions),
                        seq.get("coverage_tags") or [],
                        seq.get("source_refs") or [],
                        json.dumps(seq.get("metadata") or {}),
                    )
                except Exception as db_exc:
                    logger.warning("cold_call_sequences insert failed (non-fatal): %s", db_exc)

                for q in questions:
                    try:
                        await conn.execute(
                            """
                            INSERT INTO cold_call_questions (
                                id, sequence_id, run_id, question_label,
                                question_index, question_type, depth_position,
                                difficulty_label, question_text, target_skill,
                                expected_answer_shape, source_refs, metadata, created_at
                            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,NOW())
                            ON CONFLICT (id) DO NOTHING
                            """,
                            f"q_{db_seq_id}_{q.get('question_id', '')}",
                            db_seq_id, export_batch_id,
                            q.get("question_id", ""), q.get("question_index", 0),
                            q.get("question_type", ""), q.get("depth_position", "middle"),
                            q.get("difficulty_label", "medium"),
                            q.get("question_text", ""), q.get("target_skill", ""),
                            q.get("expected_answer_shape", ""),
                            q.get("source_refs") or [],
                            json.dumps(q.get("metadata") or {}),
                        )
                        q_exported += 1
                    except Exception as db_exc:
                        logger.warning("cold_call_questions insert failed (non-fatal): %s", db_exc)

                for ans in seq.get("answers") or []:
                    try:
                        await conn.execute(
                            """
                            INSERT INTO cold_call_answers (
                                id, sequence_id, run_id, question_id,
                                model_answer, strong_answer, common_weak_answer,
                                professor_follow_up_trap, recovery_phrase, created_at
                            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,NOW())
                            ON CONFLICT (id) DO NOTHING
                            """,
                            f"a_{db_seq_id}_{ans.get('question_id', '')}",
                            db_seq_id, export_batch_id, ans.get("question_id", ""),
                            ans.get("model_answer", ""), ans.get("strong_answer", ""),
                            ans.get("common_weak_answer", ""),
                            ans.get("professor_follow_up_trap", ""),
                            ans.get("recovery_phrase", ""),
                        )
                    except Exception as db_exc:
                        logger.warning("cold_call_answers insert failed (non-fatal): %s", db_exc)
    except Exception as outer_exc:
        logger.warning("final_formatter DB export failed (non-fatal): %s", outer_exc)

    # Deterministic count — independent of whether the DB export succeeded, so
    # the markdown stats and notes.num_questions can never disagree.
    total_questions = sum(len(s.get("questions") or []) for s in sequences)
    export_result = {
        "export_batch_id": export_batch_id,
        "question_sequences_exported": len(sequences),
        "answer_sequences_exported": sum(1 for s in sequences if s.get("answers")),
        "questions_exported": total_questions,
    }
    if q_exported != total_questions:
        logger.warning("  ⚠ [final_formatter] DB wrote %d of %d questions",
                       q_exported, total_questions)

    # ── Markdown ──────────────────────────────────────────────────────────
    lines: List[str] = [
        "# Cold-Call Question Sequences", "",
        f"**Sequences generated:** {len(sequences)}  ",
        f"**Total questions:** {total_questions}  ",
        f"**Coverage score:** {coverage_score:.0%}  ",
        f"**Depth profile:** {by_depth}  ",
        "", "---", "",
    ]
    for seq in sequences:
        ans_by_qid = {a["question_id"]: a for a in seq.get("answers") or []}
        lines.append(f"## Sequence {seq.get('sequence_index', '')}: {seq.get('sequence_theme', '')}")
        lines.append(f"*Case: {seq.get('case_name', '')}*")
        lines.append("")
        for q in seq.get("questions") or []:
            qid = q.get("question_id", "")
            lines.append(f"**{qid}. [{q.get('question_type', '')}]** {q.get('question_text', '')}")
            ans = ans_by_qid.get(qid)
            if ans:
                lines.append(f"> {ans.get('model_answer', '')}")
            lines.append("")
        lines += ["---", ""]
    if missing:
        lines += [f"*Coverage gaps: {', '.join(missing)}*", ""]
    lines.append(
        f"<!-- cold-call-agent compact | {len(sequences)} sequences | "
        f"{total_questions} Q/A | export_batch: {export_batch_id[:8]} -->"
    )
    markdown = "\n".join(lines)

    await _try_save_artifact(
        state, "export_result", dict(export_result), "tool_only",
        "final_formatter", "export_result", source_ids=state.get("source_ids"),
    )
    _node_done("final_formatter", state,
               sequences=len(sequences), questions=total_questions,
               db_rows=q_exported)
    return {
        "coverage_map": coverage_map,
        "export_result": export_result,
        "final_output": markdown,
    }
