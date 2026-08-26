# agents/flashcards/compact_nodes.py
"""
Compact 4-stage flashcards pipeline.

Replaces the 7-wired-node graph's ~21 LLM round-trips with four stages:

  1. plan_agent        — count math (deterministic) + one worker_mid call for
                         coverage strategy. Runs in PARALLEL with research.
  2. research_agent    — ONE bounded tool-calling loop. Absorbs source_profiler
                         (×N) + concept_extractor (whose 7 PROBE_MAP searches
                         become retrieval turns) + card_blueprint_planner.
                         Emits exactly num_cards specs + a concept inventory.
     sync_barrier      — no-op join.
  3. card_batch_generator — [Send × ceil(num_cards/batch_size)]. Absorbs
                         flashcard_drafter + local_card_critic +
                         card_repair_agent.
  4. final_formatter   — pure Python. Merge, DB write, notes UPDATE. Zero LLM.

Fixes folded in:
  • The blueprint prompt truncated the taxonomy to 20 of 33 types
    (`FLASHCARD_CARD_TYPES[:20]`), so 13 types were unreachable — the full list
    is injected here.
  • It also told the model to make `DISSENT_VS_MAJORITY` cards, a type that
    does not exist in the taxonomy at all — removed.
  • `card_order` was `batch_idx * batch_size + pos`, which leaves gaps when a
    batch under-delivers; it is now assigned globally at merge.
  • The legacy pipeline never grounded (`grounding_verdict` was never set by
    anything, so its accept filter was a no-op) — generators now self-ground
    against the harvested evidence.
  • Re-running a note APPENDED duplicate cards (random uuid4, no-op
    ON CONFLICT); the formatter now deletes this deck's prior rows.

Also note: three legacy nodes (`answer_backside_enricher`, `batch_commit`,
`batch_router`) were defined but never wired into the graph.

Legacy pipeline remains in nodes.py; graph.py selects via FLASHCARD_COMPACT
(default true).
"""

import json
import logging
import math
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
    APPLICATION_TYPES,
    FLASHCARD_CARD_TYPES,
    MAPPING_TYPES,
    RECALL_TYPES,
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

MAX_RESEARCH_TURNS = int(os.getenv("FLASHCARD_RESEARCH_MAX_TURNS", "5"))
BATCH_GEN_MAX_TURNS = int(os.getenv("FLASHCARD_GEN_MAX_TURNS", "2"))
HARD_OUTPUT_TOKENS = int(os.getenv("FLASHCARD_HARD_OUTPUT_TOKENS", "14000"))
DOSSIER_TARGET_WORDS = int(os.getenv("FLASHCARD_DOSSIER_TARGET_WORDS", "900"))
# Cards are moderate index-card Q/As. 5 per Send (not 7) so a deck fans out to
# more, smaller batches: generator latency under thinking=True varies widely per
# call, and with too few batches one slow draw sets the whole stage.
DEFAULT_BATCH_SIZE = int(os.getenv("FLASHCARD_BATCH_SIZE", "5"))

GENERATOR_MAX_CHUNKS = 14
GENERATOR_CHUNK_CHAR_CAP = 1200

FRONT_MAX_WORDS = 40
BACK_MAX_WORDS = 120


def _fetch_worker_model_local():
    from .worker_config import _fetch_worker_model
    return _fetch_worker_model


# ─────────────────────────────────────────────────────────────────────────────
# 1. plan_agent
# ─────────────────────────────────────────────────────────────────────────────

async def plan_agent(state: AgentState) -> Dict:
    """Count math + coverage strategy. Absorbs head_orchestrator."""
    from agents.tools.base import make_tools
    from agents.tools.registry import FLASHCARD_PLANNER_TOOLS

    logger.info("🔀 [%s] PARALLEL fire → plan_agent (alongside research_agent)",
                (state.get("job_id") or "")[:8] or "no-job")

    num_cards = max(1, state.get("num_cards") or 10)
    batch_size = min(max(state.get("batch_size") or DEFAULT_BATCH_SIZE, 1), 10)
    num_batches = math.ceil(num_cards / batch_size)

    _node_start("plan_agent", state, num_cards=num_cards,
                batch_size=batch_size, num_batches=num_batches)

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=FLASHCARD_PLANNER_TOOLS,
    )
    list_tool = next((t for t in tools if t.name == "list_sources"), None)
    sources_json = "[]"
    if list_tool:
        try:
            sources_json = await list_tool.ainvoke({})
        except Exception as exc:
            _node_warn("plan_agent", state, f"list_sources failed: {exc}")

    system = (
        "You are a T-14 law professor designing a rigorous flashcard deck. "
        "Plan the deck — do NOT write cards.\n\n"
        "Flag any documents with multi-element tests, definitional disputes, or "
        "policy debates — these yield the best card material. Note any case "
        "opinions with dissents; dissent reasoning is high-yield for comparison "
        "and contrast cards.\n\n"
        "Return JSON with keys: job_type, course_context, priority_topics "
        "(list), high_yield_notes (one sentence)."
    )
    raw = await _llm(
        "worker_mid",
        f"Sources available:\n{sources_json}\n\n"
        f"User request: {state['request']}\nCards: {num_cards}",
        system=system, max_tokens=640, _node="plan_agent",
    )
    try:
        job_plan = _parse_json(raw)
    except Exception:
        job_plan = {"job_type": "flashcards", "course_context": "",
                    "priority_topics": [], "high_yield_notes": ""}

    _node_done("plan_agent", state, topics=job_plan.get("priority_topics", [])[:4])
    return {
        "job_plan": job_plan,
        "num_cards": num_cards,
        "batch_size": batch_size,
        "num_batches": num_batches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. research_agent
# ─────────────────────────────────────────────────────────────────────────────

def _research_system(num_cards: int) -> str:
    """Absorbs source_profiler, concept_extractor, card_blueprint_planner."""
    per_type_cap = max(3, num_cards // 6)
    return (
        "You are a legal research agent building the inventory for a T-14 "
        "flashcard deck. Work in bounded steps:\n"
        "  1. Orient: the CORPUS SURVEY is ALREADY PROVIDED in the first "
        "message. Do NOT re-fetch it with list_sources / get_doc_outline / "
        "get_doc_metadata / find_docs_about / find_sections_about.\n"
        "  2. Retrieve (START HERE on turn 1): batch MANY parallel searches "
        "across these angles — rules and holdings, elements and tests, "
        "exceptions and limits, policy and rationale, procedure and burden, "
        "definitions, and trigger facts.\n"
        "  3. Inventory and blueprint: extract the atomic card material, then "
        "allocate the card specs.\n\n"
        "WHAT TO EXTRACT:\n"
        "• cases — for each: case_name, source_id, holding (one sentence), rule "
        "(standalone restatement), trigger_fact, procedural_posture. **Extract "
        "every case mentioned; do not summarise multiple into one.**\n"
        "• definitions — prefer the court's or statute's exact language.\n"
        "• rule_element_sets — for multi-part tests (e.g. duty, breach, "
        "causation, damages) list EACH element as a separate string.\n"
        "• exceptions — state the base rule, the exception, and the specific "
        "factual context that triggers it.\n"
        "• policy_points — the values the rule serves.\n"
        "• burden_assignments — precisely who bears each burden, under which "
        "standard, in which context.\n\n"
        f"BLUEPRINT — allocate EXACTLY {num_cards} card specs.\n"
        "PEDAGOGY MIX: direct comprehension, case facts, black-letter law, "
        "moderate hypotheticals, and application — roughly 40% recall, 35% "
        "application, 25% mapping, with AT LEAST 20% application or analysis.\n"
        "ALLOCATION RULES:\n"
        f"• No more than {per_type_cap} specs may share the same card_type.\n"
        "• Every identified case must appear in at least 1 card.\n"
        "• Prioritise: rule_element_sets → ELEMENT cards; exceptions → "
        "EXCEPTION cards; policy_points → POLICY cards; dissent reasoning → "
        "comparison cards.\n"
        f"• card_type MUST be one of these {len(FLASHCARD_CARD_TYPES)} types: "
        f"{FLASHCARD_CARD_TYPES}\n\n"
        "RULES:\n"
        "• Your turn budget is for RETRIEVAL. Turn 1 should already be a large "
        "batch of searches.\n"
        "• Only assert what you retrieved — every spec carries chunk_ids.\n"
        f"• STOP CRITERION: once the inventory supports {num_cards} distinct "
        "specs, STOP calling tools and emit.\n\n"
        f"FINAL ANSWER — return ONLY this JSON with EXACTLY {num_cards} specs:\n"
        "{\n"
        '  "inventory": {"cases": [...], "definitions": [...],\n'
        '     "rule_element_sets": [...], "exceptions": [...],\n'
        '     "policy_points": [...], "burden_assignments": [...]},\n'
        '  "specs": [{\n'
        '     "spec_index": 0-based int,\n'
        '     "card_type": one of the types above,\n'
        '     "topic": str, "case_names": [str],\n'
        '     "difficulty": "recall"|"application"|"analysis",\n'
        '     "evidence_chunk_ids": [6-10 chunk_ids — or every relevant chunk\n'
        "        if the corpus holds fewer; never pad with ids you did not\n"
        "        actually retrieve]\n"
        "  }]\n"
        "}"
        + word_budget_line(DOSSIER_TARGET_WORDS,
                           int(DOSSIER_TARGET_WORDS * 1.5), kind="dossier")
    )


async def research_agent(state: AgentState) -> Dict:
    from agents.tools.base import make_tools
    from agents.tools.registry import FLASHCARD_RESEARCH_TOOLS

    logger.info("🔀 [%s] PARALLEL fire → research_agent (alongside plan_agent)",
                (state.get("job_id") or "")[:8] or "no-job")

    num_cards = max(1, state.get("num_cards") or 10)
    _node_start("research_agent", state,
                max_turns=MAX_RESEARCH_TURNS, num_cards=num_cards)

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=FLASHCARD_RESEARCH_TOOLS,
    )
    base_model, sem, rebuild = await build_tool_model(
        _fetch_worker_model_local(), "orchestrator",
        provider=os.getenv("FLASHCARD_RESEARCH_PROVIDER"),
        hard_cap=HARD_OUTPUT_TOKENS,
    )

    survey, n_chunks = await prefetch_corpus_survey(state, tools)
    logger.info("  📚 [research_agent] corpus survey pre-fetched (%d chars) n_chunks=%d",
                len(survey), n_chunks)

    messages: List[Any] = [
        SystemMessage(content=_research_system(num_cards)),
        HumanMessage(content=(
            f"{survey}\n\n"
            f"Source ids in scope: {state.get('source_ids', [])}\n"
            f"User request: {state['request']}\nCards required: {num_cards}\n\n"
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
            f"{num_cards} specs. Return ONLY the JSON object."
        ),
    )
    try:
        dossier: Optional[Dict[str, Any]] = _parse_json(raw)
    except Exception as exc:
        _node_warn("research_agent", state, f"dossier JSON parse failed: {exc}")
        dossier = None
    if not isinstance(dossier, dict):
        dossier = {}

    specs = [s for s in (dossier.get("specs") or []) if isinstance(s, dict)][:num_cards]
    for i, s in enumerate(specs):
        s["spec_index"] = i
        if s.get("card_type") not in FLASHCARD_CARD_TYPES:
            s["card_type"] = "RULE_RECALL"
    dossier["specs"] = specs
    dossier.setdefault("inventory", {})

    if specs:
        n_app = sum(1 for s in specs
                    if s.get("card_type") in set(APPLICATION_TYPES + MAPPING_TYPES))
        if n_app < max(1, round(len(specs) * 0.20)):
            _node_warn("research_agent", state,
                       f"only {n_app}/{len(specs)} specs are application/mapping "
                       "— below the 20% floor")

    await _try_save_artifact(
        state, "flashcard_dossier",
        {"dossier": dossier, "evidence_chunks": len(evidence_store)},
        "orchestrator", "research_agent", "flashcard_blueprint",
        source_ids=state.get("source_ids"),
    )
    _node_done("research_agent", state,
               n_specs=len(specs), n_evidence_chunks=len(evidence_store))
    return {"flashcard_dossier": dossier, "evidence_store": evidence_store}


# ─────────────────────────────────────────────────────────────────────────────
# sync_barrier
# ─────────────────────────────────────────────────────────────────────────────

async def sync_barrier(state: AgentState) -> Dict:
    job = (state.get("job_id") or "")[:8] or "no-job"
    dossier = state.get("flashcard_dossier") or {}
    logger.info(
        "🔗 [%s] sync_barrier — joined plan_agent + research_agent  "
        "job_plan=%s n_specs=%d",
        job, "present" if state.get("job_plan") else "MISSING (join failed?)",
        len(dossier.get("specs") or []),
    )
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 3. card_batch_generator — [Send × n_batches]
# ─────────────────────────────────────────────────────────────────────────────

def dossier_to_batches(state: AgentState):
    dossier = state.get("flashcard_dossier") or {}
    specs = dossier.get("specs") or []
    if not specs:
        logger.warning("dossier_to_batches: no specs — routing to final_formatter")
        return "final_formatter"

    batch_size = min(max(state.get("batch_size") or DEFAULT_BATCH_SIZE, 1), 10)
    batches = [specs[i:i + batch_size] for i in range(0, len(specs), batch_size)]
    logger.info("flashcards(compact) x%d node fan out for card_batch_generator "
                "(%d specs, batch_size=%d)", len(batches), len(specs), batch_size)
    return [
        Send("card_batch_generator", {"batch": {"batch_index": i, "specs": b}, **state})
        for i, b in enumerate(batches)
    ]


async def card_batch_generator(state: Dict) -> Dict:
    """
    Emit one batch of complete cards. Absorbs flashcard_drafter +
    local_card_critic + card_repair_agent (fix-before-emit).
    """
    from agents.tools.base import make_tools
    from langchain_core.tools import tool as _tool_decorator

    batch: Dict[str, Any] = state["batch"]
    dossier: Dict[str, Any] = state.get("flashcard_dossier") or {}
    job_plan: Dict[str, Any] = state.get("job_plan") or {}
    evidence_store: Dict[str, Dict] = state.get("evidence_store") or {}
    specs: List[Dict] = batch["specs"]
    batch_idx = batch["batch_index"]

    _node_start("card_batch_generator", state, batch=batch_idx, n_specs=len(specs))

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
        harvested for this deck. Free and instant — use BEFORE any live database
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
        tool_names=["find_supporting_evidence"],
    )
    tools = [grep_research_corpus] + verifier_tools

    system = (
        "You are a T-14 law professor writing flashcards. This output is final "
        "— no editor runs after you.\n\n"
        "FRONT (QUESTION) STANDARDS:\n"
        "- ONE learning target per card — atomic, never compound.\n"
        "- A direct question or cloze statement.\n"
        f"- Max {FRONT_MAX_WORDS} words. Do NOT include the answer in the front.\n"
        "- RULE cards: 'What is the rule from [Case]?' / 'State the [doctrine] test.'\n"
        "- ELEMENT cards: 'List the elements of [rule].' / 'What must a plaintiff "
        "show to establish [claim]?'\n"
        "- EXCEPTION cards: 'What is the exception to [rule]?' — include the "
        "limiting condition in the front.\n"
        "- CASE_HOLDING cards: 'What did the court hold in [Case]?' — name the "
        "specific issue.\n"
        "- POLICY cards: 'What policy rationale supports [doctrine]?'\n"
        "- APPLICATION cards: a 2-3 sentence fact pattern ending in 'What "
        "result?' or 'Which rule applies?'\n\n"
        "BACK (ANSWER) STANDARDS:\n"
        "- Self-contained — a student studying the back alone gets the full "
        "answer.\n"
        "- Structure: (a) direct answer; (b) the operative legal rule; (c) any "
        "critical limiting condition (the 'unless/but').\n"
        "- Recall cards: max 80 words. Application/analysis cards: up to "
        f"{BACK_MAX_WORDS} words, including rule + application.\n"
        "- Do NOT add wrong answers or distractors — this is not a quiz.\n\n"
        "EXAM USE NOTE: for APPLICATION and ANALYSIS cards, one sentence on when "
        "to deploy this rule on an exam (e.g. 'Spot this when facts show X'). "
        "Empty string for all other card types.\n\n"
        "HINT: one memory-aid sentence pointing at the key concept WITHOUT "
        "revealing the answer.\n\n"
        "SELF-CHECK before emitting (there is no critic downstream) — every card "
        "must pass all three:\n"
        "  vagueness — is the front specific and unambiguous? 'What is "
        "negligence?' FAILS; 'What is the duty element under the reasonable "
        "person standard?' passes.\n"
        "  atomic_focus — exactly ONE concept? Combining rule + exception + "
        "policy on one back FAILS.\n"
        "  answer_quality — concise, accurate, self-contained?\n"
        "Also check the batch for near-duplicate fronts. Fix anything that "
        "fails before emitting.\n\n"
        "GROUNDING: use only the evidence provided. If something is missing, "
        "call grep_research_corpus first (free, instant). Most batches need no "
        "tool calls. Never invent a holding, a citation, or authority. Cite "
        "support inline as [chunk_id] immediately after the proposition it "
        "supports on the back of the card — these are stripped before the "
        "student sees them, so they cost you no length.\n\n"
        "Return ONLY this JSON array — one object per spec, in spec order:\n"
        "[{\n"
        '  "spec_index": int (copy from the spec),\n'
        '  "card_type": str (copy from the spec),\n'
        f'  "front_content": str (≤{FRONT_MAX_WORDS} words),\n'
        f'  "back_content": str (≤{BACK_MAX_WORDS} words),\n'
        '  "hint": str,\n'
        '  "exam_use_note": str (empty for non-application cards),\n'
        '  "source_refs": [chunk_ids]\n'
        "}]"
    )

    plan_ctx = (
        f"COURSE CONTEXT: {job_plan.get('course_context', '')}\n"
        f"PRIORITY TOPICS: {json.dumps(job_plan.get('priority_topics') or [])}\n"
        f"HIGH-YIELD NOTES: {job_plan.get('high_yield_notes', '')}\n\n"
    ) if job_plan else ""

    prompt = (
        f"{plan_ctx}"
        f"SPECS FOR THIS BATCH:\n{json.dumps(specs, indent=2)}\n\n"
        f"CONCEPT INVENTORY:\n{json.dumps(dossier.get('inventory') or {}, indent=2)[:4000]}\n\n"
        f"EVIDENCE:\n{evidence_text}"
    )

    base_model, sem, rebuild = await build_tool_model(
        _fetch_worker_model_local(), "orchestrator", hard_cap=HARD_OUTPUT_TOKENS,
    )
    messages: List[Any] = [SystemMessage(content=system), HumanMessage(content=prompt)]
    raw = await run_bounded_tool_loop(
        base_model, tools, messages, BATCH_GEN_MAX_TURNS, sem,
        "card_batch_generator", state,
        countdown_from_turn=BATCH_GEN_MAX_TURNS, rebuild=rebuild,
        forced_stop_prompt=(
            "STOP calling tools. Emit the cards JSON array NOW using only the "
            "evidence already gathered."
        ),
    )
    try:
        items = _parse_json(raw)
    except Exception as exc:
        _node_warn("card_batch_generator", state,
                   f"batch {batch_idx} JSON parse failed: {exc}")
        items = []
    if not isinstance(items, list):
        items = []

    spec_by_index = {s["spec_index"]: s for s in specs}
    cards: List[Dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        spec = spec_by_index.get(it.get("spec_index"), specs[0] if specs else {})
        back = it.get("back_content", "")
        note = (it.get("exam_use_note") or "").strip()
        if note:
            back = f"{back}\n\n*Exam tip: {note}*"
        cards.append({
            "spec_index": it.get("spec_index", spec.get("spec_index", 0)),
            "card_type": it.get("card_type") or spec.get("card_type", ""),
            "front_content": it.get("front_content", ""),
            "back_content": back,
            "hint": it.get("hint", ""),
            "source_refs": it.get("source_refs") or [],
        })

    _node_done("card_batch_generator", state, batch=batch_idx, n_cards=len(cards))
    return {"generated_cards": cards}


card_batch_generator.default_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 4. final_formatter — deterministic, zero LLM
# ─────────────────────────────────────────────────────────────────────────────

async def final_formatter(state: AgentState) -> Dict:
    """
    Merge batches, write the DB, update the note. NO LLM call.

    card_order is assigned globally here (the legacy offset scheme left gaps
    when a batch under-delivered), and the deck's prior rows are deleted so a
    re-run replaces rather than appends.
    """
    cards: List[Dict] = list(state.get("generated_cards") or [])
    job_id = state.get("job_id") or ""
    user_id = state.get("user_id") or ""
    project_id = state.get("project_id") or ""
    num_requested = state.get("num_cards") or len(cards)
    source_ids = state.get("source_ids") or []

    _node_start("final_formatter", state, n_cards=len(cards))
    cards.sort(key=lambda c: c.get("spec_index", 99))

    citations_removed = 0
    for c in cards:
        for key in ("front_content", "back_content", "hint"):
            cleaned, n = strip_chunk_citations(c.get(key, ""))
            c[key] = cleaned
            citations_removed += n
    if citations_removed:
        logger.info("  🧹 [final_formatter] stripped %d inline chunk citation(s)",
                    citations_removed)

    kept = [c for c in cards if c.get("front_content") and c.get("back_content")]
    dropped = len(cards) - len(kept)
    if dropped:
        _node_warn("final_formatter", state, f"dropped {dropped} empty card(s)")

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
                        "DELETE FROM individual_cards WHERE deck_id = $1", job_id
                    )
                    for order, c in enumerate(kept):   # gap-free global ordering
                        card_id = str(_uuid_mod.uuid4())
                        await conn.execute(
                            """
                            INSERT INTO individual_cards (
                                id, deck_id, user_id, project_id,
                                front_content, back_content, card_order,
                                created_at, is_active
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                            ON CONFLICT (id) DO NOTHING
                            """,
                            card_id, job_id, user_id or None, project_id or None,
                            c["front_content"], c["back_content"], order, now, True,
                        )
                        persisted_ids.append(card_id)
        except Exception as exc:
            logger.error("final_formatter DB write failed: %s", exc)
            await _try_save_artifact(
                state, "flashcard_write_error", {"error": str(exc)},
                "tool_only", "final_formatter", "error",
                source_ids=source_ids,
            )

    # ── Markdown + type coverage ──────────────────────────────────────────
    type_coverage: Dict[str, int] = {}
    for c in kept:
        t = c.get("card_type", "")
        type_coverage[t] = type_coverage.get(t, 0) + 1

    lines: List[str] = [
        "# Flashcard Deck", "",
        f"**Cards:** {len(kept)} (requested {num_requested})  ",
        f"**Card types covered:** {len(type_coverage)}  ",
        "", "---", "",
    ]
    for i, c in enumerate(kept, start=1):
        lines += [
            f"### {i}. {c.get('front_content', '')}",
            "",
            c.get("back_content", ""),
        ]
        if c.get("hint"):
            lines += ["", f"*Hint: {c['hint']}*"]
        lines += ["", f"`{c.get('card_type', '')}`", "", "---", ""]
    lines.append(
        f"<!-- flashcards compact | {len(kept)} cards | "
        f"{len(persisted_ids)} persisted | types: {len(type_coverage)} -->"
    )
    markdown = "\n".join(lines)

    # Notes UPDATE is owned by this agent (the dispatcher does not update).
    # Matches the legacy column set — deliberately no content_markdown.
    if job_id:
        try:
            from tasks.database import get_db_connection

            description = (
                f"AI-generated flashcard deck with {len(persisted_ids)} cards "
                f"(requested: {num_requested})"
            )
            async with get_db_connection() as conn:
                await conn.execute(
                    """
                    UPDATE notes SET
                        description          = $1,
                        num_cards            = $2,
                        is_active            = $3,
                        is_essential         = $4,
                        num_sources_based_on = $5,
                        referenced_sources   = $6,
                        note_progress_status = 'COMPLETE'
                    WHERE id = $7
                    """,
                    description, len(persisted_ids), True,
                    bool(state.get("is_essential", False)),
                    len(source_ids), [str(s) for s in source_ids], job_id,
                )
        except Exception as exc:
            logger.error("final_formatter notes UPDATE failed: %s", exc)

    _node_done("final_formatter", state,
               cards=len(kept), persisted=len(persisted_ids))
    # accepted_card_ids is the key the dispatcher reads (not persisted_card_ids).
    return {
        "accepted_card_ids": persisted_ids,
        "persisted_card_ids": persisted_ids,
        "final_output": markdown,
    }
