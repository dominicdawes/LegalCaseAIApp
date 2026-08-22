# agents/case_brief/compact_nodes.py
"""
Compact 4-stage case-brief pipeline.

Replaces the 14-node graph's ~31 LLM calls with four stages:

  1. plan_agent        — one worker_mid call → job plan (brief_mode, length,
                         retrieval depth). Runs in PARALLEL with research_agent.
  2. research_agent    — ONE bounded tool-calling ReAct loop. Compresses
                         source_profiler + retrieval_planner (2 calls) +
                         planned_retriever + evidence_card_builder +
                         legal_artifact_extractor (×4) + doctrinal_synthesizer.
                         Emits a LEAN case dossier: identity, anchors, and a
                         role→chunk_id index. Every retrieval result is
                         harvested into state["evidence_store"].
     sync_barrier      — no-op join so section writers see BOTH job_plan and
                         the dossier regardless of which branch finishes first.
                         Also removes the old graph's conditional-edge-on-a-
                         fan-out pattern, which risked T×B / S² task blowup.
  3. section_generator — [Send×N] one per BRIEF UNIT (a fixed table — case
                         brief sections are known a priori, nothing to
                         discover). Compresses section_writer + section_grounder
                         + critic + brief_revision_agent. Each unit emits FINAL
                         template markdown, not a draft.
  4. final_formatter   — pure Python. Zero LLM.

Two structural fixes fall out of this shape:

  • The old pipeline wrote 10 sections, kept only 6 flat SectionDraft fields,
    then spent a 12,000-token LLM call in final_formatter re-deriving the
    structure it had just discarded. Units now emit final form directly, so
    that call is gone entirely.
  • `[omit if absent]` pruning and Roman-numeral renumbering were left to the
    LLM while `_prune_and_renumber_sections` sat dead in nodes.py. Both are now
    deterministic here.

Legacy 14-node pipeline remains in nodes.py; graph.py selects via
CASE_BRIEF_COMPACT (default true).
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Send

# Reuse the legacy module's helpers (same package — telemetry, JSON parsing,
# ledger writes, tier resolution, DeepSeek concurrency cap).
from .nodes import (
    _llm,
    _parse_json,
    _try_save_artifact,
    _node_start,
    _node_done,
    _node_warn,
    _get_deepseek_semaphore,
)
from .state import AgentState

logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────

MAX_RESEARCH_TURNS = int(os.getenv("BRIEF_RESEARCH_MAX_TURNS", "5"))
BRIEF_UNIT_MAX_TURNS = int(os.getenv("BRIEF_UNIT_MAX_TURNS", "2"))

TOOL_RESULT_CHAR_CAP = 6000
GENERATOR_MAX_CHUNKS = 12
GENERATOR_CHUNK_CHAR_CAP = 1400

# ── Output budget ─────────────────────────────────────────────────────────────
# Budgeted in WORDS. Unlike attack_outline (which was ~10x over its 1-5 page
# spec), the legacy case brief's ~1,600 words across a 16-section study template
# is defensible — these targets HOLD that length, they do not cut it. Sum of the
# per-unit targets below ≈ 1,500-1,700 words.
DOSSIER_TARGET_WORDS = int(os.getenv("BRIEF_DOSSIER_TARGET_WORDS", "700"))
HARD_OUTPUT_TOKEN_CAP = int(os.getenv("BRIEF_HARD_OUTPUT_TOKENS", "14000"))

PREFETCH_MAX_OUTLINES = int(os.getenv("BRIEF_PREFETCH_MAX_OUTLINES", "10"))
PREFETCH_MAX_SECTIONS = int(os.getenv("BRIEF_PREFETCH_MAX_SECTIONS", "40"))
PREFETCH_SECTION_SUMMARY_CHARS = 300

# Evidence roles carried on every chunk id in the dossier index. Ported from the
# legacy evidence_card_builder's role taxonomy (nodes.py:734-741) — it is what
# lets each unit select its own chunks without re-reading the corpus.
EVIDENCE_ROLES = (
    "facts", "posture", "issue", "holding", "rule",
    "reasoning", "dissent", "citation", "pedagogy", "exam_trigger",
)


def _word_budget_line(target: int, cap: int, kind: str = "section") -> str:
    """Soft word budget stated in the system prompt so the model self-regulates."""
    if kind == "dossier":
        tail = (
            "This is an internal hand-off, not prose for a reader — terse "
            "fragments only, no explanation, no restating the same point twice."
        )
    else:
        tail = (
            "A case brief is a dense study reference, not an essay. Prefer "
            "precise legal statements over narration; never pad a section to "
            "reach the target, and never truncate mid-structure to stay under it."
        )
    return (
        f"\n\n**LENGTH LIMIT**: aim for ~{target} words; never exceed {cap}. {tail}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BRIEF UNIT TABLE — the fan-out plan
# ─────────────────────────────────────────────────────────────────────────────
# Case-brief sections are known a priori (unlike attack_outline's discovered
# doctrine clusters), so the fan-out is a constant table rather than something
# the research agent has to invent. Each unit owns a contiguous run of the
# canonical 16-section template, pulls the evidence roles it needs, and emits
# FINAL markdown for those sections.
#
# `order` fixes assembly sequence. `roles` selects chunks from the dossier's
# evidence_by_role index. `template` is the exact skeleton the unit must emit —
# ported verbatim in structure from the legacy final_formatter template
# (nodes.py:1930-2042) so the student-facing output is unchanged.

BRIEF_UNITS: List[Dict[str, Any]] = [
    {
        "unit_id": "identity",
        "order": 1,
        "roles": ["citation", "posture", "holding"],
        "target_words": 180,
        "template": (
            "## I. One-Sentence Rule\n"
            "[ONE sentence: subject + may/cannot/must + condition. The most "
            "quotable statement of what this case stands for.]\n\n"
            "---\n\n"
            "## II. Procedural Posture\n"
            "[Numbered steps: origin → trial → appeals → this court. Include "
            "what each court held and why.]"
        ),
        "guidance": (
            "The posture controls which standard of review applies — students "
            "who confuse the frame misapply the rule, so make it explicit. If "
            "the full citation is not in the evidence, OMIT it — never fabricate "
            "a citation, reporter number, or date."
        ),
    },
    {
        "unit_id": "facts",
        "order": 2,
        "roles": ["facts"],
        "target_words": 260,
        "template": (
            "## III. Facts\n"
            "### A. Parties\n"
            "[Petitioner: … | Respondent: …]\n"
            "### B. Background & Context\n"
            "[Operative facts — who, what, where, scale of the conduct]\n"
            "### C. The Dispute\n"
            "[The specific conduct at issue and why it led to litigation]"
        ),
        "guidance": (
            "Include ONLY outcome-determinative facts. A fact is material if "
            "removing it would change the legal outcome — if you cannot say why "
            "a fact matters, cut it. Do NOT characterise facts as good or bad "
            "for either party; state them neutrally."
        ),
    },
    {
        "unit_id": "issue_holding",
        "order": 3,
        "roles": ["issue", "holding", "citation"],
        "target_words": 220,
        "template": (
            "## IV. Statutory / Constitutional Framework  [OMIT-IF-ABSENT]\n"
            "[If a statute: the Act, key sections, operative language. If "
            "constitutional: quote the clause. Omit entirely if the case does "
            "not turn on a specific text.]\n\n"
            "---\n\n"
            "## V. Issues Presented\n"
            "1. Whether [issue 1]\n"
            "2. Whether [issue 2 — omit if only one issue]\n\n"
            "---\n\n"
            "## VI. Holdings\n"
            "**1.** [Direct answer to Issue 1]\n"
            "**2.** [Direct answer to Issue 2 — omit if only one issue]"
        ),
        "guidance": (
            "Every issue takes the form 'Whether [legal standard] applies when "
            "[specific facts]' — a bare legal question with no facts is too "
            "abstract to be useful. Issue and holding must MIRROR each other: "
            "each holding directly answers its issue.\n"
            "CRITICAL: the holding is the LEGAL RULE the court announced. It is "
            "NOT the disposition (affirmed / reversed / remanded). Do not "
            "conflate them — stating 'the Court reversed' as the holding is the "
            "single most common briefing error."
        ),
    },
    {
        "unit_id": "rule",
        "order": 4,
        "roles": ["rule", "holding"],
        "target_words": 240,
        "template": (
            "## VII. Rule & Legal Test\n"
            "**[Black-letter rule — bold, one full sentence]**\n\n"
            "Elements / test:\n"
            "- [element or factor]\n"
            "- [element or factor]\n\n"
            "**Limiting Principle:** [what the rule expressly does NOT cover]\n\n"
            "---\n\n"
            "## X. Black Letter Law\n"
            "**[Category]:**\n"
            "- [rule]\n"
            "- [rule]"
        ),
        "guidance": (
            "The rule must be REUSABLE STANDALONE — state it without reference "
            "to this case by name, so a student can apply it to a new fact "
            "pattern. This is the most exam-tested section of the brief.\n"
            "Do NOT include reasoning or policy here; those belong to the "
            "Reasoning section."
        ),
    },
    {
        "unit_id": "reasoning",
        "order": 5,
        "roles": ["reasoning", "rule"],
        "target_words": 280,
        "template": (
            "## VIII. Reasoning\n"
            "[The court's analytical steps in logical sequence. Use ### "
            "subsections for multi-step reasoning, e.g. ### A. The Framing Move.]\n\n"
            "---\n\n"
            "## IX. Arguments & Court's Answers  [OMIT-IF-ABSENT]\n"
            "[Format each as: **Argument:** [X] → **Court:** [Y]. Omit if the "
            "evidence contains no adversarial framing.]"
        ),
        "guidance": (
            "Trace facts → rule → holding. Tag each analytical step with its "
            "type: application, precedent, policy, textual, institutional, "
            "fairness, or administrability — and name the PRIMARY type driving "
            "the outcome. Do NOT restate the facts here. Flag reasoning the "
            "court asserted without support."
        ),
    },
    {
        "unit_id": "dissent",
        "order": 6,
        "roles": ["dissent"],
        "target_words": 160,
        "conditional_on": "has_dissent",
        "template": (
            "## XII. Dissent  [OMIT-IF-ABSENT]\n"
            "[Who dissented, on what grounds, and why it matters for "
            "understanding the majority.]"
        ),
        "guidance": (
            "Structure as majority rule vs dissent rule vs the CORE "
            "DISAGREEMENT between them, in parallel form.\n"
            "HALLUCINATION GUARD: if the evidence does not show a separate "
            "opinion, return exactly the string OMIT-SECTION and nothing else. "
            "Never invent a dissent. The dissent is a high-yield exam and "
            "cold-call source, which is precisely why a fabricated one is "
            "dangerous."
        ),
    },
    {
        "unit_id": "pedagogy",
        "order": 7,
        "roles": ["pedagogy", "reasoning"],
        "target_words": 200,
        "template": (
            "## XI. Doctrinal Significance\n"
            "[One focused paragraph: what jurisprudential shift this marks, "
            "which earlier cases it modifies or distinguishes, where it sits in "
            "the doctrinal timeline.]\n\n"
            "---\n\n"
            "## XIII. Pedagogy\n"
            "[What this case teaches, what concept it illustrates, how it fits "
            "the course arc, what a student should take away.]"
        ),
        "guidance": (
            "Say WHY a professor assigned this case and what doctrinal move "
            "they want students to learn. Name the common misreadings — the "
            "ways students get this case wrong."
        ),
    },
    {
        "unit_id": "exam",
        "order": 8,
        "roles": ["exam_trigger", "pedagogy", "holding"],
        "target_words": 320,
        "template": (
            "## XIV. Exam Translation\n"
            "**Exam Triggers:**\n"
            "- [fact pattern that should bring this case to mind]\n\n"
            "⚠️ **Do-Not-Overread:** [what the holding expressly does NOT say — "
            "the most common exam mistake]\n\n"
            "---\n\n"
            "## XV. If/Then Case Map\n"
            "- **If** [strong analogy fact], **then** this case applies because […]\n"
            "- **If** [distinguishing fact], **then** it does **not** apply because […]\n"
            "- **If** [ambiguous fact], **then** argue both sides using […]\n\n"
            "---\n\n"
            "## XVI. Cold Call Q&A\n"
            "**1.** [Question]\n"
            "> **Model Answer:** [answer]\n\n"
            "**2.** [Question]\n"
            "> **Model Answer:** [answer]\n\n"
            "**3.** [Question — include one asking how this case differs from a "
            "related or prior case]\n"
            "> **Model Answer:** [answer]"
        ),
        "guidance": (
            "Give at least 3 exam triggers and at least 2 do-not-overread "
            "warnings. Cold-call answers follow Because / Unless / But / "
            "Therefore. Target the mistakes professors most often catch "
            "students on."
        ),
    },
]

# Emitted by a conditional unit when its subject is absent from the evidence.
OMIT_SECTION_MARKER = "OMIT-SECTION"


# ─────────────────────────────────────────────────────────────────────────────
# 1. plan_agent
# ─────────────────────────────────────────────────────────────────────────────

async def plan_agent(state: AgentState) -> Dict:
    """
    Job-level controller — one fast call. Absorbs the legacy head_orchestrator.

    Keeps its real contribution (brief_mode / target_length / retrieval_depth
    calibration) and drops its tool call: the source list is pre-fetched.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import BRIEF_PLANNER_TOOLS

    logger.info("🔀 [%s] PARALLEL fire → plan_agent (alongside research_agent)",
                (state.get("job_id") or "")[:8] or "no-job")
    _node_start("plan_agent", state,
                n_sources=len(state.get("source_ids") or []),
                request=state.get("request", "")[:60])

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=BRIEF_PLANNER_TOOLS,
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
        "You are the workflow controller for a T-14 law-school case brief.\n"
        "Given the available sources, decide how the brief should be built. "
        "Do NOT write the brief itself.\n\n"
        "Return JSON with keys: job_type, brief_mode "
        "(single_case|multi_case|casebook_excerpt|doctrine_packet|mixed_source), "
        "source_ids, target_length (short|standard|long), "
        "retrieval_depth (shallow|standard|deep), "
        "primary_case_guess (best guess at the case being briefed, or ''), "
        "notes (one sentence on anything unusual about this corpus)."
    )
    raw = await _llm(
        "worker_mid",
        f"Sources available:\n{sources_json}\n\nUser request: {state['request']}",
        system=system, max_tokens=640, _node="plan_agent",
    )
    try:
        job_plan = _parse_json(raw)
    except Exception as exc:
        _node_warn("plan_agent", state, f"JSON parse failed ({exc}) — minimal plan")
        job_plan = {
            "job_type": "case_brief",
            "brief_mode": "single_case",
            "source_ids": state.get("source_ids", []),
            "target_length": "standard",
            "retrieval_depth": "standard",
            "primary_case_guess": "",
            "notes": "",
        }

    await _try_save_artifact(
        state, "job_plan", job_plan, "worker_mid", "plan_agent",
        "job_plan", source_ids=state.get("source_ids"),
    )
    _node_done("plan_agent", state,
               mode=job_plan.get("brief_mode", "?"),
               length=job_plan.get("target_length", "?"))
    return {"job_plan": job_plan}


# ─────────────────────────────────────────────────────────────────────────────
# 2. research_agent
# ─────────────────────────────────────────────────────────────────────────────

def _research_system() -> str:
    """
    Research prompt. Folds in the intent of six deleted nodes: source_profiler,
    retrieval_planner (corpus orientation + plans), planned_retriever,
    evidence_card_builder, legal_artifact_extractor (×4), doctrinal_synthesizer.
    """
    return (
        "You are a legal research agent assembling the evidence base for a "
        "T-14 law-school case brief. Work in bounded steps:\n"
        "  1. Orient: the CORPUS SURVEY (sources, outlines, section map) is "
        "ALREADY PROVIDED in the first message. Identify the PRIMARY OPINION "
        "being briefed and classify every other source "
        "(primary_opinion | casebook_excerpt | lecture_notes | secondary | "
        "duplicate | irrelevant). Do NOT re-fetch the survey with list_sources "
        "/ get_doc_outline / get_doc_metadata / find_docs_about / "
        "find_sections_about — that wastes a whole turn.\n"
        "  2. Retrieve (START HERE on turn 1): batch MANY parallel calls. "
        "Prefer CASE-NAME-ANCHORED keyword search, court-specific terms and "
        "citation patterns over generic semantic queries — you are pinning down "
        "one specific opinion, not surveying a topic. Use get_section / "
        "get_neighbors to walk an opinion's structure once you have a hit.\n"
        "  3. Index: tag the chunks you retrieved by ROLE so the section "
        "writers can select evidence without re-reading the corpus.\n\n"
        "WHAT THE BRIEF NEEDS (gather evidence for each):\n"
        "• FACTS — material facts only. A fact is material if removing it would "
        "change the legal outcome.\n"
        "• POSTURE — how the case arrived here, and the standard of review.\n"
        "• ISSUE — phrased 'Whether [legal standard] applies when [facts]'.\n"
        "• HOLDING — the legal rule announced. NOT the disposition. Holding = "
        "the rule; disposition = what the court ordered (affirmed/reversed). "
        "Never conflate them.\n"
        "• RULE — the black-letter rule stated so it is reusable WITHOUT naming "
        "this case, plus its test/elements/exceptions/limits.\n"
        "• REASONING — the court's steps, each classifiable as application, "
        "precedent, policy, textual, institutional, fairness or "
        "administrability.\n"
        "• DISSENT — only if one genuinely exists. If the sources show no "
        "separate opinion, set has_dissent=false and leave dissent evidence "
        "EMPTY. Do NOT invent a dissent.\n"
        "• PEDAGOGY — why this case is assigned: its doctrinal role "
        "(introduces|refines|limits|overrules|applies|distinguishes), the "
        "broad takeaway rule vs the narrow holding, the limits that generate "
        "wrong answers, exam triggers, and the traps professors catch students "
        "on.\n\n"
        "RULES:\n"
        "• Your turn budget is for RETRIEVAL. Turn 1 should already be a large "
        "batch of search calls, not discovery.\n"
        "• Only assert what you actually retrieved — every claim carries "
        "chunk_ids.\n"
        "• STOP CRITERION: once every bullet above has supporting chunk_ids "
        "(or is genuinely absent from the sources), STOP calling tools and emit "
        "the dossier immediately, even if turns remain.\n"
        "• SOURCE TYPE MATTERS: if a source is a casebook note, law-review "
        "article or commentary, its conclusions are ARGUMENTS about the case, "
        "not the court's holding. Keep them in `contested_positions`; never let "
        "a commentator's position become the holding or the rule.\n\n"
        "FINAL ANSWER — return ONLY this JSON object (no prose):\n"
        "{\n"
        '  "case": {"case_name", "citation", "court", "year", "disposition",\n'
        '           "opinion_author", "has_dissent": true|false},\n'
        '  "source_roles": [{"source_id", "role"}],\n'
        '  "one_line_holding": ONE sentence — the legal rule, not the disposition,\n'
        '  "black_letter_rule": ONE sentence, reusable without naming the case,\n'
        '  "doctrinal_role": "introduces|refines|limits|overrules|applies|distinguishes",\n'
        '  "contested_positions": [short strings; [] if none],\n'
        '  "evidence_by_role": {\n'
        '      "facts": [chunk_ids], "posture": [...], "issue": [...],\n'
        '      "holding": [...], "rule": [...], "reasoning": [...],\n'
        '      "dissent": [...], "citation": [...], "pedagogy": [...],\n'
        '      "exam_trigger": [...]\n'
        "  }\n"
        "}\n"
        "Put every chunk_id you retrieved under at least one role — these "
        "indexes are the ONLY way section writers find evidence. A role with no "
        "supporting chunks gets an empty list; never pad with ids you did not "
        "retrieve."
        + _word_budget_line(DOSSIER_TARGET_WORDS,
                            int(DOSSIER_TARGET_WORDS * 1.5), kind="dossier")
    )


def _harvest_evidence(tool_name: str, result_str: str, store: Dict[str, Dict]) -> None:
    """
    Capture chunk payloads returned by retrieval tools into the evidence store
    so section writers can ground without re-retrieving. Never raises.
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


async def _prefetch_corpus_survey(state: Dict, tools: List[Any]) -> Tuple[str, int]:
    """
    Deterministically fetch the corpus survey so research_agent can skip
    discovery and spend every turn on retrieval.

    Layers, all plain indexed DB reads — no LLM, no embeddings, no vector search:
      1. corpus   — list_sources
      2. document — get_doc_outline per source
      3. section  — document_sections summaries, FALLING BACK to distinct
                    section_paths from document_vector_store.

    The fallback is the normal path, not an edge case: ingest dispatches
    build_section_summaries fire-and-forget at the moment a doc goes COMPLETE
    and the note fires ~0.8s later, so document_sections is reliably empty at
    survey time. Section paths ship with the chunks, so the structural map
    survives even when the summaries do not.

    Returns (survey_text, n_chunks); 0 means unknown. Never raises.
    """
    source_ids: List[str] = list(state.get("source_ids") or [])
    parts: List[str] = []
    n_chunks = 0
    tool_by_name = {t.name: t for t in tools}

    list_tool = tool_by_name.get("list_sources")
    if list_tool is not None:
        try:
            parts.append(f"## Sources in this project\n{await list_tool.ainvoke({})}")
        except Exception as exc:
            _node_warn("research_agent", state, f"prefetch list_sources failed: {exc}")

    outline_tool = tool_by_name.get("get_doc_outline")
    if outline_tool is not None and source_ids:
        targets = source_ids[:PREFETCH_MAX_OUTLINES]
        results = await asyncio.gather(
            *(outline_tool.ainvoke({"source_id": sid}) for sid in targets),
            return_exceptions=True,
        )
        outlines = [
            f"### {sid}\n{res}"
            for sid, res in zip(targets, results)
            if not isinstance(res, Exception)
        ]
        if outlines:
            parts.append("## Document outlines (TOC + key concepts)\n" + "\n".join(outlines))

    if source_ids:
        try:
            from tasks.database import get_global_async_db_pool, init_async_pools

            await init_async_pools()
            pool = get_global_async_db_pool()
            async with pool.acquire() as conn:
                n_chunks = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM document_vector_store
                    WHERE project_id = $1 AND source_id = ANY($2::uuid[])
                    """,
                    state["project_id"], source_ids,
                ) or 0

                rows = await conn.fetch(
                    """
                    SELECT source_id, section_path, section_summary
                    FROM document_sections
                    WHERE project_id = $1
                      AND source_id = ANY($2::uuid[])
                    ORDER BY source_id, start_chunk_idx
                    LIMIT $3
                    """,
                    state["project_id"], source_ids, PREFETCH_MAX_SECTIONS,
                )

                if not rows:
                    rows = await conn.fetch(
                        """
                        SELECT DISTINCT ON (source_id, section_path)
                               source_id, section_path,
                               NULL::text AS section_summary
                        FROM document_vector_store
                        WHERE project_id = $1
                          AND source_id = ANY($2::uuid[])
                          AND section_path IS NOT NULL
                          AND section_path <> ''
                        ORDER BY source_id, section_path, chunk_index
                        LIMIT $3
                        """,
                        state["project_id"], source_ids, PREFETCH_MAX_SECTIONS,
                    )
                    if rows:
                        logger.info(
                            "  📚 [research_agent] document_sections empty — "
                            "derived %d section paths from chunks instead", len(rows)
                        )

            if rows:
                lines = []
                for r in rows:
                    line = f"[{str(r['source_id'])[:8]}] {r['section_path']}"
                    summary = (r["section_summary"] or "")[:PREFETCH_SECTION_SUMMARY_CHARS]
                    if summary:
                        line += f"\n    {summary}"
                    lines.append(line)
                parts.append("## Section map\n" + "\n".join(lines))
                logger.info("  📚 [research_agent] prefetched %d section entries", len(rows))
        except Exception as exc:
            _node_warn("research_agent", state, f"prefetch section map failed: {exc}")

    if not parts:
        return (
            "(Corpus survey pre-fetch unavailable — use list_sources / "
            "get_doc_outline / find_sections_about to orient yourself first.)"
        ), n_chunks

    return (
        "=== CORPUS SURVEY (pre-fetched for you) ===\n"
        + "\n\n".join(parts)
        + "\n=== END CORPUS SURVEY ==="
    ), n_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Shared tool-calling infrastructure (research_agent + section_generator)
# ─────────────────────────────────────────────────────────────────────────────

async def _build_tool_model(worker_class: str, provider: Optional[str] = None):
    """
    Resolve a tool-bindable chat model through the tiered WORKER_MODEL_MAP
    (_fetch_worker_model) rather than hardcoding a provider, so escalating a
    node to a flagship model is a config change.

    Returns (base_model, semaphore). base_model is UNBOUND — callers bind tools
    themselves, which lets the same instance serve as the no-tools model for the
    reserved emission turn.
    """
    from .worker_config import _fetch_worker_model

    resolved_provider, model_name, thinking = _fetch_worker_model(worker_class, provider)

    # HARD cap sits deliberately above the prompt's word budget: the prompt does
    # the budgeting, this only bounds pathology. Setting it near the target is
    # what causes mid-sentence clipping.
    if resolved_provider == "deepseek":
        from utils.llm_clients.deepseek_client import DeepSeekClient
        client = DeepSeekClient(
            model_name=model_name, temperature=0.3,
            max_output_tokens=HARD_OUTPUT_TOKEN_CAP, thinking=thinking,
        )
        base_model = client._client
        sem = _get_deepseek_semaphore()
    elif resolved_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from utils.llm_clients.anthropic_rate_limits import get_llm_semaphore
        base_model = ChatAnthropic(
            model=model_name, max_tokens=HARD_OUTPUT_TOKEN_CAP, temperature=0.3
        )
        sem = await get_llm_semaphore()
    else:
        raise NotImplementedError(
            f"_build_tool_model: provider '{resolved_provider}' is not wired "
            "for tool-calling here (only deepseek/anthropic are supported)."
        )

    logger.info("  🤖 [_build_tool_model] worker_class=%s provider=%s model=%s thinking=%s",
                worker_class, resolved_provider, model_name, thinking)
    return base_model, sem


async def _run_bounded_tool_loop(
    base_model: Any,
    tools: List[Any],
    messages: List[Any],
    max_turns: int,
    sem: Any,
    node_name: str,
    state: Dict,
    evidence_store: Optional[Dict[str, Dict]] = None,
    countdown_from_turn: int = 3,
    forced_stop_prompt: str = "STOP calling tools. Answer NOW using only what you've gathered.",
) -> str:
    """
    Bounded ReAct loop. `max_turns` is the RETRIEVAL budget; the final answer is
    emitted in an additional reserved turn via the UNBOUND base_model, so it
    cannot emit further tool calls. The model is warned the turn is coming, which
    makes emission a planned hand-off rather than an interrupt. If it finishes
    early (a turn with no tool calls) that response IS the answer.
    """
    tool_map = {t.name: t for t in tools}
    bound_model = base_model.bind_tools(tools)

    for turn in range(1, max_turns + 1):
        # Ladder gated on countdown_from_turn so short budgets stay quiet early:
        # section_generator runs a 2-turn budget whose common case is answering
        # on turn 1, and a "one turn remains" nudge there would invite tool calls
        # it would not otherwise make.
        if turn >= countdown_from_turn:
            if turn == max_turns:
                messages.append(HumanMessage(content=(
                    f"(⚠ FINAL RETRIEVAL TURN — turn {turn} of {max_turns}. Make any "
                    "last tool calls now. Your NEXT response must be the final "
                    "answer itself, with no tool calls.)"
                )))
            elif turn == max_turns - 1:
                messages.append(HumanMessage(content=(
                    f"(Turn {turn} of {max_turns} — one retrieval turn remains after "
                    "this one, then you must emit the final answer.)"
                )))
            else:
                remaining = max_turns - turn + 1
                messages.append(HumanMessage(content=(
                    f"({remaining} of {max_turns} retrieval turns remain — wrap up and "
                    "answer as soon as you have sufficient evidence.)"
                )))
        async with sem:
            try:
                resp: AIMessage = await bound_model.ainvoke(messages)
            except Exception as exc:
                _node_warn(node_name, state, f"turn {turn} LLM error ({exc}) — retrying once")
                await asyncio.sleep(5)
                resp = await bound_model.ainvoke(messages)
        messages.append(resp)

        tool_calls = getattr(resp, "tool_calls", None) or []
        if not tool_calls:
            return resp.content

        logger.info("  🔧 [%s] turn %d/%d → %d tool call(s): %s",
                    node_name, turn, max_turns, len(tool_calls),
                    [tc.get("name") for tc in tool_calls][:8])
        results = await asyncio.gather(
            *(_exec_tool_call(tool_map, tc) for tc in tool_calls)
        )
        for (tool_msg, full_result), tc in zip(results, tool_calls):
            messages.append(tool_msg)
            if evidence_store is not None:
                _harvest_evidence(tc.get("name", ""), full_result, evidence_store)

    logger.info("  📝 [%s] reserved emission turn (retrieval budget %d/%d used)",
                node_name, max_turns, max_turns)
    messages.append(HumanMessage(content=forced_stop_prompt))
    async with sem:
        final = await base_model.ainvoke(messages)  # unbound — cannot call tools
    return final.content


async def research_agent(state: AgentState) -> Dict:
    """
    The all-in-one research stage. Runs in PARALLEL with plan_agent — it does
    NOT depend on job_plan; its own corpus survey stands in for that.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import BRIEF_RESEARCH_TOOLS

    logger.info("🔀 [%s] PARALLEL fire → research_agent (alongside plan_agent)",
                (state.get("job_id") or "")[:8] or "no-job")
    _node_start("research_agent", state, max_turns=MAX_RESEARCH_TURNS)

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=BRIEF_RESEARCH_TOOLS,
    )

    base_model, sem = await _build_tool_model(
        "orchestrator", provider=os.getenv("BRIEF_RESEARCH_PROVIDER")
    )

    survey, n_chunks = await _prefetch_corpus_survey(state, tools)
    logger.info("  📚 [research_agent] corpus survey pre-fetched (%d chars) n_chunks=%d",
                len(survey), n_chunks)

    messages: List[Any] = [
        SystemMessage(content=_research_system()),
        HumanMessage(content=(
            f"{survey}\n\n"
            f"Source ids in scope: {state.get('source_ids', [])}\n"
            f"User request: {state['request']}\n\n"
            "You already have the corpus survey above — do not re-fetch it. "
            "Begin RETRIEVAL immediately: identify the primary opinion and "
            "issue a large batch of parallel, case-name-anchored searches."
        )),
    ]

    evidence_store: Dict[str, Dict] = {}
    raw_answer = await _run_bounded_tool_loop(
        base_model, tools, messages, MAX_RESEARCH_TURNS, sem,
        "research_agent", state, evidence_store=evidence_store,
        countdown_from_turn=3,
        forced_stop_prompt=(
            "Retrieval is complete. Emit the final case dossier JSON now, using "
            "the evidence gathered. Return ONLY the JSON object."
        ),
    )
    try:
        dossier: Optional[Dict[str, Any]] = _parse_json(raw_answer)
    except Exception as exc:
        _node_warn("research_agent", state, f"dossier JSON parse failed: {exc}")
        dossier = None

    if not isinstance(dossier, dict):
        dossier = {}
    dossier.setdefault("case", {})
    dossier.setdefault("evidence_by_role", {})

    case = dossier["case"] if isinstance(dossier["case"], dict) else {}
    dossier["case"] = case

    await _try_save_artifact(
        state, "case_dossier",
        {"dossier": dossier, "evidence_chunks": len(evidence_store)},
        "orchestrator", "research_agent", "case_dossier",
        source_ids=state.get("source_ids"),
    )
    _node_done("research_agent", state,
               case=str(case.get("case_name", "?"))[:40],
               has_dissent=case.get("has_dissent"),
               n_evidence_chunks=len(evidence_store),
               n_roles_filled=sum(
                   1 for v in dossier["evidence_by_role"].values() if v
               ))
    return {"case_dossier": dossier, "evidence_store": evidence_store}


# ─────────────────────────────────────────────────────────────────────────────
# sync_barrier — join for the parallel plan_agent / research_agent branches
# ─────────────────────────────────────────────────────────────────────────────

async def sync_barrier(state: AgentState) -> Dict:
    """
    Join node. plan_agent and research_agent run in parallel from START; a node
    with two incoming static edges waits for both (standard LangGraph fan-in),
    so routing the unit fan-out from HERE guarantees every section writer sees
    both job_plan and the dossier.

    It also keeps conditional edges off fan-out nodes — the legacy graph hung
    routers directly on Send-parallel nodes, where the branch re-evaluates per
    task over an accumulating list.
    """
    job = (state.get("job_id") or "")[:8] or "no-job"
    has_plan = bool(state.get("job_plan"))
    case = (state.get("case_dossier") or {}).get("case") or {}
    logger.info(
        "🔗 [%s] sync_barrier — joined plan_agent + research_agent  "
        "job_plan=%s case=%s has_dissent=%s",
        job, "present" if has_plan else "MISSING (join failed?)",
        str(case.get("case_name", "?"))[:40], case.get("has_dissent"),
    )
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 3. section_generator — [Send×N] one per brief unit
# ─────────────────────────────────────────────────────────────────────────────

def dossier_to_generators(state: AgentState):
    """
    Fan out over the fixed BRIEF_UNITS table. Sections are known a priori, so
    nothing is discovered here — the only dynamic decision is dropping a
    conditional unit whose subject is absent (preserving brief_drafter's one
    real judgement: suppress `dissent` when the case has none).
    """
    dossier = state.get("case_dossier") or {}
    if not dossier.get("case"):
        logger.warning("dossier_to_generators: no dossier — routing to final_formatter")
        return "final_formatter"

    case = dossier.get("case") or {}
    units = [
        u for u in BRIEF_UNITS
        if not u.get("conditional_on") or bool(case.get(u["conditional_on"]))
    ]
    skipped = [u["unit_id"] for u in BRIEF_UNITS if u not in units]
    if skipped:
        logger.info("case_brief(compact) skipping unit(s) %s (absent in case)", skipped)

    logger.info("case_brief(compact) x%d node fan out for section_generator", len(units))
    return [Send("section_generator", {"unit": u, **state}) for u in units]


async def section_generator(state: Dict) -> Dict:
    """
    Write ONE brief unit in FINAL template form, self-grounded.

    Absorbs section_writer + section_grounder + critic + brief_revision_agent:
    the unit drafts and then re-checks its own claims against the evidence,
    with tools to close gaps rather than a downstream critic pass.

    Emitting final markdown (not a draft) is what removes the legacy pipeline's
    12,000-token re-synthesis call in final_formatter.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import BRIEF_VERIFIER_TOOLS
    from langchain_core.tools import tool as _tool_decorator

    unit: Dict[str, Any] = state["unit"]
    dossier: Dict[str, Any] = state.get("case_dossier") or {}
    job_plan: Dict[str, Any] = state.get("job_plan") or {}
    evidence_store: Dict[str, Dict] = state.get("evidence_store") or {}
    case: Dict[str, Any] = dossier.get("case") or {}
    unit_id = unit["unit_id"]

    _node_start("section_generator", state, unit=unit_id,
                roles=unit["roles"])

    # Evidence selection: union the chunk ids indexed under this unit's roles.
    by_role: Dict[str, Any] = dossier.get("evidence_by_role") or {}
    chunk_ids: List[str] = []
    for role in unit["roles"]:
        ids = by_role.get(role) or []
        if isinstance(ids, list):
            chunk_ids.extend(str(i) for i in ids)

    seen: set = set()
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
    ) or "(no evidence indexed for this unit — use grep_research_corpus to find support)"

    # Local, zero-cost grep over the FULL harvested corpus (not just this unit's
    # roles) — closure-defined, no ToolContext, no DB round trip.
    @_tool_decorator
    def grep_research_corpus(keyword: str) -> str:
        """Case-insensitive substring search over the research corpus already
        harvested for this brief (every chunk retrieved, not just the ones
        indexed to this section). Use this FIRST — it is free and instant —
        before any live database tool. Returns up to 10 matching excerpts with
        their chunk_id."""
        kw = keyword.lower()
        matches = [
            {"chunk_id": cid, "excerpt": e["content"][:400]}
            for cid, e in evidence_store.items()
            if kw in e["content"].lower()
        ][:10]
        return json.dumps(matches) if matches else json.dumps({"result": "no matches"})

    # verify_claim is INCLUDED here (attack_outline excludes it for latency):
    # a wrong holding is fatal in a brief, and because units run in parallel its
    # nested LLM call costs one call's latency, not one per unit serially.
    verifier_tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=[t for t in BRIEF_VERIFIER_TOOLS
                    if t in ("verify_claim", "find_supporting_evidence",
                             "get_citations_for")],
    )
    tools = [grep_research_corpus] + verifier_tools

    target = unit["target_words"]
    system = (
        "You are a T-14 law professor writing ONE unit of a law-school case "
        "brief. Emit the section(s) below in FINAL form — this output goes "
        "straight into the student's brief with no further rewriting.\n\n"
        "EMIT EXACTLY THIS SKELETON (same headings, same order, nothing "
        "outside it):\n\n"
        f"{unit['template']}\n\n"
        "SECTION GUIDANCE:\n"
        f"{unit['guidance']}\n\n"
        "FORMATTING:\n"
        "• `##` for Roman-numeral sections, `###` for subsections.\n"
        "• **Bold** every black-letter rule and every holding.\n"
        "• Bullet lists for elements, triggers, and factors.\n"
        "• No raw JSON, no code fences, no template artifacts, no preamble.\n"
        "• A heading marked [OMIT-IF-ABSENT] must be dropped ENTIRELY — "
        "heading and body — when the evidence does not support it. Never emit "
        "placeholder text. If EVERY section in your skeleton would be omitted, "
        f"return exactly `{OMIT_SECTION_MARKER}` and nothing else.\n"
        "• Leave the [OMIT-IF-ABSENT] marker itself out of your output.\n\n"
        "GROUNDING: every fact, rule, holding and quotation must be supported "
        "by the evidence provided. If something you want to state is not "
        "covered, call grep_research_corpus first (free, instant); use "
        "find_supporting_evidence / get_citations_for to reach the source, and "
        "verify_claim when you are unsure a legal assertion is actually "
        "supported. Most units need no tool calls. After drafting, RE-CHECK "
        "each assertion against the evidence and delete or soften anything "
        "unsupported — there is no editor downstream. Cite support inline as "
        "[chunk_id] after the proposition (these are stripped before the "
        "student sees them, so they cost you no length). Never invent a "
        "citation, a quotation, or a judge's name."
        + _word_budget_line(target, int(target * 1.4))
    )

    prompt = (
        f"CASE: {json.dumps(case, indent=2)}\n\n"
        f"One-line holding: {dossier.get('one_line_holding', '')}\n"
        f"Black-letter rule: {dossier.get('black_letter_rule', '')}\n"
        f"Doctrinal role: {dossier.get('doctrinal_role', '')}\n"
        f"Brief mode: {job_plan.get('brief_mode', 'single_case')} | "
        f"target length: {job_plan.get('target_length', 'standard')}\n"
        f"Contested positions (arguments ABOUT the case, not holdings): "
        f"{json.dumps(dossier.get('contested_positions') or [])}\n\n"
        f"EVIDENCE for this unit ({', '.join(unit['roles'])}):\n{evidence_text}"
    )

    base_model, sem = await _build_tool_model("orchestrator")
    messages: List[Any] = [SystemMessage(content=system), HumanMessage(content=prompt)]
    markdown = await _run_bounded_tool_loop(
        base_model, tools, messages, BRIEF_UNIT_MAX_TURNS, sem,
        "section_generator", state,
        countdown_from_turn=BRIEF_UNIT_MAX_TURNS,
        forced_stop_prompt=(
            "STOP calling tools. Write the final markdown for your section(s) "
            "NOW using only the evidence already gathered."
        ),
    )
    markdown = (markdown or "").strip()

    omitted = markdown.upper().startswith(OMIT_SECTION_MARKER) or not markdown
    if omitted:
        markdown = ""
        _node_warn("section_generator", state,
                   f"unit '{unit_id}' self-omitted (no supporting evidence)")

    block = {
        "unit_id": unit_id,
        "order": unit["order"],
        "markdown": markdown,
    }
    await _try_save_artifact(
        state, f"brief_unit:{unit_id}", block,
        "orchestrator", "section_generator", "brief_unit",
        source_ids=state.get("source_ids"),
    )
    _node_done("section_generator", state, unit=unit_id, chars=len(markdown))
    return {"brief_units": [block]}


section_generator.default_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 4. final_formatter — deterministic, zero LLM
# ─────────────────────────────────────────────────────────────────────────────

_MIN_UNIT_CHARS = 40

# Inline [chunk-uuid] citations make the generator's self-grounding checkable,
# but they are internal retrieval ids — stripped here at the presentation
# boundary. Raw markdown keeps them in state["brief_units"] and the ledger, so
# grounding stays auditable after the fact.
_UUID_PAT = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_CHUNK_CITE_RE = re.compile(
    r"[ \t]*\[\s*" + _UUID_PAT + r"(?:\s*[;,]\s*" + _UUID_PAT + r")*\s*\]"
)

_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
          "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]

# "## VII. Rule & Legal Test" -> captures the numeral and the title
_ROMAN_HEADING_RE = re.compile(r"^##\s+([IVXL]+)\.\s+(.*)$", re.M)


def _strip_chunk_citations(markdown: str) -> Tuple[str, int]:
    """Remove inline [chunk-uuid] citations and tidy the punctuation they leave
    behind. Returns (cleaned_markdown, n_removed)."""
    if not markdown:
        return markdown, 0
    n = len(_CHUNK_CITE_RE.findall(markdown))
    cleaned = _CHUNK_CITE_RE.sub("", markdown)
    cleaned = re.sub(r"[ \t]+([.,;:!?)])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+$", "", cleaned, flags=re.M)
    return cleaned, n


def _renumber_roman_headings(markdown: str) -> Tuple[str, int]:
    """
    Renumber `## <ROMAN>. Title` headings contiguously from I.

    The legacy pipeline emitted a fixed I-XVI template and asked the LLM to
    'omit if absent', which leaves gaps (…XI, XIII…) whenever a section drops
    out. nodes.py carried `_prune_and_renumber_sections` for exactly this but
    never called it. Here the omissions are deterministic, so the renumbering
    can be too.
    """
    idx = {"n": 0}

    def _sub(m: "re.Match") -> str:
        idx["n"] += 1
        numeral = _ROMAN[idx["n"] - 1] if idx["n"] <= len(_ROMAN) else str(idx["n"])
        return f"## {numeral}. {m.group(2)}"

    out = _ROMAN_HEADING_RE.sub(_sub, markdown)
    return out, idx["n"]


async def final_formatter(state: AgentState) -> Dict:
    """
    Deterministic assembly: title block, ordering, citation stripping, pruning,
    Roman renumbering, word count. Makes NO LLM call — the units already emitted
    final-form markdown, which is what retired the legacy 12,000-token
    re-synthesis pass.
    """
    dossier = state.get("case_dossier") or {}
    case = dossier.get("case") or {}
    units: List[Dict] = list(state.get("brief_units") or [])

    _node_start("final_formatter", state, n_units=len(units))

    units.sort(key=lambda u: u.get("order", 99))

    citations_removed = 0
    kept: List[Dict] = []
    dropped: List[str] = []
    for u in units:
        cleaned, n = _strip_chunk_citations(u.get("markdown", ""))
        citations_removed += n
        if len(cleaned.strip()) >= _MIN_UNIT_CHARS:
            u["clean_markdown"] = cleaned.strip()
            kept.append(u)
        else:
            dropped.append(u.get("unit_id", "?"))
    if citations_removed:
        logger.info("  🧹 [final_formatter] stripped %d inline chunk citation(s)",
                    citations_removed)
    if dropped:
        logger.info("  ✂️ [final_formatter] pruned empty unit(s): %s", dropped)

    case_name = case.get("case_name") or "Case Brief"
    header: List[str] = [f"# {case_name}"]
    meta_bits = []
    if case.get("citation"):
        meta_bits.append(f"**Citation:** {case['citation']}  ")
    court_line = " ".join(
        p for p in (
            f"**Court:** {case['court']} " if case.get("court") else "",
            f"**Decided:** {case['year']}" if case.get("year") else "",
        ) if p
    ).strip()
    if court_line:
        meta_bits.append(court_line + "  ")
    disp_line = " ".join(
        p for p in (
            f"**Disposition:** {case['disposition']} " if case.get("disposition") else "",
            f"**Opinion:** {case['opinion_author']}" if case.get("opinion_author") else "",
        ) if p
    ).strip()
    if disp_line:
        meta_bits.append(disp_line)
    header += meta_bits

    if kept:
        body = "\n\n---\n\n".join(u["clean_markdown"] for u in kept)
    else:
        # Total generation failure — surface what research found rather than
        # returning an empty note.
        body = (
            "*(Brief generation produced no sections — research summary below.)*\n\n"
            f"**Holding:** {dossier.get('one_line_holding', '(not established)')}\n\n"
            f"**Rule:** {dossier.get('black_letter_rule', '(not established)')}"
        )

    final_md = "\n".join(header) + "\n\n---\n\n" + body
    final_md, n_sections = _renumber_roman_headings(final_md)

    total_words = len(final_md.split())
    est_pages = max(1, round(total_words / 250))
    # Token/cost figures are deliberately omitted: the legacy footer printed
    # them from `budget`, which _add_budget never populated, so it always
    # reported 0 tokens / $0.0000.
    final_md += (
        f"\n\n<!-- case-brief-agent compact | {n_sections} sections | "
        f"{len(kept)}/{len(units)} units | words: {total_words} (~{est_pages}p) -->"
    )

    await _try_save_artifact(
        state, "final_output",
        {"markdown_length": len(final_md), "word_count": total_words,
         "sections": n_sections},
        "tool_only", "final_formatter", "final_output",
        source_ids=state.get("source_ids"),
    )
    _node_done("final_formatter", state,
               sections=n_sections, total_words=total_words, est_pages=est_pages)
    # assembled_brief kept populated for any legacy reader of that key.
    return {"final_output": final_md, "assembled_brief": final_md}
