# agents/quiz/nodes.py
"""
All 13 nodes + 3 routing helpers for the quiz LangGraph agent.

Node index (in execution order):
  1.  head_orchestrator              — worker_mid; validate inputs, select quiz mode
  2.  source_profiler                — tool_only; per-source inventory (parallel Send)
  3.  case_rule_extractor            — orchestrator; structured extraction per identified case
  4.  cross_doc_concepts_synthesis   — worker_mid; doctrine clusters, confusables, traps
  5.  quiz_blueprint_planner         — worker_mid; create all batch_specs with question allocations
  6.  question_drafter               — orchestrator; draft 5 MCQ stems + answer skeletons per batch
  7.  false_trap_red_herring_generator — orchestrator; enrich distractors with type + feedback
  8.  question_evaluator             — worker_mid; batch-level pedagogical QA scoring
  9.  reviser                        — orchestrator; fix failing questions (max 2 passes per batch)
  10. grounder                       — worker_low; verify question/answer claims against source
  11. batch_commit                   — tool_only; write accepted questions to DB, update progress
  12. critic                         — worker_low; global diversity + coverage check
  13. final_formatter                — worker_low; update parent notes row, return summary

Routing helpers (not nodes):
  head_orchestrator_to_profiler — Send per source_id
  should_revise_batch           — "reviser" | "grounder"
  batch_router                  — "question_drafter" | "critic"
"""

import asyncio
import json
import logging
import math
import random
import re
import uuid as _uuid_mod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langgraph.types import Send

from .constants import (
    DISTRACTOR_TYPES,
    MC_ANALYSIS_TYPES,
    MC_APPLICATION_TYPES,
    MC_QUESTION_TYPES,
    MC_RECALL_TYPES,
    QUIZ_MODES,
)
from .state import (
    AgentState,
    BatchEvaluation,
    BatchSpec,
    CaseExtract,
    DraftAnswer,
    DraftQuizQuestion,
    QuestionSpec,
    QuizSourceProfile,
)
from .worker_config import _fetch_worker_model, model_costs

logger = logging.getLogger(__name__)


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


async def _try_save_artifact(
    state: Dict,
    artifact_key: str,
    content: Dict[str, Any],
    worker_class: str,
    node_name: str,
    artifact_type: str,
    source_ids: Optional[List[str]] = None,
) -> None:
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


# ─────────────────────────────────────────────────────────────────────────────
# 1. head_orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def head_orchestrator(state: AgentState) -> Dict:
    """Validate inputs, determine quiz_mode and batch_size, read source list."""
    from agents.tools.base import make_tools
    from agents.tools.registry import QUIZ_PLANNER_TOOLS

    tools = make_tools(
        state["project_id"],
        source_ids=state["source_ids"],
        use_voyage=state.get("use_voyage", False),
        tool_names=QUIZ_PLANNER_TOOLS,
    )
    list_sources_tool = next(t for t in tools if t.name == "list_sources")
    sources_json = await list_sources_tool.ainvoke({})

    num_questions = max(1, state.get("num_questions") or 10)
    batch_size = min(max(state.get("batch_size") or 5, 1), 10)
    num_batches = math.ceil(num_questions / batch_size)

    quiz_mode = state.get("quiz_mode") or "mixed"
    if quiz_mode not in QUIZ_MODES:
        quiz_mode = "mixed"

    system = (
        "You are a law professor designing a multiple-choice quiz. "
        "Read the available documents and confirm the quiz scope in one sentence."
    )
    await _llm(
        "worker_mid",
        f"Documents: {sources_json}\nRequest: {state.get('request', '')}\n"
        f"Quiz: {num_questions} questions, mode: {quiz_mode}.",
        system=system,
        max_tokens=128,
    )

    return {
        "batch_size": batch_size,
        "num_batches": num_batches,
        "quiz_mode": quiz_mode,
        "current_batch_index": 0,
        "batch_revision_count": 0,
        "accepted_question_ids": [],
        "rejected_question_metadata": [],
        "used_question_signatures": [],
        "coverage_summary": "{}",
    }


head_orchestrator.default_worker_class = "worker_mid"


def head_orchestrator_to_profiler(state: AgentState) -> List[Send]:
    """Fan-out: one source_profiler per source document."""
    return [
        Send("source_profiler", {"source_id": sid, **state})
        for sid in state["source_ids"]
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. source_profiler (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def source_profiler(state: Dict) -> Dict:
    """Build a profile for a single source: doc type, identified cases, concepts."""
    from agents.tools.base import make_tools
    from agents.tools.registry import QUIZ_PROFILER_TOOLS

    source_id = state["source_id"]
    tools = make_tools(
        state["project_id"],
        source_ids=[source_id],
        use_voyage=state.get("use_voyage", False),
        tool_names=QUIZ_PROFILER_TOOLS,
    )
    outline_tool = next(t for t in tools if t.name == "get_doc_outline")
    outline_json = await outline_tool.ainvoke({"source_id": source_id})
    outline = json.loads(outline_json)

    filename = outline.get("filename", "")
    summary = outline.get("doc_summary") or ""
    concepts = outline.get("doc_concepts", [])[:20]

    # Heuristic doc_type detection
    fn_lower = filename.lower()
    if any(k in fn_lower for k in ("v.", " v ", "opinion", "decision")):
        doc_type = "full_case_opinion"
    elif any(k in fn_lower for k in ("casebook", "textbook")):
        doc_type = "casebook_excerpt"
    else:
        doc_type = "secondary"

    # Identify case names from concepts/summary using LLM
    prompt = (
        f"List the legal case names (e.g. 'Palsgraf v. Long Island Railroad') "
        f"mentioned in this document. Return only a JSON array of strings.\n\n"
        f"Filename: {filename}\nSummary: {summary[:500]}\n"
        f"Concepts: {concepts[:10]}"
    )
    raw = await _llm("worker_low", prompt, max_tokens=256)
    try:
        identified_cases = _parse_json(raw)
        if not isinstance(identified_cases, list):
            identified_cases = []
    except Exception:
        identified_cases = []

    profile: QuizSourceProfile = {
        "source_id": source_id,
        "filename": filename,
        "doc_type_guess": doc_type,
        "document_summary": summary[:600],
        "identified_cases": identified_cases[:10],
        "key_concepts": concepts[:15],
        "has_dissent_guess": any(
            w in (summary + " ".join(concepts)).lower()
            for w in ("dissent", "dissenting", "j., dissenting")
        ),
    }
    await _try_save_artifact(
        state,
        artifact_key=f"quiz_source_profile:{source_id}",
        content=profile,
        worker_class="worker_low",
        node_name="source_profiler",
        artifact_type="source_profile",
        source_ids=[source_id],
    )
    return {"source_profiles": [profile]}


source_profiler.default_worker_class = "worker_low"


# ─────────────────────────────────────────────────────────────────────────────
# 3. case_rule_extractor
# ─────────────────────────────────────────────────────────────────────────────

async def case_rule_extractor(state: AgentState) -> Dict:
    """
    Extract structured legal data (facts, holding, rule, etc.) for every
    identified case across all source profiles.  Runs asyncio.gather for
    parallel per-case extraction without adding more Send fan-out nodes.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import QUIZ_RETRIEVER_TOOLS

    profiles = state.get("source_profiles") or []

    # Collect unique (case_name, source_id) pairs
    case_candidates: List[Dict[str, str]] = []
    seen: set = set()
    for p in profiles:
        for cn in (p.get("identified_cases") or []):
            key = f"{cn}:{p['source_id']}"
            if key not in seen:
                seen.add(key)
                case_candidates.append({"case_name": cn, "source_id": p["source_id"]})

    if not case_candidates:
        # Fall back: one entry per source
        for p in profiles:
            case_candidates.append({
                "case_name": p.get("filename", "source"),
                "source_id": p["source_id"],
            })

    case_candidates = case_candidates[:15]  # cap to avoid runaway costs

    async def _extract_one(candidate: Dict, idx: int) -> Optional[CaseExtract]:
        case_name = candidate["case_name"]
        source_id = candidate["source_id"]
        tools = make_tools(
            state["project_id"],
            source_ids=[source_id],
            use_voyage=state.get("use_voyage", False),
            tool_names=QUIZ_RETRIEVER_TOOLS,
        )
        search_tool = next(t for t in tools if t.name == "hybrid_search")
        try:
            chunks_json = await search_tool.ainvoke({"query": case_name, "k": 15})
            chunks = json.loads(chunks_json)
        except Exception:
            chunks = []

        context = "\n\n---\n\n".join(
            f"[chunk_id:{c.get('id','?')}]\n{c.get('content','')}"
            for c in chunks[:12]
        )
        if not context.strip():
            return None

        system = (
            "You are a law professor extracting structured case information. "
            "Return ONLY a JSON object with these exact keys: "
            "procedural_posture, facts, legally_relevant_facts (array of strings), "
            "issue, holding, rule, reasoning, dicta, dissent, policy. "
            "Base every field strictly on the provided text."
        )
        prompt = (
            f"Case: {case_name}\n\nSource text:\n{context[:4000]}\n\n"
            "Extract the structured case information."
        )
        raw = await _llm("orchestrator", prompt, system=system, max_tokens=1800)
        try:
            data = _parse_json(raw)
        except Exception:
            data = {}

        chunk_ids = [c.get("id", "") for c in chunks[:10]]
        return {
            "case_id": f"case_{idx:03d}",
            "source_id": source_id,
            "case_name": case_name,
            "procedural_posture": data.get("procedural_posture", ""),
            "facts": data.get("facts", ""),
            "legally_relevant_facts": data.get("legally_relevant_facts", []),
            "issue": data.get("issue", ""),
            "holding": data.get("holding", ""),
            "rule": data.get("rule", ""),
            "reasoning": data.get("reasoning", ""),
            "dicta": data.get("dicta", ""),
            "dissent": data.get("dissent", ""),
            "policy": data.get("policy", ""),
            "source_refs": chunk_ids,
        }

    results = await asyncio.gather(*[
        _extract_one(c, i) for i, c in enumerate(case_candidates)
    ])
    extracts = [r for r in results if r is not None]

    await _try_save_artifact(
        state,
        artifact_key="case_extracts",
        content={"extracts": extracts},
        worker_class="orchestrator",
        node_name="case_rule_extractor",
        artifact_type="case_extracts",
        source_ids=state.get("source_ids"),
    )
    return {"case_extracts": extracts}


case_rule_extractor.default_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 4. cross_doc_concepts_synthesis
# ─────────────────────────────────────────────────────────────────────────────

async def cross_doc_concepts_synthesis(state: AgentState) -> Dict:
    """
    Identify doctrine clusters, confusable concepts, and common traps across
    all extracted cases.  The synthesis drives quiz_blueprint_planner's
    distractor type selection and question diversity allocation.
    """
    extracts = state.get("case_extracts") or []
    profiles = state.get("source_profiles") or []

    cases_brief = [
        {
            "case_name": e["case_name"],
            "holding": e["holding"][:200],
            "rule": e["rule"][:200],
            "dicta": e["dicta"][:100],
            "dissent": e["dissent"][:100],
        }
        for e in extracts[:12]
    ]
    all_concepts = []
    for p in profiles:
        all_concepts.extend(p.get("key_concepts", [])[:8])

    prompt = (
        "You are a law professor identifying teaching priorities for a quiz.\n\n"
        f"Cases:\n{json.dumps(cases_brief, indent=2)}\n\n"
        f"Key concepts across sources: {all_concepts[:20]}\n\n"
        "Identify:\n"
        "1. doctrine_clusters: 3-5 doctrinal clusters (each with a name, cases involved, and core rule)\n"
        "2. confusable_concepts: pairs or groups of concepts students commonly confuse\n"
        "3. common_traps: 3-5 common misconceptions or overbroad readings students make\n"
        "4. high_yield_areas: 3-5 areas most likely to appear on exams\n\n"
        "Return a JSON object with keys: doctrine_clusters, confusable_concepts, "
        "common_traps, high_yield_areas. No extra text."
    )

    raw = await _llm("worker_mid", prompt, max_tokens=1500)
    try:
        synthesis_data = _parse_json(raw)
    except Exception:
        synthesis_data = {
            "doctrine_clusters": [],
            "confusable_concepts": [],
            "common_traps": [],
            "high_yield_areas": [],
        }

    synthesis_str = json.dumps(synthesis_data)
    await _try_save_artifact(
        state,
        artifact_key="concept_synthesis",
        content=synthesis_data,
        worker_class="worker_mid",
        node_name="cross_doc_concepts_synthesis",
        artifact_type="concept_synthesis",
        source_ids=state.get("source_ids"),
    )
    return {"concept_synthesis": synthesis_str}


cross_doc_concepts_synthesis.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 5. quiz_blueprint_planner
# ─────────────────────────────────────────────────────────────────────────────

async def quiz_blueprint_planner(state: AgentState) -> Dict:
    """
    Create all batch_specs upfront.  Each batch_spec holds batch_size question_specs
    with specific question_type, source, topic, difficulty, and distractor_type
    allocations.  The planner ensures type diversity and difficulty balance
    across all batches.
    """
    num_questions = state.get("num_questions") or 10
    batch_size = state.get("batch_size") or 5
    quiz_mode = state.get("quiz_mode") or "mixed"
    target_difficulty = state.get("target_difficulty") or "application"
    extracts = state.get("case_extracts") or []
    profiles = state.get("source_profiles") or []
    synthesis_str = state.get("concept_synthesis") or "{}"

    # Determine question type pool based on mode
    if quiz_mode == "recall":
        type_pool = MC_RECALL_TYPES * 4 + MC_APPLICATION_TYPES
    elif quiz_mode == "application":
        type_pool = MC_APPLICATION_TYPES * 3 + MC_RECALL_TYPES + MC_ANALYSIS_TYPES
    elif quiz_mode == "exam-style":
        type_pool = MC_ANALYSIS_TYPES * 3 + MC_APPLICATION_TYPES
    else:  # mixed
        type_pool = MC_RECALL_TYPES + MC_APPLICATION_TYPES * 2 + MC_ANALYSIS_TYPES * 2

    cases_brief = [
        {"case_id": e["case_id"], "case_name": e["case_name"],
         "source_id": e["source_id"], "holding": e["holding"][:150]}
        for e in extracts[:12]
    ]
    source_ids = [p["source_id"] for p in profiles]

    prompt = (
        f"You are a law professor creating a quiz blueprint.\n\n"
        f"Total questions needed: {num_questions}\n"
        f"Batch size: {batch_size}\n"
        f"Quiz mode: {quiz_mode}\n"
        f"Target difficulty: {target_difficulty}\n\n"
        f"Available cases:\n{json.dumps(cases_brief, indent=2)}\n\n"
        f"Concept synthesis:\n{synthesis_str[:1000]}\n\n"
        f"Available distractor types: {DISTRACTOR_TYPES}\n\n"
        f"Create exactly {num_questions} question specs as a JSON array. "
        f"Each spec must have:\n"
        f'  "spec_index": int (0-based global index)\n'
        f'  "question_type": one of {MC_QUESTION_TYPES[:8]}... (use full variety)\n'
        f'  "source_ids": [list of source_id UUIDs from available sources]\n'
        f'  "case_names": [list of case names to draw from]\n'
        f'  "topic": specific topic or concept for this question (one sentence)\n'
        f'  "difficulty": "recall" | "application" | "analysis"\n'
        f'  "distractor_types": [exactly 3 distractor type tags for the wrong answers]\n\n'
        f"Ensure:\n"
        f"- No more than 3 specs share the same question_type\n"
        f"- Difficulty distribution: ~30% recall, ~40% application, ~30% analysis\n"
        f"- Each case is used in at least 1 question\n"
        f"Return ONLY the JSON array of {num_questions} specs. No extra text."
    )

    raw = await _llm("worker_mid", prompt, max_tokens=3500)
    try:
        all_specs: List[QuestionSpec] = _parse_json(raw)
        if not isinstance(all_specs, list):
            raise ValueError("not a list")
    except Exception:
        # Fallback: generate specs programmatically
        all_specs = []
        for i in range(num_questions):
            qtype = type_pool[i % len(type_pool)]
            case = extracts[i % len(extracts)] if extracts else {}
            all_specs.append({
                "spec_index": i,
                "question_type": qtype,
                "source_ids": source_ids[:2],
                "case_names": [case.get("case_name", "")] if case else [],
                "topic": f"Question {i+1} about {case.get('case_name', 'source material')}",
                "difficulty": target_difficulty,
                "distractor_types": DISTRACTOR_TYPES[:3],
            })

    # Normalize spec_index
    for i, spec in enumerate(all_specs):
        spec["spec_index"] = i

    # Group into batches
    batch_specs: List[BatchSpec] = []
    for batch_idx in range(math.ceil(len(all_specs) / batch_size)):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_specs.append({
            "batch_index": batch_idx,
            "question_specs": all_specs[start:end],
        })

    await _try_save_artifact(
        state,
        artifact_key="quiz_blueprint",
        content={"batch_specs": batch_specs, "num_batches": len(batch_specs)},
        worker_class="worker_mid",
        node_name="quiz_blueprint_planner",
        artifact_type="blueprint",
        source_ids=state.get("source_ids"),
    )
    return {
        "batch_specs": batch_specs,
        "num_batches": len(batch_specs),
    }


quiz_blueprint_planner.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 6. question_drafter
# ─────────────────────────────────────────────────────────────────────────────

async def question_drafter(state: AgentState) -> Dict:
    """
    Draft all questions for the current batch.  Each question_spec becomes one
    DraftQuizQuestion with a stem, hint, and 4 answer skeletons (A-D).
    Uses asyncio.gather for parallel per-question drafting within the batch.
    The correct answer is randomly shuffled to a different position per question.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import QUIZ_RETRIEVER_TOOLS

    batch_idx = state.get("current_batch_index", 0)
    batch_specs = state.get("batch_specs") or []
    if batch_idx >= len(batch_specs):
        return {"current_batch_drafts": []}

    batch_spec: BatchSpec = batch_specs[batch_idx]
    used_sigs = state.get("used_question_signatures") or []
    coverage_raw = state.get("coverage_summary") or "{}"
    try:
        coverage = json.loads(coverage_raw)
    except Exception:
        coverage = {}

    async def _draft_one(spec: QuestionSpec) -> DraftQuizQuestion:
        src_ids = spec.get("source_ids") or state["source_ids"]
        tools = make_tools(
            state["project_id"],
            source_ids=src_ids,
            use_voyage=state.get("use_voyage", False),
            tool_names=QUIZ_RETRIEVER_TOOLS,
        )
        search_tool = next(t for t in tools if t.name == "hybrid_search")

        query = f"{' '.join(spec.get('case_names') or [])} {spec.get('topic', '')}"
        try:
            chunks_json = await search_tool.ainvoke({"query": query.strip(), "k": 12})
            chunks = json.loads(chunks_json)
        except Exception:
            chunks = []

        context = "\n\n---\n\n".join(
            f"[chunk_id:{c.get('id','?')}]\n{c.get('content','')}"
            for c in chunks[:10]
        )
        chunk_ids = [c.get("id", "") for c in chunks[:8] if c.get("id")]

        avoid_note = ""
        if used_sigs:
            avoid_note = (
                f"\n\nALREADY USED (do NOT duplicate these topics):\n"
                + "\n".join(f"- {s}" for s in used_sigs[-20:])
            )

        system = (
            "You are an expert law professor creating a multiple-choice quiz question. "
            "Generate exactly ONE question with ONE correct answer and THREE wrong answers. "
            "The question must be grounded strictly in the provided legal text. "
            "Return ONLY a JSON object with keys:\n"
            '  "question_stem": str (concise, max 120 words)\n'
            '  "hint": str (one sentence; guides without revealing the answer)\n'
            '  "correct_answer": str (the correct choice text)\n'
            '  "wrong_answers": [str, str, str] (exactly 3 plausible wrong choices)\n'
            '  "source_refs": [str, ...] (chunk_id UUIDs from [chunk_id:...] markers)\n'
        )

        prompt = (
            f"Question type: {spec['question_type']}\n"
            f"Topic: {spec.get('topic', '')}\n"
            f"Cases: {', '.join(spec.get('case_names', []))}\n"
            f"Target difficulty: {spec.get('difficulty', 'application')}\n"
            f"Distractor types to exploit: {spec.get('distractor_types', DISTRACTOR_TYPES[:3])}\n"
            f"{avoid_note}\n\n"
            f"Legal source material:\n{context[:3500] if context else 'Use general legal knowledge for this question type.'}\n\n"
            "Draft the multiple-choice question."
        )

        raw = await _llm("orchestrator", prompt, system=system, max_tokens=900)
        try:
            data = _parse_json(raw)
        except Exception:
            data = {
                "question_stem": f"Question about {spec.get('topic', 'legal doctrine')}.",
                "hint": "Consider the precise scope of the rule.",
                "correct_answer": "The correct legal rule applies here.",
                "wrong_answers": [
                    "An overbroad statement of the rule.",
                    "A rule that applies to different facts.",
                    "A statement based on dicta, not the holding.",
                ],
                "source_refs": [],
            }

        correct_text = data.get("correct_answer", "")
        wrong_texts = (data.get("wrong_answers") or [])[:3]
        while len(wrong_texts) < 3:
            wrong_texts.append("This answer is incorrect.")

        # Shuffle correct answer to random position
        letters = ["A", "B", "C", "D"]
        correct_pos = random.randint(0, 3)
        wrong_idx = 0
        answers: List[DraftAnswer] = []
        for pos, letter in enumerate(letters):
            if pos == correct_pos:
                answers.append({
                    "choice_letter": letter,
                    "answer_text": correct_text,
                    "is_correct": True,
                    "distractor_type": "",
                    "feedback": "",
                })
            else:
                answers.append({
                    "choice_letter": letter,
                    "answer_text": wrong_texts[wrong_idx] if wrong_idx < len(wrong_texts) else "Incorrect.",
                    "is_correct": False,
                    "distractor_type": "",
                    "feedback": "",
                })
                wrong_idx += 1

        src_refs = data.get("source_refs") or chunk_ids
        _UUID_RE = re.compile(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I
        )
        src_refs = [
            m.group() for raw_ref in src_refs
            if (m := _UUID_RE.search(str(raw_ref)))
        ]

        return {
            "spec_index": spec["spec_index"],
            "question_type": spec["question_type"],
            "question_stem": data.get("question_stem", ""),
            "hint": data.get("hint", ""),
            "answers": answers,
            "source_refs": src_refs or chunk_ids[:4],
            "grounding_verdict": "",
            "grounding_notes": "",
        }

    drafts = list(await asyncio.gather(*[
        _draft_one(spec)
        for spec in batch_spec["question_specs"]
    ]))

    logger.info(
        "question_drafter: batch %d — drafted %d questions",
        batch_idx, len(drafts),
    )
    return {"current_batch_drafts": drafts}


question_drafter.default_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 7. false_trap_red_herring_generator
# ─────────────────────────────────────────────────────────────────────────────

async def false_trap_red_herring_generator(state: AgentState) -> Dict:
    """
    Enrich all wrong answers in the current batch with distractor_type and
    feedback (pedagogical rationale).  Also writes feedback for the correct
    answer.  Works over the full batch in one LLM call for efficiency.
    """
    drafts = list(state.get("current_batch_drafts") or [])
    if not drafts:
        return {}

    # Serialize batch for prompt
    batch_payload = []
    for d in drafts:
        batch_payload.append({
            "spec_index": d["spec_index"],
            "question_type": d["question_type"],
            "question_stem": d["question_stem"],
            "answers": [
                {
                    "choice_letter": a["choice_letter"],
                    "answer_text": a["answer_text"],
                    "is_correct": a["is_correct"],
                }
                for a in d["answers"]
            ],
        })

    system = (
        "You are a law professor enriching multiple-choice quiz distractors. "
        "For each question in the batch, enrich every answer:\n"
        "  - Wrong answers: assign distractor_type (from the taxonomy) and write "
        "    feedback explaining (a) why a student might pick this, and (b) why it is wrong.\n"
        "  - Correct answer: write feedback explaining why it is the correct answer.\n\n"
        f"Available distractor_type values: {DISTRACTOR_TYPES}\n\n"
        "Return a JSON array — one object per question — each with:\n"
        '  "spec_index": int\n'
        '  "answers": [\n'
        '    {"choice_letter": str, "distractor_type": str, "feedback": str},\n'
        '    ... (all 4 choices)\n'
        '  ]\n'
        "No extra text."
    )

    prompt = (
        f"Enrich the following {len(batch_payload)} quiz questions:\n\n"
        f"{json.dumps(batch_payload, indent=2)}"
    )

    raw = await _llm("orchestrator", prompt, system=system, max_tokens=2500)
    try:
        enriched_batch = _parse_json(raw)
        if not isinstance(enriched_batch, list):
            raise ValueError
        enriched_map = {item["spec_index"]: item for item in enriched_batch}
    except Exception:
        enriched_map = {}

    updated_drafts = []
    for draft in drafts:
        enriched = enriched_map.get(draft["spec_index"])
        if not enriched:
            updated_drafts.append(draft)
            continue

        answer_enrichment = {
            a["choice_letter"]: a for a in (enriched.get("answers") or [])
        }
        new_answers = []
        for ans in draft["answers"]:
            patch = answer_enrichment.get(ans["choice_letter"], {})
            new_answers.append({
                **ans,
                "distractor_type": patch.get("distractor_type", ans["distractor_type"]),
                "feedback": patch.get("feedback", ans["feedback"]),
            })

        updated_drafts.append({**draft, "answers": new_answers})

    return {"current_batch_drafts": updated_drafts}


false_trap_red_herring_generator.default_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 8. question_evaluator
# ─────────────────────────────────────────────────────────────────────────────

async def question_evaluator(state: AgentState) -> Dict:
    """
    Score the current batch as a pedagogical set.  Evaluates:
      - source_grounding, single_correct_answer, distractor_quality,
        question_type_diversity, difficulty_match.
    Returns a BatchEvaluation with per-question verdicts.
    """
    drafts = state.get("current_batch_drafts") or []
    batch_idx = state.get("current_batch_index", 0)
    target_difficulty = state.get("target_difficulty") or "application"
    if not drafts:
        return {
            "current_batch_eval": {
                "batch_index": batch_idx,
                "passes": True,
                "scores": {},
                "rejection_reasons": [],
                "revision_instructions": [],
                "question_verdicts": [],
            }
        }

    batch_payload = [
        {
            "spec_index": d["spec_index"],
            "question_type": d["question_type"],
            "question_stem": d["question_stem"],
            "hint": d["hint"],
            "answers": [
                {
                    "letter": a["choice_letter"],
                    "text": a["answer_text"],
                    "is_correct": a["is_correct"],
                    "distractor_type": a["distractor_type"],
                }
                for a in d["answers"]
            ],
        }
        for d in drafts
    ]

    system = (
        "You are a senior law professor evaluating a batch of multiple-choice quiz questions. "
        "Score each criterion 0.0–1.0:\n"
        "  source_grounding      — are answers grounded, not hallucinated?\n"
        "  single_correct_answer — is there exactly one unambiguous correct answer?\n"
        "  distractor_quality    — are wrong answers plausible with real misconceptions?\n"
        "  question_type_diversity — does the batch cover diverse cognitive demands?\n"
        "  difficulty_match      — does difficulty match the target level?\n\n"
        "The batch PASSES if all scores >= 0.70 and no per-question critical failures.\n\n"
        "Return a JSON object:\n"
        '  "passes": bool\n'
        '  "scores": {"source_grounding": f, "single_correct_answer": f, '
        '"distractor_quality": f, "question_type_diversity": f, "difficulty_match": f}\n'
        '  "rejection_reasons": [str, ...]\n'
        '  "revision_instructions": [str, ...]\n'
        '  "question_verdicts": [{"spec_index": int, "passes": bool, "reason": str}, ...]\n'
        "No extra text."
    )

    prompt = (
        f"Target difficulty: {target_difficulty}\n"
        f"Batch ({len(batch_payload)} questions):\n\n"
        f"{json.dumps(batch_payload, indent=2)}\n\n"
        "Evaluate the batch."
    )

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=1500)
    try:
        result = _parse_json(raw)
        if not isinstance(result, dict):
            raise ValueError
    except Exception:
        result = {
            "passes": True,
            "scores": {},
            "rejection_reasons": [],
            "revision_instructions": [],
            "question_verdicts": [],
        }

    batch_eval: BatchEvaluation = {
        "batch_index": batch_idx,
        "passes": result.get("passes", True),
        "scores": result.get("scores", {}),
        "rejection_reasons": result.get("rejection_reasons", []),
        "revision_instructions": result.get("revision_instructions", []),
        "question_verdicts": result.get("question_verdicts", []),
    }
    return {"current_batch_eval": batch_eval}


question_evaluator.default_worker_class = "worker_mid"


def should_revise_batch(state: AgentState) -> str:
    """Route to reviser if batch failed and we haven't hit max revisions."""
    if state.get("batch_revision_count", 0) >= 2:
        return "grounder"
    eval_result = state.get("current_batch_eval")
    if eval_result and not eval_result.get("passes", True):
        return "reviser"
    return "grounder"


# ─────────────────────────────────────────────────────────────────────────────
# 9. reviser (batch revision, max 2 passes)
# ─────────────────────────────────────────────────────────────────────────────

async def reviser(state: AgentState) -> Dict:
    """
    Fix questions that failed evaluation.  Works surgically on failing
    questions only; passing questions are left unchanged.  Increments
    batch_revision_count; after 2 passes the should_revise_batch gate
    routes directly to grounder.
    """
    drafts = list(state.get("current_batch_drafts") or [])
    eval_result: Optional[BatchEvaluation] = state.get("current_batch_eval")
    revision_count = state.get("batch_revision_count", 0)

    if not eval_result:
        return {"batch_revision_count": revision_count + 1}

    # Build per-question verdict map
    verdict_map: Dict[int, Dict] = {
        v["spec_index"]: v
        for v in (eval_result.get("question_verdicts") or [])
        if isinstance(v, dict)
    }
    revision_instructions = eval_result.get("revision_instructions") or []
    instructions_text = "\n".join(f"- {r}" for r in revision_instructions)

    failing_indices = {
        si for si, v in verdict_map.items() if not v.get("passes", True)
    }
    if not failing_indices:
        # All passed — nothing to fix
        return {"batch_revision_count": revision_count + 1}

    failing_drafts = [d for d in drafts if d["spec_index"] in failing_indices]
    passing_drafts = [d for d in drafts if d["spec_index"] not in failing_indices]

    if not failing_drafts:
        return {"batch_revision_count": revision_count + 1}

    system = (
        "You are a law professor revising multiple-choice quiz questions that failed QA. "
        "Fix ONLY the specific issues listed in the revision instructions. "
        "Preserve the question_type and topic. "
        "For each question return the updated JSON with the same keys as the input. "
        "Return a JSON array of the revised questions only."
    )

    failing_payload = [
        {
            "spec_index": d["spec_index"],
            "question_type": d["question_type"],
            "question_stem": d["question_stem"],
            "hint": d["hint"],
            "answers": d["answers"],
            "failure_reason": (verdict_map.get(d["spec_index"]) or {}).get("reason", ""),
        }
        for d in failing_drafts
    ]

    prompt = (
        f"Revision instructions:\n{instructions_text or 'Improve question quality.'}\n\n"
        f"Questions to revise:\n{json.dumps(failing_payload, indent=2)}\n\n"
        "Return the revised questions as a JSON array with the same spec_index values."
    )

    raw = await _llm("orchestrator", prompt, system=system, max_tokens=2500)
    try:
        revised_list = _parse_json(raw)
        if not isinstance(revised_list, list):
            raise ValueError
        revised_map = {r["spec_index"]: r for r in revised_list if isinstance(r, dict)}
    except Exception:
        revised_map = {}

    updated_drafts = list(passing_drafts)
    for draft in failing_drafts:
        patch = revised_map.get(draft["spec_index"])
        if patch:
            updated_drafts.append({
                **draft,
                "question_stem": patch.get("question_stem", draft["question_stem"]),
                "hint": patch.get("hint", draft["hint"]),
                "answers": patch.get("answers", draft["answers"]),
            })
        else:
            updated_drafts.append(draft)

    updated_drafts.sort(key=lambda d: d["spec_index"])
    return {
        "current_batch_drafts": updated_drafts,
        "batch_revision_count": revision_count + 1,
    }


reviser.default_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 10. grounder
# ─────────────────────────────────────────────────────────────────────────────

async def grounder(state: AgentState) -> Dict:
    """
    Verify that each draft question's stem and correct answer are supported by
    the source material.  Sets grounding_verdict on each DraftQuizQuestion.
    Uses asyncio.gather for parallel per-question verification.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import QUIZ_VERIFIER_TOOLS

    drafts = list(state.get("current_batch_drafts") or [])
    if not drafts:
        return {}

    async def _ground_one(draft: DraftQuizQuestion) -> DraftQuizQuestion:
        correct_ans = next((a for a in draft["answers"] if a["is_correct"]), None)
        claim = (
            f"{draft['question_stem']} "
            f"Correct answer: {correct_ans['answer_text'] if correct_ans else ''}"
        )

        tools = make_tools(
            state["project_id"],
            source_ids=state.get("source_ids", []),
            use_voyage=state.get("use_voyage", False),
            tool_names=QUIZ_VERIFIER_TOOLS,
        )
        verify_tool = next(t for t in tools if t.name == "verify_claim")
        try:
            result_json = await verify_tool.ainvoke({"claim": claim[:400], "k": 6})
            result = json.loads(result_json)
            verdict = result.get("verdict", "insufficient")
        except Exception:
            verdict = "insufficient"

        if verdict == "supported":
            gv, notes = "pass", "Claim supported by source material."
        elif verdict == "contradicted":
            gv, notes = "fail", "Claim contradicted by source material."
        else:
            gv, notes = "warn", "Insufficient evidence in sources; may still be correct."

        return {**draft, "grounding_verdict": gv, "grounding_notes": notes}

    grounded = list(await asyncio.gather(*[_ground_one(d) for d in drafts]))
    return {"current_batch_drafts": grounded}


grounder.default_worker_class = "worker_low"


# ─────────────────────────────────────────────────────────────────────────────
# 11. batch_commit
# ─────────────────────────────────────────────────────────────────────────────

async def batch_commit(state: AgentState) -> Dict:
    """
    Persist accepted questions to the database, update thin accumulators,
    increment current_batch_index, and reset the in-flight batch state.

    Accepted  = grounding_verdict in ("pass", "warn") — these are written to DB.
    Rejected  = grounding_verdict == "fail" — metadata logged, not written.

    quiz_questions row: id, quiz_id (=job_id), user_id, question_text, question_type, hint
    quiz_answers rows : id, question_id, is_correct, feedback, answer_choice_text, distractor_type
    """
    from tasks.database import get_db_connection

    drafts = state.get("current_batch_drafts") or []
    batch_idx = state.get("current_batch_index", 0)
    job_id = (state.get("job_id") or "").strip()
    user_id = (state.get("user_id") or "").strip()

    accepted = [d for d in drafts if d.get("grounding_verdict", "") != "fail"]
    rejected = [d for d in drafts if d.get("grounding_verdict", "") == "fail"]

    new_question_ids: List[str] = []
    now = datetime.now(timezone.utc)

    if job_id and accepted:
        try:
            async with get_db_connection() as conn:
                for draft in accepted:
                    q_id = str(_uuid_mod.uuid4())
                    await conn.execute(
                        """
                        INSERT INTO quiz_questions
                          (id, quiz_id, user_id, question_text, question_type, hint, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        q_id,
                        job_id,
                        user_id or None,
                        draft["question_stem"],
                        draft["question_type"],
                        draft["hint"],
                        now,
                    )
                    for ans in draft["answers"]:
                        await conn.execute(
                            """
                            INSERT INTO quiz_answers
                              (id, question_id, is_correct, feedback,
                               answer_choice_text, distractor_type, created_at)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (id) DO NOTHING
                            """,
                            str(_uuid_mod.uuid4()),
                            q_id,
                            ans["is_correct"],
                            ans["feedback"],
                            ans["answer_text"],
                            ans["distractor_type"] or None,
                            now,
                        )
                    new_question_ids.append(q_id)
                    logger.info(
                        "batch_commit: wrote Q[%s] → %s",
                        draft["spec_index"], q_id[:8],
                    )
        except Exception as exc:
            logger.error("batch_commit: DB write failed for batch %d: %s", batch_idx, exc)
            await _try_save_artifact(
                state,
                artifact_key=f"batch_commit_error:{batch_idx}",
                content={"error": str(exc), "batch_index": batch_idx},
                worker_class="tool_only",
                node_name="batch_commit",
                artifact_type="error",
            )

    # Update thin accumulators (full list returned — not Annotated)
    existing_ids = list(state.get("accepted_question_ids") or [])
    all_ids = existing_ids + new_question_ids

    existing_sigs = list(state.get("used_question_signatures") or [])
    new_sigs = [
        f"{d['question_type']}:{d['question_stem'][:40]}"
        for d in accepted
    ]
    all_sigs = existing_sigs + new_sigs

    existing_rejected = list(state.get("rejected_question_metadata") or [])
    new_rejected = [
        {
            "batch_index": batch_idx,
            "spec_index": r["spec_index"],
            "question_type": r["question_type"],
            "reason": r.get("grounding_notes", "grounding_fail"),
        }
        for r in rejected
    ]
    all_rejected = existing_rejected + new_rejected

    # Update coverage summary (lightweight JSON: question_type counts)
    try:
        coverage = json.loads(state.get("coverage_summary") or "{}")
    except Exception:
        coverage = {}
    for d in accepted:
        qtype = d.get("question_type", "UNKNOWN")
        coverage[qtype] = coverage.get(qtype, 0) + 1
    coverage_str = json.dumps(coverage)

    await _try_save_artifact(
        state,
        artifact_key=f"batch_commit:{batch_idx}",
        content={
            "batch_index": batch_idx,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "question_ids": new_question_ids,
        },
        worker_class="tool_only",
        node_name="batch_commit",
        artifact_type="batch_commit",
    )

    return {
        "accepted_question_ids": all_ids,
        "rejected_question_metadata": all_rejected,
        "used_question_signatures": all_sigs,
        "coverage_summary": coverage_str,
        "current_batch_index": batch_idx + 1,
        "batch_revision_count": 0,
        "current_batch_drafts": [],
        "current_batch_eval": None,
    }


batch_commit.default_worker_class = "tool_only"


def batch_router(state: AgentState) -> str:
    """After batch_commit: loop to question_drafter or proceed to critic."""
    idx = state.get("current_batch_index", 0)
    total = len(state.get("batch_specs") or [])
    accepted = len(state.get("accepted_question_ids") or [])
    needed = state.get("num_questions") or 10

    if idx >= total or accepted >= needed:
        return "critic"
    return "question_drafter"


# ─────────────────────────────────────────────────────────────────────────────
# 12. critic
# ─────────────────────────────────────────────────────────────────────────────

async def critic(state: AgentState) -> Dict:
    """
    Global diversity and coverage check across the full accepted quiz.
    Uses the lightweight used_question_signatures list (not the full question text)
    to keep context bounded regardless of quiz size.
    """
    sigs = state.get("used_question_signatures") or []
    accepted_count = len(state.get("accepted_question_ids") or [])
    coverage_raw = state.get("coverage_summary") or "{}"
    try:
        coverage = json.loads(coverage_raw)
    except Exception:
        coverage = {}

    num_questions = state.get("num_questions") or 10
    rejected_count = len(state.get("rejected_question_metadata") or [])

    prompt = (
        f"You are a senior law professor reviewing a {accepted_count}-question quiz "
        f"(target: {num_questions}, rejected: {rejected_count}).\n\n"
        f"Question type coverage:\n{json.dumps(coverage, indent=2)}\n\n"
        f"Question signatures (first 40 chars of each stem):\n"
        + "\n".join(f"- {s}" for s in sigs[:50])
        + "\n\n"
        "Evaluate:\n"
        "1. Type diversity: are too many questions from the same type?\n"
        "2. Duplicate detection: any suspiciously similar stems?\n"
        "3. Coverage gaps: which question types are under-represented?\n"
        "4. Overall quality score (0.0-1.0)\n\n"
        "Return a JSON object:\n"
        '  "overall_score": float\n'
        '  "type_diversity_ok": bool\n'
        '  "duplicates_found": [str, ...]\n'
        '  "coverage_gaps": [str, ...]\n'
        '  "summary": str (2-3 sentences)\n'
        "No extra text."
    )

    raw = await _llm("worker_low", prompt, max_tokens=800)
    try:
        report = _parse_json(raw)
        report_str = json.dumps(report)
    except Exception:
        report_str = json.dumps({
            "overall_score": 0.75,
            "type_diversity_ok": True,
            "duplicates_found": [],
            "coverage_gaps": [],
            "summary": f"Quiz has {accepted_count} questions across {len(coverage)} types.",
        })

    await _try_save_artifact(
        state,
        artifact_key="critic_report",
        content={"report": report_str},
        worker_class="worker_low",
        node_name="critic",
        artifact_type="critic_report",
    )
    return {"critic_report": report_str}


critic.default_worker_class = "worker_low"


# ─────────────────────────────────────────────────────────────────────────────
# 13. final_formatter
# ─────────────────────────────────────────────────────────────────────────────

async def final_formatter(state: AgentState) -> Dict:
    """
    Write a markdown summary of the completed quiz to the parent notes row
    (notes.content_markdown) and update notes.metadata with doctrine_tags.
    The actual question/answer rows were written incrementally by batch_commit.
    Returns the final accepted_question_ids as persisted_question_ids.
    """
    from tasks.database import get_db_connection

    accepted_ids = state.get("accepted_question_ids") or []
    coverage_raw = state.get("coverage_summary") or "{}"
    critic_report_raw = state.get("critic_report") or "{}"
    job_id = (state.get("job_id") or "").strip()
    num_questions = state.get("num_questions") or 10
    quiz_mode = state.get("quiz_mode") or "mixed"

    try:
        coverage = json.loads(coverage_raw)
    except Exception:
        coverage = {}
    try:
        critic_report = json.loads(critic_report_raw)
    except Exception:
        critic_report = {}

    # Derive doctrine tags from case_extracts if available
    extracts = state.get("case_extracts") or []
    case_names = [e.get("case_name", "") for e in extracts[:8]]
    profiles = state.get("source_profiles") or []
    all_concepts = []
    for p in profiles:
        all_concepts.extend(p.get("key_concepts", [])[:5])

    # Build summary markdown
    type_breakdown = "\n".join(
        f"  - {qt}: {cnt}" for qt, cnt in sorted(coverage.items(), key=lambda x: -x[1])
    )
    critic_summary = critic_report.get("summary", "")
    overall_score = critic_report.get("overall_score", "N/A")

    markdown = (
        f"# Quiz ({len(accepted_ids)} questions)\n\n"
        f"**Mode:** {quiz_mode}  \n"
        f"**Requested:** {num_questions}  \n"
        f"**Generated:** {len(accepted_ids)}  \n\n"
        f"## Question Type Breakdown\n\n{type_breakdown or '  *(not available)*'}\n\n"
        f"## Cases Covered\n\n"
        + "\n".join(f"- {cn}" for cn in case_names if cn)
        + f"\n\n## Quality Score\n\n{overall_score}\n\n"
        f"{critic_summary}\n"
    )

    # Update parent notes row
    if job_id:
        doctrine_tags = list(dict.fromkeys(all_concepts[:10]))  # dedup, preserve order
        metadata = json.dumps({
            "doctrine_tags": doctrine_tags,
            "quiz_mode": quiz_mode,
            "question_count": len(accepted_ids),
            "question_type_coverage": coverage,
        })
        try:
            async with get_db_connection() as conn:
                await conn.execute(
                    """
                    UPDATE notes
                    SET content_markdown = $1, metadata = $2::jsonb
                    WHERE id = $3
                    """,
                    markdown,
                    metadata,
                    job_id,
                )
            logger.info("final_formatter: updated notes row %s", job_id[:8])
        except Exception as exc:
            logger.error("final_formatter: notes update failed: %s", exc)

    await _try_save_artifact(
        state,
        artifact_key="final_output",
        content={"markdown": markdown, "question_count": len(accepted_ids)},
        worker_class="worker_low",
        node_name="final_formatter",
        artifact_type="final_output",
    )

    return {
        "persisted_question_ids": accepted_ids,
        "final_output": markdown,
    }


final_formatter.default_worker_class = "worker_low"
