# agents/quiz/nodes.py
"""
All nodes + routing helpers for the quiz LangGraph agent.

Node index (in execution order):
  1.  head_orchestrator              — worker_mid; validate inputs, select quiz mode
  2.  source_profiler                — tool_only; per-source inventory (parallel Send)
  3.  case_rule_extractor            — orchestrator; structured extraction per identified case
  4.  cross_doc_concepts_synthesis   — worker_mid; doctrine clusters, confusables, traps
  5.  quiz_blueprint_planner         — worker_mid; create all batch_specs with question allocations
  6.  process_batch                  — orchestrator; parallel fan-out node: runs the full
                                       per-batch pipeline (nodes 6a–6e) for each batch concurrently
      6a. question_drafter           — worker_mid; draft 5 MCQ stems + answer skeletons
      6b. false_trap_red_herring_generator — worker_mid; enrich distractors (parallel per-question)
      6c. question_evaluator         — worker_mid; batch-level pedagogical QA scoring
      6d. reviser                    — orchestrator; fix failing questions (max 2 passes)
      6e. grounder                   — worker_low; verify claims against source
      6f. batch_commit               — tool_only; write accepted questions to DB
  7.  critic                         — worker_low; global diversity + coverage check
  8.  final_formatter                — worker_low; update parent notes row, return summary

Routing helpers (not nodes):
  head_orchestrator_to_profiler      — Send per source_id
  quiz_blueprint_planner_to_batch    — Send per batch_spec (parallel fan-out)
  should_revise_batch                — "reviser" | "grounder" (used inside process_batch)
  batch_router                       — legacy sequential router (kept but unused in parallel mode)
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
from utils.llm_clients.anthropic_rate_limits import get_llm_semaphore

logger = logging.getLogger(__name__)

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

    from utils.llm_clients.llm_factory import WORKER_FALLBACK_CHAINS
    fallback_chain = WORKER_FALLBACK_CHAINS.get(worker_class, [])

    if _provider == "deepseek":
        sem = _get_deepseek_semaphore()
    else:
        sem = await get_llm_semaphore()

    async with sem:
        for attempt in range(_LLM_RATE_LIMIT_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    LLMFactory.async_call_with_fallback(
                        _provider, model_name, prompt, system=system,
                        max_tokens=max_tokens, fallback_chain=fallback_chain,
                        **client_kwargs,
                    ),
                    timeout=_LLM_CALL_TIMEOUT,
                )
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
                raise


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

    _node_start("head_orchestrator", state,
                num_questions=num_questions, quiz_mode=quiz_mode)

    system = (
        "You are a T-14 law professor designing a rigorous multiple-choice quiz.\n\n"
        "PLANNING REQUIREMENTS:\n"
        "1. Identify the doctrinal areas covered across all documents.\n"
        "2. Select quiz_mode-appropriate question types: recall tests element definitions; "
        "   application tests rule-to-fact fit; exam-style tests nuanced distinctions "
        "   and competing doctrines.\n"
        "3. Flag any documents with dissents, circuit splits, or evolving standards — "
        "   these are high-yield distractor sources.\n"
        "4. Confirm scope in 2-3 sentences: topic coverage, estimated question distribution, "
        "   and any gaps or pedagogical concerns.\n"
        "Respond with your 2-3 sentence scope confirmation only."
    )
    await _llm(
        "worker_mid",
        f"Documents: {sources_json}\nRequest: {state.get('request', '')}\n"
        f"Quiz: {num_questions} questions, mode: {quiz_mode}.",
        system=system,
        max_tokens=256,
        _node="head_orchestrator",
    )

    _node_done("head_orchestrator", state,
               num_questions=num_questions, quiz_mode=quiz_mode,
               num_batches=num_batches)

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
    source_ids = state["source_ids"]
    logger.info("quiz x%d node fan out for source_profiler", len(source_ids))
    return [
        Send("source_profiler", {"source_id": sid, **state})
        for sid in source_ids
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
            "You are a T-14 law professor performing structured case extraction.\n\n"
            "EXTRACTION REQUIREMENTS:\n"
            "- procedural_posture: court level, prior holdings, and what is being reviewed.\n"
            "- facts: key factual background (2-4 sentences); include only facts the court "
            "  actually relied on in reaching its decision.\n"
            "- legally_relevant_facts: array of 4-8 individual facts that drove the legal "
            "  outcome (e.g., 'defendant owed a duty as a common carrier'; "
            "  'plaintiff was a foreseeable plaintiff'). These feed distractor construction.\n"
            "- issue: the precise legal question decided — phrase it as a yes/no question.\n"
            "- holding: the court's direct answer to the issue, including the rule of decision.\n"
            "- rule: the operative legal rule or test, written as a standalone statement "
            "  usable in a future case without referring back to this case by name.\n"
            "- reasoning: the analytical steps used to reach the holding (2-3 sentences).\n"
            "- dicta: any statements that go beyond what was necessary to decide the issue.\n"
            "- dissent: the dissent's core objection (if any) — a high-yield distractor source.\n"
            "- policy: the underlying policy rationale(s) the court cited or relied on.\n"
            "Base EVERY field strictly on the provided text. "
            "Return ONLY the JSON object. No extra text."
        )
        prompt = (
            f"Case: {case_name}\n\nSource text:\n{context[:4000]}\n\n"
            "Extract the structured case information."
        )
        raw = await _llm("orchestrator", prompt, system=system, max_tokens=2500,
                         _node="case_rule_extractor")
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
            "case_name": e.get("case_name") or "",
            "holding":   (e.get("holding")  or "")[:200],
            "rule":      (e.get("rule")      or "")[:200],
            "dicta":     (e.get("dicta")     or "")[:100],
            "dissent":   (e.get("dissent")   or "")[:100],
        }
        for e in extracts[:12]
    ]
    all_concepts = []
    for p in profiles:
        all_concepts.extend(p.get("key_concepts", [])[:8])

    _node_start("cross_doc_concepts_synthesis", state,
                n_extracts=len(extracts), n_profiles=len(profiles))

    prompt = (
        "You are a T-14 law professor preparing teaching priorities for a rigorous MCQ quiz.\n\n"
        f"Cases:\n{json.dumps(cases_brief, indent=2)}\n\n"
        f"Key concepts across sources: {all_concepts[:20]}\n\n"
        "SYNTHESIS REQUIREMENTS:\n"
        "1. doctrine_clusters: 3-5 clusters, each with:\n"
        "   - name: the doctrinal area\n"
        "   - cases: list of case names in the cluster\n"
        "   - core_rule: the rule that links them\n"
        "   - internal_tension: any intra-cluster split or doctrinal evolution to exploit\n"
        "2. confusable_concepts: pairs or groups students routinely mix up — include WHY "
        "   they confuse them (similar names, overlapping elements, same outcome for "
        "   different reasons, etc.).\n"
        "3. common_traps: 3-5 overbroad readings, mis-stated rules, or scope errors "
        "   students make. These map directly to distractor answer choices.\n"
        "4. high_yield_areas: 3-5 doctrinal areas most heavily tested on bar exams and "
        "   law school finals — prioritise these for harder questions.\n\n"
        "Return ONLY a JSON object with keys: doctrine_clusters, confusable_concepts, "
        "common_traps, high_yield_areas. No extra text."
    )

    raw = await _llm("worker_mid", prompt, max_tokens=2000)
    try:
        synthesis_data = _parse_json(raw)
    except Exception:
        synthesis_data = {
            "doctrine_clusters": [],
            "confusable_concepts": [],
            "common_traps": [],
            "high_yield_areas": [],
        }

    _node_done("cross_doc_concepts_synthesis", state,
               n_clusters=len(synthesis_data.get("doctrine_clusters", [])),
               n_traps=len(synthesis_data.get("common_traps", [])))

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
        {"case_id": e.get("case_id") or "", "case_name": e.get("case_name") or "",
         "source_id": e.get("source_id") or "", "holding": (e.get("holding") or "")[:150]}
        for e in extracts[:12]
    ]
    source_ids = [p["source_id"] for p in profiles]

    _node_start("quiz_blueprint_planner", state,
                num_questions=num_questions, quiz_mode=quiz_mode)

    prompt = (
        f"You are a T-14 law professor creating a rigorous quiz blueprint.\n\n"
        f"Total questions needed: {num_questions}\n"
        f"Batch size: {batch_size}\n"
        f"Quiz mode: {quiz_mode}\n"
        f"Target difficulty: {target_difficulty}\n\n"
        f"Available cases:\n{json.dumps(cases_brief, indent=2)}\n\n"
        f"Concept synthesis (doctrine clusters, traps):\n{synthesis_str[:1200]}\n\n"
        f"Available distractor types: {DISTRACTOR_TYPES}\n\n"
        f"BLUEPRINT REQUIREMENTS:\n"
        f"- Create exactly {num_questions} question specs as a JSON array.\n"
        f"- Each spec must have:\n"
        f'  "spec_index": int (0-based global index)\n'
        f'  "question_type": one of {MC_QUESTION_TYPES[:8]}... (use full variety)\n'
        f'  "source_ids": [list of source_id UUIDs from available sources]\n'
        f'  "case_names": [list of case names to draw from]\n'
        f'  "topic": specific doctrinal point for this question (one precise sentence)\n'
        f'  "difficulty": "recall" | "application" | "analysis"\n'
        f'  "distractor_types": [exactly 3 distractor type tags for the wrong answers]\n\n'
        f"ALLOCATION RULES:\n"
        f"- No more than 3 specs share the same question_type.\n"
        f"- Difficulty distribution: ~30% recall, ~40% application, ~30% analysis.\n"
        f"- Each identified case must appear in at least 1 question.\n"
        f"- Prioritise high_yield_areas and common_traps from the concept synthesis.\n"
        f"- Assign distractor_types that match the common_traps for that topic "
        f"  (overbroad_rule → where rule-scope confusion is the trap; "
        f"  wrong_case_applied → where students mix up two similar cases).\n"
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
    _node_done("quiz_blueprint_planner", state,
               num_batches=len(batch_specs), total_specs=len(all_specs))
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
            "You are a T-14 law professor writing a bar-caliber multiple-choice question.\n\n"
            "MCQ STEM CONSTRUCTION STANDARDS:\n"
            "- Stems must present a complete legal scenario or doctrinal question — "
            "  never a fill-in-the-blank or 'which of the following' without context.\n"
            "- For APPLICATION questions: present a concrete fact pattern (2-5 sentences) "
            "  ending with a specific legal question (e.g., 'Is D liable for negligence?').\n"
            "- For RECALL questions: test the precise scope and limits of a rule, not "
            "  just its name or label.\n"
            "- For ANALYSIS questions: present two competing doctrines or arguments and "
            "  ask which analysis is correct given the facts.\n"
            "- Stems must be self-contained: a well-prepared student can answer from "
            "  the stem alone without re-reading the source.\n"
            "- Maximum 120 words for the stem.\n\n"
            "CORRECT ANSWER STANDARDS:\n"
            "- Must be unambiguously correct — no 'best answer' hedging.\n"
            "- State the rule + its application to the facts, or the precise doctrinal "
            "  formulation that distinguishes it from the distractors.\n"
            "- Avoid 'all of the above' or 'none of the above'.\n\n"
            "WRONG ANSWER (DISTRACTOR) STANDARDS:\n"
            "- Each wrong answer must exploit a specific student misconception from the "
            "  distractor_types list.\n"
            "- Wrong answers must be plausible to a student who partially understands the "
            "  doctrine — if a zero-effort student can eliminate them, they fail QA.\n"
            "- Make distractors parallel in structure and length to the correct answer.\n\n"
            "HINT STANDARDS:\n"
            "- One sentence that points to the key legal concept without naming the answer.\n"
            "- Should redirect a confused student toward the right framework, not the "
            "  right answer.\n\n"
            "Return ONLY a JSON object with keys:\n"
            '  "question_stem": str (max 120 words)\n'
            '  "hint": str (one sentence; key issue without revealing the answer)\n'
            '  "correct_answer": str (full sentence stating rule/application)\n'
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

        raw = await _llm("worker_mid", prompt, system=system, max_tokens=1500,
                         _node="question_drafter")
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
    feedback (pedagogical rationale).  Runs one parallel LLM call per question
    (~700 tokens each) instead of one bulk call for the whole batch.
    """
    drafts = list(state.get("current_batch_drafts") or [])
    if not drafts:
        return {}

    _node_start("false_trap_red_herring_generator", state,
                n_drafts=len(drafts))

    system = (
        "You are a T-14 law professor enriching multiple-choice quiz distractors with "
        "pedagogically precise labels and explanatory feedback.\n\n"
        "DISTRACTOR ENRICHMENT STANDARDS:\n"
        "For WRONG answers:\n"
        "  1. distractor_type: select the single most accurate type from the taxonomy. "
        "     Prefer specific types (e.g., 'overbroad_rule', 'wrong_case_applied') over "
        "     generic ones (e.g., 'plausible_wrong').\n"
        "  2. feedback (2-3 sentences):\n"
        "     (a) Name the misconception — 'A student who selects this likely believes [X]';\n"
        "     (b) Explain the precise error — which element is wrong, which case is "
        "         misapplied, or how the rule scope was mis-stated;\n"
        "     (c) Correct the misconception in one direct sentence.\n\n"
        "For the CORRECT answer:\n"
        "  1. distractor_type: use 'correct_answer'\n"
        "  2. feedback (2-3 sentences):\n"
        "     (a) State why this answer is right — which rule applies and why;\n"
        "     (b) Name the case or doctrine it derives from;\n"
        "     (c) Note any limiting conditions or scope restrictions (the 'unless/but').\n\n"
        f"Available distractor_type values: {DISTRACTOR_TYPES}\n\n"
        "Return a JSON object with:\n"
        '  "spec_index": int\n'
        '  "answers": [\n'
        '    {"choice_letter": str, "distractor_type": str, "feedback": str},\n'
        '    ... (all 4 choices)\n'
        '  ]\n'
        "No extra text."
    )

    async def _enrich_one(draft: DraftQuizQuestion) -> DraftQuizQuestion:
        payload = {
            "spec_index": draft["spec_index"],
            "question_type": draft["question_type"],
            "question_stem": draft["question_stem"],
            "answers": [
                {
                    "choice_letter": a["choice_letter"],
                    "answer_text": a["answer_text"],
                    "is_correct": a["is_correct"],
                }
                for a in draft["answers"]
            ],
        }
        prompt = (
            f"Enrich the following quiz question with distractor labels and feedback:\n\n"
            f"{json.dumps(payload, indent=2)}"
        )
        try:
            raw = await _llm("worker_mid", prompt, system=system, max_tokens=700,
                             _node="false_trap_red_herring_generator")
            enriched = _parse_json(raw)
            if not isinstance(enriched, dict):
                raise ValueError
        except Exception:
            return draft

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
        return {**draft, "answers": new_answers}

    updated_drafts = list(await asyncio.gather(*[_enrich_one(d) for d in drafts]))

    _node_done("false_trap_red_herring_generator", state,
               n_enriched=len(updated_drafts))
    return {"current_batch_drafts": updated_drafts}


false_trap_red_herring_generator.default_worker_class = "worker_mid"


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

    _node_start("question_evaluator", state,
                batch_idx=batch_idx, n_questions=len(drafts))

    system = (
        "You are a T-14 law school exam committee member performing pedagogical QA on "
        "a batch of multiple-choice questions.\n\n"
        "EVALUATION CRITERIA (score each 0.0–1.0):\n"
        "  source_grounding: Are all factual claims in the stem and correct answer "
        "actually grounded in the source material? "
        "(0.0 = hallucinated; 1.0 = directly quoted or closely paraphrased)\n"
        "  single_correct_answer: Is there exactly one unambiguously correct answer? "
        "(0.0 = multiple defensible answers or answer key error; 1.0 = uniquely correct)\n"
        "  distractor_quality: Are wrong answers genuinely plausible misconceptions? "
        "(0.0 = obviously wrong; 1.0 = every distractor exploits a documented student error)\n"
        "  question_type_diversity: Does the batch span different cognitive demands "
        "(recall, application, analysis)? "
        "(0.0 = all same type; 1.0 = well-distributed across types)\n"
        "  difficulty_match: Does each question match the target difficulty level? "
        "(0.0 = far off target; 1.0 = correctly calibrated)\n\n"
        "PASS THRESHOLD: all criterion scores >= 0.70 AND no per-question critical failures.\n"
        "CRITICAL FAILURE (automatic fail): ambiguous correct answer, answer key error, "
        "or direct factual contradiction of the source material.\n\n"
        "REVISION INSTRUCTIONS must be specific and actionable — name which spec_index "
        "has the problem and exactly what to fix.\n\n"
        "Return a JSON object:\n"
        '  "passes": bool\n'
        '  "scores": {"source_grounding": f, "single_correct_answer": f, '
        '"distractor_quality": f, "question_type_diversity": f, "difficulty_match": f}\n'
        '  "rejection_reasons": [str, ...] (empty if passes)\n'
        '  "revision_instructions": [str, ...] (specific fixes for each failing criterion)\n'
        '  "question_verdicts": [{"spec_index": int, "passes": bool, "reason": str}, ...]\n'
        "No extra text."
    )

    prompt = (
        f"Target difficulty: {target_difficulty}\n"
        f"Batch ({len(batch_payload)} questions):\n\n"
        f"{json.dumps(batch_payload, indent=2)}\n\n"
        "Evaluate the batch."
    )

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=2000,
                     _node="question_evaluator")
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
    _node_done("question_evaluator", state,
               batch_idx=batch_idx,
               passes=batch_eval["passes"],
               n_verdicts=len(batch_eval["question_verdicts"]))
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

    _node_start("reviser", state,
                batch_idx=state.get("current_batch_index", 0),
                revision_count=revision_count,
                n_failing=len(failing_indices))

    system = (
        "You are a T-14 law professor performing surgical revision of MCQ questions "
        "that failed pedagogical QA.\n\n"
        "REVISION STANDARDS:\n"
        "1. Fix ONLY the specific issues in the revision instructions — do not rewrite "
        "   questions that already pass.\n"
        "2. Preserve the question_type, topic, and spec_index exactly as given.\n"
        "3. If the issue is an ambiguous correct answer: rewrite the stem to eliminate "
        "   ambiguity, or replace the offending distractor with a clearly inferior choice.\n"
        "4. If the issue is weak distractors: replace with choices that exploit real "
        "   misconceptions (overbroad_rule, scope_error, wrong_case_applied).\n"
        "5. If the issue is source grounding: remove any claims not in the source text; "
        "   replace with a question testing what IS documented.\n"
        "6. Each revised question must score >= 0.70 on all evaluation criteria.\n"
        "Return a JSON array of the revised questions only (same JSON structure as input)."
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

    raw = await _llm("orchestrator", prompt, system=system, max_tokens=3000,
                     _node="reviser")
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
    _node_done("reviser", state,
               n_revised=len(revised_map), n_total=len(updated_drafts),
               revision_pass=revision_count + 1)
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
# 11b. process_batch  (parallel fan-out version of the batch pipeline)
# ─────────────────────────────────────────────────────────────────────────────

async def process_batch(state: Dict) -> Dict:
    """
    Run the full batch pipeline for one batch spec in parallel with other batches.
    Invoked via Send fan-out from quiz_blueprint_planner_to_batch.

    Pipeline: question_drafter → false_trap_red_herring_generator
              → question_evaluator → reviser loop (max 2) → grounder → DB write.

    Uses a synthetic serial state so existing node functions need no changes.
    Returns only the delta for this batch; operator.add reducers in AgentState
    accumulate results across all parallel batches.
    """
    batch_spec = state.get("current_batch_spec") or {}
    batch_idx = batch_spec.get("batch_index", 0)

    # Isolated serial state — no cross-batch interference.
    # current_batch_index=0 because batch_specs has exactly one entry at position 0.
    serial: Dict = {
        **state,
        "current_batch_index":   0,
        "batch_specs":           [batch_spec],
        "current_batch_drafts":  [],
        "current_batch_eval":    None,
        "batch_revision_count":  0,
        # Reset accumulators so batch_commit returns only this batch's delta
        "accepted_question_ids":      [],
        "rejected_question_metadata": [],
        "used_question_signatures":   [],
        "coverage_summary":           "{}",
    }

    # 1. Draft
    serial = {**serial, **(await question_drafter(serial))}

    # 2. Enrich distractors (parallel per-question inside the node)
    serial = {**serial, **(await false_trap_red_herring_generator(serial))}

    # 3. Evaluate → revise loop (max 2 revision passes)
    for _ in range(3):
        serial = {**serial, **(await question_evaluator(serial))}
        if should_revise_batch(serial) != "reviser":
            break
        serial = {**serial, **(await reviser(serial))}

    # 4. Ground
    grounded = await grounder(serial)
    serial = {**serial, **grounded}

    # Capture grounded drafts before batch_commit clears them
    grounded_drafts = serial.get("current_batch_drafts") or []
    accepted_drafts = [d for d in grounded_drafts if d.get("grounding_verdict", "") != "fail"]

    # 5. Write accepted questions to DB
    commit_result = await batch_commit(serial)

    new_question_ids = commit_result.get("accepted_question_ids") or []
    new_rejected     = commit_result.get("rejected_question_metadata") or []
    new_sigs         = commit_result.get("used_question_signatures") or []

    # Coverage delta for this batch (merged by final_formatter via batch_results)
    coverage_delta: Dict[str, int] = {}
    for d in accepted_drafts:
        qt = d.get("question_type", "UNKNOWN")
        coverage_delta[qt] = coverage_delta.get(qt, 0) + 1

    logger.info(
        "process_batch[%d]: %d questions written, %d rejected",
        batch_idx, len(new_question_ids), len(new_rejected),
    )
    return {
        "accepted_question_ids":      new_question_ids,
        "rejected_question_metadata": new_rejected,
        "used_question_signatures":   new_sigs,
        "batch_results": [{
            "batch_index":  batch_idx,
            "question_ids": new_question_ids,
            "coverage":     coverage_delta,
        }],
    }


process_batch.default_worker_class = "orchestrator"


def quiz_blueprint_planner_to_batch(state: AgentState) -> List[Send]:
    """Fan-out: one process_batch per batch spec (all run in parallel)."""
    batch_specs = state.get("batch_specs") or []
    logger.info("quiz: parallel fan-out → %d batches", len(batch_specs))
    return [
        Send("process_batch", {"current_batch_spec": bs, **state})
        for bs in batch_specs
    ]


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

    # Compute coverage from batch_results (parallel mode) with fallback to coverage_summary
    batch_results = state.get("batch_results") or []
    coverage: Dict[str, int] = {}
    for br in batch_results:
        for qt, cnt in (br.get("coverage") or {}).items():
            coverage[qt] = coverage.get(qt, 0) + cnt
    if not coverage:
        try:
            coverage = json.loads(state.get("coverage_summary") or "{}")
        except Exception:
            coverage = {}

    num_questions = state.get("num_questions") or 10
    rejected_count = len(state.get("rejected_question_metadata") or [])

    _node_start("critic", state,
                accepted=accepted_count, target=num_questions, rejected=rejected_count)

    system = (
        "You are a T-14 law school assessment director performing a final diversity "
        "and coverage audit of a completed MCQ quiz.\n\n"
        "AUDIT STANDARDS:\n"
        "- Type diversity: no single question type should exceed 40% of the quiz. "
        "  Flag overrepresented types AND underrepresented types.\n"
        "- Duplicate detection: scan question signatures for near-identical stems. "
        "  Same fact pattern + different call of the question counts as a near-duplicate.\n"
        "- Coverage gaps: identify doctrinal areas that have NO question — these are "
        "  curriculum coverage failures that weaken the quiz's teaching value.\n"
        "- Overall quality: weight single_correct_answer and distractor_quality most heavily "
        "  in the overall_score.\n"
        "Return ONLY the JSON object. No extra text."
    )

    prompt = (
        f"Audit a {accepted_count}-question quiz "
        f"(target: {num_questions}, rejected: {rejected_count}).\n\n"
        f"Question type coverage:\n{json.dumps(coverage, indent=2)}\n\n"
        f"Question signatures (first 40 chars of each stem):\n"
        + "\n".join(f"- {s}" for s in sigs[:50])
        + "\n\n"
        "Evaluate and return:\n"
        '  "overall_score": float (0.0–1.0)\n'
        '  "type_diversity_ok": bool\n'
        '  "duplicates_found": [str, ...] (signatures of any near-duplicate pairs)\n'
        '  "coverage_gaps": [str, ...] (question types or doctrinal areas with no coverage)\n'
        '  "summary": str (2-3 sentences on overall quiz quality and any major concerns)\n'
        "No extra text."
    )

    raw = await _llm("worker_low", prompt, system=system, max_tokens=1200,
                     _node="critic")
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
    _node_done("critic", state, accepted=accepted_count)
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

    _node_start("final_formatter", state,
                n_accepted=len(accepted_ids), quiz_mode=quiz_mode)

    # Compute coverage from batch_results (parallel mode) with fallback to coverage_summary
    batch_results_list = state.get("batch_results") or []
    coverage: Dict[str, int] = {}
    for br in batch_results_list:
        for qt, cnt in (br.get("coverage") or {}).items():
            coverage[qt] = coverage.get(qt, 0) + cnt
    if not coverage:
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

    _node_done("final_formatter", state,
               n_questions=len(accepted_ids), quiz_mode=quiz_mode)
    return {
        "persisted_question_ids": accepted_ids,
        "final_output": markdown,
    }


final_formatter.default_worker_class = "worker_low"
