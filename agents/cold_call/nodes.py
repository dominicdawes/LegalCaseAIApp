# agents/cold_call/nodes.py
"""
All nodes for the cold-call LangGraph agent.

Node index (in execution order):
  1.  head_orchestrator            — orchestrator; plans run, identifies scope
  2.  source_profiler              — worker_mid; per-source inventory (parallel Send)
  3.  corpus_synthesizer           — worker_mid; merges profiles → identified case list
  4.  case_rule_extractor          — orchestrator; full structured extraction per case (parallel Send)
  5.  doctrine_mapper              — orchestrator; builds doctrinal relationship network
  6.  compare_distinguish_mapper   — worker_mid; generates compare/distinguish prompts for case pairs
  7.  question_type_bank_selector  — worker_mid; selects question types for this run
  8.  cold_call_seed_generator     — orchestrator; generates N diverse seeds per case (parallel Send)
  9.  seed_diversity_agent         — worker_mid; checks diversity, approves seeds (merge + retry gate)
  10. socratic_thread_builder      — orchestrator; builds 8-deep question sequence per seed (parallel Send)
  11. socratic_answer_agent        — orchestrator; generates Because/Unless/But answers per sequence (parallel Send)
  12. grounder_agent               — worker_mid; verifies answer grounding per sequence (parallel Send)
  13. critic_coverage_agent        — orchestrator; coverage QA, difficulty labelling (merge point)
  14. formatter_export_agent       — worker_low; exports to Supabase cold_call tables + markdown summary

Fan-out routing helpers (not nodes):
  head_orchestrator_to_profiler, corpus_synthesizer_to_extractor,
  doctrine_mapper_to_seeds, seeds_routing,
  thread_builders_to_answer_agents, answer_agents_to_grounders
"""

import asyncio
import json
import logging
import re
import uuid as _uuid_mod
from typing import Any, Dict, List, Optional

from langgraph.types import Send

from .state import (
    AgentState,
    AnswerSequence,
    CaseRuleObject,
    CompareDistinguishPrompt,
    CoverageMap,
    DoctrineEdge,
    DoctrineMap,
    ExportResult,
    GroundingResult,
    QuestionAnswer,
    QuestionSequence,
    QuestionTypeSelection,
    Seed,
    SequenceQuestion,
    SourceProfile,
)
from .constants import (
    QUESTION_TYPES_EARLY,
    QUESTION_TYPES_MIDDLE,
    QUESTION_TYPES_DEEP,
    QUESTION_TYPES_REQUIRED_ALWAYS,
    QUESTION_TYPES_CONDITIONAL_MULTI_CASE_DEEP,
    QUESTION_TYPES_CONDITIONAL_DISSENT_MIDDLE,
    QUESTION_TYPES_CONDITIONAL_ADVANCED_DEEP,
    QUESTION_TYPES_CONDITIONAL_ADVANCED_MIDDLE,
    QUESTION_TYPES_OPTIONAL_MULTI_CASE,
    QUESTION_TYPES_OPTIONAL_UPPER_LEVEL,
    SEED_THEMES,
)
from .worker_config import _fetch_worker_model, model_costs, DEFAULT_PROVIDER, PROVIDER_FALLBACK
from utils.llm_clients.anthropic_rate_limits import get_llm_semaphore

logger = logging.getLogger(__name__)

MAX_SEED_RETRIES = 2
DIVERSITY_THRESHOLD = 0.65
SEEDS_PER_CASE_MULTIPLIER = 3   # generate 3× requested count per case; diversity agent prunes

# Question depth buckets by label position in the A-I sequence
DEPTH_MAP = {
    "A": "early", "B": "early",
    "C": "middle", "D": "middle", "E": "middle",
    "F": "deep", "G": "deep", "H": "deep", "I": "deep",
}

# ── Rate-limit guard ──────────────────────────────────────────────────────────
# The semaphore is DYNAMIC — sized from the org's actual Anthropic tier by
# utils/llm_clients/anthropic_rate_limits.py (hourly heartbeat probe).
# These constants control the 429-retry loop that fires when the semaphore alone
# isn't enough (e.g. burst of parallel calls that exceeds the minute token bucket).
_LLM_RATE_LIMIT_RETRIES = 3   # additional attempts after the first 429
_LLM_RATE_LIMIT_DELAY   = 60  # seconds to wait before each retry (linear: 60, 120, 180 s)
_LLM_CALL_TIMEOUT       = 300  # seconds before a hung LLM call is aborted
_LLM_TIMEOUT_RETRIES    = 1   # extra attempts on timeout before giving up

# DeepSeek has much higher RPM than Anthropic Tier 1 — allow up to 10 concurrent
# calls instead of the 2 returned by the Anthropic-probe semaphore.
_DEEPSEEK_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _get_deepseek_semaphore() -> asyncio.Semaphore:
    """Return (creating if needed) the per-loop DeepSeek concurrency semaphore."""
    global _DEEPSEEK_SEMAPHORE
    if _DEEPSEEK_SEMAPHORE is None:
        _DEEPSEEK_SEMAPHORE = asyncio.Semaphore(10)
    return _DEEPSEEK_SEMAPHORE


# ── Debug helpers ─────────────────────────────────────────────────────────────

def _node_start(name: str, state: Dict, **extras: Any) -> None:
    """Log node entry with job context and any caller-supplied key/value pairs."""
    job = (state.get("job_id") or "")[:8] or "no-job"
    parts = " ".join(f"{k}={v}" for k, v in extras.items())
    logger.info("▶ [%s] %s  %s", job, name, parts)


def _node_done(name: str, state: Dict, **extras: Any) -> None:
    """Log node completion with result summary."""
    job = (state.get("job_id") or "")[:8] or "no-job"
    parts = " ".join(f"{k}={v}" for k, v in extras.items())
    logger.info("✓ [%s] %s  %s", job, name, parts)


def _node_warn(name: str, state: Dict, msg: str) -> None:
    """Log a non-fatal warning inside a node."""
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
    """Call the LLM with rate-limit protection and provider fallback.

    Tries the primary provider first (default: DeepSeek). If that provider
    exhausts its timeout retries, falls back to the next provider in
    PROVIDER_FALLBACK (DeepSeek → OpenAI → Anthropic) before giving up.

    On a 429 RateLimitError, waits _LLM_RATE_LIMIT_DELAY seconds and retries
    up to _LLM_RATE_LIMIT_RETRIES times (same provider — rate limits are not
    a signal to switch providers).
    """
    from utils.llm_clients.llm_factory import LLMFactory

    _primary = (provider or DEFAULT_PROVIDER).lower()
    provider_chain = [_primary]
    if _fallback := PROVIDER_FALLBACK.get(_primary):
        provider_chain.append(_fallback)

    # Auto-append token budget soft limit if not already present
    if system and "IMPORTANT" not in system:
        system = system + f"\n\n**IMPORTANT**: keep your response under {max_tokens} tokens."

    last_exc: BaseException = RuntimeError("_llm: no providers tried")

    for p_idx, _pname in enumerate(provider_chain):
        _provider, model_name, thinking = _fetch_worker_model(worker_class, _pname)

        if _node:
            if p_idx == 0:
                _llm_call(_node, worker_class, model_name, max_tokens)
            else:
                logger.warning(
                    "🔀 [%s] %s timed out — falling back to %s/%s",
                    _node or worker_class, provider_chain[0], _provider, model_name,
                )

        client_kwargs: Dict[str, Any] = {}
        if thinking is not None:
            client_kwargs["thinking"] = thinking

        client = LLMFactory.get_client_for(
            _provider, model_name,
            temperature=0.7, streaming=False, max_output_tokens=max_tokens,
            **client_kwargs,
        )

        if _provider == "deepseek":
            sem = _get_deepseek_semaphore()
        else:
            sem = await get_llm_semaphore()

        timeout_attempts = 0
        timed_out = False
        async with sem:
            for attempt in range(_LLM_RATE_LIMIT_RETRIES + 1):
                try:
                    if hasattr(client, "achat"):
                        return await asyncio.wait_for(
                            client.achat(prompt, system_prompt=system or None),
                            timeout=_LLM_CALL_TIMEOUT,
                        )
                    async def _stream() -> str:
                        chunks: List[str] = []
                        async for chunk in client.stream_chat(prompt, system_prompt=system or None):
                            chunks.append(chunk)
                        return "".join(chunks)
                    return await asyncio.wait_for(_stream(), timeout=_LLM_CALL_TIMEOUT)
                except asyncio.TimeoutError as exc:
                    timeout_attempts += 1
                    logger.warning(
                        "⏱️ [%s] LLM call timed out after %ds (timeout attempt %d/%d)",
                        _node or worker_class, _LLM_CALL_TIMEOUT,
                        timeout_attempts, _LLM_TIMEOUT_RETRIES + 1,
                    )
                    last_exc = exc
                    if timeout_attempts > _LLM_TIMEOUT_RETRIES:
                        timed_out = True
                        break  # exhausted for this provider; try next in chain
                    continue
                except Exception as exc:
                    err = str(exc)
                    is_rate_limit = (
                        "429" in err
                        or "rate_limit" in err.lower()
                        or "rate limit" in err.lower()
                    )
                    if is_rate_limit and attempt < _LLM_RATE_LIMIT_RETRIES:
                        wait = _LLM_RATE_LIMIT_DELAY * (attempt + 1)  # 60, 120, 180 s
                        logger.warning(
                            "🚦 [%s] Rate limit hit (429) — waiting %ds before retry %d/%d",
                            _node or worker_class, wait, attempt + 1, _LLM_RATE_LIMIT_RETRIES,
                        )
                        await asyncio.sleep(wait)
                        continue
                    raise  # non-timeout, non-429: propagate immediately, no provider switch

        if not timed_out:
            break  # exited inner loop without timing out (shouldn't reach here)

    raise last_exc


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
        from agents.ledger import AgentLedgerService
        ledger = AgentLedgerService()
        await ledger.save_artifact(
            job_id=_uuid_mod.UUID(job_id_str),
            artifact_key=artifact_key,
            content=content,
            worker_class=worker_class,
            node_name=node_name,
            artifact_type=artifact_type,
            source_ids=[_uuid_mod.UUID(s) for s in (source_ids or []) if s],
        )
    except Exception as exc:
        logger.warning("Ledger save '%s' failed (non-fatal): %s", artifact_key, exc)


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
    Plans the cold-call generation run. Reads source_ids, requested_sequence_count,
    target_difficulty, and corpus metadata to produce a job_plan.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import COLD_CALL_PLANNER_TOOLS

    _node_start("head_orchestrator", state,
                n_sources=len(state.get("source_ids") or []),
                request=repr(state.get("request", "")[:80]))

    tools = make_tools(
        state["project_id"],
        source_ids=state["source_ids"],
        use_voyage=state.get("use_voyage", False),
        tool_names=COLD_CALL_PLANNER_TOOLS,
    )
    list_sources_tool = next(t for t in tools if t.name == "list_sources")
    sources_json = await list_sources_tool.ainvoke({})
    logger.info("  📋 [head_orchestrator] list_sources → %d chars", len(sources_json))
    node_max_tokens = 800

    system = (
        "You are the orchestrator for a T-14 law-school cold-call question generation pipeline. "
        "Your job is to plan the construction of a complete Socratic question set that matches "
        "the depth and rigor of a T-14 classroom — covering legally relevant facts, rule extraction, "
        "holding vs. dicta, fact-change hypotheticals, rule boundary testing, counterarguments, "
        "and policy analysis.\n\n"
        "Given the available source documents, create a job plan that identifies:\n"
        "(1) the primary cases and doctrines present,\n"
        "(2) the course context and difficulty tier,\n"
        "(3) whether multi-case comparison threads are warranted,\n"
        "(4) the optimal depth and coverage mode.\n\n"
        "Return JSON with keys:\n"
        "  job_type: 'cold_call'\n"
        "  source_ids: list of source UUIDs in scope\n"
        "  requested_sequence_count: int\n"
        "  target_difficulty: 'law_1l' | 'day_one_t14' | 'advanced'\n"
        "  course_context: string (e.g. 'Torts / Negligence', 'Contracts / Formation')\n"
        "  coverage_mode: 'single_case' | 'multi_case' | 'doctrine_survey'\n"
        "  retrieval_depth: 'standard' | 'deep'\n"
        "  expected_cases: list of case names visible in the sources\n"
        "Return only JSON."
        f"\n\n**IMPORTANT**: keep your response under {node_max_tokens} tokens."
    )

    raw = await _llm(
        "orchestrator",
        f"Sources available:\n{sources_json}\n\nUser request: {state['request']}\n"
        f"Requested sequences: {state.get('requested_sequence_count', 5)}\n"
        f"Target difficulty: {state.get('target_difficulty', 'day_one_t14')}",
        system=system,
        max_tokens=node_max_tokens,
        _node="head_orchestrator",
    )
    try:
        job_plan = _parse_json(raw)
    except Exception as exc:
        _check_token_limit(exc, "head_orchestrator", state)
        _node_warn("head_orchestrator", state, f"JSON parse failed ({exc}) — using default plan")
        job_plan = {
            "job_type": "cold_call",
            "source_ids": state["source_ids"],
            "requested_sequence_count": state.get("requested_sequence_count", 5),
            "target_difficulty": state.get("target_difficulty", "day_one_t14"),
            "course_context": "",
            "coverage_mode": "single_case",
            "retrieval_depth": "deep",
            "expected_cases": [],
        }

    await _try_save_artifact(
        state, "job_plan", job_plan, "orchestrator", "head_orchestrator", "job_plan",
        source_ids=state.get("source_ids"),
    )
    _node_done("head_orchestrator", state,
               mode=job_plan.get("coverage_mode"),
               difficulty=job_plan.get("target_difficulty"),
               context=job_plan.get("course_context"))
    return {"job_plan": job_plan}


head_orchestrator.default_worker_class = "orchestrator"


def head_orchestrator_to_profiler(state: AgentState) -> List[Send]:
    """Fan-out: one source_profiler per source document."""
    source_ids = state["source_ids"]
    logger.info("cold_call x%d node fan out for source_profiler", len(source_ids))
    return [
        Send("source_profiler", {"source_id": sid, **state})
        for sid in source_ids
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. source_profiler  (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def source_profiler(state: Dict) -> Dict:
    """
    Profile one source: identify cases, statutes, professor notes, doctrine sections.
    Light pass — does not deeply extract; only inventories.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import COLD_CALL_PROFILER_TOOLS

    source_id = state["source_id"]
    project_id = state["project_id"]

    tools = make_tools(
        project_id,
        source_ids=[source_id],
        use_voyage=state.get("use_voyage", False),
        tool_names=COLD_CALL_PROFILER_TOOLS,
    )
    outline_tool = next(t for t in tools if t.name == "get_doc_outline")
    outline_json = await outline_tool.ainvoke({"source_id": source_id})
    outline = json.loads(outline_json)

    sections_brief = json.dumps(outline.get("toc", [])[:20], indent=2)
    doc_summary = outline.get("doc_summary") or ""
    concepts = outline.get("doc_concepts", [])[:15]
    node_max_tokens = 1000

    system = (
        "You are inventorying a legal source document for cold-call question generation. "
        "Identify all cases, statutes, and doctrinal sections present.\n\n"
        "Return JSON with:\n"
        "  doc_type_guess: 'full_case_opinion' | 'casebook_excerpt' | 'lecture_notes' | 'secondary' | 'other'\n"
        "  case_name_guess: string (primary case name if single case, else '')\n"
        "  court_guess: string\n"
        "  year_guess: string\n"
        "  has_dissent_guess: bool\n"
        "  document_summary: string (2-3 sentences)\n"
        "  identified_cases: list of case name strings found in this source (max 10)\n"
        "  identified_statutes: list of statute/rule references (max 5)\n"
        "  confidence: float 0.0-1.0\n"
        "Return only JSON."
        f"\n\n**IMPORTANT**: keep your response under {node_max_tokens} tokens."
    )

    raw = await _llm(
        "worker_mid",
        f"Document summary: {doc_summary}\n\nKey concepts: {json.dumps(concepts)}\n\n"
        f"Table of contents:\n{sections_brief}\n\nInventory this document.",
        system=system,
        max_tokens=node_max_tokens,
    )
    try:
        data = _parse_json(raw)
    except Exception:
        data = {}

    profile: SourceProfile = {
        "source_id": source_id,
        "doc_type_guess": data.get("doc_type_guess", "other"),
        "case_name_guess": data.get("case_name_guess", ""),
        "court_guess": data.get("court_guess", ""),
        "year_guess": data.get("year_guess", ""),
        "has_dissent_guess": bool(data.get("has_dissent_guess", False)),
        "document_summary": data.get("document_summary", doc_summary[:400]),
        "identified_cases": data.get("identified_cases", [])[:10],
        "identified_statutes": data.get("identified_statutes", [])[:5],
        "confidence": float(data.get("confidence", 0.5)),
    }

    await _try_save_artifact(
        state, f"source_profile:{source_id}", profile,
        "worker_mid", "source_profiler", "source_profile", source_ids=[source_id],
    )
    return {"source_profiles": [profile]}


source_profiler.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 3. corpus_synthesizer
# ─────────────────────────────────────────────────────────────────────────────

async def corpus_synthesizer(state: AgentState) -> Dict:
    """
    Merge all source profiles into a consolidated inventory.
    Produces corpus_analysis: deduplicated list of cases with source refs.
    """
    profiles = state.get("source_profiles") or []
    job_plan = state.get("job_plan") or {}

    profiles_brief = [
        {
            "source_id": p["source_id"],
            "doc_type": p["doc_type_guess"],
            "primary_case": p["case_name_guess"],
            "cases_found": p["identified_cases"],
            "summary": p["document_summary"][:200],
        }
        for p in profiles
    ]
    node_max_tokens = 1500

    system = (
        "You are synthesising source profiles for a cold-call generation pipeline. "
        "Deduplicate and consolidate the identified cases and doctrinal sections.\n\n"
        "Return JSON with:\n"
        "  cases: [{case_name, case_id (e.g. 'case_001'), source_ids, court, year, has_dissent, is_primary}]\n"
        "  doctrine_sections: [{section_name, source_ids, description}] (max 5)\n"
        "  course_context: string (e.g. 'Torts / Negligence')\n"
        "  total_case_count: int\n"
        "Return only JSON."
        f"\n\n**IMPORTANT**: keep your response under {node_max_tokens} tokens."
    )

    raw = await _llm(
        "worker_mid",
        f"Source profiles:\n{json.dumps(profiles_brief, indent=2)}\n\n"
        f"Course context from plan: {job_plan.get('course_context', '')}\n"
        f"Consolidate into a case inventory.",
        system=system,
        max_tokens=node_max_tokens,
    )
    try:
        data = _parse_json(raw)
    except Exception:
        data = {}

    # Fallback: build one pseudo-case from the first profile
    if not data.get("cases") and profiles:
        first = profiles[0]
        data["cases"] = [{
            "case_name": first.get("case_name_guess") or "Unnamed Case",
            "case_id": "case_001",
            "source_ids": [first["source_id"]],
            "court": first.get("court_guess", ""),
            "year": first.get("year_guess", ""),
            "has_dissent": first.get("has_dissent_guess", False),
            "is_primary": True,
        }]

    # Ensure unique, sequential case_ids
    for i, case in enumerate(data.get("cases", []), start=1):
        if not case.get("case_id"):
            case["case_id"] = f"case_{i:03d}"

    corpus_analysis = {
        "cases": data.get("cases", []),
        "doctrine_sections": data.get("doctrine_sections", [])[:5],
        "course_context": data.get("course_context", ""),
        "total_case_count": data.get("total_case_count", len(data.get("cases", []))),
    }

    await _try_save_artifact(
        state, "corpus_analysis", corpus_analysis,
        "worker_mid", "corpus_synthesizer", "corpus_analysis",
        source_ids=state.get("source_ids"),
    )
    return {"corpus_analysis": corpus_analysis}


corpus_synthesizer.default_worker_class = "worker_mid"


def corpus_synthesizer_to_extractor(state: AgentState) -> List[Send]:
    """Fan-out: one case_rule_extractor per identified case."""
    cases = (state.get("corpus_analysis") or {}).get("cases", [])
    logger.info("cold_call x%d node fan out for case_rule_extractor", len(cases))
    if not cases:
        return []
    return [
        Send("case_rule_extractor", {"case_item": case, **state})
        for case in cases
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 4. case_rule_extractor  (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def case_rule_extractor(state: Dict) -> Dict:
    """
    Extract a full structured CaseRuleObject for one identified case.
    Retrieves targeted chunks for: facts, posture, issue, holding, rule,
    reasoning, dicta, dissent, policy.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import COLD_CALL_RETRIEVER_TOOLS

    case_item: Dict = state["case_item"]
    case_name = case_item.get("case_name", "")
    case_id = case_item.get("case_id", "case_001")
    source_ids = case_item.get("source_ids") or state.get("source_ids") or []
    project_id = state["project_id"]

    tools = make_tools(
        project_id,
        source_ids=source_ids,
        use_voyage=state.get("use_voyage", False),
        tool_names=COLD_CALL_RETRIEVER_TOOLS,
    )
    hybrid_tool = next((t for t in tools if t.name == "hybrid_search"), None)
    search_tool = next((t for t in tools if t.name == "search_passages"), None)

    # Pull targeted chunks for each brief element
    queries = [
        f"{case_name} procedural posture facts",
        f"{case_name} issue holding rule",
        f"{case_name} reasoning policy dissent",
        f"{case_name} legally relevant facts elements",
    ]
    seen_ids: set = set()
    all_chunks: List[Dict] = []
    for q in queries:
        try:
            tool = hybrid_tool or search_tool
            if tool:
                raw = await tool.ainvoke({"query": q, "k": 12})
                for c in json.loads(raw):
                    cid = c.get("id") or c.get("chunk_id") or ""
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        all_chunks.append(c)
        except Exception as exc:
            logger.debug("case_rule_extractor query='%s' failed: %s", q[:60], exc)

    context = "\n\n---\n\n".join(
        f"[chunk_id:{c.get('id', c.get('chunk_id', '?'))} p.{c.get('page_number', '?')}]\n"
        f"{c.get('content', '')}"
        for c in all_chunks[:25]
    )

    _node_start("case_rule_extractor", state,
                case=case_name[:40], source_ids=[s[:8] for s in source_ids[:3]])
    node_max_tokens = 3000

    system = (
        "You are a T-14 law professor extracting a complete structured case analysis for "
        "cold-call question generation. Your extraction must be thorough enough to support "
        "8-deep Socratic questioning — from direct fact comprehension through rule boundary "
        "testing and policy analysis.\n\n"
        "T-14 EXTRACTION REQUIREMENTS:\n"
        "• legally_relevant_facts: identify ONLY the facts the court's rule actually hinges on "
        "(not background narrative); these become the FACT_CHANGE_HYPO targets\n"
        "• rule.elements: list ALL required elements in the order courts apply them\n"
        "• rule.exceptions: list every exception, carve-out, and limiting doctrine\n"
        "• reasoning: each step must be a distinct analytical move the court made, "
        "not a summary\n"
        "• dicta: flag statements about what the rule IS NOT or future cases\n"
        "• policy_concerns: name the specific policy values at stake "
        "(efficiency, fairness, administrability, notice, etc.)\n\n"
        "Return JSON with:\n"
        "  case_name: str\n"
        "  court: str\n"
        "  year: str\n"
        "  procedural_posture: str\n"
        "  facts: [str] (all relevant facts, max 8)\n"
        "  legally_relevant_facts: [str] (ONLY facts the rule's outcome hinges on, max 5)\n"
        "  issue: 'Whether ...' string (narrow, precise)\n"
        "  holding: str (narrow answer to the issue, one sentence)\n"
        "  rule: {\n"
        "    rule_statement: str (full black-letter rule — not a fragment),\n"
        "    elements: [str] (ALL required elements in order),\n"
        "    exceptions: [str] (ALL exceptions and carve-outs),\n"
        "    burdens: [str] (who bears each burden),\n"
        "    rule_type: categorical|balancing|element_based|factor_based\n"
        "  }\n"
        "  reasoning: [str] (distinct analytical steps, max 6)\n"
        "  dicta: [str] (statements not necessary to the holding, max 3)\n"
        "  dissent: str (dissent summary with its core objection, '' if none)\n"
        "  policy_concerns: [str] (policy rationales with the specific values at stake, max 4)\n"
        "  source_refs: [str] (chunk_ids from context)\n"
        "Return only JSON."
        f"\n\n**IMPORTANT**: keep your response under {node_max_tokens} tokens."
    )

    raw = await _llm(
        "orchestrator",
        f"Case to extract: {case_name}\n\nSource context:\n{context}\n\nExtract the full case/rule object.",
        system=system,
        max_tokens=node_max_tokens,
        _node="case_rule_extractor",
    )
    try:
        data = _parse_json(raw)
    except Exception as exc:
        _check_token_limit(exc, "case_rule_extractor", state)
        _node_warn("case_rule_extractor", state, f"JSON parse failed ({exc}) — using empty dict")
        data = {}

    obj: CaseRuleObject = {
        "case_id": case_id,
        "source_id": source_ids[0] if source_ids else "",
        "case_name": data.get("case_name", case_name),
        "court": data.get("court", case_item.get("court", "")),
        "year": data.get("year", case_item.get("year", "")),
        "procedural_posture": data.get("procedural_posture", ""),
        "facts": data.get("facts", [])[:8],
        "legally_relevant_facts": data.get("legally_relevant_facts", [])[:5],
        "issue": data.get("issue", ""),
        "holding": data.get("holding", ""),
        "rule": {
            "rule_statement": (data.get("rule") or {}).get("rule_statement", ""),
            "elements": (data.get("rule") or {}).get("elements", [])[:6],
            "exceptions": (data.get("rule") or {}).get("exceptions", [])[:4],
            "burdens": (data.get("rule") or {}).get("burdens", [])[:3],
            "rule_type": (data.get("rule") or {}).get("rule_type", "element_based"),
        },
        "reasoning": data.get("reasoning", [])[:6],
        "dicta": data.get("dicta", [])[:3],
        "dissent": data.get("dissent", ""),
        "policy_concerns": data.get("policy_concerns", [])[:4],
        "source_refs": [c.get("id") or c.get("chunk_id", "") for c in all_chunks[:10]],
    }

    await _try_save_artifact(
        state, f"case_rule_object:{case_id}", obj,
        "orchestrator", "case_rule_extractor", "case_rule_object",
        source_ids=source_ids,
    )
    _node_done("case_rule_extractor", state,
               case=obj.get("case_name", "")[:40],
               n_elements=len((obj.get("rule") or {}).get("elements", [])),
               n_facts=len(obj.get("legally_relevant_facts", [])))
    return {"case_rule_objects": [obj]}


case_rule_extractor.default_worker_class = "orchestrator"
case_rule_extractor.escalation_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 5. doctrine_mapper
# ─────────────────────────────────────────────────────────────────────────────

async def doctrine_mapper(state: AgentState) -> Dict:
    """
    Build a doctrinal network across all extracted case/rule objects.
    Identifies relationships: DEFINES, DISTINGUISHES, EXPANDS, LIMITS, CONFLICTS_WITH.
    """
    cases = state.get("case_rule_objects") or []
    corpus_analysis = state.get("corpus_analysis") or {}

    if len(cases) <= 1:
        # Single case — minimal map with one node
        single = cases[0] if cases else {}
        doctrine_map: DoctrineMap = {
            "topic": corpus_analysis.get("course_context", "Unknown Doctrine"),
            "subdoctrine": single.get("rule", {}).get("rule_statement", "")[:120],
            "nodes": [{"id": single.get("case_id", "case_001"), "type": "case", "label": single.get("case_name", "")}] if single else [],
            "edges": [],
        }
        await _try_save_artifact(
            state, "doctrine_map", dict(doctrine_map),
            "orchestrator", "doctrine_mapper", "doctrine_map",
            source_ids=state.get("source_ids"),
        )
        return {"doctrine_map": doctrine_map}

    cases_summary = [
        {
            "case_id": c["case_id"],
            "case_name": c["case_name"],
            "holding": c["holding"][:200],
            "rule_statement": c["rule"].get("rule_statement", "")[:200],
            "elements": c["rule"].get("elements", [])[:4],
        }
        for c in cases
    ]
    node_max_tokens = 2000

    system = (
        "You are a legal doctrine mapper. Given multiple case/rule objects, "
        "build a doctrinal network showing how the cases relate.\n\n"
        "Return JSON with:\n"
        "  topic: str (overall doctrine area, e.g. 'Negligence')\n"
        "  subdoctrine: str (more specific topic, e.g. 'Duty / Foreseeability')\n"
        "  nodes: [{id, type: case|rule|doctrine, label}]\n"
        "  edges: [{from_id, to_id, rel_type: DEFINES|DISTINGUISHES|EXPANDS|LIMITS|EXCEPTION_TO|ANALOGOUS_TO|CONFLICTS_WITH, rationale, source_refs:[]}]\n"
        "Return only JSON."
        f"\n\n**IMPORTANT**: keep your response under {node_max_tokens} tokens."
    )

    raw = await _llm(
        "orchestrator",
        f"Cases:\n{json.dumps(cases_summary, indent=2)}\n\n"
        f"Course context: {corpus_analysis.get('course_context', '')}\n\n"
        f"Build the doctrinal map.",
        system=system,
        max_tokens=node_max_tokens,
    )
    try:
        data = _parse_json(raw)
    except Exception:
        data = {}

    # Ensure all cases appear as nodes
    existing_node_ids = {n.get("id") for n in (data.get("nodes") or [])}
    all_nodes = list(data.get("nodes") or [])
    for c in cases:
        if c["case_id"] not in existing_node_ids:
            all_nodes.append({"id": c["case_id"], "type": "case", "label": c["case_name"]})

    doctrine_map = {
        "topic": data.get("topic", corpus_analysis.get("course_context", "Unknown")),
        "subdoctrine": data.get("subdoctrine", ""),
        "nodes": all_nodes,
        "edges": data.get("edges", []),
    }

    await _try_save_artifact(
        state, "doctrine_map", doctrine_map,
        "orchestrator", "doctrine_mapper", "doctrine_map",
        source_ids=state.get("source_ids"),
    )
    return {"doctrine_map": doctrine_map}


doctrine_mapper.default_worker_class = "orchestrator"
doctrine_mapper.escalation_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 6. compare_distinguish_mapper
# ─────────────────────────────────────────────────────────────────────────────

async def compare_distinguish_mapper(state: AgentState) -> Dict:
    """
    Generate compare-and-distinguish question prompts for case pairs identified
    in the doctrine map. Runs once sequentially after doctrine_mapper.
    """
    cases = state.get("case_rule_objects") or []
    doctrine_map = state.get("doctrine_map") or {}
    edges = doctrine_map.get("edges") or []
    node_max_tokens = 600

    if len(cases) < 2:
        return {"compare_distinguish_prompts": []}

    case_by_id = {c["case_id"]: c for c in cases}

    # Only generate prompts for pairs connected by a doctrine edge
    prompted_pairs: set = set()
    prompts: List[CompareDistinguishPrompt] = []

    for edge in edges[:8]:  # cap to avoid O(n²) explosion
        a_id = edge.get("from_id", "")
        b_id = edge.get("to_id", "")
        pair_key = tuple(sorted([a_id, b_id]))
        if pair_key in prompted_pairs:
            continue
        prompted_pairs.add(pair_key)

        ca = case_by_id.get(a_id)
        cb = case_by_id.get(b_id)
        if not ca or not cb:
            continue

        system = (
            "You are generating a compare-and-distinguish cold-call question for two cases.\n\n"
            "Return JSON with:\n"
            "  relationship: str (DISTINGUISHES|ANALOGOUS_TO|CONFLICTS_WITH|EXPANDS|LIMITS)\n"
            "  question: str (the professor's compare/distinguish question, 1-2 sentences)\n"
            "  model_distinction: str (the answer students should give, 2-3 sentences)\n"
            "Return only JSON."
            f"\n\n**IMPORTANT**: keep your response under {node_max_tokens} tokens."
        )

        raw = await _llm(
            "worker_mid",
            f"Case A: {ca['case_name']}\nHolding: {ca['holding']}\nRule: {ca['rule'].get('rule_statement','')}\n\n"
            f"Case B: {cb['case_name']}\nHolding: {cb['holding']}\nRule: {cb['rule'].get('rule_statement','')}\n\n"
            f"Doctrine relationship: {edge.get('rel_type','')}\nRationale: {edge.get('rationale','')}\n\n"
            f"Generate a compare/distinguish question.",
            system=system,
            max_tokens=node_max_tokens,
        )
        try:
            d = _parse_json(raw)
        except Exception:
            d = {}

        cmp_id = f"cmp_{len(prompts)+1:03d}"
        prompts.append({
            "comparison_id": cmp_id,
            "case_a_id": a_id,
            "case_b_id": b_id,
            "case_a_name": ca["case_name"],
            "case_b_name": cb["case_name"],
            "relationship": d.get("relationship", edge.get("rel_type", "ANALOGOUS_TO")),
            "question": d.get("question", f"How would you distinguish {ca['case_name']} from {cb['case_name']}?"),
            "model_distinction": d.get("model_distinction", ""),
            "source_refs": edge.get("source_refs", []),
        })

    await _try_save_artifact(
        state, "compare_distinguish_prompts", {"prompts": prompts},
        "worker_mid", "compare_distinguish_mapper", "compare_distinguish_prompts",
        source_ids=state.get("source_ids"),
    )
    return {"compare_distinguish_prompts": prompts}


compare_distinguish_mapper.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 7. question_type_bank_selector
# ─────────────────────────────────────────────────────────────────────────────

async def question_type_bank_selector(state: AgentState) -> Dict:
    """
    Select which question types to use for this run based on case count,
    target difficulty, and requested depth. Deterministic / low-temperature.
    """
    target_difficulty = (state.get("job_plan") or {}).get("target_difficulty", state.get("target_difficulty", "day_one_t14"))
    cases = state.get("case_rule_objects") or []
    has_multi_case = len(cases) > 1
    has_dissent = any(c.get("dissent") for c in cases)

    early = list(QUESTION_TYPES_EARLY)
    middle = list(QUESTION_TYPES_MIDDLE)
    deep = list(QUESTION_TYPES_DEEP)

    if has_multi_case:
        deep += QUESTION_TYPES_CONDITIONAL_MULTI_CASE_DEEP
    if has_dissent:
        middle += QUESTION_TYPES_CONDITIONAL_DISSENT_MIDDLE
    if target_difficulty == "advanced":
        deep += QUESTION_TYPES_CONDITIONAL_ADVANCED_DEEP
        middle += QUESTION_TYPES_CONDITIONAL_ADVANCED_MIDDLE

    optional = []
    if has_multi_case:
        optional += QUESTION_TYPES_OPTIONAL_MULTI_CASE
    if target_difficulty != "law_1l":
        optional += QUESTION_TYPES_OPTIONAL_UPPER_LEVEL

    selection: QuestionTypeSelection = {
        "selected_question_types": {
            "early": early,
            "middle": middle,
            "deep": deep,
        },
        "required_types": QUESTION_TYPES_REQUIRED_ALWAYS,
        "optional_types": optional,
    }

    await _try_save_artifact(
        state, "question_type_selection", dict(selection),
        "worker_mid", "question_type_bank_selector", "question_type_selection",
        source_ids=state.get("source_ids"),
    )
    return {"question_type_selection": selection}


question_type_bank_selector.default_worker_class = "worker_mid"


def type_selector_to_seed_generators(state: AgentState) -> List[Send]:
    """Fan-out: one cold_call_seed_generator per extracted case."""
    cases = state.get("case_rule_objects") or []
    logger.info("cold_call x%d node fan out for cold_call_seed_generator", len(cases))
    if not cases:
        return []
    return [
        Send("cold_call_seed_generator", {"case_obj": case, **state})
        for case in cases
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 8. cold_call_seed_generator  (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def cold_call_seed_generator(state: Dict) -> Dict:
    """
    Generate diverse cold-call seed questions for one case.
    Produces SEEDS_PER_CASE_MULTIPLIER × requested seeds to give the
    diversity agent a larger pool to draw from.
    """
    case_obj: CaseRuleObject = state.get("case_obj") or (state.get("case_rule_objects") or [{}])[0]
    requested = state.get("requested_sequence_count", 5)
    q_selection: QuestionTypeSelection = state.get("question_type_selection") or {
        "selected_question_types": {"early": [], "middle": [], "deep": []},
        "required_types": [],
        "optional_types": [],
    }
    n_seeds = max(requested, SEEDS_PER_CASE_MULTIPLIER * max(1, requested // max(1, len(state.get("case_rule_objects") or [1]))))

    _node_start("cold_call_seed_generator", state,
                case=case_obj.get("case_name", "")[:40], n_seeds=n_seeds)
    node_max_tokens = 3000

    system = (
        "You are a T-14 law professor generating diverse cold-call seed questions. "
        "Each seed is the OPENING QUESTION of a Socratic thread — it must be concrete, "
        "anchored to the specific case, and designed to reveal a student's depth of "
        "preparation at the targeted difficulty level.\n\n"
        "T-14 SEED REQUIREMENTS:\n"
        "• Each seed must target a DIFFERENT angle so the resulting threads do not overlap\n"
        "• Early seeds: test direct case comprehension (facts, posture, holding)\n"
        "• Middle seeds: test rule application (elements, exceptions, fact probes)\n"
        "• Deep seeds: test analysis (rule boundary, policy, compare/distinguish, hypos)\n"
        "• Questions must name specific case facts — never generic ('What is the rule?')\n"
        "• The question must be phrased as a professor would ask it in class, "
        "not as a textbook prompt\n\n"
        "Required seed themes (cover as many as requested count allows):\n"
        f"  {json.dumps(SEED_THEMES)}\n\n"
        "Question type bank (early/middle/deep):\n"
        f"  {json.dumps(q_selection.get('selected_question_types', {}))}\n\n"
        f"Return a JSON array of exactly {n_seeds} seed objects. Each:\n"
        "  seed_id: 'seed_NNN'\n"
        "  sequence_theme: one of the required themes\n"
        "  question_type: from the question type bank\n"
        "  question: the opening question text (concrete, case-specific, professor-voiced)\n"
        "  target_skill: what this question tests (e.g. 'holding identification', "
        "'rule boundary', 'policy analysis')\n"
        "  difficulty: 'early' | 'middle' | 'deep'\n"
        "Return only the JSON array."
        f"\n\n**IMPORTANT**: keep your response under {node_max_tokens} tokens."
    )

    case_ctx = (
        f"Case: {case_obj.get('case_name', '')}\n"
        f"Issue: {case_obj.get('issue', '')}\n"
        f"Holding: {case_obj.get('holding', '')}\n"
        f"Rule: {(case_obj.get('rule') or {}).get('rule_statement', '')}\n"
        f"Legally relevant facts: {json.dumps(case_obj.get('legally_relevant_facts', []))}\n"
        f"Policy concerns: {json.dumps(case_obj.get('policy_concerns', []))}\n"
    )

    raw = await _llm("orchestrator", f"{case_ctx}\n\nGenerate {n_seeds} diverse seeds.", system=system, max_tokens=node_max_tokens, _node="cold_call_seed_generator")
    try:
        seeds_raw = _parse_json(raw)
        if not isinstance(seeds_raw, list):
            seeds_raw = []
    except Exception:
        seeds_raw = []

    case_id = case_obj.get("case_id", "case_001")
    seeds: List[Seed] = []
    for i, s in enumerate(seeds_raw):
        if not isinstance(s, dict):
            continue
        seeds.append({
            "seed_id": s.get("seed_id") or f"seed_{case_id}_{i:03d}",
            "case_id": case_id,
            "sequence_theme": s.get("sequence_theme", SEED_THEMES[i % len(SEED_THEMES)]),
            "question_type": s.get("question_type", "LEGALLY_RELEVANT_FACTS"),
            "question": s.get("question", ""),
            "target_skill": s.get("target_skill", ""),
            "difficulty": s.get("difficulty", "early"),
        })

    await _try_save_artifact(
        state, f"seeds:{case_id}", {"case_id": case_id, "seeds": seeds},
        "orchestrator", "cold_call_seed_generator", "seeds",
        source_ids=state.get("source_ids"),
    )
    _node_done("cold_call_seed_generator", state,
               case_id=case_id[:8], n_seeds=len(seeds))
    return {"seeds": seeds}


cold_call_seed_generator.default_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 9. seed_diversity_agent  (merge + retry gate)
# ─────────────────────────────────────────────────────────────────────────────

async def seed_diversity_agent(state: AgentState) -> Dict:
    """
    Check diversity of accumulated seeds. Approve a diverse subset of
    requested_sequence_count seeds. Increment seed_attempt_count for retry tracking.
    """
    all_seeds = state.get("seeds") or []
    requested = state.get("requested_sequence_count", 5)
    attempt = (state.get("seed_attempt_count") or 0) + 1

    if not all_seeds:
        return {"approved_seeds": [], "seed_attempt_count": attempt}
    node_max_tokens = 1200

    seeds_brief = [
        {"seed_id": s["seed_id"], "theme": s["sequence_theme"], "type": s["question_type"], "question": s["question"][:80]}
        for s in all_seeds
    ]

    system = (
        "You are a seed diversity evaluator for cold-call question generation.\n"
        "Evaluate whether the seeds cover diverse angles (core understanding, rule boundary, "
        "policy, hypo, compare/distinguish, counterargument, posture, exam transfer).\n\n"
        "Return JSON with:\n"
        f"  approved_seed_ids: list of exactly {requested} (or fewer if < {requested} available) "
        "seed_id strings — the most diverse set\n"
        "  diversity_score: float 0.0-1.0 (how well the approved set covers different angles)\n"
        "  coverage_by_theme: {theme: count}\n"
        "  duplicates_flagged: [{seed_id, reason}]\n"
        "Return only JSON."
        f"\n\n**IMPORTANT**: keep your response under {node_max_tokens} tokens."
    )

    raw = await _llm(
        "worker_mid",
        f"All generated seeds ({len(all_seeds)} total):\n{json.dumps(seeds_brief, indent=2)}\n\n"
        f"Select the best {requested} diverse seeds.",
        system=system,
        max_tokens=node_max_tokens,
    )
    try:
        data = _parse_json(raw)
    except Exception:
        data = {}

    approved_ids: set = set(data.get("approved_seed_ids") or [])
    diversity_score = float(data.get("diversity_score", 0.5))

    # If no explicit selection, take first N by theme coverage
    if not approved_ids:
        theme_seen: set = set()
        for s in all_seeds:
            if len(approved_ids) >= requested:
                break
            t = s["sequence_theme"]
            if t not in theme_seen or len(approved_ids) < requested:
                approved_ids.add(s["seed_id"])
                theme_seen.add(t)

    approved_seeds = [s for s in all_seeds if s["seed_id"] in approved_ids][:requested]

    await _try_save_artifact(
        state, f"seed_diversity:{attempt}", {
            "diversity_score": diversity_score,
            "approved_count": len(approved_seeds),
            "total_seeds": len(all_seeds),
            "attempt": attempt,
        },
        "worker_mid", "seed_diversity_agent", "seed_diversity",
        source_ids=state.get("source_ids"),
    )
    return {
        "approved_seeds": approved_seeds,
        "seed_attempt_count": attempt,
    }


seed_diversity_agent.default_worker_class = "worker_mid"


def seeds_routing(state: AgentState) -> List[Send]:
    """
    Route after seed_diversity_agent:
    - If diversity insufficient AND under retry limit → re-fan-out to cold_call_seed_generator
    - Otherwise → fan-out to socratic_thread_builder (one per approved seed)
    Also injects compare/distinguish prompts as synthetic thread seeds.
    """
    approved_seeds = state.get("approved_seeds") or []
    requested = state.get("requested_sequence_count", 5)
    attempt = state.get("seed_attempt_count") or 1
    cases = state.get("case_rule_objects") or []

    needs_retry = (
        len(approved_seeds) < min(requested, 2)
        and attempt < MAX_SEED_RETRIES
        and cases
    )

    if needs_retry:
        logger.info("seed_diversity_agent: retry %d/%d — insufficient diversity", attempt, MAX_SEED_RETRIES)
        logger.info("cold_call x%d node fan out for cold_call_seed_generator (diversity retry)", len(cases))
        return [
            Send("cold_call_seed_generator", {"case_obj": case, **state})
            for case in cases
        ]

    # Proceed to thread builders
    sends = [
        Send("socratic_thread_builder", {"seed": seed, **state})
        for seed in approved_seeds
    ]

    # Also fan-out compare/distinguish prompts as extra thread seeds
    cmp_prompts = state.get("compare_distinguish_prompts") or []
    for cmp in cmp_prompts[:max(0, requested - len(approved_seeds))]:
        synthetic_seed: Seed = {
            "seed_id": f"seed_cmp_{cmp['comparison_id']}",
            "case_id": cmp["case_a_id"],
            "sequence_theme": "COMPARE_DISTINGUISH",
            "question_type": "COMPARE_DISTINGUISH",
            "question": cmp["question"],
            "target_skill": "compare-and-distinguish two cases",
            "difficulty": "deep",
        }
        sends.append(Send("socratic_thread_builder", {"seed": synthetic_seed, **state}))

    logger.info("cold_call x%d node fan out for socratic_thread_builder", len(sends))
    return sends


# ─────────────────────────────────────────────────────────────────────────────
# 10. socratic_thread_builder  (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def socratic_thread_builder(state: Dict) -> Dict:
    """
    Build a full 8-deep Socratic question sequence (1A–1H) from one seed.
    Depth progression:
      A,B → early (direct comprehension, reasoning chain)
      C,D,E → middle (fact probe, holding vs dicta, hypo)
      F,G,H → deep (rule boundary, counterargument/losing side, policy/exam)
    Includes inline hypothetical generation for the FACT_CHANGE_HYPO slot.
    """
    node_max_tokens = 4000
    seed: Seed = state["seed"]
    case_id = seed.get("case_id", "")
    case_obj = next(
        (c for c in (state.get("case_rule_objects") or []) if c["case_id"] == case_id),
        (state.get("case_rule_objects") or [{}])[0] if state.get("case_rule_objects") else {},
    )
    q_selection: QuestionTypeSelection = state.get("question_type_selection") or {}
    doctrine_map = state.get("doctrine_map") or {}
    compare_prompts = state.get("compare_distinguish_prompts") or []

    # Find any compare/distinguish prompt relevant to this case
    relevant_cmp = next(
        (p for p in compare_prompts if p.get("case_a_id") == case_id or p.get("case_b_id") == case_id),
        None,
    )

    _node_start("socratic_thread_builder", state,
                seed_id=seed.get("seed_id", "?"), case_id=case_id[:8],
                theme=seed.get("sequence_theme", "?"))

    system = (
        "You are a T-14 law professor building a Socratic cold-call question sequence. "
        "The sequence must escalate methodically — each question should be harder to answer "
        "than the one before, and should build on what a strong student would have said.\n\n"
        "T-14 DEPTH STRUCTURE — 8 questions (A through H):\n"
        "  A: Direct comprehension — 'What are the legally relevant facts?' "
        "(Do not ask for a conclusion; ask for facts only)\n"
        "  B: Reasoning chain — 'Why did the court reach that conclusion?' "
        "(Ask for the court's analytical steps, not just the holding)\n"
        "  C: Legally relevant fact probe — 'Which of those facts actually mattered to the rule?' "
        "(Force the student to distinguish material from background facts)\n"
        "  D: Holding vs. dicta — 'Was that statement necessary to the holding?' "
        "(Test whether the student understands the scope of the precedent)\n"
        "  E: Fact-change hypothetical — 'What if [specific key fact] had been different — "
        "would the result change?' (Name the exact fact being changed)\n"
        "  F: Rule boundary — 'Where does the rule stop? Give me a case where it wouldn't apply.'\n"
        "  G: Counterargument / losing side — 'What is the best argument for the losing party?'\n"
        "  H: Policy or exam application — 'What policy value does this rule serve?' "
        "or 'How would you argue this issue on an exam?'\n\n"
        "T-14 QUESTION STANDARDS:\n"
        "• Each question must be phrased as a professor would ask it aloud in class\n"
        "• Each question must reference specific case facts — never generic\n"
        "• Each question must logically follow from the prior (do not reset context)\n"
        "• expected_answer_shape: describe what an A student would say in 2-3 sentences\n\n"
        "Return a JSON array of exactly 8 question objects. Each:\n"
        "  question_id: '1A' through '1H'\n"
        "  question_index: 1-8\n"
        "  question_type: the type tag (e.g. LEGALLY_RELEVANT_FACTS)\n"
        "  depth_position: 'early' | 'middle' | 'deep'\n"
        "  question_text: str (the professor's actual spoken question)\n"
        "  target_skill: str (what this question tests)\n"
        "  expected_answer_shape: str (what an A student says, 2-3 sentences)\n"
        "  source_refs: [] (leave empty; filled by grounder)\n"
        "  metadata: {}\n"
        "Return only the JSON array."
        f"\n\n**IMPORTANT**: keep your response under {node_max_tokens} tokens."
    )

    cmp_ctx = ""
    if relevant_cmp:
        cmp_ctx = (
            f"\nCompare/distinguish context: {relevant_cmp.get('question', '')}\n"
            f"Model distinction: {relevant_cmp.get('model_distinction', '')}\n"
        )

    case_ctx = (
        f"Case: {case_obj.get('case_name', '')}\n"
        f"Procedural posture: {case_obj.get('procedural_posture', '')}\n"
        f"Legally relevant facts: {json.dumps(case_obj.get('legally_relevant_facts', []))}\n"
        f"Issue: {case_obj.get('issue', '')}\n"
        f"Holding: {case_obj.get('holding', '')}\n"
        f"Rule: {(case_obj.get('rule') or {}).get('rule_statement', '')}\n"
        f"Elements: {json.dumps((case_obj.get('rule') or {}).get('elements', []))}\n"
        f"Reasoning: {json.dumps(case_obj.get('reasoning', []))}\n"
        f"Dissent: {case_obj.get('dissent', '')[:200]}\n"
        f"Policy concerns: {json.dumps(case_obj.get('policy_concerns', []))}\n"
        f"{cmp_ctx}"
    )

    seed_ctx = (
        f"\nSeed theme: {seed.get('sequence_theme', '')}\n"
        f"Opening question type: {seed.get('question_type', '')}\n"
        f"Opening question: {seed.get('question', '')}\n"
        f"Target skill: {seed.get('target_skill', '')}\n"
    )

    raw = await _llm(
        "orchestrator",
        f"{case_ctx}{seed_ctx}\nBuild the 8-question Socratic thread.",
        system=system,
        max_tokens=node_max_tokens,
        _node="socratic_thread_builder",
    )
    try:
        questions_raw = _parse_json(raw)
        if not isinstance(questions_raw, list):
            questions_raw = []
    except Exception:
        questions_raw = []

    seq_idx = len(state.get("socratic_sequences") or []) + 1
    seq_id = f"seq_{case_id}_{seed.get('seed_id', 'unknown')}_{seq_idx:03d}"

    questions: List[SequenceQuestion] = []
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    for i, q in enumerate(questions_raw[:9]):
        if not isinstance(q, dict):
            continue
        label = labels[i] if i < len(labels) else str(i + 1)
        depth = DEPTH_MAP.get(label, "deep")
        questions.append({
            "question_id": q.get("question_id") or f"1{label}",
            "question_index": i + 1,
            "question_type": q.get("question_type", "LEGALLY_RELEVANT_FACTS"),
            "depth_position": q.get("depth_position", depth),
            "question_text": q.get("question_text", ""),
            "target_skill": q.get("target_skill", ""),
            "difficulty_label": "",  # filled by critic_coverage_agent
            "expected_answer_shape": q.get("expected_answer_shape", ""),
            "source_refs": q.get("source_refs", []),
            "metadata": q.get("metadata", {}),
        })

    coverage_tags = list({q["question_type"] for q in questions})

    sequence: QuestionSequence = {
        "sequence_id": seq_id,
        "case_id": case_id,
        "case_name": case_obj.get("case_name", ""),
        "seed_id": seed.get("seed_id", ""),
        "sequence_theme": seed.get("sequence_theme", ""),
        "sequence_index": seq_idx,
        "questions": questions,
        "coverage_tags": coverage_tags,
        "source_refs": case_obj.get("source_refs", []),
        "metadata": {
            "generated_by": "socratic_thread_builder",
            "doctrine": (state.get("doctrine_map") or {}).get("topic", ""),
            "subdoctrine": (state.get("doctrine_map") or {}).get("subdoctrine", ""),
        },
    }

    await _try_save_artifact(
        state, f"question_sequence:{seq_id}", dict(sequence),
        "orchestrator", "socratic_thread_builder", "question_sequence",
        source_ids=state.get("source_ids"),
    )
    _node_done("socratic_thread_builder", state,
               seq_id=seq_id[:16], n_questions=len(questions),
               theme=seed.get("sequence_theme", "?"))
    return {"socratic_sequences": [sequence]}


socratic_thread_builder.default_worker_class = "orchestrator"
socratic_thread_builder.escalation_worker_class = "orchestrator"


def thread_builders_to_answer_agents(state: AgentState) -> List[Send]:
    """Fan-out: one socratic_answer_agent per completed question sequence."""
    seqs = state.get("socratic_sequences") or []
    logger.info("cold_call x%d node fan out for socratic_answer_agent", len(seqs))
    return [
        Send("socratic_answer_agent", {"sequence": seq, **state})
        for seq in seqs
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 11. socratic_answer_agent  (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def socratic_answer_agent(state: Dict) -> Dict:
    """
    Generate Socratic model answers for every question in one sequence.
    Answer format: Answer / Because / Unless / But / Therefore
    Also generates strong answer, common weak answer, professor trap, recovery phrase.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import COLD_CALL_RETRIEVER_TOOLS

    node_max_tokens = 6000
    sequence: QuestionSequence = state["sequence"]
    seq_id = sequence["sequence_id"]
    case_id = sequence.get("case_id", "")
    case_obj = next(
        (c for c in (state.get("case_rule_objects") or []) if c["case_id"] == case_id),
        (state.get("case_rule_objects") or [{}])[0] if state.get("case_rule_objects") else {},
    )

    # Retrieve source context for grounding
    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=COLD_CALL_RETRIEVER_TOOLS,
    )
    hybrid_tool = next((t for t in tools if t.name == "hybrid_search"), None)

    source_ctx = ""
    if hybrid_tool and case_obj.get("case_name"):
        try:
            raw_chunks = await hybrid_tool.ainvoke({
                "query": f"{case_obj['case_name']} holding rule reasoning policy",
                "k": 10,
            })
            chunks = json.loads(raw_chunks)
            source_ctx = "\n\n".join(c.get("content", "") for c in chunks[:6])
        except Exception:
            pass

    questions_ctx = json.dumps(
        [{"id": q["question_id"], "type": q["question_type"], "text": q["question_text"], "expected": q["expected_answer_shape"]} for q in sequence["questions"]],
        indent=2,
    )

    _node_start("socratic_answer_agent", state,
                seq_id=sequence.get("sequence_id", "?")[:16],
                case_id=case_id[:8])

    system = (
        "You are a T-14 law professor generating model Socratic answers for cold-call questions. "
        "Your answers must match T-14 caliber — precise rule statements, case-specific "
        "application, and acknowledgment of the best counterarguments.\n\n"
        "ANSWER FORMAT (Because / Unless / But / Therefore):\n"
        "  model_answer: 'The [party] probably [wins/loses] on [issue]. "
        "Because [precise rule application to specific facts]. "
        "Unless [key exception or competing fact]. "
        "But [the most important limiting doctrine or counterweight]. "
        "Therefore [one-sentence conclusion with confidence level].'\n\n"
        "T-14 ANSWER STANDARDS:\n"
        "• model_answer: must cite the specific rule elements and apply them to the named facts; "
        "do NOT use generic statements like 'the rule applies here'\n"
        "• strong_answer: what an A student says — more precise, acknowledges exceptions, "
        "names the legally relevant fact; 3-4 sentences minimum\n"
        "• common_weak_answer: the surface-level answer an unprepared student gives — "
        "usually correct conclusion but missing the rule mechanics\n"
        "• professor_follow_up_trap: the NEXT question a professor asks after a strong answer "
        "to probe for deeper understanding (e.g. 'What if the plaintiff had been warned?')\n"
        "• recovery_phrase: a rule-focused sentence the student can say if they blank "
        "(not just 'I need more time'; it must move the analysis forward)\n\n"
        "Return a JSON array where each object corresponds to one question_id:\n"
        "  [{question_id, model_answer, strong_answer, common_weak_answer, "
        "professor_follow_up_trap, recovery_phrase}]\n"
        "Return only the JSON array."
        f"\n\n**IMPORTANT**: keep your response under {node_max_tokens} tokens."
    )

    case_ctx = (
        f"Case: {case_obj.get('case_name', '')}\n"
        f"Holding: {case_obj.get('holding', '')}\n"
        f"Rule: {(case_obj.get('rule') or {}).get('rule_statement', '')}\n"
        f"Elements: {json.dumps((case_obj.get('rule') or {}).get('elements', []))}\n"
        f"Policy concerns: {json.dumps(case_obj.get('policy_concerns', []))}\n"
    )

    raw = await _llm(
        "orchestrator",
        f"{case_ctx}\n\nSource material:\n{source_ctx[:2000]}\n\n"
        f"Questions to answer:\n{questions_ctx}\n\n"
        f"Generate Socratic answers for each question.",
        system=system,
        max_tokens=node_max_tokens,
        _node="socratic_answer_agent",
    )
    try:
        answers_raw = _parse_json(raw)
        if not isinstance(answers_raw, list):
            answers_raw = []
    except Exception:
        answers_raw = []

    answers: List[QuestionAnswer] = []
    for a in answers_raw:
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

    answer_seq: AnswerSequence = {
        "sequence_id": seq_id,
        "answers": answers,
    }

    await _try_save_artifact(
        state, f"answer_sequence:{seq_id}", dict(answer_seq),
        "orchestrator", "socratic_answer_agent", "answer_sequence",
        source_ids=state.get("source_ids"),
    )
    _node_done("socratic_answer_agent", state,
               seq_id=seq_id[:16], n_answers=len(answers))
    return {"answer_sequences": [answer_seq]}


socratic_answer_agent.default_worker_class = "orchestrator"
socratic_answer_agent.escalation_worker_class = "orchestrator"


def answer_agents_to_grounders(state: AgentState) -> List[Send]:
    """Fan-out: one grounder_agent per answer sequence."""
    answers = state.get("answer_sequences") or []
    logger.info("cold_call x%d node fan out for grounder_agent", len(answers))
    return [
        Send("grounder_agent", {"answer_seq": ans, **state})
        for ans in answers
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 12. grounder_agent  (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def grounder_agent(state: Dict) -> Dict:
    """
    Verify that every model answer in one sequence is supported by the source corpus.
    Flags unsupported claims, overbroad rules, and invented facts.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import COLD_CALL_VERIFIER_TOOLS

    answer_seq: AnswerSequence = state["answer_seq"]
    seq_id = answer_seq["sequence_id"]

    tools = make_tools(
        state["project_id"],
        source_ids=state.get("source_ids", []),
        use_voyage=state.get("use_voyage", False),
        tool_names=COLD_CALL_VERIFIER_TOOLS,
    )
    verify_tool = next((t for t in tools if t.name == "verify_claim"), None)

    claims_checked: List[Dict] = []
    unsupported_count = 0

    for ans in answer_seq.get("answers") or []:
        claim_text = ans.get("model_answer", "")[:300]
        if not claim_text:
            continue

        status = "unverified"
        if verify_tool:
            try:
                result_json = await verify_tool.ainvoke({"claim": claim_text, "k": 6})
                result = json.loads(result_json)
                verdict = result.get("verdict", "insufficient")
                status = "supported" if verdict == "supported" else "unsupported" if verdict == "contradicted" else "weak"
                if status == "unsupported":
                    unsupported_count += 1
            except Exception:
                status = "unverified"
        else:
            # Lightweight heuristic: flag obviously invented facts
            HALLUCINATION_PATTERNS = [r'\b(always|never|universally)\b', r'\bfound liable in all\b']
            for pat in HALLUCINATION_PATTERNS:
                if re.search(pat, claim_text, re.IGNORECASE):
                    status = "weak"
                    break
            else:
                status = "supported"

        claims_checked.append({
            "question_id": ans.get("question_id", ""),
            "claim": claim_text[:120],
            "status": status,
        })

    unsupported_count = sum(1 for c in claims_checked if c["status"] == "unsupported")
    grounding_status = (
        "passed" if unsupported_count == 0
        else "passed_with_warnings" if unsupported_count <= 1
        else "failed"
    )

    result: GroundingResult = {
        "sequence_id": seq_id,
        "grounding_status": grounding_status,
        "claims_checked": claims_checked,
        "unsupported_claim_count": unsupported_count,
    }

    await _try_save_artifact(
        state, f"grounding:{seq_id}", dict(result),
        "worker_mid", "grounder_agent", "grounding_result",
        source_ids=state.get("source_ids"),
    )
    return {"grounding_results": [result]}


grounder_agent.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 13. critic_coverage_agent
# ─────────────────────────────────────────────────────────────────────────────

async def critic_coverage_agent(state: AgentState) -> Dict:
    """
    Merge point. Compute coverage across all sequences, add difficulty_label to
    every question, and produce a CoverageMap + final_sequences list.
    """
    sequences = state.get("socratic_sequences") or []
    grounding_results = state.get("grounding_results") or []

    # Index grounding by sequence_id
    grounding_by_seq = {r["sequence_id"]: r for r in grounding_results}

    # Count coverage by question type and depth
    type_counts: Dict[str, int] = {}
    depth_counts: Dict[str, int] = {"early": 0, "middle": 0, "deep": 0}

    for seq in sequences:
        for q in seq.get("questions") or []:
            qt = q.get("question_type", "UNKNOWN")
            type_counts[qt] = type_counts.get(qt, 0) + 1
            dp = q.get("depth_position", "middle")
            depth_counts[dp] = depth_counts.get(dp, 0) + 1

    total_q = sum(type_counts.values()) or 1
    must_have = [
        "PROCEDURAL_POSTURE", "RULE_EXTRACTION", "FACT_CHANGE_HYPO",
        "COUNTERARGUMENT", "POLICY_ANALYSIS",
    ]
    missing = [t for t in must_have if type_counts.get(t, 0) == 0]

    coverage_score = max(0.0, min(1.0, 1.0 - (len(missing) / len(must_have)) * 0.5))

    # Stamp difficulty_label onto each question using LLM for a quick pass
    node_max_tokens = 2000
    system = (
        "You are stamping difficulty labels on cold-call questions. "
        "For each question assign: 'easy' | 'medium' | 'hard' based on "
        "depth_position and question_type.\n\n"
        "Rules:\n"
        "  early → mostly 'easy', occasionally 'medium'\n"
        "  middle → mostly 'medium', occasionally 'hard'\n"
        "  deep → mostly 'hard', occasionally 'medium'\n"
        "  FACT_CHANGE_HYPO, RULE_BOUNDARY, COMPARE_DISTINGUISH, POLICY_ANALYSIS → bump up one level\n\n"
        "Return JSON array: [{question_id, sequence_id, difficulty_label}].\n"
        "Return only JSON."
        f"\n\n**IMPORTANT**: keep your response under {node_max_tokens} tokens."
    )

    q_list = []
    for seq in sequences:
        for q in seq.get("questions") or []:
            q_list.append({"question_id": q["question_id"], "sequence_id": seq["sequence_id"],
                           "question_type": q["question_type"], "depth_position": q["depth_position"]})

    difficulty_map: Dict[str, Dict[str, str]] = {}  # seq_id → {q_id → label}
    if q_list:
        raw = await _llm("worker_mid", f"Questions:\n{json.dumps(q_list, indent=2)}\n\nLabel difficulties.", system=system, max_tokens=node_max_tokens)
        try:
            labels = _parse_json(raw)
            if isinstance(labels, list):
                for item in labels:
                    sid = item.get("sequence_id", "")
                    qid = item.get("question_id", "")
                    if sid and qid:
                        if sid not in difficulty_map:
                            difficulty_map[sid] = {}
                        difficulty_map[sid][qid] = item.get("difficulty_label", "medium")
        except Exception:
            pass

    # Build final_sequences with difficulty_label stamped
    DEPTH_LABEL_DEFAULT = {"early": "easy", "middle": "medium", "deep": "hard"}
    final_sequences: List[QuestionSequence] = []
    for seq in sequences:
        seq_labels = difficulty_map.get(seq["sequence_id"], {})
        updated_questions: List[SequenceQuestion] = []
        for q in seq.get("questions") or []:
            updated_q = dict(q)
            updated_q["difficulty_label"] = seq_labels.get(q["question_id"]) or DEPTH_LABEL_DEFAULT.get(q.get("depth_position", "middle"), "medium")
            updated_questions.append(updated_q)  # type: ignore[arg-type]
        final_seq = dict(seq)
        final_seq["questions"] = updated_questions
        grounding = grounding_by_seq.get(seq["sequence_id"], {})
        final_seq["metadata"] = {
            **seq.get("metadata", {}),
            "grounding_status": grounding.get("grounding_status", "unverified"),
            "coverage_score": coverage_score,
            "answer_status": "answers_generated",
        }
        final_sequences.append(final_seq)  # type: ignore[arg-type]

    coverage_map: CoverageMap = {
        "overall_coverage_score": coverage_score,
        "coverage_by_question_type": type_counts,
        "coverage_by_depth": depth_counts,
        "missing_or_weak_areas": missing,
        "difficulty_labels_added": True,
    }

    await _try_save_artifact(
        state, "coverage_map", dict(coverage_map),
        "orchestrator", "critic_coverage_agent", "coverage_map",
        source_ids=state.get("source_ids"),
    )
    return {
        "coverage_map": coverage_map,
        "final_sequences": final_sequences,
    }


critic_coverage_agent.default_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 14. formatter_export_agent
# ─────────────────────────────────────────────────────────────────────────────

async def formatter_export_agent(state: AgentState) -> Dict:
    """
    Export final question + answer sequences to Supabase cold_call tables and
    produce a markdown summary for the notes stub.

    Exports to:
      public.cold_call_sequences  — one row per sequence (container)
      public.cold_call_questions  — individual question parts
      public.cold_call_answers    — individual answer parts
    """
    final_sequences = state.get("final_sequences") or state.get("socratic_sequences") or []
    answer_sequences = state.get("answer_sequences") or []
    coverage_map = state.get("coverage_map") or {}
    note_id = state.get("note_id") or ""
    budget = state.get("budget") or {}

    answer_by_seq: Dict[str, AnswerSequence] = {a["sequence_id"]: a for a in answer_sequences}

    export_batch_id = str(_uuid_mod.uuid4())
    q_exported = 0
    a_exported = 0

    # ── Supabase export ───────────────────────────────────────────────────────
    try:
        from tasks.database import get_db_connection
        import uuid as _uuid

        async with get_db_connection() as conn:
            for seq in final_sequences:
                seq_id = seq["sequence_id"]
                ans_seq = answer_by_seq.get(seq_id, {"sequence_id": seq_id, "answers": []})

                # Insert cold_call_sequences row
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
                        seq_id,
                        export_batch_id,
                        note_id or None,
                        state.get("user_id") or None,
                        seq.get("case_id", ""),
                        seq.get("case_name", ""),
                        seq.get("sequence_theme", ""),
                        seq.get("sequence_index", 1),
                        (seq.get("questions") or [{}])[-1].get("difficulty_label", "medium"),
                        len(seq.get("questions") or []),
                        seq.get("coverage_tags") or [],
                        seq.get("source_refs") or [],
                        seq.get("metadata") or {},
                    )
                except Exception as db_exc:
                    logger.warning("cold_call_sequences insert failed (non-fatal): %s", db_exc)

                # Insert cold_call_questions rows
                for q in seq.get("questions") or []:
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
                            f"q_{seq_id}_{q.get('question_id', '')}",
                            seq_id,
                            export_batch_id,
                            q.get("question_id", ""),
                            q.get("question_index", 0),
                            q.get("question_type", ""),
                            q.get("depth_position", "middle"),
                            q.get("difficulty_label", "medium"),
                            q.get("question_text", ""),
                            q.get("target_skill", ""),
                            q.get("expected_answer_shape", ""),
                            q.get("source_refs") or [],
                            q.get("metadata") or {},
                        )
                        q_exported += 1
                    except Exception as db_exc:
                        logger.warning("cold_call_questions insert failed (non-fatal): %s", db_exc)

                # Insert cold_call_answers rows
                for ans in ans_seq.get("answers") or []:
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
                            f"a_{seq_id}_{ans.get('question_id', '')}",
                            seq_id,
                            export_batch_id,
                            ans.get("question_id", ""),
                            ans.get("model_answer", ""),
                            ans.get("strong_answer", ""),
                            ans.get("common_weak_answer", ""),
                            ans.get("professor_follow_up_trap", ""),
                            ans.get("recovery_phrase", ""),
                        )
                        a_exported += 1
                    except Exception as db_exc:
                        logger.warning("cold_call_answers insert failed (non-fatal): %s", db_exc)

    except Exception as outer_exc:
        logger.warning("formatter_export_agent DB export failed (non-fatal): %s", outer_exc)

    export_result: ExportResult = {
        "export_batch_id": export_batch_id,
        "question_sequences_exported": len(final_sequences),
        "answer_sequences_exported": len(answer_sequences),
    }

    # ── Markdown summary for notes stub ──────────────────────────────────────
    lines: List[str] = [
        "# Cold-Call Question Sequences",
        "",
        f"**Sequences generated:** {len(final_sequences)}  ",
        f"**Total questions:** {q_exported}  ",
        f"**Coverage score:** {coverage_map.get('overall_coverage_score', 0):.0%}  ",
        f"**Depth profile:** {coverage_map.get('coverage_by_depth', {})}  ",
        "",
        "---",
        "",
    ]

    for seq in final_sequences:
        case_name = seq.get("case_name", "")
        theme = seq.get("sequence_theme", "")
        lines.append(f"## Sequence {seq.get('sequence_index', '')}: {theme}")
        lines.append(f"*Case: {case_name}*")
        lines.append("")
        ans_seq = answer_by_seq.get(seq["sequence_id"], {"answers": []})
        ans_by_qid = {a["question_id"]: a for a in ans_seq.get("answers") or []}

        for q in seq.get("questions") or []:
            qid = q.get("question_id", "")
            lines.append(f"**{qid}. [{q.get('question_type','')}]** {q.get('question_text','')}")
            ans = ans_by_qid.get(qid)
            if ans:
                lines.append(f"> {ans.get('model_answer','')}")
            lines.append("")
        lines.append("---")
        lines.append("")

    missing = coverage_map.get("missing_or_weak_areas") or []
    if missing:
        lines.append(f"*Coverage gaps: {', '.join(missing)}*")
        lines.append("")

    lines.append(
        f"<!-- cold-call-agent | "
        f"tokens: {budget.get('input_tokens', 0)} in / {budget.get('output_tokens', 0)} out | "
        f"est. ${budget.get('cost_usd', 0):.4f} | "
        f"export_batch: {export_batch_id[:8]} -->"
    )

    markdown = "\n".join(lines)

    await _try_save_artifact(
        state, "export_result", dict(export_result),
        "worker_low", "formatter_export_agent", "export_result",
        source_ids=state.get("source_ids"),
    )
    return {
        "export_result": export_result,
        "final_output": markdown,
    }


formatter_export_agent.default_worker_class = "worker_low"
