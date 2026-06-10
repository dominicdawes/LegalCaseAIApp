# agents/flashcards/nodes.py
"""
All 12 nodes + 3 routing helpers for the flashcard LangGraph agent.

Node index (execution order):
  1.  head_orchestrator              — worker_mid; validate inputs, batch plan, survey sources
  2.  source_profiler                — worker_low; per-source inventory (parallel Send)
  3.  concept_extractor             — orchestrator; multi-probe retrieval + atomic concept inventory
  4.  card_blueprint_planner        — worker_mid; assign card types + create batch specs
  5.  flashcard_drafter             — orchestrator; draft front/back pairs for current batch
  6.  answer_backside_enricher      — worker_mid; enrich backsides with explanation + exam note
  7.  local_card_critic             — worker_mid; batch pedagogical QA + grounding check
  8.  card_repair_agent             — orchestrator; fix failing cards surgically (max 2 passes)
  9.  batch_commit                  — tool_only; write accepted cards to DB, advance batch index
  10. global_deck_critic            — worker_low; coverage + duplicate check over full deck
  11. deterministic_formatter_persister — tool_only; update notes row, return summary

Routing helpers (not graph nodes):
  head_orchestrator_to_profiler — Send per source_id
  should_repair_batch           — "card_repair_agent" | "batch_commit"
  batch_router                  — "flashcard_drafter" | "global_deck_critic"
"""

import asyncio
import json
import logging
import math
import re
import uuid as _uuid_mod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langgraph.types import Send

from .constants import (
    APPLICATION_TYPES,
    DEFAULT_CARD_MIX,
    FLASHCARD_CARD_TYPES,
    MAPPING_TYPES,
    RECALL_TYPES,
)
from .state import (
    AgentState,
    DraftFlashcard,
    FlashcardBatchEvaluation,
    FlashcardBatchSpec,
    FlashcardCardSpec,
    FlashcardSourceProfile,
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
    """Survey sources, determine batch plan, validate num_cards."""
    from agents.tools.base import make_tools
    from agents.tools.registry import FLASHCARD_PLANNER_TOOLS

    tools = make_tools(
        state["project_id"],
        source_ids=state["source_ids"],
        use_voyage=state.get("use_voyage", False),
        tool_names=FLASHCARD_PLANNER_TOOLS,
    )
    list_sources_tool = next(t for t in tools if t.name == "list_sources")
    sources_json = await list_sources_tool.ainvoke({})

    num_cards = max(1, state.get("num_cards") or 10)
    batch_size = min(max(state.get("batch_size") or 8, 1), 10)
    num_batches = math.ceil(num_cards / batch_size)

    _node_start("head_orchestrator", state,
                num_cards=num_cards, num_batches=num_batches)

    system = (
        "You are a T-14 law professor designing a rigorous flashcard deck.\n\n"
        "PLANNING REQUIREMENTS:\n"
        "1. Survey the documents and identify the primary doctrinal areas covered.\n"
        "2. Flag any documents with multi-element tests, definitional disputes, or "
        "   policy debates — these yield the best card material.\n"
        "3. Note any case opinions with dissents — dissent reasoning is high-yield "
        "   for comparison and contrast cards.\n"
        "4. Confirm scope in 2-3 sentences: doctrinal coverage, recommended card type "
        "   mix, and any limitations of the source material.\n"
        "Respond with your 2-3 sentence scope confirmation only."
    )
    await _llm(
        "worker_mid",
        f"Documents: {sources_json}\nRequest: {state.get('request', '')}\n"
        f"Deck: {num_cards} flashcards, batch size {batch_size}.",
        system=system,
        max_tokens=256,
        _node="head_orchestrator",
    )

    _node_done("head_orchestrator", state,
               num_cards=num_cards, num_batches=num_batches)

    return {
        "num_cards": num_cards,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "current_batch_index": 0,
        "batch_revision_count": 0,
        "accepted_card_ids": [],
        "rejected_card_metadata": [],
        "used_card_signatures": [],
        "coverage_summary": "{}",
    }


head_orchestrator.default_worker_class = "worker_mid"


def head_orchestrator_to_profiler(state: AgentState) -> List[Send]:
    """Fan-out: one source_profiler per source document."""
    source_ids = state["source_ids"]
    logger.info("flashcards x%d node fan out for source_profiler", len(source_ids))
    return [
        Send("source_profiler", {"source_id": sid, **state})
        for sid in source_ids
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. source_profiler (parallel leaf)
# ─────────────────────────────────────────────────────────────────────────────

async def source_profiler(state: Dict) -> Dict:
    """
    Build a profile for a single source: type, cases, key concepts, key rules.
    Runs in parallel via Send fan-out from head_orchestrator.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import FLASHCARD_PROFILER_TOOLS

    source_id = state["source_id"]
    tools = make_tools(
        state["project_id"],
        source_ids=[source_id],
        use_voyage=state.get("use_voyage", False),
        tool_names=FLASHCARD_PROFILER_TOOLS,
    )
    outline_tool = next(t for t in tools if t.name == "get_doc_outline")
    outline_json = await outline_tool.ainvoke({"source_id": source_id})
    outline = json.loads(outline_json)

    filename = outline.get("filename", "")
    summary = outline.get("doc_summary") or ""
    concepts = outline.get("doc_concepts", [])[:20]

    # Heuristic source_type
    fn_lower = filename.lower()
    if any(k in fn_lower for k in ("v.", " v ", "opinion", "decision")):
        source_type = "case_opinion"
    elif any(k in fn_lower for k in ("casebook", "textbook")):
        source_type = "casebook_excerpt"
    elif any(k in fn_lower for k in ("outline", "attack")):
        source_type = "attack_outline"
    elif any(k in fn_lower for k in ("notes", "class", "lecture")):
        source_type = "class_notes"
    elif any(k in fn_lower for k in ("statute", "code", "restatement")):
        source_type = "statute"
    else:
        source_type = "secondary"

    # LLM identifies cases + rules
    prompt = (
        "From the document below, extract:\n"
        "1. case_names: a JSON array of legal case names mentioned\n"
        "2. key_rules: a JSON array of up to 8 one-sentence legal rules\n\n"
        "Return ONLY a JSON object with keys 'case_names' and 'key_rules'.\n\n"
        f"Filename: {filename}\nSummary: {summary[:500]}\n"
        f"Concepts: {concepts[:12]}"
    )
    raw = await _llm("worker_low", prompt, max_tokens=512)
    try:
        extracted = _parse_json(raw)
        identified_cases = extracted.get("case_names") or []
        key_rules = extracted.get("key_rules") or []
        if not isinstance(identified_cases, list):
            identified_cases = []
        if not isinstance(key_rules, list):
            key_rules = []
    except Exception:
        identified_cases = []
        key_rules = []

    profile: FlashcardSourceProfile = {
        "source_id": source_id,
        "filename": filename,
        "source_type": source_type,
        "document_summary": summary[:600],
        "identified_cases": identified_cases[:10],
        "key_concepts": concepts[:15],
        "key_rules": key_rules[:8],
    }
    await _try_save_artifact(
        state,
        artifact_key=f"flashcard_source_profile:{source_id}",
        content=profile,
        worker_class="worker_low",
        node_name="source_profiler",
        artifact_type="source_profile",
        source_ids=[source_id],
    )
    return {"source_profiles": [profile]}


source_profiler.default_worker_class = "worker_low"


# ─────────────────────────────────────────────────────────────────────────────
# 3. concept_extractor
# ─────────────────────────────────────────────────────────────────────────────

async def concept_extractor(state: AgentState) -> Dict:
    """
    Multi-probe retrieval to surface atomic study targets across all sources.

    Runs a set of targeted retrieval queries — one per flashcard concept category
    (rules, holdings, elements, exceptions, policy, etc.).  The output is a compact
    concept_inventory JSON that card_blueprint_planner uses to decide the card mix.

    Also produces concept_synthesis: cross-doc throughlines for multi-source decks.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import FLASHCARD_RETRIEVER_TOOLS

    profiles = state.get("source_profiles") or []
    project_id = state["project_id"]
    source_ids = state["source_ids"]
    use_voyage = state.get("use_voyage", False)

    tools = make_tools(
        project_id,
        source_ids=source_ids,
        use_voyage=use_voyage,
        tool_names=FLASHCARD_RETRIEVER_TOOLS,
    )
    search_tool = next(t for t in tools if t.name == "hybrid_search")

    # Retrieval probes keyed by concept category
    PROBE_MAP = {
        "rules_and_holdings": "legal rule holding doctrine standard",
        "elements_and_tests": "elements of the test required showing factors",
        "exceptions_and_limits": "exception limitation narrow exception carve-out",
        "policy_and_rationale": "policy rationale purpose behind the rule",
        "procedure_and_burden": "procedural posture burden of proof standard of review",
        "definitions": "definition meaning term doctrine concept",
        "trigger_facts": "key fact decisive fact trigger outcome pivotal",
    }

    async def _probe(category: str, query: str) -> Dict:
        try:
            result_json = await search_tool.ainvoke({"query": query, "k": 10})
            chunks = json.loads(result_json)
            return {"category": category, "chunks": chunks[:6]}
        except Exception:
            return {"category": category, "chunks": []}

    probe_results = await asyncio.gather(*[
        _probe(cat, qry) for cat, qry in PROBE_MAP.items()
    ])

    # Flatten context per category for the LLM extraction pass
    context_by_category = {}
    for pr in probe_results:
        texts = [c.get("content", "") for c in pr["chunks"][:4]]
        context_by_category[pr["category"]] = "\n\n".join(texts)[:1500]

    # Build compact concept inventory via LLM
    profiles_brief = [
        {
            "source_id": p["source_id"],
            "filename": p["filename"],
            "source_type": p["source_type"],
            "cases": p.get("identified_cases", [])[:5],
            "rules": p.get("key_rules", [])[:5],
            "concepts": p.get("key_concepts", [])[:8],
        }
        for p in profiles
    ]

    _node_start("concept_extractor", state,
                n_profiles=len(profiles), n_probes=len(PROBE_MAP))

    system = (
        "You are a T-14 law professor extracting atomic study targets for a flashcard deck.\n\n"
        "EXTRACTION REQUIREMENTS:\n"
        "- cases: For each case extract: case_name, source_id, holding (one sentence), "
        "  rule (standalone restatement), trigger_fact (the fact that drove the outcome), "
        "  procedural_posture. Extract every case mentioned; do not summarise multiple into one.\n"
        "- definitions: For every defined legal term, extract the precise formulation used "
        "  in the source. Prefer the court's or statute's exact language.\n"
        "- rule_element_sets: For multi-part tests (e.g. duty, breach, causation, damages) "
        "  list each element as a separate string. Every Restatement section or multi-factor "
        "  test should have its own entry.\n"
        "- exceptions: For each exception or carve-out, state the base rule, the exception, "
        "  and the specific factual context that triggers it.\n"
        "- policy_points: For each policy rationale, name the doctrine and the policy goal. "
        "  These become high-yield policy/rationale flashcards.\n"
        "- burden_assignments: State precisely who bears each burden, under which standard, "
        "  and in which context (summary judgment vs. trial, etc.).\n"
        "Base ALL entries strictly on the provided source material. "
        "Return ONLY the JSON object. No extra text."
    )

    prompt = (
        f"Source profiles:\n{json.dumps(profiles_brief, indent=2)}\n\n"
        f"Retrieved passages by category:\n"
        + "\n\n".join(
            f"[{cat}]:\n{txt[:800]}"
            for cat, txt in context_by_category.items()
            if txt.strip()
        )
        + "\n\nExtract the concept inventory."
    )

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=3000,
                     _node="concept_extractor")
    try:
        inventory = _parse_json(raw)
    except Exception:
        inventory = {
            "cases": [],
            "definitions": [],
            "rule_element_sets": [],
            "exceptions": [],
            "policy_points": [],
            "burden_assignments": [],
        }

    # Cross-doc synthesis for multi-source decks
    synthesis: Dict[str, Any] = {}
    if len(profiles) > 1:
        all_concepts = []
        for p in profiles:
            all_concepts.extend(p.get("key_concepts", [])[:6])
        syn_prompt = (
            f"Identify 3-5 conceptual throughlines that span these {len(profiles)} sources "
            "and would make high-yield flashcard material.\n\n"
            f"Sources: {json.dumps(profiles_brief, indent=2)}\n\n"
            "Return a JSON array of objects with keys: concept, summary, source_ids."
        )
        syn_raw = await _llm("worker_mid", syn_prompt, max_tokens=800)
        try:
            synthesis = {"throughlines": _parse_json(syn_raw)}
        except Exception:
            synthesis = {"throughlines": []}

    inventory_str = json.dumps(inventory)
    synthesis_str = json.dumps(synthesis)

    await _try_save_artifact(
        state,
        artifact_key="concept_inventory",
        content=inventory,
        worker_class="orchestrator",
        node_name="concept_extractor",
        artifact_type="concept_inventory",
        source_ids=source_ids,
    )
    _node_done("concept_extractor", state,
               n_cases=len(inventory.get("cases", [])),
               n_definitions=len(inventory.get("definitions", [])),
               n_rule_sets=len(inventory.get("rule_element_sets", [])))
    return {
        "concept_inventory": inventory_str,
        "concept_synthesis": synthesis_str,
    }


concept_extractor.default_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 4. card_blueprint_planner
# ─────────────────────────────────────────────────────────────────────────────

async def card_blueprint_planner(state: AgentState) -> Dict:
    """
    Create all FlashcardBatchSpecs upfront.

    Allocates card types across batches to enforce diversity — prevents 50-card
    decks from becoming 50 RULE_RECALL cards.  Blueprint is saved as an artifact.
    """
    num_cards = state.get("num_cards") or 10
    batch_size = state.get("batch_size") or 5
    profiles = state.get("source_profiles") or []
    inventory_str = state.get("concept_inventory") or "{}"
    synthesis_str = state.get("concept_synthesis") or "{}"

    try:
        inventory = json.loads(inventory_str)
    except Exception:
        inventory = {}

    cases = inventory.get("cases") or []
    cases_brief = [
        {
            "case_name": c.get("case_name", ""),
            "source_id": c.get("source_id", ""),
            "holding": (c.get("holding") or "")[:120],
            "rule": (c.get("rule") or "")[:100],
        }
        for c in cases[:12]
    ]
    source_ids_available = [p["source_id"] for p in profiles]

    _node_start("card_blueprint_planner", state,
                num_cards=num_cards, n_cases=len(cases_brief))

    prompt = (
        f"You are a T-14 law professor creating a rigorous flashcard deck blueprint.\n\n"
        f"Total cards needed: {num_cards}\n"
        f"Batch size: {batch_size}\n\n"
        f"Available cases:\n{json.dumps(cases_brief, indent=2)}\n\n"
        f"Concept synthesis:\n{synthesis_str[:700]}\n\n"
        f"Available card types: {FLASHCARD_CARD_TYPES[:20]} ... "
        f"(full list has {len(FLASHCARD_CARD_TYPES)} types)\n\n"
        f"BLUEPRINT REQUIREMENTS:\n"
        f"- Create exactly {num_cards} card specs as a JSON array.\n"
        f"- Each spec must have:\n"
        f'  "spec_index":  int (0-based global index)\n'
        f'  "card_type":   one from the card type list\n'
        f'  "source_ids":  [list of source_id UUIDs from available sources]\n'
        f'  "case_names":  [list of case names; empty list if not case-based]\n'
        f'  "topic":       specific doctrinal point for this card (one precise sentence)\n'
        f'  "difficulty":  "recall" | "application" | "analysis"\n\n'
        f"ALLOCATION RULES:\n"
        f"- No more than {max(3, num_cards // 6)} specs may share the same card_type.\n"
        f"- Difficulty distribution: ~40% recall, ~35% application, ~25% analysis.\n"
        f"- Every identified case must appear in at least 1 card.\n"
        f"- Prioritise: rule_element_sets → ELEMENTS cards; exceptions → EXCEPTION cards; "
        f"  dissent reasoning → DISSENT_VS_MAJORITY cards; policy_points → POLICY cards.\n"
        f"- Ensure at least 20% of cards are application or analysis difficulty.\n"
        f"Return ONLY the JSON array of {num_cards} specs. No extra text."
    )

    raw = await _llm("worker_mid", prompt, max_tokens=max(2000, num_cards * 80))
    try:
        all_specs: List[FlashcardCardSpec] = _parse_json(raw)
        if not isinstance(all_specs, list):
            raise ValueError("not a list")
    except Exception:
        # Fallback: programmatic distribution using DEFAULT_CARD_MIX
        all_specs = []
        for i in range(num_cards):
            card_type = DEFAULT_CARD_MIX[i % len(DEFAULT_CARD_MIX)]
            case = cases[i % len(cases)] if cases else {}
            all_specs.append({
                "spec_index": i,
                "card_type": card_type,
                "source_ids": source_ids_available[:2],
                "case_names": [case.get("case_name", "")] if case else [],
                "topic": f"Card {i+1}: {case.get('rule', 'key doctrine from source material')[:80]}",
                "difficulty": "recall" if card_type in RECALL_TYPES else "application",
            })

    # Normalise spec_index
    for i, spec in enumerate(all_specs):
        spec["spec_index"] = i

    # Group into batches
    batch_specs: List[FlashcardBatchSpec] = []
    for batch_idx in range(math.ceil(len(all_specs) / batch_size)):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_specs.append({
            "batch_index": batch_idx,
            "card_specs": all_specs[start:end],
        })

    await _try_save_artifact(
        state,
        artifact_key="flashcard_blueprint",
        content={"batch_specs": batch_specs, "num_batches": len(batch_specs)},
        worker_class="worker_mid",
        node_name="card_blueprint_planner",
        artifact_type="blueprint",
        source_ids=state.get("source_ids"),
    )
    _node_done("card_blueprint_planner", state,
               num_batches=len(batch_specs), total_specs=len(all_specs))
    return {
        "batch_specs": batch_specs,
        "num_batches": len(batch_specs),
    }


card_blueprint_planner.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 5. flashcard_drafter
# ─────────────────────────────────────────────────────────────────────────────

async def flashcard_drafter(state: AgentState) -> Dict:
    """
    Draft front/back pairs for all cards in the current batch.

    Each card_spec becomes one DraftFlashcard.  Parallel asyncio.gather within
    the batch.  Deduplication guard uses used_card_signatures to avoid re-using
    topics already committed in prior batches.
    """
    from agents.tools.base import make_tools
    from agents.tools.registry import FLASHCARD_RETRIEVER_TOOLS

    batch_idx = state.get("current_batch_index", 0)
    batch_specs = state.get("batch_specs") or []
    if batch_idx >= len(batch_specs):
        return {"current_batch_drafts": []}

    batch_spec: FlashcardBatchSpec = batch_specs[batch_idx]
    used_sigs = state.get("used_card_signatures") or []
    coverage_raw = state.get("coverage_summary") or "{}"
    try:
        coverage = json.loads(coverage_raw)
    except Exception:
        coverage = {}

    async def _draft_one(spec: FlashcardCardSpec) -> DraftFlashcard:
        src_ids = spec.get("source_ids") or state["source_ids"]
        tools = make_tools(
            state["project_id"],
            source_ids=src_ids,
            use_voyage=state.get("use_voyage", False),
            tool_names=FLASHCARD_RETRIEVER_TOOLS,
        )
        search_tool = next(t for t in tools if t.name == "hybrid_search")

        query = f"{' '.join(spec.get('case_names') or [])} {spec.get('topic', '')}".strip()
        try:
            chunks_json = await search_tool.ainvoke({"query": query, "k": 10})
            chunks = json.loads(chunks_json)
        except Exception:
            chunks = []

        context = "\n\n---\n\n".join(
            f"[chunk_id:{c.get('id','?')}]\n{c.get('content','')}"
            for c in chunks[:8]
        )
        chunk_ids = [c.get("id", "") for c in chunks[:6] if c.get("id")]

        avoid_note = ""
        if used_sigs:
            avoid_note = (
                "\n\nALREADY COVERED (do NOT duplicate these topics):\n"
                + "\n".join(f"- {s}" for s in used_sigs[-25:])
            )

        _max_tokens = 900
        system = (
            "You are a T-14 law professor creating a single bar-caliber flashcard.\n\n"
            "FRONT (QUESTION) STANDARDS:\n"
            "- ONE learning target per card — atomic, not compound.\n"
            "- State the prompt as a direct question or cloze (fill-in) statement.\n"
            "- Max 40 words. Do NOT include the answer in the front.\n"
            "- For RULE cards: 'What is the rule from [Case]?' or "
            "  'State the [doctrine] test.'\n"
            "- For ELEMENT cards: 'List the elements of [rule].' or "
            "  'What must a plaintiff show to establish [claim]?'\n"
            "- For EXCEPTION cards: 'What is the exception to [rule]?' — "
            "  include the limiting condition in the front.\n"
            "- For CASE_HOLDING cards: 'What did the court hold in [Case]?' — "
            "  name the specific issue.\n"
            "- For POLICY cards: 'What policy rationale supports [doctrine]?'\n"
            "- For APPLICATION cards: present a 2-3 sentence fact pattern and ask "
            "  'What result?' or 'Which rule applies?'\n\n"
            "BACK (ANSWER) STANDARDS:\n"
            "- Self-contained — a student studying the back alone gets the full answer.\n"
            "- Structure: (a) direct answer; (b) operative legal rule/formulation; "
            "  (c) any critical limiting condition (the 'unless/but').\n"
            "- For recall: max 80 words; one focused answer.\n"
            "- For application/analysis: up to 120 words; include the rule + application.\n"
            "- Do NOT add wrong answers or distractors — just the correct answer.\n\n"
            "EXAM USE NOTE:\n"
            "- For APPLICATION and ANALYSIS cards: one sentence explaining when to deploy "
            "  this rule on an exam (e.g. 'Spot this when facts show X').\n"
            "- For all other card types: empty string.\n\n"
            "HINT STANDARDS:\n"
            "- One memory-aid sentence that points to the key concept WITHOUT revealing the answer.\n"
            "- Leave empty ('') if not useful for this card type.\n\n"
            "Return ONLY a JSON object with keys:\n"
            '  "front_content": str (max 40 words)\n'
            '  "back_content":  str (max 120 words)\n'
            '  "hint":          str (empty string if not applicable)\n'
            '  "exam_use_note": str (one sentence for APPLICATION/ANALYSIS, else "")\n'
            '  "source_refs":   [chunk_id UUID strings from [chunk_id:...] markers]\n'
            f"\n\n**IMPORTANT**: keep your response under {_max_tokens} tokens."
        )

        prompt = (
            f"Card type: {spec['card_type']}\n"
            f"Topic: {spec.get('topic', '')}\n"
            f"Cases: {', '.join(spec.get('case_names', []) or [])}\n"
            f"Difficulty: {spec.get('difficulty', 'recall')}\n"
            f"{avoid_note}\n\n"
            f"Legal source material:\n"
            f"{context[:3000] if context else 'Use source profiles and general knowledge.'}\n\n"
            "Draft the flashcard."
        )

        raw = await _llm("orchestrator", prompt, system=system, max_tokens=_max_tokens,
                         _node="flashcard_drafter")
        try:
            data = _parse_json(raw)
        except Exception:
            data = {
                "front_content": f"What is the rule from {spec.get('topic', 'this doctrine')}?",
                "back_content": "See source material for the applicable rule.",
                "hint": "",
                "exam_use_note": "",
                "source_refs": [],
            }

        _UUID_RE = re.compile(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I
        )
        src_refs_raw = data.get("source_refs") or []
        src_refs = [
            m.group() for raw_ref in src_refs_raw
            if (m := _UUID_RE.search(str(raw_ref)))
        ]

        back = (data.get("back_content") or "").strip()
        exam_note = (data.get("exam_use_note") or "").strip()
        if exam_note:
            back = back.rstrip() + f"\n\n*Exam tip: {exam_note}*"

        return {
            "spec_index": spec["spec_index"],
            "card_type": spec["card_type"],
            "front_content": (data.get("front_content") or "").strip(),
            "back_content": back,
            "hint": (data.get("hint") or "").strip(),
            "source_refs": src_refs or chunk_ids[:4],
            "grounding_verdict": "",
            "grounding_notes": "",
        }

    drafts = list(await asyncio.gather(*[
        _draft_one(spec) for spec in batch_spec["card_specs"]
    ]))

    logger.info(
        "flashcard_drafter: batch %d — drafted %d cards",
        batch_idx, len(drafts),
    )
    return {"current_batch_drafts": drafts}


flashcard_drafter.default_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 6. answer_backside_enricher
# ─────────────────────────────────────────────────────────────────────────────

async def answer_backside_enricher(state: AgentState) -> Dict:
    """
    Improve the back content of all cards in the current batch.

    Enriches each backside with:
      - Concise, self-contained answer
      - Explanation of why the answer is correct
      - Optional exam-use note (when the card type warrants it)
      - Source/case reference hint

    Works over the whole batch in one LLM call to reduce API round trips.
    """
    drafts = list(state.get("current_batch_drafts") or [])
    if not drafts:
        return {}

    batch_payload = [
        {
            "spec_index": d["spec_index"],
            "card_type": d["card_type"],
            "front_content": d["front_content"],
            "back_content": d["back_content"],
        }
        for d in drafts
    ]

    _node_start("answer_backside_enricher", state,
                n_drafts=len(drafts))

    system = (
        "You are a T-14 law professor enriching flashcard back sides to bar-exam caliber.\n\n"
        "ENRICHMENT STANDARDS:\n"
        "For EVERY card:\n"
        "1. Rewrite back_content to be self-contained — a student studying from the back "
        "   alone should get the full answer without needing the front or source text.\n"
        "2. Structure: (a) the direct answer to the question; (b) the operative legal rule "
        "   or formulation; (c) any critical limiting conditions (the 'unless/but').\n"
        "3. Keep the answer atomic — one concept per card, max 120 words.\n"
        "4. Do NOT add MCQ-style wrong answers or distractors.\n\n"
        "For APPLICATION and ANALYSIS cards also add:\n"
        "  exam_use_note: one sentence explaining when to deploy this rule on an exam "
        "  (e.g. 'Spot this when the facts show a plaintiff injured outside the zone of "
        "  danger but claims emotional distress').\n\n"
        "For RECALL and DEFINITION cards:\n"
        "  exam_use_note: empty string.\n\n"
        "Return a JSON array — one object per card — with:\n"
        '  "spec_index":    int\n'
        '  "back_content":  str (max 120 words)\n'
        '  "exam_use_note": str (one sentence or empty string)\n'
        "No extra text."
    )

    prompt = (
        f"Enrich the back sides of the following {len(batch_payload)} flashcards:\n\n"
        f"{json.dumps(batch_payload, indent=2)}"
    )

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=2500,
                     _node="answer_backside_enricher")
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

        new_back = enriched.get("back_content", draft["back_content"])
        exam_note = enriched.get("exam_use_note", "")
        # Append exam note if present
        if exam_note and exam_note.strip():
            new_back = new_back.rstrip() + f"\n\n*Exam tip: {exam_note.strip()}*"

        updated_drafts.append({**draft, "back_content": new_back})

    _node_done("answer_backside_enricher", state,
               n_enriched=len(updated_drafts))
    return {"current_batch_drafts": updated_drafts}


answer_backside_enricher.default_worker_class = "worker_mid"


# ─────────────────────────────────────────────────────────────────────────────
# 7. local_card_critic
# ─────────────────────────────────────────────────────────────────────────────

async def local_card_critic(state: AgentState) -> Dict:
    """
    Evaluate the current batch as a pedagogical set.

    Scores five dimensions per batch (0.0–1.0):
      vagueness          — are fronts specific and unambiguous?
      uniqueness         — do cards cover distinct concepts?
      source_support     — are answers grounded in source material?
      atomic_focus       — does each card test exactly one thing?
      answer_quality     — are backsides concise, accurate, complete?

    Also checks each card individually and flags any that need repair.
    Batch passes if all dimension scores >= 0.70.
    """
    drafts = state.get("current_batch_drafts") or []
    batch_idx = state.get("current_batch_index", 0)
    used_sigs = (state.get("used_card_signatures") or [])[-30:]

    if not drafts:
        return {
            "current_batch_eval": {
                "batch_index": batch_idx,
                "passes": True,
                "scores": {},
                "rejection_reasons": [],
                "revision_instructions": [],
                "card_verdicts": [],
            }
        }

    batch_payload = [
        {
            "spec_index": d["spec_index"],
            "card_type": d["card_type"],
            "front_content": d["front_content"],
            "back_content": d["back_content"],
        }
        for d in drafts
    ]

    _node_start("local_card_critic", state,
                batch_idx=batch_idx, n_cards=len(drafts))

    system = (
        "You are a T-14 law school curriculum director performing pedagogical QA on a "
        "batch of flashcards.\n\n"
        "EVALUATION CRITERIA (score each 0.0–1.0):\n"
        "  vagueness: Are fronts specific and unambiguous? A vague front like 'What is "
        "  negligence?' fails; 'What is the duty element of negligence under the "
        "  reasonable person standard?' passes. (1.0 = all fronts precise)\n"
        "  uniqueness: Do cards cover distinct concepts with no near-duplicate content? "
        "  Same rule tested two different ways is fine; same rule stated identically is not. "
        "(1.0 = fully distinct)\n"
        "  source_support: Are back answers grounded in the provided material, not "
        "  hallucinated from general knowledge? (1.0 = fully grounded)\n"
        "  atomic_focus: Does each card test exactly ONE learning target? "
        "  A card combining rule + exception + policy in one back fails atomic. "
        "(1.0 = fully atomic)\n"
        "  answer_quality: Are backs concise, accurate, and self-contained? "
        "  Does each back answer the specific question on the front? (1.0 = excellent)\n\n"
        "PASS THRESHOLD: all scores >= 0.70.\n"
        "REVISION INSTRUCTIONS must be specific: name the spec_index and the exact fix.\n\n"
        "Return a JSON object:\n"
        '  "passes": bool\n'
        '  "scores": {"vagueness": f, "uniqueness": f, "source_support": f, '
        '"atomic_focus": f, "answer_quality": f}\n'
        '  "rejection_reasons": [str, ...] (empty if passes)\n'
        '  "revision_instructions": [str, ...] (specific fixes per failing card)\n'
        '  "card_verdicts": [{"spec_index": int, "passes": bool, "reason": str}, ...]\n'
        "No extra text."
    )

    avoid_context = ""
    if used_sigs:
        avoid_context = (
            f"\n\nAlready-committed card signatures (check for duplicates):\n"
            + "\n".join(f"- {s}" for s in used_sigs)
        )

    prompt = (
        f"Batch ({len(batch_payload)} cards):{avoid_context}\n\n"
        f"{json.dumps(batch_payload, indent=2)}\n\n"
        "Evaluate the batch."
    )

    raw = await _llm("worker_mid", prompt, system=system, max_tokens=2000,
                     _node="local_card_critic")
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
            "card_verdicts": [],
        }

    batch_eval: FlashcardBatchEvaluation = {
        "batch_index": batch_idx,
        "passes": result.get("passes", True),
        "scores": result.get("scores", {}),
        "rejection_reasons": result.get("rejection_reasons", []),
        "revision_instructions": result.get("revision_instructions", []),
        "card_verdicts": result.get("card_verdicts", []),
    }
    _node_done("local_card_critic", state,
               batch_idx=batch_idx,
               passes=batch_eval["passes"],
               n_verdicts=len(batch_eval["card_verdicts"]))
    return {"current_batch_eval": batch_eval}


local_card_critic.default_worker_class = "worker_mid"
local_card_critic.escalation_worker_class = "orchestrator"


def should_repair_batch(state: AgentState) -> str:
    """Route to card_repair_agent if batch failed and revision budget remains."""
    if state.get("batch_revision_count", 0) >= 1:
        return "batch_commit"
    eval_result = state.get("current_batch_eval")
    if eval_result and not eval_result.get("passes", True):
        return "card_repair_agent"
    return "batch_commit"


# ─────────────────────────────────────────────────────────────────────────────
# 8. card_repair_agent
# ─────────────────────────────────────────────────────────────────────────────

async def card_repair_agent(state: AgentState) -> Dict:
    """
    Repair only cards that failed local critique.  Passing cards are untouched.

    Preserves card spec_index and card_type.  Increments batch_revision_count;
    after 2 passes should_repair_batch routes directly to batch_commit.
    """
    drafts = list(state.get("current_batch_drafts") or [])
    eval_result: Optional[FlashcardBatchEvaluation] = state.get("current_batch_eval")
    revision_count = state.get("batch_revision_count", 0)

    if not eval_result:
        return {"batch_revision_count": revision_count + 1}

    verdict_map: Dict[int, Dict] = {
        v["spec_index"]: v
        for v in (eval_result.get("card_verdicts") or [])
        if isinstance(v, dict)
    }
    revision_instructions = eval_result.get("revision_instructions") or []
    instructions_text = "\n".join(f"- {r}" for r in revision_instructions)

    failing_indices = {
        si for si, v in verdict_map.items() if not v.get("passes", True)
    }
    if not failing_indices:
        return {"batch_revision_count": revision_count + 1}

    failing_drafts = [d for d in drafts if d["spec_index"] in failing_indices]
    passing_drafts = [d for d in drafts if d["spec_index"] not in failing_indices]

    if not failing_drafts:
        return {"batch_revision_count": revision_count + 1}

    _node_start("card_repair_agent", state,
                batch_idx=state.get("current_batch_index", 0),
                revision_count=revision_count,
                n_failing=len(failing_indices))

    system = (
        "You are a T-14 law professor performing surgical repair of flashcards that "
        "failed pedagogical QA.\n\n"
        "REPAIR STANDARDS:\n"
        "1. Fix ONLY the specific issues listed — do not rewrite passing cards.\n"
        "2. Preserve card_type, topic, and spec_index exactly.\n"
        "3. If vagueness failure: rewrite the front to name the specific rule, case, "
        "   or element being tested. Avoid overly broad fronts.\n"
        "4. If atomic_focus failure: split the card if it covers two concepts, OR remove "
        "   the secondary concept from the back and keep only the primary answer.\n"
        "5. If source_support failure: remove any content not in the source material and "
        "   replace with a question about what IS documented.\n"
        "6. If answer_quality failure: rewrite the back to be self-contained, precise, "
        "   and capped at 120 words.\n"
        "Return a JSON array of the repaired cards only (same structure as input)."
    )

    failing_payload = [
        {
            "spec_index": d["spec_index"],
            "card_type": d["card_type"],
            "front_content": d["front_content"],
            "back_content": d["back_content"],
            "hint": d["hint"],
            "failure_reason": (verdict_map.get(d["spec_index"]) or {}).get("reason", ""),
        }
        for d in failing_drafts
    ]

    prompt = (
        f"Repair instructions:\n{instructions_text or 'Improve card quality.'}\n\n"
        f"Cards to repair:\n{json.dumps(failing_payload, indent=2)}\n\n"
        "Return the repaired cards as a JSON array with the same spec_index values."
    )

    raw = await _llm("orchestrator", prompt, system=system, max_tokens=2500,
                     _node="card_repair_agent")
    try:
        repaired_list = _parse_json(raw)
        if not isinstance(repaired_list, list):
            raise ValueError
        repaired_map = {r["spec_index"]: r for r in repaired_list if isinstance(r, dict)}
    except Exception:
        repaired_map = {}

    updated_drafts = list(passing_drafts)
    for draft in failing_drafts:
        patch = repaired_map.get(draft["spec_index"])
        if patch:
            updated_drafts.append({
                **draft,
                "front_content": patch.get("front_content", draft["front_content"]),
                "back_content": patch.get("back_content", draft["back_content"]),
                "hint": patch.get("hint", draft["hint"]),
            })
        else:
            updated_drafts.append(draft)

    updated_drafts.sort(key=lambda d: d["spec_index"])
    _node_done("card_repair_agent", state,
               n_repaired=len(repaired_map), n_total=len(updated_drafts),
               repair_pass=revision_count + 1)
    return {
        "current_batch_drafts": updated_drafts,
        "batch_revision_count": revision_count + 1,
    }


card_repair_agent.default_worker_class = "orchestrator"
card_repair_agent.escalation_worker_class = "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 9. batch_commit
# ─────────────────────────────────────────────────────────────────────────────

async def batch_commit(state: AgentState) -> Dict:
    """
    Persist accepted cards to individual_cards, update thin accumulators, advance index.

    Accepted = grounding_verdict not "fail" OR grounding_verdict empty (default accept).
    Rejected  = grounding_verdict == "fail".

    individual_cards schema:
      id, deck_id (=job_id), user_id, project_id, front_content,
      back_content, card_order, created_at, is_active
    """
    from tasks.database import get_db_connection

    drafts = state.get("current_batch_drafts") or []
    batch_idx = state.get("current_batch_index", 0)
    job_id = (state.get("job_id") or "").strip()
    user_id = (state.get("user_id") or "").strip()
    project_id = (state.get("project_id") or "").strip()

    # Accept everything except explicit fail (grounding_verdict "" counts as pass)
    accepted = [d for d in drafts if d.get("grounding_verdict", "") != "fail"]
    rejected = [d for d in drafts if d.get("grounding_verdict", "") == "fail"]

    new_card_ids: List[str] = []
    now = datetime.now(timezone.utc)

    if job_id and accepted:
        # Global card_order: sum of all previously accepted cards + position in this batch
        existing_count = len(state.get("accepted_card_ids") or [])

        try:
            async with get_db_connection() as conn:
                for pos, draft in enumerate(accepted):
                    card_id = str(_uuid_mod.uuid4())
                    card_order = existing_count + pos
                    await conn.execute(
                        """
                        INSERT INTO individual_cards (
                            id, deck_id, user_id, project_id,
                            front_content, back_content, card_order,
                            created_at, is_active
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        card_id,
                        job_id,
                        user_id or None,
                        project_id or None,
                        draft["front_content"],
                        draft["back_content"],
                        card_order,
                        now,
                        True,
                    )
                    new_card_ids.append(card_id)
                    logger.info(
                        "batch_commit: wrote card[%s] → %s (order %d)",
                        draft["spec_index"], card_id[:8], card_order,
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

    # Update thin accumulators (return full accumulated list — not Annotated)
    all_ids = list(state.get("accepted_card_ids") or []) + new_card_ids

    new_sigs = [
        f"{d['card_type']}:{d['front_content'][:40]}"
        for d in accepted
    ]
    all_sigs = list(state.get("used_card_signatures") or []) + new_sigs

    new_rejected = [
        {
            "batch_index": batch_idx,
            "spec_index": r["spec_index"],
            "card_type": r["card_type"],
            "reason": r.get("grounding_notes", "critic_fail"),
        }
        for r in rejected
    ]
    all_rejected = list(state.get("rejected_card_metadata") or []) + new_rejected

    # Update coverage summary
    try:
        coverage = json.loads(state.get("coverage_summary") or "{}")
    except Exception:
        coverage = {}
    for d in accepted:
        ctype = d.get("card_type", "UNKNOWN")
        coverage[ctype] = coverage.get(ctype, 0) + 1
    coverage_str = json.dumps(coverage)

    await _try_save_artifact(
        state,
        artifact_key=f"batch_commit:{batch_idx}",
        content={
            "batch_index": batch_idx,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "card_ids": new_card_ids,
        },
        worker_class="tool_only",
        node_name="batch_commit",
        artifact_type="batch_commit",
    )

    return {
        "accepted_card_ids": all_ids,
        "rejected_card_metadata": all_rejected,
        "used_card_signatures": all_sigs,
        "coverage_summary": coverage_str,
        "current_batch_index": batch_idx + 1,
        "batch_revision_count": 0,
        "current_batch_drafts": [],
        "current_batch_eval": None,
    }


batch_commit.default_worker_class = "tool_only"


def batch_router(state: AgentState) -> str:
    """After batch_commit: loop to flashcard_drafter or proceed to global_deck_critic."""
    idx = state.get("current_batch_index", 0)
    total = len(state.get("batch_specs") or [])
    accepted = len(state.get("accepted_card_ids") or [])
    needed = state.get("num_cards") or 10

    if idx >= total or accepted >= needed:
        return "global_deck_critic"
    return "flashcard_drafter"


# ─────────────────────────────────────────────────────────────────────────────
# 10. global_deck_critic
# ─────────────────────────────────────────────────────────────────────────────

async def global_deck_critic(state: AgentState) -> Dict:
    """
    Review the full accepted deck using thin state only (signatures + coverage).

    Checks:
      - Card type diversity (no dominant type)
      - Duplicate concept detection
      - Coverage of major source material
      - Whether the accepted count matches the requested num_cards
    """
    sigs = state.get("used_card_signatures") or []
    accepted_count = len(state.get("accepted_card_ids") or [])
    coverage_raw = state.get("coverage_summary") or "{}"
    num_cards = state.get("num_cards") or 10
    rejected_count = len(state.get("rejected_card_metadata") or [])

    try:
        coverage = json.loads(coverage_raw)
    except Exception:
        coverage = {}

    _node_start("global_deck_critic", state,
                accepted=accepted_count, target=num_cards, rejected=rejected_count)

    system = (
        "You are a T-14 law school curriculum director performing a final audit of a "
        "completed flashcard deck.\n\n"
        "AUDIT STANDARDS:\n"
        "- type_diversity: flag any card type that exceeds 40% of the deck — "
        "  name the over-represented type and the under-represented types.\n"
        "- duplicates: scan card front signatures for near-identical fronts. "
        "  Same rule stated two different ways is fine; same rule + same question wording "
        "  is a duplicate.\n"
        "- coverage_gaps: identify flashcard-worthy material (rule_element_sets, exceptions, "
        "  policy_points, dissent reasoning) with NO card in the deck.\n"
        "- count_ok: flag if accepted_count falls more than 15% below the target.\n"
        "- overall_score: weight answer_quality and atomic_focus most heavily.\n"
        "Return ONLY the JSON object. No extra text."
    )

    prompt = (
        f"Audit a {accepted_count}-card flashcard deck "
        f"(target: {num_cards}, rejected: {rejected_count}).\n\n"
        f"Card type coverage:\n{json.dumps(coverage, indent=2)}\n\n"
        "Card front signatures (first 40 chars):\n"
        + "\n".join(f"- {s}" for s in sigs[:60])
        + "\n\n"
        "Return:\n"
        '  "overall_score": float (0.0–1.0)\n'
        '  "type_diversity_ok": bool\n'
        '  "duplicates_found": [str, ...] (signatures of near-duplicate pairs)\n'
        '  "coverage_gaps": [str, ...] (card types or doctrinal areas missing)\n'
        '  "count_ok": bool\n'
        '  "summary": str (2-3 sentences on deck quality and major concerns)\n'
        "No extra text."
    )

    raw = await _llm("worker_low", prompt, system=system, max_tokens=1200,
                     _node="global_deck_critic")
    try:
        report = _parse_json(raw)
        report_str = json.dumps(report)
    except Exception:
        report_str = json.dumps({
            "overall_score": 0.75,
            "type_diversity_ok": True,
            "duplicates_found": [],
            "coverage_gaps": [],
            "count_ok": accepted_count >= num_cards,
            "summary": f"Deck has {accepted_count} cards across {len(coverage)} card types.",
        })

    await _try_save_artifact(
        state,
        artifact_key="global_deck_report",
        content={"report": report_str},
        worker_class="worker_low",
        node_name="global_deck_critic",
        artifact_type="deck_report",
    )
    _node_done("global_deck_critic", state, accepted=accepted_count)
    return {"global_deck_report": report_str}


global_deck_critic.default_worker_class = "worker_low"


# ─────────────────────────────────────────────────────────────────────────────
# 11. deterministic_formatter_persister
# ─────────────────────────────────────────────────────────────────────────────

async def deterministic_formatter_persister(state: AgentState) -> Dict:
    """
    Finalise the deck run.

    No LLM call — deterministic only.

    1. Builds a markdown summary of the completed deck.
    2. Updates the parent notes row: sets description, num_cards,
       referenced_sources, note_progress_status = 'COMPLETE'.
    3. Returns persisted_card_ids (final list) and final_output markdown.

    The individual_cards rows were written incrementally by batch_commit.
    This node only updates the notes/deck metadata row.
    """
    from tasks.database import get_db_connection

    accepted_ids = state.get("accepted_card_ids") or []
    coverage_raw = state.get("coverage_summary") or "{}"
    deck_report_raw = state.get("global_deck_report") or "{}"
    job_id = (state.get("job_id") or "").strip()
    num_cards_requested = state.get("num_cards") or 10
    source_ids = state.get("source_ids") or []
    is_essential = state.get("is_essential") or False

    _node_start("deterministic_formatter_persister", state,
                n_accepted=len(accepted_ids))

    try:
        coverage = json.loads(coverage_raw)
    except Exception:
        coverage = {}
    try:
        deck_report = json.loads(deck_report_raw)
    except Exception:
        deck_report = {}

    # Derive card type breakdown for summary
    type_breakdown = "\n".join(
        f"  - {ct}: {cnt}" for ct, cnt in sorted(coverage.items(), key=lambda x: -x[1])
    )
    overall_score = deck_report.get("overall_score", "N/A")
    deck_summary_text = deck_report.get("summary", "")
    coverage_gaps = deck_report.get("coverage_gaps") or []

    # Profiles for case coverage section
    profiles = state.get("source_profiles") or []
    all_cases: List[str] = []
    for p in profiles:
        all_cases.extend(p.get("identified_cases", [])[:5])
    all_cases = list(dict.fromkeys(all_cases))[:15]  # dedup, preserve order

    deck_description = (
        f"AI-generated flashcard deck with {len(accepted_ids)} cards "
        f"(requested: {num_cards_requested})"
    )

    markdown = (
        f"# Flashcard Deck ({len(accepted_ids)} cards)\n\n"
        f"**Requested:** {num_cards_requested}  \n"
        f"**Generated:** {len(accepted_ids)}  \n\n"
        f"## Card Type Breakdown\n\n{type_breakdown or '  *(not available)*'}\n\n"
    )
    if all_cases:
        markdown += (
            "## Cases Covered\n\n"
            + "\n".join(f"- {c}" for c in all_cases)
            + "\n\n"
        )
    markdown += f"## Deck Quality Score\n\n{overall_score}\n\n{deck_summary_text}\n"
    if coverage_gaps:
        markdown += (
            "\n## Coverage Gaps\n\n"
            + "\n".join(f"- {g}" for g in coverage_gaps)
            + "\n"
        )

    # Update parent notes row
    if job_id:
        try:
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
                    deck_description,
                    len(accepted_ids),
                    True,
                    is_essential,
                    len(source_ids),
                    [str(s) for s in source_ids],
                    job_id,
                )
            logger.info(
                "deterministic_formatter_persister: updated notes row %s (%d cards)",
                job_id[:8], len(accepted_ids),
            )
        except Exception as exc:
            logger.error(
                "deterministic_formatter_persister: notes update failed: %s", exc
            )

    await _try_save_artifact(
        state,
        artifact_key="final_output",
        content={"markdown": markdown, "card_count": len(accepted_ids)},
        worker_class="tool_only",
        node_name="deterministic_formatter_persister",
        artifact_type="final_output",
    )

    _node_done("deterministic_formatter_persister", state,
               n_cards=len(accepted_ids))
    return {
        "persisted_card_ids": accepted_ids,
        "final_output": markdown,
    }


deterministic_formatter_persister.default_worker_class = "tool_only"
