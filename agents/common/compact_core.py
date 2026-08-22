# agents/common/compact_core.py
"""
Shared machinery for the compact 4-stage agent pipelines
(plan ∥ research → sync_barrier → [Send×N] generate → deterministic format).

Extracted from agents/attack_outline/compact_nodes.py and
agents/case_brief/compact_nodes.py after the same ~450 lines had been copied
twice and were about to be copied four more times (cold_call, exam_questions,
quiz, flashcards). This module is the single home for the loop/model/prefetch/
stripper logic; the per-agent compact_nodes keep only their prompts, dossier
schemas, fan-out tables, and formatters.

What lives here (agent-agnostic):
  • build_tool_model()        — tiered, tool-bindable model resolution through
                                each agent's own _fetch_worker_model
  • run_bounded_tool_loop()   — the bounded ReAct loop: countdown ladder,
                                reserved unbound-model emission turn, parallel
                                tool execution, evidence harvesting, and
                                capacity-error fallback to the tier's backup
                                model
  • exec_tool_call()          — never-raise tool execution
  • harvest_evidence()        — chunk capture into an evidence store
  • prefetch_corpus_survey()  — deterministic corpus orientation (sources,
                                outlines, section map w/ vector-store fallback)
  • strip_chunk_citations()   — presentation-boundary [chunk-uuid] removal
  • word_budget_line()        — soft word-budget prompt fragment

What stays per-agent: _llm (string calls), _parse_json, _try_save_artifact,
node telemetry, prompts, dossier schemas, unit/spec tables, DB exporters.

Deliberate change vs the per-agent copies: ONE process-wide DeepSeek semaphore
here instead of one Semaphore(10) per agent module. Provider rate limits are
per-account, not per-agent, so per-module semaphores under-throttled whenever
two agents ran concurrently in the same worker.
"""

import asyncio
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

logger = logging.getLogger(__name__)

# ── Shared tunable defaults (callers may override per call) ───────────────────

TOOL_RESULT_CHAR_CAP = 6000
PREFETCH_MAX_OUTLINES = 10
PREFETCH_MAX_SECTIONS = 40
PREFETCH_SECTION_SUMMARY_CHARS = 300
DEFAULT_HARD_OUTPUT_TOKENS = 14000

# ── Process-wide DeepSeek concurrency cap ─────────────────────────────────────

_DEEPSEEK_SEMAPHORE: Optional[asyncio.Semaphore] = None


def get_deepseek_semaphore() -> asyncio.Semaphore:
    """One cap for every compact agent in this worker process (rate limits are
    per-account, not per-agent)."""
    global _DEEPSEEK_SEMAPHORE
    if _DEEPSEEK_SEMAPHORE is None:
        _DEEPSEEK_SEMAPHORE = asyncio.Semaphore(10)
    return _DEEPSEEK_SEMAPHORE


def _warn(node_name: str, state: Dict, msg: str) -> None:
    """Matches the per-agent `_node_warn` log format so dashboards keep working."""
    job = (state.get("job_id") or "")[:8] or "no-job"
    logger.warning("⚠ [%s] %s  %s", job, node_name, msg)


# ── Prompt fragments ──────────────────────────────────────────────────────────

def word_budget_line(target: int, cap: int, kind: str = "section") -> str:
    """Soft word budget stated in the system prompt so the model self-regulates.
    Budgeting by prompt (with a high max_tokens runaway guard) is what prevents
    mid-sentence clipping."""
    if kind == "dossier":
        tail = (
            "This is an internal hand-off, not prose for a reader — terse "
            "fragments only, no explanation, no restating the same point twice."
        )
    elif kind == "outline":
        tail = (
            "This is an attack outline — a memorisable checklist a student scans "
            "under exam pressure, NOT an essay or a case brief. Compress "
            "ruthlessly: fragments, not full sentences; bullets, not paragraphs. "
            "If you are running long, cut explanation and keep the rule, the "
            "triggers, and the IF/THEN logic."
        )
    else:
        tail = (
            "Never pad to reach the target, and never truncate mid-structure to "
            "stay under it — tighten wording instead."
        )
    return (
        f"\n\n**LENGTH LIMIT**: aim for ~{target} words; never exceed {cap}. {tail}"
    )


# ── Evidence harvesting ───────────────────────────────────────────────────────

_HARVEST_TOOLS = {
    "search_passages", "hybrid_search", "get_section",
    "get_neighbors", "find_concept_across_docs", "get_parents",
}


def harvest_evidence(tool_name: str, result_str: str, store: Dict[str, Dict]) -> None:
    """Capture chunk payloads returned by retrieval tools into the evidence
    store so generators can ground without re-retrieving. Never raises."""
    if tool_name not in _HARVEST_TOOLS:
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


async def exec_tool_call(
    tool_map: Dict[str, Any],
    tc: Dict[str, Any],
    char_cap: int = TOOL_RESULT_CHAR_CAP,
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
        content=result_str[:char_cap],
        tool_call_id=tc.get("id", ""),
        name=name,
    ), result_str


# ── Corpus survey pre-fetch ───────────────────────────────────────────────────

async def prefetch_corpus_survey(
    state: Dict,
    tools: List[Any],
    *,
    max_outlines: int = PREFETCH_MAX_OUTLINES,
    max_sections: int = PREFETCH_MAX_SECTIONS,
    summary_chars: int = PREFETCH_SECTION_SUMMARY_CHARS,
    node_name: str = "research_agent",
) -> Tuple[str, int]:
    """
    Deterministically fetch the corpus survey so a research agent can skip
    discovery and spend every turn on retrieval.

    Layers, all plain indexed DB reads — no LLM, no embeddings, no vector search:
      1. corpus   — list_sources
      2. document — get_doc_outline per source (concurrent)
      3. section  — document_sections summaries, FALLING BACK to distinct
                    section_paths from document_vector_store.

    The fallback is the normal path, not an edge case: ingest dispatches
    build_section_summaries fire-and-forget at the moment a doc goes COMPLETE
    and the note fires ~0.8s later, so document_sections is reliably empty at
    survey time for speculative uploads. Section paths ship with the chunks,
    so the structural map survives even when the summaries do not.

    Returns (survey_text, n_chunks); n_chunks==0 means unknown. Never raises.
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
            _warn(node_name, state, f"prefetch list_sources failed: {exc}")

    outline_tool = tool_by_name.get("get_doc_outline")
    if outline_tool is not None and source_ids:
        targets = source_ids[:max_outlines]
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
                    state["project_id"], source_ids, max_sections,
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
                        state["project_id"], source_ids, max_sections,
                    )
                    if rows:
                        logger.info(
                            "  📚 [%s] document_sections empty — derived %d "
                            "section paths from chunks instead", node_name, len(rows)
                        )

            if rows:
                lines = []
                for r in rows:
                    line = f"[{str(r['source_id'])[:8]}] {r['section_path']}"
                    summary = (r["section_summary"] or "")[:summary_chars]
                    if summary:
                        line += f"\n    {summary}"
                    lines.append(line)
                parts.append("## Section map\n" + "\n".join(lines))
                logger.info("  📚 [%s] prefetched %d section entries", node_name, len(rows))
        except Exception as exc:
            _warn(node_name, state, f"prefetch section map failed: {exc}")

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


# ── Tiered, tool-bindable model resolution ────────────────────────────────────

async def build_tool_model(
    fetch_worker_model: Callable[..., Tuple[str, str, Optional[bool]]],
    worker_class: str,
    provider: Optional[str] = None,
    hard_cap: int = DEFAULT_HARD_OUTPUT_TOKENS,
):
    """
    Resolve a tool-bindable LangChain chat model via the calling agent's tiered
    WORKER_MODEL_MAP (its `_fetch_worker_model`), instead of hardcoding a
    provider. Escalating a node to a flagship model is a config change.

    Returns (base_model, semaphore, rebuild) where `rebuild(provider)` is an
    async callable producing a fresh (base_model, semaphore) for the same tier
    on a different provider — used by run_bounded_tool_loop for capacity-error
    fallback to the tier's backup model.

    `hard_cap` sits deliberately ABOVE the prompts' soft word budgets: the
    prompt does the budgeting, this only bounds pathology.
    """

    async def _build(_provider: Optional[str]):
        resolved_provider, model_name, thinking = fetch_worker_model(worker_class, _provider)
        if resolved_provider == "deepseek":
            from utils.llm_clients.deepseek_client import DeepSeekClient
            client = DeepSeekClient(
                model_name=model_name, temperature=0.3,
                max_output_tokens=hard_cap, thinking=thinking,
            )
            base_model = client._client
            sem = get_deepseek_semaphore()
        elif resolved_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            from utils.llm_clients.anthropic_rate_limits import get_llm_semaphore
            base_model = ChatAnthropic(
                model=model_name, max_tokens=hard_cap, temperature=0.3
            )
            sem = await get_llm_semaphore()
        else:
            raise NotImplementedError(
                f"build_tool_model: provider '{resolved_provider}' is not wired "
                "for tool-calling (only deepseek/anthropic are supported)."
            )
        logger.info(
            "  🤖 [build_tool_model] worker_class=%s provider=%s model=%s thinking=%s",
            worker_class, resolved_provider, model_name, thinking,
        )
        return base_model, sem, resolved_provider

    base_model, sem, resolved = await _build(provider)

    async def rebuild(fallback_provider: str):
        bm, s, _ = await _build(fallback_provider)
        return bm, s

    # Stash the resolved provider so the loop knows what NOT to fall back to.
    rebuild.resolved_provider = resolved  # type: ignore[attr-defined]
    return base_model, sem, rebuild


def _is_capacity_error(exc: Exception) -> bool:
    from utils.llm_clients.llm_factory import _is_capacity_error as _f
    return _f(exc)


# ── The bounded ReAct loop ────────────────────────────────────────────────────

async def run_bounded_tool_loop(
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
    tool_result_char_cap: int = TOOL_RESULT_CHAR_CAP,
    rebuild: Optional[Callable] = None,
) -> str:
    """
    Bounded ReAct loop shared by every compact research agent and generator.

    `max_turns` is the RETRIEVAL budget; the final answer is emitted in an
    additional reserved turn via the UNBOUND base_model (no tools attached, so
    it cannot emit further tool calls). The model is warned the reserved turn
    is coming — countdown, then an explicit final-turn notice — so emission is
    a planned hand-off, not an interrupt. Early finish (a turn with no tool
    calls) IS the answer and skips the reserved turn.

    Failure handling per LLM call: one plain retry after 5s; if the failure is
    a capacity error (503/overloaded) and `rebuild` is provided, fall back to
    the tier's anthropic backup model (per each agent's WORKER_MODEL_MAP
    column) and continue the same turn — honoring the tier's defined backup
    model inside the tool loop, which string-based `_llm` calls already get
    via WORKER_FALLBACK_CHAINS.
    """
    tool_map = {t.name: t for t in tools}
    bound_model = base_model.bind_tools(tools)

    async def _invoke(model, current_sem):
        async with current_sem:
            return await model.ainvoke(messages)

    async def _call_with_fallback(bound: bool) -> AIMessage:
        nonlocal base_model, bound_model, sem
        model = bound_model if bound else base_model
        try:
            return await _invoke(model, sem)
        except Exception as exc:
            if (
                rebuild is not None
                and _is_capacity_error(exc)
                and getattr(rebuild, "resolved_provider", "") != "anthropic"
            ):
                _warn(node_name, state,
                      f"capacity error ({exc}) — falling back to anthropic backup tier")
                base_model, sem = await rebuild("anthropic")
                bound_model = base_model.bind_tools(tools)
                model = bound_model if bound else base_model
                return await _invoke(model, sem)
            _warn(node_name, state, f"LLM error ({exc}) — retrying once")
            await asyncio.sleep(5)
            return await _invoke(model, sem)

    for turn in range(1, max_turns + 1):
        # Turn-budget ladder, gated on countdown_from_turn so short budgets
        # stay quiet early (a 2-turn generator whose common case is answering
        # on turn 1 must not get a "one turn remains" nudge that invites tool
        # calls it would not otherwise make).
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

        resp = await _call_with_fallback(bound=True)
        messages.append(resp)

        tool_calls = getattr(resp, "tool_calls", None) or []
        if not tool_calls:
            return resp.content

        logger.info("  🔧 [%s] turn %d/%d → %d tool call(s): %s",
                    node_name, turn, max_turns, len(tool_calls),
                    [tc.get("name") for tc in tool_calls][:8])
        results = await asyncio.gather(
            *(exec_tool_call(tool_map, tc, tool_result_char_cap) for tc in tool_calls)
        )
        for (tool_msg, full_result), tc in zip(results, tool_calls):
            messages.append(tool_msg)
            if evidence_store is not None:
                harvest_evidence(tc.get("name", ""), full_result, evidence_store)

    # Reserved emission turn — planned hand-off; unbound model cannot call tools.
    logger.info("  📝 [%s] reserved emission turn (retrieval budget %d/%d used)",
                node_name, max_turns, max_turns)
    messages.append(HumanMessage(content=forced_stop_prompt))
    final = await _call_with_fallback(bound=False)
    return final.content


# ── Chunk-citation stripping (presentation boundary) ──────────────────────────

_UUID_PAT = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
# Matches [uuid], [uuid; uuid], [uuid, uuid] and runs of adjacent brackets.
# Leading whitespace is [ \t] only — never \s — so a citation at the start of a
# line cannot swallow the preceding newline and weld two lines together.
_CHUNK_CITE_RE = re.compile(
    r"[ \t]*\[\s*" + _UUID_PAT + r"(?:\s*[;,]\s*" + _UUID_PAT + r")*\s*\]"
)


def strip_chunk_citations(markdown: str) -> Tuple[str, int]:
    """Remove inline [chunk-uuid] citations and tidy the punctuation they leave
    behind. Returns (cleaned_text, n_citations_removed). Real bracketed legal
    citations ([Rule 23(a)], [ECF No. 323]) are untouched — only UUID forms
    match."""
    if not markdown:
        return markdown, 0
    n = len(_CHUNK_CITE_RE.findall(markdown))
    cleaned = _CHUNK_CITE_RE.sub("", markdown)
    cleaned = re.sub(r"[ \t]+([.,;:!?)])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+$", "", cleaned, flags=re.M)
    return cleaned, n
