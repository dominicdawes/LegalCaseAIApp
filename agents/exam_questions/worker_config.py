# agents/exam_questions/worker_config.py
"""
Worker-class abstraction for exam-agent nodes.

WorkerClass tiers map to model capability levels so provider/model can be
swapped via env var without touching individual node implementations.

Usage:
    provider, model = _fetch_worker_model("worker_mid")
    client = LLMFactory.get_client_for(provider, model, ...)
"""

import os
from typing import Dict, Optional, Tuple

# ── Type alias ────────────────────────────────────────────────────────────────
# "tool_only"   — no LLM, purely tool/deterministic logic
# "worker_low"  — cheapest model; fast routing, grounding, claim extraction
# "worker_mid"  — default mid-range; planning, clustering, revision
# "orchestrator"— flagship; drafting, answer-key generation, final polish

WorkerClass = str  # Literal["tool_only", "worker_low", "worker_mid", "orchestrator"]

DEFAULT_PROVIDER: str = os.getenv("EXAM_AGENT_PROVIDER", "anthropic")

# ── Model map: tier → provider → full model name ──────────────────────────────
WORKER_MODEL_MAP: Dict[str, Dict[str, str]] = {
    "tool_only": {},
    "worker_low": {
        "anthropic": "claude-haiku-4-5-20251001",
        "openai":    "gpt-4o-mini",
        "gemini":    "gemini-2.0-flash",
        "deepseek":  "deepseek-chat",
    },
    "worker_mid": {
        "anthropic": "claude-sonnet-4-6",
        "openai":    "gpt-4o",
        "gemini":    "gemini-2.5-flash",
        "deepseek":  "deepseek-reasoner",
    },
    "orchestrator": {
        "anthropic": "claude-opus-4-7",
        "openai":    "o4-mini",
        "gemini":    "gemini-2.5-pro",
        "deepseek":  "deepseek-reasoner",
    },
}

# ── Cost table ($/token) for budget tracking ──────────────────────────────────
# Keyed by full model name: (input_cost_per_token, output_cost_per_token)
MODEL_COST_MAP: Dict[str, Tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.25e-6, 1.25e-6),
    "claude-sonnet-4-6":         (3e-6,   15e-6),
    "claude-opus-4-7":           (15e-6,  75e-6),
    "gpt-4o-mini":               (0.15e-6, 0.6e-6),
    "gpt-4o":                    (2.5e-6, 10e-6),
    "o4-mini":                   (1.1e-6,  4.4e-6),
}


def _fetch_worker_model(
    worker_class: WorkerClass,
    provider: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Return (provider, model_name) for the given worker class.

    Falls back to anthropic if the requested provider has no entry for the
    given tier. Falls back to worker_mid if the tier is unrecognised.
    """
    _provider = (provider or DEFAULT_PROVIDER).lower()
    tier = WORKER_MODEL_MAP.get(worker_class, WORKER_MODEL_MAP["worker_mid"])
    model_name = tier.get(_provider) or tier.get("anthropic", "claude-sonnet-4-6")
    return _provider, model_name


def model_costs(model_name: str) -> Tuple[float, float]:
    """Return (input $/tok, output $/tok) for a model name, defaulting to Sonnet rates."""
    return MODEL_COST_MAP.get(model_name, (3e-6, 15e-6))
