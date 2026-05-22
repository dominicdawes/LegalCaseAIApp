# agents/attack_outline/worker_config.py
"""
Worker-class abstraction for the attack-outline agent nodes.

Mirrors the exam_questions worker_config; controlled by a separate env var
so exam and attack-outline model tiers can be tuned independently.

WorkerClass tiers:
  "tool_only"   — no LLM; deterministic retrieval, DB writes, JSON parsing
  "worker_low"  — cheapest model; fast claim extraction, reranking, formatting
  "worker_mid"  — default mid-range; profiling, extraction, normalisation, block building
  "orchestrator"— flagship; planning, graph building, critique, revision
"""

import os
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv

# ——— Logging & Env Load ───────────────────────────────────────────────────────
load_dotenv()

WorkerClass = str  # Literal["tool_only", "worker_low", "worker_mid", "orchestrator"]

DEFAULT_PROVIDER: str = os.getenv("ATTACK_AGENT_PROVIDER", "anthropic")

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
    _provider = (provider or DEFAULT_PROVIDER).lower()
    tier = WORKER_MODEL_MAP.get(worker_class, WORKER_MODEL_MAP["worker_mid"])
    model_name = tier.get(_provider) or tier.get("anthropic", "claude-sonnet-4-6")
    return _provider, model_name


def model_costs(model_name: str) -> Tuple[float, float]:
    return MODEL_COST_MAP.get(model_name, (3e-6, 15e-6))
