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

DEFAULT_PROVIDER: str = os.getenv("ATTACK_AGENT_PROVIDER", "deepseek")

WORKER_MODEL_MAP: Dict[str, Dict[str, str]] = {
    "tool_only": {},
    "worker_low": {
        "anthropic": "claude-haiku-4-5-20251001",
        "openai":    "gpt-4o-mini",
        "gemini":    "gemini-2.0-flash",
        "deepseek":  "deepseek-v4-flash",
    },
    "worker_mid": {
        "anthropic": "claude-sonnet-4-6",
        "openai":    "gpt-4o",
        "gemini":    "gemini-2.5-flash",
        "deepseek":  "deepseek-v4-pro",   # default — cost-effective alternative to sonnet
    },
    "orchestrator": {
        "anthropic": "claude-opus-4-7",
        "openai":    "o4-mini",
        "gemini":    "gemini-2.5-pro",
        "deepseek":  "deepseek-v4-pro",
    },
}

MODEL_COST_MAP: Dict[str, Tuple[float, float]] = {
    # Anthropic
    "claude-haiku-4-5-20251001": (0.25e-6,  1.25e-6),
    "claude-sonnet-4-6":         (3e-6,    15e-6),
    "claude-opus-4-7":           (15e-6,   75e-6),
    # OpenAI
    "gpt-4o-mini":               (0.15e-6,  0.6e-6),
    "gpt-4o":                    (2.5e-6,  10e-6),
    "o4-mini":                   (1.1e-6,   4.4e-6),
    # DeepSeek (prices per token, approximate)
    "deepseek-v4-flash":         (0.07e-6,  0.28e-6),
    "deepseek-v4-pro":           (0.27e-6,  1.10e-6),
    "deepseek-chat":             (0.27e-6,  1.10e-6),
    "deepseek-reasoner":         (0.55e-6,  2.19e-6),
}


# ——— Thinking-mode per worker class (DeepSeek only) ──────────────────────────
# orchestrator nodes need deep reasoning (planning, critique, graph building)
# worker_mid nodes do structured extraction — thinking overhead hurts throughput
WORKER_THINKING_MAP: Dict[str, Dict[str, Optional[bool]]] = {
    "orchestrator": {"deepseek": True},
    "worker_mid":   {"deepseek": False},
    "worker_low":   {"deepseek": False},
}


def _fetch_worker_model(
    worker_class: WorkerClass,
    provider: Optional[str] = None,
) -> Tuple[str, str, Optional[bool]]:
    """Return (provider, model_name, thinking) for the given worker class.

    ``thinking`` is ``True``/``False`` for DeepSeek models that support the
    thinking flag, or ``None`` for all other providers (no-op kwarg).
    """
    _provider = (provider or DEFAULT_PROVIDER).lower()
    tier = WORKER_MODEL_MAP.get(worker_class, WORKER_MODEL_MAP["worker_mid"])
    model_name = tier.get(_provider) or tier.get("anthropic", "claude-sonnet-4-6")
    thinking: Optional[bool] = WORKER_THINKING_MAP.get(worker_class, {}).get(_provider)
    return _provider, model_name, thinking


def model_costs(model_name: str) -> Tuple[float, float]:
    return MODEL_COST_MAP.get(model_name, (3e-6, 15e-6))
