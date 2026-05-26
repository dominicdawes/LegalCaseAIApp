# agents/flashcards/worker_config.py
"""
Worker-class abstraction for flashcard-agent nodes.

Tier map:
  tool_only   — no LLM; deterministic DB writes / formatting
  worker_low  — cheapest; global critic, coverage checks
  worker_mid  — default; blueprint planner, card evaluator, backside enricher
  orchestrator— flagship; concept extraction, card drafting, repair
"""

import os
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv

# ——— Logging & Env Load ───────────────────────────────────────────────────────
load_dotenv()

WorkerClass = str  # Literal["tool_only", "worker_low", "worker_mid", "orchestrator"]

DEFAULT_PROVIDER: str = os.getenv("FLASHCARD_AGENT_PROVIDER", "deepseek")

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
        "deepseek":  "deepseek-v4-pro",   # cost-effective alternative to sonnet
    },
    "orchestrator": {
        "anthropic": "claude-opus-4-7",
        "openai":    "o4-mini",
        "gemini":    "gemini-2.5-pro",
        "deepseek":  "deepseek-v4-pro",   # thinking=True passed via extra_body
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


# ——— Thinking-mode per worker class (DeepSeek V4, April 2026+) ───────────────
# DeepSeek V4 models default to thinking=enabled.  Must be explicitly disabled for
# worker_mid/low nodes (structured extraction — no reasoning depth needed) and kept
# enabled for orchestrator nodes (planning, critique, graph building).
# DeepSeekClient translates True/False → extra_body={"thinking": {"type": ...}}.
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

    ``thinking`` is True/False for DeepSeek (forwarded via extra_body to the V4 API),
    or None for all other providers (ignored — not forwarded).
    """
    _provider = (provider or DEFAULT_PROVIDER).lower()
    tier = WORKER_MODEL_MAP.get(worker_class, WORKER_MODEL_MAP["worker_mid"])
    model_name = tier.get(_provider) or tier.get("anthropic", "claude-sonnet-4-6")
    thinking: Optional[bool] = WORKER_THINKING_MAP.get(worker_class, {}).get(_provider)
    return _provider, model_name, thinking


def model_costs(model_name: str) -> Tuple[float, float]:
    return MODEL_COST_MAP.get(model_name, (3e-6, 15e-6))
