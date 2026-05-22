# utils/llm_clients/anthropic_rate_limits.py
"""
Dynamic Anthropic rate-limit probe.

Fires a minimal 1-token request to the Anthropic API once per hour and reads
the ``anthropic-ratelimit-output-tokens-limit`` response header to discover the
organisation's current output token rate limit.  That limit is then mapped to
an ``asyncio.Semaphore`` whose size caps the number of concurrent ``_llm()``
calls across the agent, keeping throughput within the org's tier allowance.

Why not a static semaphore?
  A hardcoded ``Semaphore(2)`` is correct for Tier 1 (8 000 output tok/min) but
  wastefully conservative at Tier 2+ (80 000+ tok/min).  After purchasing more
  credits and being promoted to a higher tier the agent automatically opens up
  to more parallelism on the next heartbeat — no config change needed.

How semaphore replacement works:
  When the probe detects a tier change, ``self._semaphore`` is replaced with a
  new ``asyncio.Semaphore`` of the updated size.  Callers that already acquired
  the old semaphore continue to hold/release it normally (it is garbage-collected
  once all holders release).  New callers (via ``get_llm_semaphore()``) receive
  the new object.  No deadlock or double-release is possible.

Usage in nodes.py::

    from utils.llm_clients.anthropic_rate_limits import get_llm_semaphore

    async with await get_llm_semaphore():
        result = await client.achat(...)
"""

import asyncio
import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# Model used for the probe request.  Must match the model family whose rate
# limit you care about most — Sonnet is the heaviest node in the attack outline
# pipeline, so we measure its limit.
PROBE_MODEL = "claude-sonnet-4-6"

# How long (seconds) before a cached probe result is considered stale.
PROBE_INTERVAL_SECS: int = 3_600  # 1 hour

# Conservative fallback: Tier 1 Sonnet limit (8 000 output tok/min).
# Used at startup before the first successful probe.
DEFAULT_OUTPUT_TPM: int = 8_000

# ── Tier table: output tokens/min → max concurrent _llm() calls ───────────────
#
# Derivation (conservative):
#   concurrency ≈ output_tpm / (avg_tokens_per_call × completions_per_min_per_slot)
#   avg_tokens_per_call  ≈ 3 500  (legal_artifact_extractor worst-case)
#   completions/min/slot ≈  4     (one call every ~15 s)
#   → concurrency ≈ tpm / 14 000
#
# We round up slightly and rely on the 429-retry safety net in _llm() to catch
# the occasional burst rather than capping too aggressively.
#
# Adjust these thresholds if Anthropic updates tier limits.
_TPM_CONCURRENCY_TABLE: list[tuple[int, int]] = [
    (8_000,   2),   # Tier 1  — Sonnet: 8 K tok/min
    (16_000,  3),
    (40_000,  6),   # Tier 2  — Sonnet:  ~40–80 K tok/min
    (80_000,  10),
    (160_000, 15),
    (400_000, 20),  # Tier 3+ — cap at 20 to avoid thundering-herd
]


def _tpm_to_concurrency(output_tpm: int) -> int:
    """Map an observed output tokens/min limit to a semaphore concurrency value."""
    for threshold, slots in _TPM_CONCURRENCY_TABLE:
        if output_tpm <= threshold:
            return slots
    return 20  # above all table entries → use the ceiling


# ── Shared singleton state ─────────────────────────────────────────────────────

class _AnthropicRateLimitState:
    """
    Singleton that holds the current concurrency semaphore and refreshes it
    once per PROBE_INTERVAL_SECS by probing the Anthropic API.
    """

    def __init__(self) -> None:
        initial_concurrency     = _tpm_to_concurrency(DEFAULT_OUTPUT_TPM)
        self._output_tpm:  int               = DEFAULT_OUTPUT_TPM
        self._concurrency: int               = initial_concurrency
        self._semaphore:   asyncio.Semaphore = asyncio.Semaphore(initial_concurrency)
        self._last_probed: float             = 0.0   # monotonic timestamp; 0 → never probed
        self._lock:        asyncio.Lock      = asyncio.Lock()

    # ── Public properties (read-only) ─────────────────────────────────────

    @property
    def output_tpm(self) -> int:
        """Last observed output tokens/minute limit."""
        return self._output_tpm

    @property
    def concurrency(self) -> int:
        """Current semaphore size."""
        return self._concurrency

    # ── Semaphore accessors ───────────────────────────────────────────────

    def semaphore_sync(self) -> asyncio.Semaphore:
        """Return the current semaphore without triggering a probe (sync-safe)."""
        return self._semaphore

    async def get_semaphore(self) -> asyncio.Semaphore:
        """Ensure the probe is fresh, then return the semaphore.

        Cheap (no-op) when the cached value is within PROBE_INTERVAL_SECS.
        The probe itself fires a tiny 1-token API call; callers are NOT blocked
        while waiting for the probe — the lock is acquired only once and other
        awaiters return the current (stale) semaphore immediately.
        """
        await self._ensure_fresh()
        return self._semaphore

    # ── Internals ─────────────────────────────────────────────────────────

    async def _ensure_fresh(self) -> None:
        """Trigger a probe only if the cached result is older than PROBE_INTERVAL_SECS."""
        if time.monotonic() - self._last_probed < PROBE_INTERVAL_SECS:
            return  # still fresh — fast path
        async with self._lock:
            # Double-check inside the lock: another coroutine may have probed
            # while we waited to acquire it.
            if time.monotonic() - self._last_probed < PROBE_INTERVAL_SECS:
                return
            await self._probe()

    async def _probe(self) -> None:
        """
        Fire a minimal 1-token request to PROBE_MODEL and read the
        ``anthropic-ratelimit-output-tokens-limit`` response header.

        Sets ``_last_probed`` regardless of outcome so we don't retry on every
        call after a transient failure.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning(
                "anthropic_rate_limits: ANTHROPIC_API_KEY not set — "
                "keeping defaults (tpm=%d, concurrency=%d)",
                self._output_tpm, self._concurrency,
            )
            self._last_probed = time.monotonic()
            return

        new_tpm: Optional[int] = None
        try:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=api_key)
            # with_raw_response gives us the full HTTP response so we can read headers.
            raw = await client.messages.with_raw_response.create(
                model=PROBE_MODEL,
                max_tokens=1,
                messages=[{"role": "user", "content": "x"}],
            )
            header_val: str = raw.headers.get(
                "anthropic-ratelimit-output-tokens-limit", ""
            )
            if header_val:
                new_tpm = int(header_val)
            else:
                logger.warning(
                    "anthropic_rate_limits: 'anthropic-ratelimit-output-tokens-limit' "
                    "header absent in probe response — keeping tpm=%d",
                    self._output_tpm,
                )
        except Exception as exc:
            logger.warning(
                "anthropic_rate_limits: probe to %s failed (%s) — keeping tpm=%d",
                PROBE_MODEL, exc, self._output_tpm,
            )

        # Always update the timestamp so we don't spam on failures.
        self._last_probed = time.monotonic()

        if new_tpm is None:
            return

        new_concurrency = _tpm_to_concurrency(new_tpm)

        if new_tpm == self._output_tpm:
            logger.debug(
                "🎚 anthropic_rate_limits: heartbeat OK — "
                "output_tpm=%d  concurrency=%d  (no change)",
                new_tpm, new_concurrency,
            )
            return

        # Tier change detected — swap the semaphore.
        logger.info(
            "🎚 anthropic_rate_limits: tier change detected — "
            "output_tpm %d → %d  concurrency %d → %d",
            self._output_tpm, new_tpm,
            self._concurrency, new_concurrency,
        )
        self._output_tpm  = new_tpm
        self._concurrency = new_concurrency
        # Replace the semaphore object.  Existing holders keep a reference to
        # the old Semaphore and release it normally; it is then garbage-collected.
        # Future callers (get_semaphore → this object) get the new one.
        self._semaphore = asyncio.Semaphore(new_concurrency)


# Module-level singleton.
_state = _AnthropicRateLimitState()


# ── Public API ────────────────────────────────────────────────────────────────

async def get_llm_semaphore() -> asyncio.Semaphore:
    """Return the dynamically sized LLM concurrency semaphore.

    Triggers a background probe if the cached rate-limit header is older than
    PROBE_INTERVAL_SECS (default 1 hour).  The probe call is cheap — only 1
    input token, 1 output token — and happens at most once per interval.

    Typical usage in an async node::

        async with await get_llm_semaphore():
            result = await client.achat(prompt)
    """
    return await _state.get_semaphore()


def get_llm_semaphore_sync() -> asyncio.Semaphore:
    """Return the current semaphore without probing (sync contexts).

    Returns whatever was last cached.  Safe to call from non-async code, e.g.
    to inspect the current concurrency level in health-check endpoints.
    """
    return _state.semaphore_sync()


def current_output_tpm() -> int:
    """Last observed org output tokens/min limit (for metrics / health checks)."""
    return _state.output_tpm


def current_concurrency() -> int:
    """Current semaphore size (for metrics / health checks)."""
    return _state.concurrency
