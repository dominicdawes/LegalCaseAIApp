# utils/ingest_telemetry.py

"""
Memory and fan-out telemetry for the ingest pipeline.

Written after a production incident where six concurrent uploads each built
their own Docling converter, loaded six copies of the layout model, and
OOM-killed a 4GB Render worker in a loop for two days. The logs at the time
showed six "Loading weights" bars and nothing else — no fan-out count, no RSS,
no indication that the six were concurrent rather than sequential.

Everything here exists to make that diagnosable from the log alone:

  * fan-out   — how many documents are in the pipeline right now, and the peak
  * pressure  — how many are stuck waiting for the Docling parse slot
  * memory    — RSS before/after each phase, and the delta attributable to it
  * ceiling   — the CONTAINER limit (cgroup), not the host's RAM, so the
                percentage means something on Render

Log lines are prefixed [MEM] so `grep '\\[MEM\\]'` gives a complete picture of a
worker's memory story in order.

All measurement is best-effort: if psutil or cgroup files are unavailable the
helpers degrade to returning None and logging without the numbers. Telemetry
must never be the reason an ingest fails.
"""

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import psutil
    _PROCESS = psutil.Process(os.getpid())
except Exception:  # pragma: no cover - psutil should be installed, but never fail on it
    psutil = None
    _PROCESS = None

# Warn once RSS crosses this share of the container limit, so there is a signal
# in the log BEFORE the OOM killer produces a silent restart.
MEM_WARN_PCT = float(os.getenv("MEM_WARN_PCT", "75"))
MEM_CRITICAL_PCT = float(os.getenv("MEM_CRITICAL_PCT", "88"))

# Fan-out at or above this many concurrent documents gets a WARNING on each new
# high-water mark. Set to roughly the point where the worker got into trouble.
FANOUT_ALERT_THRESHOLD = int(os.getenv("FANOUT_ALERT_THRESHOLD", "4"))

# RSS still held after a document that ran ALONE finishes. Torch caches arenas
# and lazily initialises clients, so small positive deltas are normal early on;
# what matters is whether this keeps growing batch after batch.
RETAINED_ALERT_MB = float(os.getenv("RETAINED_ALERT_MB", "100"))


# ——— Container Memory Limit ——————————————————————————————————————————————————


def _detect_memory_limit_mb() -> Optional[float]:
    """
    Read the container's memory ceiling.

    psutil.virtual_memory().total reports the HOST's RAM, which on Render is far
    larger than the instance limit — using it makes every percentage look
    harmlessly low right up until the OOM kill. The real limit lives in cgroup.
    """
    candidates = [
        "/sys/fs/cgroup/memory.max",                      # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",    # cgroup v1
    ]
    for path in candidates:
        try:
            with open(path) as fh:
                raw = fh.read().strip()
            if raw in ("max", ""):
                continue
            value = int(raw)
            # cgroup v1 reports an absurd sentinel when unlimited.
            if value <= 0 or value >= (1 << 62):
                continue
            return value / (1024 * 1024)
        except Exception:
            continue

    # Fall back to host RAM — better than nothing, but flagged as such by the
    # caller logging "host" instead of "limit".
    if psutil is not None:
        try:
            return psutil.virtual_memory().total / (1024 * 1024)
        except Exception:
            pass
    return None


MEMORY_LIMIT_MB = _detect_memory_limit_mb()


def rss_mb() -> Optional[float]:
    """Current resident set size of this process, in MB."""
    if _PROCESS is None:
        return None
    try:
        return _PROCESS.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _fmt_mb(value: Optional[float]) -> str:
    return f"{value:,.0f}MB" if value is not None else "?MB"


def _pct_of_limit(value: Optional[float]) -> Optional[float]:
    if value is None or not MEMORY_LIMIT_MB:
        return None
    return 100.0 * value / MEMORY_LIMIT_MB


# ——— Fan-out Tracker —————————————————————————————————————————————————————————


class IngestTelemetry:
    """
    Process-wide counters for concurrent ingest work.

    Cheap by design: a lock, a few ints, and one psutil read per event. Safe to
    call from the worker's threads and from its event loop.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._inflight = 0          # documents between task start and task end
        self._peak_inflight = 0
        self._parsing = 0           # documents inside converter.convert()
        self._waiting = 0           # documents queued for a parse slot
        self._peak_rss = 0.0
        self._warned_at_pct = 0.0   # de-dupes the pressure warnings

    # ── State ────────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            state = {
                "inflight": self._inflight,
                "peak_inflight": self._peak_inflight,
                "parsing": self._parsing,
                "waiting": self._waiting,
            }
        current = rss_mb()
        if current is not None:
            with self._lock:
                self._peak_rss = max(self._peak_rss, current)
                state["peak_rss_mb"] = round(self._peak_rss, 1)
            state["rss_mb"] = round(current, 1)
            pct = _pct_of_limit(current)
            if pct is not None:
                state["pct_of_limit"] = round(pct, 1)
        return state

    def format_state(self) -> str:
        """One compact, greppable summary of where the worker stands."""
        s = self.snapshot()
        rss = s.get("rss_mb")
        pct = s.get("pct_of_limit")
        mem = _fmt_mb(rss)
        if MEMORY_LIMIT_MB:
            mem += f"/{_fmt_mb(MEMORY_LIMIT_MB)}"
        if pct is not None:
            mem += f" ({pct:.0f}%)"
        return (
            f"docs={s['inflight']} (peak {s['peak_inflight']}) "
            f"parsing={s['parsing']} queued={s['waiting']} rss={mem}"
        )

    # ── Pressure ─────────────────────────────────────────────────────────────

    def check_pressure(self, context: str = "") -> None:
        """
        Emit a warning when RSS climbs toward the container limit.

        This is the line that should appear in the log a few seconds before an
        OOM kill, naming exactly how many documents were in flight at the time.
        """
        current = rss_mb()
        pct = _pct_of_limit(current)
        if pct is None:
            return

        if pct >= MEM_CRITICAL_PCT:
            level, tag = logger.error, "🚨 [MEM] CRITICAL"
        elif pct >= MEM_WARN_PCT:
            level, tag = logger.warning, "⚠️ [MEM] HIGH"
        else:
            with self._lock:
                self._warned_at_pct = 0.0
            return

        # Only re-warn after a further 5 percentage points, so a long parse near
        # the limit doesn't flood the log.
        with self._lock:
            if pct < self._warned_at_pct + 5.0:
                return
            self._warned_at_pct = pct

        suffix = f" during {context}" if context else ""
        level(
            f"{tag}{suffix} — {self.format_state()}. "
            f"Concurrent documents are the usual cause; "
            f"lower CELERY_CONCURRENCY or DOCLING_PARSE_CONCURRENCY."
        )

    # ── Document lifecycle ───────────────────────────────────────────────────

    def document_started(self, doc_id: str, filename: str = "") -> Optional[float]:
        """Record a document entering the pipeline. Returns starting RSS."""
        with self._lock:
            self._inflight += 1
            is_new_peak = self._inflight > self._peak_inflight
            self._peak_inflight = max(self._peak_inflight, self._inflight)
            inflight = self._inflight

        start_rss = rss_mb()
        name = f" '{filename}'" if filename else ""
        logger.info(f"📈 [MEM] doc={doc_id[:8]}{name} START — {self.format_state()}")

        # Only shout on a NEW high-water mark, and only once fan-out is actually
        # interesting. The running count is already in every line above; this
        # line exists to make a dangerous burst impossible to miss.
        if is_new_peak and inflight >= FANOUT_ALERT_THRESHOLD:
            logger.warning(
                f"🔀 [MEM] FAN-OUT HIGH: {inflight} documents processing concurrently "
                f"(new peak) — each holds file bytes, chunks and embeddings in memory. "
                f"Cap with CELERY_CONCURRENCY."
            )
        self.check_pressure("document start")
        return start_rss

    def document_finished(
        self,
        doc_id: str,
        start_rss: Optional[float] = None,
        status: str = "",
    ) -> None:
        """
        Record a document leaving the pipeline, with its memory delta.

        The delta is process-wide RSS between this document's start and end, so
        it is only attributable to THIS document when nothing else was running.
        With concurrent documents it necessarily includes their allocations too —
        labelling it "retained" regardless would cry leak on every busy batch.
        """
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            others_running = self._inflight

        end_rss = rss_mb()
        delta = ""
        if start_rss is not None and end_rss is not None:
            change = end_rss - start_rss
            if others_running > 0:
                # Confounded: other documents allocated during this window.
                delta = (
                    f" rss{change:+,.0f}MB over its lifetime "
                    f"(shared with {others_running} concurrent doc(s), not attributable)"
                )
            elif change > RETAINED_ALERT_MB:
                # Sole occupant and RSS still climbed — the real leak signature.
                delta = (
                    f" net={change:+,.0f}MB ⬆ RETAINED "
                    f"(ran alone — this memory was not released)"
                )
            else:
                delta = f" net={change:+,.0f}MB"

        label = f" {status}" if status else ""
        logger.info(f"📉 [MEM] doc={doc_id[:8]} END{label}{delta} — {self.format_state()}")

    # ── Parse slot ───────────────────────────────────────────────────────────

    @contextmanager
    def parse_slot(self, doc_id: str, semaphore):
        """
        Acquire `semaphore` for a Docling conversion, logging the queue wait and
        the memory the parse costs.

        Owns the semaphore so the counters can't drift: whatever happens inside,
        the waiting/parsing gauges and the release both unwind correctly.
        """
        with self._lock:
            self._waiting += 1
            waiting, parsing = self._waiting, self._parsing

        if waiting > 1 or parsing > 0:
            logger.info(
                f"⏳ [MEM] doc={doc_id[:8]} waiting for a Docling parse slot "
                f"({parsing} parsing, {waiting} queued) — this is the "
                f"DOCLING_PARSE_CONCURRENCY cap doing its job, not a hang"
            )

        queued_at = time.perf_counter()
        acquired = False
        try:
            semaphore.acquire()
            acquired = True

            with self._lock:
                self._waiting = max(0, self._waiting - 1)
                self._parsing += 1

            waited = time.perf_counter() - queued_at
            start_rss = rss_mb()
            wait_note = f" after queueing {waited:.1f}s" if waited > 0.5 else ""
            logger.info(
                f"🦆 [MEM] doc={doc_id[:8]} PARSE start{wait_note} — {self.format_state()}"
            )
            self.check_pressure("docling parse")

            parse_started_at = time.perf_counter()
            try:
                yield
            finally:
                with self._lock:
                    self._parsing = max(0, self._parsing - 1)

                end_rss = rss_mb()
                elapsed = time.perf_counter() - parse_started_at
                delta = ""
                if start_rss is not None and end_rss is not None:
                    delta = (
                        f" rss {_fmt_mb(start_rss)}→{_fmt_mb(end_rss)} "
                        f"({end_rss - start_rss:+,.0f}MB)"
                    )
                logger.info(
                    f"🦆 [MEM] doc={doc_id[:8]} PARSE done in {elapsed:.1f}s{delta} "
                    f"— {self.format_state()}"
                )
                self.check_pressure("post-parse")
        finally:
            if acquired:
                semaphore.release()
            else:
                # Never got the slot (interrupt/shutdown) — undo the gauge.
                with self._lock:
                    self._waiting = max(0, self._waiting - 1)

    @contextmanager
    def model_load(self, what: str):
        """
        Wrap a one-time model/converter build and report what it costs.

        This is the number that explains an OOM: if the Docling converters cost
        ~500MB and that shows up once, the singleton is working; if it shows up
        per document, it isn't.
        """
        before = rss_mb()
        started = time.perf_counter()
        try:
            yield
        finally:
            after = rss_mb()
            cost = ""
            if before is not None and after is not None:
                cost = f" cost {after - before:+,.0f}MB (rss {_fmt_mb(before)}→{_fmt_mb(after)})"
            logger.info(
                f"🧠 [MEM] {what} loaded in {time.perf_counter() - started:.1f}s{cost} "
                f"— one-time, shared by every document in this worker"
            )
            self.check_pressure("model load")


# Single process-wide instance.
telemetry = IngestTelemetry()


def log_startup_memory() -> None:
    """Log the memory ceiling once at worker boot, so later percentages parse."""
    limit = _fmt_mb(MEMORY_LIMIT_MB) if MEMORY_LIMIT_MB else "unknown"
    logger.info(
        f"🧮 [MEM] Worker memory ceiling: {limit} | baseline rss={_fmt_mb(rss_mb())} | "
        f"warn at {MEM_WARN_PCT:.0f}%, critical at {MEM_CRITICAL_PCT:.0f}%"
    )
