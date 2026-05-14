# agents/ledger/agent_ledger.py
"""
AgentLedgerService — relational state-management layer for LangGraph agent runs.

Tracks three levels of state:
  agent_jobs      — the overarching user request (survives across retry attempts)
  agent_runs      — one row per graph execution attempt, carries the LangGraph thread_id
  agent_artifacts — versioned intermediate outputs (concept maps, blueprints, Q&A, etc.)

Design note on client choice:
  We use asyncpg directly rather than the supabase-py sync client because:
    1. All caller code is async; run_in_executor wrapping adds noise.
    2. save_artifact() requires SELECT … FOR UPDATE for safe versioned upserts,
       which PostgREST (supabase-py) cannot express.
    3. The project's asyncpg pool (tasks.database) is already initialised per Celery
       worker — zero extra connection overhead.
  The existing global pool from tasks.database is used by default; an explicit pool
  can be injected for testing.

Artifact TTL:
  All artifacts are written with expires_at = NOW() + 30 days.
  A Supabase scheduled function (cron_job) handles physical deletion.
"""

from __future__ import annotations

import hashlib
import json
import logging
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import asyncpg
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_ARTIFACT_TTL_DAYS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class LedgerError(Exception):
    """Base for all AgentLedgerService errors."""


class LedgerDatabaseError(LedgerError):
    """Raised when a database operation fails unexpectedly."""


class ArtifactNotFoundError(LedgerError):
    """Raised when get_latest_artifact finds no matching artifact."""


class RunNotFoundError(LedgerError):
    """Raised when no active run exists for a given job_id."""


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class RunMetadata(BaseModel):
    """Returned by initialize_run() and create_retry_run()."""
    run_id: UUID
    job_id: UUID
    langgraph_thread_id: str
    attempt: int
    graph_name: str
    created_at: datetime

    model_config = {"from_attributes": True}


class SaveArtifactInput(BaseModel):
    """Validated input for save_artifact()."""
    job_id: UUID
    artifact_key: str                           # logical name, e.g. "source_profile:doc_abc"
    content: Dict[str, Any]
    worker_class: str                           # "tool_only"|"worker_low"|"worker_mid"|"orchestrator"
    node_name: str = ""                         # LangGraph node that produced this artifact
    artifact_type: str = "intermediate"         # free-form label: "concept_map", "blueprint", etc.
    input_artifact_ids: List[UUID] = Field(default_factory=list)
    model_id: Optional[str] = None
    source_ids: List[UUID] = Field(default_factory=list)
    chunk_ids: List[UUID] = Field(default_factory=list)
    lifecycle: str = "intermediate"             # "durable"|"intermediate"|"debug"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[int] = None
    token_usage: Optional[Dict[str, Any]] = None
    prompt_hash: Optional[str] = None

    @field_validator("worker_class")
    @classmethod
    def _valid_worker_class(cls, v: str) -> str:
        allowed = {"tool_only", "worker_low", "worker_mid", "orchestrator"}
        if v not in allowed:
            raise ValueError(f"worker_class must be one of {allowed}")
        return v

    @field_validator("lifecycle")
    @classmethod
    def _valid_lifecycle(cls, v: str) -> str:
        allowed = {"durable", "intermediate", "debug", "expired"}
        if v not in allowed:
            raise ValueError(f"lifecycle must be one of {allowed}")
        return v


class ArtifactRecord(BaseModel):
    """Matches the agent_artifacts table row."""
    id: UUID
    job_id: UUID
    node_name: str
    artifact_type: str
    artifact_key: Optional[str]
    content: Dict[str, Any]
    version: int
    is_latest: bool
    worker_class: Optional[str]
    model_id: Optional[str]
    parent_artifact_id: Optional[UUID]
    input_artifact_ids: List[UUID] = Field(default_factory=list)
    source_ids: List[UUID] = Field(default_factory=list)
    created_at: datetime

    model_config = {"from_attributes": True}


class CheckpointInfo(BaseModel):
    """Summarises the latest LangGraph checkpoint for a thread."""
    thread_id: str
    checkpoint_id: Optional[str]
    checkpoint_ns: str = ""
    exists: bool


class ResumptionContext(BaseModel):
    """Full crash-recovery payload returned by get_resumption_context()."""
    run: RunMetadata
    checkpoint: CheckpointInfo
    # artifact_key → content dict for every is_latest=True artifact in the job
    latest_artifacts: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # Human-readable label of where the graph was when it crashed
    last_node: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────

class AgentLedgerService:
    """
    Async service layer for managing agent_jobs, agent_runs, and agent_artifacts.

    Usage (inside a Celery async worker):
        ledger = AgentLedgerService()
        run = await ledger.initialize_run(job_id=..., graph_name="exam_questions")
        artifact_id = await ledger.save_artifact(
            job_id=job_id,
            artifact_key="source_profile:doc_abc",
            content={"key_concepts": [...], ...},
            worker_class="tool_only",
            node_name="source_profiler",
            artifact_type="source_profile",
        )
        ...
        ctx = await ledger.get_resumption_context(job_id)
        # Pass ctx.run.langgraph_thread_id into run_exam_agent() to resume.
    """

    def __init__(self, pool: Optional[asyncpg.Pool] = None) -> None:
        self._explicit_pool = pool

    # ── Pool accessor ─────────────────────────────────────────────────────────

    async def _pool(self) -> asyncpg.Pool:
        if self._explicit_pool is not None:
            return self._explicit_pool
        from tasks.database import get_global_async_db_pool, init_async_pools
        pool = get_global_async_db_pool()
        if pool is None:
            await init_async_pools()
            pool = get_global_async_db_pool()
        if pool is None:
            raise LedgerDatabaseError("asyncpg pool is not initialised")
        return pool

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _expires_at() -> datetime:
        return datetime.now(timezone.utc) + timedelta(days=_ARTIFACT_TTL_DAYS)

    @staticmethod
    def _prompt_hash(content: dict) -> str:
        payload = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @staticmethod
    def _row_to_dict(row: asyncpg.Record) -> dict:
        return dict(row)

    # ── Job helpers ───────────────────────────────────────────────────────────

    async def ensure_job(
        self,
        job_id: UUID,
        project_id: str,
        source_ids: Optional[List[str]] = None,
        job_type: str = "exam_questions",
    ) -> None:
        """
        Idempotent upsert of an agent_jobs row.

        Safe to call on every attempt — ON CONFLICT only updates the status back
        to 'running' and the updated_at timestamp; all other fields are left alone.
        """
        now = self._now_utc()
        src_uuids = [s for s in (source_ids or []) if s]
        try:
            pool = await self._pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO agent_jobs (
                        id, project_id, job_type_enum, status_enum,
                        source_ids, requested_output_type,
                        started_at, created_at, updated_at
                    ) VALUES (
                        $1, $2, $3::agent_job_type, 'running'::agent_job_status,
                        $4, $5,
                        $6, $6, $6
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        status_enum = 'running'::agent_job_status,
                        updated_at  = EXCLUDED.updated_at
                    """,
                    job_id,
                    UUID(project_id),
                    job_type,
                    src_uuids,
                    job_type,
                    now,
                )
        except asyncpg.PostgresError as exc:
            raise LedgerDatabaseError(f"ensure_job failed: {exc}") from exc

    async def set_job_status(self, job_id: UUID, status: str) -> None:
        """
        Update agent_jobs.status_enum.

        Terminal statuses ('succeeded', 'failed', 'cancelled', 'expired') also
        set completed_at.
        """
        now = self._now_utc()
        try:
            pool = await self._pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE agent_jobs
                    SET status_enum  = $1::agent_job_status,
                        completed_at = CASE WHEN $1::text = ANY(ARRAY['succeeded', 'failed', 'cancelled', 'expired'])
                                           THEN $2 ELSE completed_at END,
                        updated_at   = $2
                    WHERE id = $3
                    """,
                    status,
                    now,
                    job_id,
                )
        except asyncpg.PostgresError as exc:
            raise LedgerDatabaseError(f"set_job_status failed: {exc}") from exc

    async def update_job_run_pointer(self, job_id: UUID, run_id: UUID) -> None:
        """Keep agent_jobs.run_id pointing at the current active run."""
        try:
            pool = await self._pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE agent_jobs
                    SET run_id = $1, updated_at = $2
                    WHERE id = $3
                    """,
                    run_id, self._now_utc(), job_id,
                )
        except Exception as exc:
            logger.warning("update_job_run_pointer failed (non-fatal): %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. initialize_run
    # ─────────────────────────────────────────────────────────────────────────

    async def initialize_run(
        self,
        job_id: UUID,
        graph_name: str,
        graph_version: Optional[str] = None,
        max_retries: int = 2,
    ) -> RunMetadata:
        """
        Create a new agent_runs row for this job attempt.

        Generates a fresh langgraph_thread_id (UUID4 string) that must be
        passed into the LangGraph config payload so AsyncPostgresSaver writes
        checkpoints under that key.

        Returns:
            RunMetadata containing run_id and langgraph_thread_id.
        """
        thread_id = str(uuid4())
        now = self._now_utc()

        try:
            pool = await self._pool()
            async with pool.acquire() as conn:
                # Count existing attempts for this job to set attempt number
                attempt: int = await conn.fetchval(
                    "SELECT COALESCE(MAX(attempt), 0) + 1 FROM agent_runs WHERE job_id = $1",
                    job_id,
                )

                row = await conn.fetchrow(
                    """
                    INSERT INTO agent_runs (
                        job_id, run_status, graph_name, graph_version,
                        langgraph_thread_id, attempt, max_retries, started_at,
                        created_at, updated_at
                    ) VALUES (
                        $1, 'running', $2, $3,
                        $4, $5, $6, $7,
                        $7, $7
                    )
                    RETURNING id, job_id, langgraph_thread_id, attempt, graph_name, created_at
                    """,
                    job_id, graph_name, graph_version,
                    thread_id, attempt, max_retries, now,
                )

        except asyncpg.PostgresError as exc:
            raise LedgerDatabaseError(f"initialize_run failed: {exc}") from exc

        run = RunMetadata(
            run_id=row["id"],
            job_id=row["job_id"],
            langgraph_thread_id=row["langgraph_thread_id"],
            attempt=row["attempt"],
            graph_name=row["graph_name"],
            created_at=row["created_at"],
        )

        # Best-effort pointer update — don't fail the run if this errors
        await self.update_job_run_pointer(job_id, run.run_id)
        logger.info(
            "Run %s initialised (job=%s attempt=%d thread=%s)",
            run.run_id, job_id, attempt, thread_id,
        )
        return run

    # ─────────────────────────────────────────────────────────────────────────
    # 2. save_artifact
    # ─────────────────────────────────────────────────────────────────────────

    async def save_artifact(
        self,
        job_id: UUID,
        artifact_key: str,
        content: Dict[str, Any],
        worker_class: str,
        node_name: str = "",
        artifact_type: str = "intermediate",
        input_artifact_ids: Optional[List[UUID]] = None,
        model_id: Optional[str] = None,
        source_ids: Optional[List[UUID]] = None,
        chunk_ids: Optional[List[UUID]] = None,
        lifecycle: str = "intermediate",
        metadata: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[int] = None,
        token_usage: Optional[Dict[str, Any]] = None,
        prompt_hash: Optional[str] = None,
    ) -> UUID:
        """
        Versioned upsert for an agent artifact.

        If an artifact with (job_id, artifact_key, is_latest=True) already exists:
          - marks the old row is_latest = False
          - inserts a new row with version = old_version + 1, parent_artifact_id = old_id
        Otherwise inserts as version = 1.

        The upsert runs inside a single transaction with SELECT … FOR UPDATE to
        prevent duplicate version numbers from concurrent writers.

        Returns:
            UUID of the newly inserted artifact row.
        """
        inp = SaveArtifactInput(
            job_id=job_id,
            artifact_key=artifact_key,
            content=content,
            worker_class=worker_class,
            node_name=node_name,
            artifact_type=artifact_type,
            input_artifact_ids=input_artifact_ids or [],
            model_id=model_id,
            source_ids=source_ids or [],
            chunk_ids=chunk_ids or [],
            lifecycle=lifecycle,
            metadata=metadata or {},
            latency_ms=latency_ms,
            token_usage=token_usage,
            prompt_hash=prompt_hash or self._prompt_hash(content),
        )

        expires = self._expires_at()

        try:
            pool = await self._pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    # Lock any existing latest row for this key
                    existing = await conn.fetchrow(
                        """
                        SELECT id, version
                        FROM agent_artifacts
                        WHERE job_id = $1
                          AND artifact_key = $2
                          AND is_latest = TRUE
                        FOR UPDATE
                        """,
                        inp.job_id, inp.artifact_key,
                    )

                    if existing:
                        old_id: UUID = existing["id"]
                        new_version: int = existing["version"] + 1
                        await conn.execute(
                            "UPDATE agent_artifacts SET is_latest = FALSE WHERE id = $1",
                            old_id,
                        )
                    else:
                        old_id = None
                        new_version = 1

                    row = await conn.fetchrow(
                        """
                        INSERT INTO agent_artifacts (
                            job_id, node_name, artifact_type, artifact_key,
                            content, source_ids, chunk_ids,
                            lifecycle, version, parent_artifact_id, is_latest,
                            worker_class, model_id, prompt_hash,
                            input_artifact_ids, metadata,
                            latency_ms, token_usage, expires_at, created_at
                        ) VALUES (
                            $1,  $2,  $3,  $4,
                            $5,  $6,  $7,
                            $8,  $9,  $10, TRUE,
                            $11, $12, $13,
                            $14, $15,
                            $16, $17, $18, NOW()
                        )
                        RETURNING id
                        """,
                        inp.job_id,
                        inp.node_name,
                        inp.artifact_type,
                        inp.artifact_key,
                        json.dumps(inp.content),
                        [str(s) for s in inp.source_ids],
                        [str(c) for c in inp.chunk_ids],
                        inp.lifecycle,
                        new_version,
                        old_id,
                        inp.worker_class,
                        inp.model_id,
                        inp.prompt_hash,
                        [str(a) for a in inp.input_artifact_ids],
                        json.dumps(inp.metadata),
                        inp.latency_ms,
                        json.dumps(inp.token_usage) if inp.token_usage else None,
                        expires,
                    )

        except asyncpg.PostgresError as exc:
            raise LedgerDatabaseError(f"save_artifact failed for key '{artifact_key}': {exc}") from exc

        new_id: UUID = row["id"]
        logger.debug(
            "Artifact saved: key=%s version=%d id=%s job=%s",
            artifact_key, new_version, new_id, job_id,
        )
        return new_id

    # ─────────────────────────────────────────────────────────────────────────
    # 3. get_latest_artifact
    # ─────────────────────────────────────────────────────────────────────────

    async def get_latest_artifact(
        self,
        job_id: UUID,
        artifact_key: str,
    ) -> Dict[str, Any]:
        """
        Return the content dict of the latest artifact for (job_id, artifact_key).

        Raises:
            ArtifactNotFoundError if no matching artifact exists.
        """
        try:
            pool = await self._pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT content
                    FROM agent_artifacts
                    WHERE job_id = $1
                      AND artifact_key = $2
                      AND is_latest = TRUE
                    LIMIT 1
                    """,
                    job_id, artifact_key,
                )
        except asyncpg.PostgresError as exc:
            raise LedgerDatabaseError(f"get_latest_artifact failed: {exc}") from exc

        if row is None:
            raise ArtifactNotFoundError(
                f"No artifact found for job_id={job_id} key='{artifact_key}'"
            )
        return json.loads(row["content"])

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Lifecycle helpers
    # ─────────────────────────────────────────────────────────────────────────

    async def update_run_node(self, run_id: UUID, node_name: str) -> None:
        """
        Record which graph node is currently executing.
        Call this at the start of each LangGraph node for fine-grained progress tracking.
        """
        try:
            pool = await self._pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE agent_runs
                    SET current_node = $1, updated_at = $2
                    WHERE id = $3
                    """,
                    node_name, self._now_utc(), run_id,
                )
        except Exception as exc:
            # Non-fatal — a failed progress update must never crash a node
            logger.warning("update_run_node failed (non-fatal): %s", exc)

    async def complete_run(self, run_id: UUID) -> None:
        """Mark the run as succeeded and record completion time."""
        try:
            pool = await self._pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE agent_runs
                    SET run_status = 'succeeded',
                        completed_at = $1,
                        updated_at   = $1
                    WHERE id = $2
                    """,
                    self._now_utc(), run_id,
                )
        except asyncpg.PostgresError as exc:
            raise LedgerDatabaseError(f"complete_run failed: {exc}") from exc

    async def mark_run_failed(self, run_id: UUID, error: Exception) -> None:
        """
        Record an error on the run row.

        Captures the exception type, message, and truncated traceback.
        Safe to call from an except block — will not raise.
        """
        tb = traceback.format_exception(type(error), error, error.__traceback__)
        error_message = f"{type(error).__name__}: {error}\n{''.join(tb)}"[:4000]
        try:
            pool = await self._pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE agent_runs
                    SET run_status    = 'failed',
                        error_message = $1,
                        completed_at  = $2,
                        updated_at    = $2
                    WHERE id = $3
                    """,
                    error_message, self._now_utc(), run_id,
                )
            logger.error("Run %s marked failed: %s", run_id, error)
        except Exception as exc:
            logger.critical("mark_run_failed itself failed: %s", exc)

    async def update_run_state_summary(
        self,
        run_id: UUID,
        summary: Dict[str, Any],
    ) -> None:
        """
        Write a lightweight summary of the current graph state to the run row.

        Useful for quick observability without deserialising the full LangGraph
        checkpoint.  Example: {"completed_nodes": ["planner", "source_profiler"],
                                "n_questions": 3, "revision_count": 1}
        """
        try:
            pool = await self._pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE agent_runs
                    SET state_summary = $1, updated_at = $2
                    WHERE id = $3
                    """,
                    json.dumps(summary), self._now_utc(), run_id,
                )
        except Exception as exc:
            logger.warning("update_run_state_summary failed (non-fatal): %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Crash-recovery helpers
    # ─────────────────────────────────────────────────────────────────────────

    async def _query_latest_checkpoint(self, thread_id: str) -> CheckpointInfo:
        """
        Query the LangGraph checkpoint tables written by AsyncPostgresSaver.

        Table created by langgraph-checkpoint-postgres:
            checkpoints(thread_id, checkpoint_ns, checkpoint_id, ...)

        checkpoint_id is a UUID v1 (timestamp-based), so DESC ordering gives the
        most recent checkpoint.  Returns CheckpointInfo(exists=False) if the table
        does not exist or no checkpoint is found.
        """
        try:
            pool = await self._pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT checkpoint_id, checkpoint_ns
                    FROM checkpoints
                    WHERE thread_id = $1
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                    """,
                    thread_id,
                )
        except asyncpg.UndefinedTableError:
            # langgraph-checkpoint-postgres not installed or tables not yet created
            return CheckpointInfo(thread_id=thread_id, checkpoint_id=None, exists=False)
        except asyncpg.PostgresError as exc:
            logger.warning("_query_latest_checkpoint failed: %s", exc)
            return CheckpointInfo(thread_id=thread_id, checkpoint_id=None, exists=False)

        if row is None:
            return CheckpointInfo(thread_id=thread_id, checkpoint_id=None, exists=False)

        return CheckpointInfo(
            thread_id=thread_id,
            checkpoint_id=row["checkpoint_id"],
            checkpoint_ns=row["checkpoint_ns"] or "",
            exists=True,
        )

    async def get_all_latest_artifacts(
        self,
        job_id: UUID,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return {artifact_key: content} for every is_latest artifact in the job.
        Used by get_resumption_context to hand all intermediate outputs to the caller.
        """
        try:
            pool = await self._pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT artifact_key, content
                    FROM agent_artifacts
                    WHERE job_id = $1
                      AND is_latest = TRUE
                    ORDER BY created_at ASC
                    """,
                    job_id,
                )
        except asyncpg.PostgresError as exc:
            raise LedgerDatabaseError(f"get_all_latest_artifacts failed: {exc}") from exc

        return {
            row["artifact_key"]: json.loads(row["content"])
            for row in rows
            if row["artifact_key"]
        }

    async def get_resumption_context(self, job_id: UUID) -> ResumptionContext:
        """
        Build a full crash-recovery payload for a job.

        Steps:
          1. Find the most recent run for this job (highest attempt number).
          2. Retrieve its langgraph_thread_id.
          3. Check whether a LangGraph checkpoint exists for that thread.
          4. Fetch all is_latest artifacts so the caller can inspect completed stages.

        Returns:
            ResumptionContext with run metadata, checkpoint info, and artifacts.

        Raises:
            RunNotFoundError if no run exists for this job.
        """
        try:
            pool = await self._pool()
            async with pool.acquire() as conn:
                run_row = await conn.fetchrow(
                    """
                    SELECT id, job_id, langgraph_thread_id, attempt,
                           graph_name, current_node, created_at
                    FROM agent_runs
                    WHERE job_id = $1
                    ORDER BY attempt DESC
                    LIMIT 1
                    """,
                    job_id,
                )
        except asyncpg.PostgresError as exc:
            raise LedgerDatabaseError(f"get_resumption_context failed: {exc}") from exc

        if run_row is None:
            raise RunNotFoundError(f"No runs found for job_id={job_id}")

        run = RunMetadata(
            run_id=run_row["id"],
            job_id=run_row["job_id"],
            langgraph_thread_id=run_row["langgraph_thread_id"],
            attempt=run_row["attempt"],
            graph_name=run_row["graph_name"],
            created_at=run_row["created_at"],
        )

        checkpoint = await self._query_latest_checkpoint(run.langgraph_thread_id)
        artifacts  = await self.get_all_latest_artifacts(job_id)

        return ResumptionContext(
            run=run,
            checkpoint=checkpoint,
            latest_artifacts=artifacts,
            last_node=run_row["current_node"],
        )

    async def create_retry_run(
        self,
        job_id: UUID,
        graph_name: str,
        graph_version: Optional[str] = None,
        max_retries: int = 2,
    ) -> RunMetadata:
        """
        Create a new run for a retry attempt.

        Call this after marking the previous run as failed.  The new run gets a
        fresh langgraph_thread_id — LangGraph will start a new checkpoint sequence
        while all previous artifacts remain queryable for recovery logic.

        Returns:
            RunMetadata for the new run.
        """
        return await self.initialize_run(
            job_id=job_id,
            graph_name=graph_name,
            graph_version=graph_version,
            max_retries=max_retries,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Convenience: bulk artifact access by type
    # ─────────────────────────────────────────────────────────────────────────

    async def get_artifacts_by_type(
        self,
        job_id: UUID,
        artifact_type: str,
    ) -> List[ArtifactRecord]:
        """
        Return all is_latest artifacts of a given type for a job.
        Useful for recovering all source_profiles or question_blueprints at once.
        """
        try:
            pool = await self._pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, job_id, node_name, artifact_type, artifact_key,
                           content, version, is_latest, worker_class, model_id,
                           parent_artifact_id, input_artifact_ids, source_ids, created_at
                    FROM agent_artifacts
                    WHERE job_id = $1
                      AND artifact_type = $2
                      AND is_latest = TRUE
                    ORDER BY created_at ASC
                    """,
                    job_id, artifact_type,
                )
        except asyncpg.PostgresError as exc:
            raise LedgerDatabaseError(f"get_artifacts_by_type failed: {exc}") from exc

        return [
            ArtifactRecord(
                id=row["id"],
                job_id=row["job_id"],
                node_name=row["node_name"],
                artifact_type=row["artifact_type"],
                artifact_key=row["artifact_key"],
                content=json.loads(row["content"]),
                version=row["version"],
                is_latest=row["is_latest"],
                worker_class=row["worker_class"],
                model_id=row["model_id"],
                parent_artifact_id=row["parent_artifact_id"],
                input_artifact_ids=row["input_artifact_ids"] or [],
                source_ids=row["source_ids"] or [],
                created_at=row["created_at"],
            )
            for row in rows
        ]
