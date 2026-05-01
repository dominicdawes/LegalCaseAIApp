# agents/tools/verification.py
"""
Verification tools — fact-checking and citation grounding.
Use in the Grounder / Verifier node to validate claims made in drafts.
"""

import asyncio
import json
import logging
import os
from typing import Optional

from langchain_core.tools import tool

from .base import ToolContext

logger = logging.getLogger(__name__)


def build_verification_tools(ctx: ToolContext) -> list:

    @tool
    async def verify_claim(claim: str, k: int = 10) -> str:
        """
        Verify whether a legal claim or statement is supported by the documents.

        Use in the Grounder node to check draft assertions before finalising
        an answer.  Runs hybrid search for the claim text and asks Haiku to
        assess support/contradiction/insufficient-evidence.

        Prefer find_supporting_evidence when you just need passages without
        a verdict.

        Args:
            claim: A specific factual or legal statement to verify
            k: Number of supporting passages to retrieve (5-20, default 10)

        Returns JSON: {verdict: "supported"|"contradicted"|"insufficient",
                       confidence: 0-1, supporting_chunks: [...], reasoning: str}
        """
        from utils.retrieval import vector_search, bm25_search, rrf_fuse, SearchFilters
        import httpx

        k = max(5, min(k, 20))

        loop = asyncio.get_event_loop()
        if ctx.use_voyage:
            from utils.llm_clients.voyage_client import VoyageEmbeddingsClient
            client = VoyageEmbeddingsClient()
            embedding = await loop.run_in_executor(None, client.embed_query, claim)
        else:
            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"},
                    json={"model": "text-embedding-ada-002", "input": claim},
                )
                resp.raise_for_status()
                embedding = resp.json()["data"][0]["embedding"]

        filters: SearchFilters = {}
        if ctx.source_ids:
            filters["source_ids"] = ctx.source_ids

        vec_task = vector_search(embedding, ctx.project_id, k=k, filters=filters)
        bm25_task = bm25_search(claim, ctx.project_id, k=k, filters=filters)
        vec_res, bm25_res = await asyncio.gather(vec_task, bm25_task)
        chunks = rrf_fuse(vec_res, bm25_res, top_n=k)

        if not chunks:
            return json.dumps({
                "verdict": "insufficient",
                "confidence": 0.0,
                "supporting_chunks": [],
                "reasoning": "No relevant passages found.",
            })

        context = "\n\n---\n\n".join(
            f"[chunk {c['id']}]\n{c['content']}" for c in chunks[:8]
        )
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not anthropic_key:
            return json.dumps({"error": "ANTHROPIC_API_KEY not set"})

        prompt = (
            "You are a legal fact-checker.  Given the passages below and the claim, "
            "return a JSON object with:\n"
            '  "verdict": "supported" | "contradicted" | "insufficient"\n'
            '  "confidence": float 0-1\n'
            '  "reasoning": one sentence\n\n'
            f"Claim: {claim}\n\nPassages:\n{context}\n\n"
            "Return only valid JSON, no extra text."
        )

        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 256,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            raw = resp.json()["content"][0]["text"].strip()

        try:
            verdict_obj = json.loads(raw)
        except Exception:
            verdict_obj = {"verdict": "insufficient", "confidence": 0.0, "reasoning": raw}

        verdict_obj["supporting_chunks"] = [
            {"id": c["id"], "content": c["content"][:300]} for c in chunks[:5]
        ]
        return json.dumps(verdict_obj)

    @tool
    async def find_supporting_evidence(claim: str, k: int = 8) -> str:
        """
        Find passages that support a specific legal claim without verdict analysis.

        Use when building citations for a draft answer or when you need raw
        evidence passages to include in a prompt.  Returns passages ranked
        by relevance; does not perform AI verdict analysis (use verify_claim
        for that).

        Args:
            claim: Legal statement or assertion to find evidence for
            k: Max passages (5-20, default 8)

        Returns JSON array: [{id, content, source_id, page_number, similarity}]
        """
        from utils.retrieval import vector_search, SearchFilters
        import httpx

        k = max(5, min(k, 20))
        loop = asyncio.get_event_loop()

        if ctx.use_voyage:
            from utils.llm_clients.voyage_client import VoyageEmbeddingsClient
            client = VoyageEmbeddingsClient()
            embedding = await loop.run_in_executor(None, client.embed_query, claim)
        else:
            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"},
                    json={"model": "text-embedding-ada-002", "input": claim},
                )
                resp.raise_for_status()
                embedding = resp.json()["data"][0]["embedding"]

        filters: SearchFilters = {}
        if ctx.source_ids:
            filters["source_ids"] = ctx.source_ids

        chunks = await vector_search(embedding, ctx.project_id, k=k, filters=filters)
        return json.dumps([
            {
                "id": c["id"],
                "source_id": str(c["source_id"]),
                "page_number": c.get("page_number"),
                "similarity": round(c.get("similarity", 0.0), 4),
                "content": c["content"][:500],
            }
            for c in chunks
        ])

    @tool
    async def get_citations_for(chunk_ids: list) -> str:
        """
        Build formatted citation strings for a list of chunk IDs.

        Use when assembling the final answer to provide inline citations.
        Looks up source filename, page number, and section path for each
        chunk and formats them as citation references.

        Args:
            chunk_ids: List of chunk UUIDs to cite

        Returns JSON array: [{chunk_id, citation_string, source_id,
                              filename, page_number, section_path}]
        """
        from tasks.database import get_global_async_db_pool, init_async_pools

        if not chunk_ids:
            return json.dumps([])

        await init_async_pools()
        pool = get_global_async_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT dvs.id, dvs.source_id, dvs.page_number, dvs.section_path,
                       ds.source_filename
                FROM document_vector_store dvs
                JOIN document_sources ds ON ds.id = dvs.source_id
                WHERE dvs.id = ANY($1::uuid[])
                  AND dvs.project_id = $2
                """,
                chunk_ids,
                ctx.project_id,
            )

        results = []
        for r in rows:
            fname = r["source_filename"] or "Unknown"
            page = r["page_number"]
            sec = r["section_path"] or ""
            parts = [fname]
            if sec:
                parts.append(f"§ {sec}")
            if page:
                parts.append(f"p. {page}")
            results.append({
                "chunk_id": str(r["id"]),
                "source_id": str(r["source_id"]),
                "filename": fname,
                "page_number": page,
                "section_path": sec,
                "citation_string": ", ".join(parts),
            })
        return json.dumps(results)

    return [verify_claim, find_supporting_evidence, get_citations_for]
