import os
import uuid
import tempfile
import requests
import logging
import asyncio

from datetime import datetime, timezone
from celery import chord, group
from celery.exceptions import MaxRetriesExceededError

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import httpx
import tiktoken

from tasks.celery_app import celery_app
from utils.s3_utils import upload_to_s3, s3_client
from utils.cloudfront_utils import get_cloudfront_url
from utils.supabase_utils import supabase_client
from utils.document_loaders.loader_factory import get_loader_for

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ——— Configuration —————————————————————————————————————————
INGEST_QUEUE = 'ingest'
PARSE_QUEUE  = 'parsing'
EMBED_QUEUE  = 'embedding'
FINAL_QUEUE  = 'finalize'

MAX_RETRIES           = 3
DEFAULT_RETRY_DELAY   = 10  # seconds
RATE_LIMIT            = '60/m'  # max embed_batch_task calls per minute

BATCH_SIZE            = 20
CHUNK_SIZE            = 1000
CHUNK_OVERLAP         = 200
EXPECTED_EMBEDDING_LEN= 1536

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Use async-capable embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=OPENAI_API_KEY,
)

# Async HTTP client for potential direct OpenAI calls
async_client = httpx.AsyncClient(timeout=60.0)

# Tokenizer for token counting
try:
    tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
except KeyError:
    tokenizer = tiktoken.get_encoding("cl100k_base")

# ——— Helpers ———————————————————————————————————————————————
def _clean(text: str) -> str:
    """Strip nulls and control characters"""
    return text.replace("\x00", "")


def download_stream_to_temp(url: str, chunk_size: int = 1024*1024) -> str:
    """
    Stream remote URL to a tempfile, return path
    """
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    suffix = os.path.splitext(url)[1] or ''
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with open(tf.name, 'wb') as f:
        for c in resp.iter_content(chunk_size):
            if c:
                f.write(c)
    return tf.name

# ——— Task 1: Ingestion → dedupe, S3 + Supabase + schedule parsing —————————————————
@celery_app.task(bind=True, queue=INGEST_QUEUE)
def process_document_task(self, file_urls: list[str], metadata: dict):
    """
    1. Download → S3 → CloudFront URL
    2. Deduplicate by cdn_url
    3. Bulk insert new rows (INITIALIZING→PENDING)
    4. Schedule parse_document_task per-document
    """
    new_rows, new_paths, existing_ids = [], {}, []

    # --- download & dedupe ---
    for url in file_urls:
        tmp = download_stream_to_temp(url)
        new_paths[url] = tmp

        ext = os.path.splitext(url)[1].lower()
        key = f"{metadata['project_id']}/{uuid.uuid4()}{ext}"
        upload_to_s3(s3_client, tmp, key)
        cdn_url = get_cloudfront_url(key)

        # check existing
        exists = supabase_client.table('document_sources') \
            .select('id') \
            .eq('cdn_url', cdn_url) \
            .execute().data
        if exists:
            existing_ids.append(exists[0]['id'])
            logger.info(f"Skipping duplicate cdn_url {cdn_url} → doc_id {exists[0]['id']}")
            continue

        new_rows.append({
            'cdn_url': cdn_url,
            'project_id': str(metadata['project_id']),
            'content_tags': metadata.get('content_tags', []),
            'uploaded_by': str(metadata['user_id']),
            'vector_embed_status': 'INITIALIZING',
            'filename': os.path.basename(url),
            'file_size_bytes': os.path.getsize(tmp),
            'file_extension': ext,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'total_batches': 0,
            'total_chunks': 0,
            'processed_batches': 0,
            'processed_chunks': 0,
            'last_batch_duration_ms': None,
            'error_message': None
        })

    # --- insert new docs ---
    inserted_ids = []
    if new_rows:
        resp = supabase_client.table('document_sources') \
            .insert(new_rows, returning='id').execute()
        new_ids = [r['id'] for r in resp.data]

        # mark new as pending
        supabase_client.table('document_sources') \
            .update({'vector_embed_status': 'PENDING'}) \
            .in_('id', new_ids).execute()
        inserted_ids = new_ids

    # combine all doc_ids
    all_doc_ids = existing_ids + inserted_ids
    # schedule parsing for new docs only
    for idx, doc_id in enumerate(inserted_ids):
        parse_document_task.apply_async(
            (doc_id, new_rows[idx]['cdn_url'], new_rows[idx]['project_id']),
            queue=PARSE_QUEUE
        )

    # cleanup temp files
    for p in new_paths.values():
        try: os.remove(p)
        except: pass

    return {'doc_ids': all_doc_ids}

# ——— Task 2: Parse → chunk → schedule embed_batch_task per-batch ———————————————————
@celery_app.task(bind=True, queue=PARSE_QUEUE)
def parse_document_task(self, source_id: str, cdn_url: str, project_id: str):
    """
    Stream, chunk, then schedule embed_batch_task per chunk-batch
    Also records total_batches & total_chunks for telemetry

    Step by step:
        - Download → local temp
        - get_loader_for (PDF/DOCX/EPUB/etc) → stream_documents()
        - split into semantic chunks
        - accumulate batches up to BATCH_SIZE → enque `embed_batch_task` to embed & insert into Supabase
    """
    # status update
    supabase_client.table('document_sources') \
        .update({'vector_embed_status': 'PARSING'}).eq('id', source_id).execute()

    # prepare splitter & loader
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    local = download_stream_to_temp(cdn_url)
    loader = get_loader_for(local)
    pages = loader.stream_documents(local)

    # accumulate batches
    batch_texts, batch_metas = [], []
    headers = []
    total_chunks, total_batches = 0, 0

    for page in pages:
        for chunk in splitter.split_documents([page]):
            batch_texts.append(chunk.page_content)
            batch_metas.append(chunk.metadata)
            # for exact batches of BATCH_SIZE
            if len(batch_texts) >= BATCH_SIZE:
                total_chunks += len(batch_texts)
                total_batches += 1
                headers.append(
                    # Signature for per-batch Celery task
                    embed_batch_task.s(source_id, project_id, batch_texts, batch_metas)
                )
                batch_texts, batch_metas = [], []

    # final partial batch
    if batch_texts:
        total_chunks += len(batch_texts)
        total_batches += 1
        headers.append(
            # Signature for per-batch Celery task
            embed_batch_task.s(source_id, project_id, batch_texts, batch_metas)
        )

    # UPDATE TELEMETRY: expected workload
    supabase_client.table('document_sources') \
        .update({
            'vector_embed_status': 'EMBEDDING',
            'total_chunks': total_chunks,
            'total_batches': total_batches
        }).eq('id', source_id).execute()

    # schedule embed batches in EMBED_QUEUE
    chord(headers, queue=EMBED_QUEUE)(
        finalize_embeddings.s([source_id])
    )

    try: os.remove(local)
    except: pass

# ——— Task 3: Embed batch —————————————————————————————————————————————
@celery_app.task(
    bind=True,
    queue=EMBED_QUEUE,
    max_retries=MAX_RETRIES,
    default_retry_delay=DEFAULT_RETRY_DELAY,
    rate_limit=RATE_LIMIT,
    acks_late=True
)
async def embed_batch_task(self, source_id: str, project_id: str, texts: list[str], metas: list[dict]):
    """
    Async Celery task: embed a batch (size=BATCH_SIZE), insert vectors, record metrics

    Step-by-step:
        1) call OpenAIEmbeddings via asyncio.run
        2) clean text & metadata, count tokens
        3) bulk-insert into document_vector_store
    """
    start = datetime.now(timezone.utc)
    try:
        # 1) call OpenAI asynchronously
        embeddings = await embedding_model.aembed_documents(texts)

        rows = []
        for txt, meta, vec in zip(texts, metas, embeddings):
            if len(vec) != EXPECTED_EMBEDDING_LEN:
                raise ValueError(
                    f"Embedding len {len(vec)} != expected {EXPECTED_EMBEDDING_LEN}"
                )
            
            # Clean rogue chars and blank spaces
            clean_txt = _clean(txt)
            num_tokens = len(tokenizer.encode(clean_txt))
            clean_meta = {
                k: (_clean(v) if isinstance(v, str) else v) for k, v in meta.items()
            }
            
            # prep row for document vector store
            rows.append({
                'source_id': source_id,
                'content': clean_txt,
                'metadata': clean_meta,
                'embedding': vec,
                'project_id': project_id,
                'num_tokens': num_tokens,
            })

        # 2) bulk insert vectors (num_rows=BATCH_SIZE)
        supabase_client.table('document_vector_store').insert(rows).execute()

        # 3) UPDATE TELEMETRY: record telemetry via RPC
        duration = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        supabase_client.rpc('increment_doc_metrics', {
            'doc_id': source_id,
            'batch_inc': 1,
            'chunk_inc': len(texts),
            'duration_ms': duration
        }).execute()

        return source_id

    except MaxRetriesExceededError:
        supabase_client.table('document_sources') \
            .update({
                'vector_embed_status': 'FAILED_EMBEDDING',
                'error_message': f"Batch failed after {MAX_RETRIES} retries"
            }).eq('id', source_id).execute()
        raise
    except Exception as exc:
        logger.exception(f"embed_batch_task error for {source_id}, retrying...")
        raise self.retry(exc=exc)

# ——— Task 4: Finalize —————————————————————————————————————————————————
@celery_app.task(bind=True, queue=FINAL_QUEUE)
def finalize_embeddings(self, header_results: list[str], doc_ids: list[str]):
    """
    Mark document COMPLETE or PARTIAL based on processed_batches vs total_batches
    """
    for doc_id in doc_ids:
        rec = supabase_client.table('document_sources') \
            .select('processed_batches,total_batches') \
            .eq('id', doc_id).single().execute().data or {}

        if rec.get('processed_batches') == rec.get('total_batches'):
            supabase_client.table('document_sources') \
                .update({'vector_embed_status': 'COMPLETE'}) \
                .eq('id', doc_id).execute()
        else:
            supabase_client.table('document_sources') \
                .update({
                    'vector_embed_status': 'PARTIAL',
                    'error_message': 'Some batches failed'
                }).eq('id', doc_id).execute()
    return {'completed_ids': header_results}
