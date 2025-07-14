import os
import uuid
import asyncio
import tempfile
import requests
from datetime import datetime, timezone
import logging

# Celery imports
from collections import deque
from celery import chord, group
from celery.exceptions import MaxRetriesExceededError

# Langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import tiktoken

# Project modules
from tasks.celery_app import celery_app
from utils.s3_utils import upload_to_s3, s3_client
from utils.cloudfront_utils import get_cloudfront_url
from utils.supabase_utils import supabase_client
from utils.document_loaders.loader_factory import get_loader_for    # → module contains document streaming ability

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ——— Configuration —————————————————————————————————————————————————————————

# Retry policy for embedding tasks
MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 10  # seconds

# How many text chunks to send in one batch to OpenAI
BATCH_SIZE = 20  # break into smaller per-batch tasks

# Semantic chunker settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ada-002 embeddings are 1,536-dimensional
EXPECTED_EMBEDDING_LEN = 1536

# Instantiate once per worker process
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=OPENAI_API_KEY,
)

# Tokenizer for token counting
try:
    tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
except KeyError:
    tokenizer = tiktoken.get_encoding("cl100k_base")

# ——— Helper: download any URL to a temp file ——————————————————————————————————————

def _clean(s: str) -> str:
    """Remove problematic nulls and control chars from text."""
    return s.replace("\x00", "")

def download_stream_to_temp(url: str, chunk_size: int = 1024*1024) -> str:
    """
    Stream a remote URL (HTTP or CloudFront) into a NamedTemporaryFile.
    Returns the local filepath.
    """
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    suffix = os.path.splitext(url)[1] or ""
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with open(tf.name, "wb") as f:
        for chunk in resp.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
    return tf.name

# ——— Task 1: Ingestion → dedupe, S3 + Supabase + schedule parsing —————————————————————

@celery_app.task(bind=True)
def process_document_task(self, file_urls: list[str], metadata: dict):
    """
    1) Download each file_url → upload_to_s3 → get_cloudfront_url
    2) Bulk-insert rows into public.document_sources with INITIALIZING → PENDING
    3) Kick off chunk_and_embed_task for each document via a Celery chord
    """
    rows_to_insert = []
    tmp_paths = {}

    for file_url in file_urls:
        # ——— Step 1a:  download locally —————————————————————
        tmp_path = download_stream_to_temp(file_url)
        tmp_paths[file_url] = tmp_path

        # file metadata
        ext = os.path.splitext(file_url)[1].lower()
        key = f"{metadata['project_id']}/{uuid.uuid4()}{ext}"

        # ——— Step 1b: upload to S3 & get CloudFront URL
        upload_to_s3(s3_client, tmp_path, key)
        cdn_url = get_cloudfront_url(key)

        # prepare Supabase row `public.document_sources`
        rows_to_insert.append({
            "cdn_url":          cdn_url,
            "project_id":       str(metadata["project_id"]),
            "content_tags":     metadata.get("content_tags", []),
            "uploaded_by":      str(metadata["user_id"]),
            "vector_embed_status": "INITIALIZING",
            "filename":         os.path.basename(file_url),
            "file_size_bytes":  os.path.getsize(tmp_path),
            "file_extension":   ext,
            "created_at":       datetime.now(timezone.utc).isoformat(),
        })

    # ——— Step 2a: insert & collect generated IDs
    insert_resp = supabase_client.table("document_sources") \
        .insert(rows_to_insert, returning="id") \
        .execute()
    source_ids = [r["id"] for r in insert_resp.data]

    # ——— Step 2b: update all to PENDING
    supabase_client.table("document_sources") \
        .update({ "vector_embed_status": "PENDING" }) \
        .in_("id", source_ids) \
        .execute()
    
    # ——— Step 3: Schedule parsing (fire off one Celery task per-document)
    for i, source_id in enumerate(source_ids):
        parse_document_task.apply_async(
            (
                source_id, 
                rows_to_insert[i]['cdn_url'], 
                rows_to_insert[i]['project_id']
            ),
            queue='parsing'
        )

    # clean up local temp files
    for p in tmp_paths.values():
        try:
            os.remove(p)
        except OSError:
            pass

    return { "submitted_ids": source_ids }


# ——— Task 2: for one document → load, chunk, batch-embed & insert ——————————————————

@celery_app.task(bind=True)
def parse_document_task(self, source_id: str, cdn_url: str, project_id: str):
    """
    Stream, chunk, then schedule embed_batch_task per chunk-batch
    
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
    local_path = download_stream_to_temp(cdn_url)
    loader = get_loader_for(local_path)
    pages = loader.stream_documents(local_path)

    # accumulate batches
    batch_texts, batch_metas = [], []
    headers = []
    for page in pages:
        for chunk in splitter.split_documents([page]):
            batch_texts.append(chunk.page_content)
            batch_metas.append(chunk.metadata)
            # for exact batches of BATCH_SIZE
            if len(batch_texts) >= BATCH_SIZE:
                headers.append(
                    embed_batch_task.s(
                        source_id, 
                        project_id, 
                        batch_texts, 
                        batch_metas
                    )
                )
                # reset for the next batch
                batch_texts, batch_metas = [], []
    # For a final partial batch (if the total number of chunks isn’t an exact multiple of BATCH_SIZE)
    if batch_texts:
        headers.append(
            # Per-batch Celery tasks
            embed_batch_task.s(
                source_id, 
                project_id, 
                batch_texts, 
                batch_metas
            )
        )

    # schedule batches as a chord → finalize per-document
    chord(headers)(
        finalize_embeddings.s([source_id])
    )

    # cleanup
    try: os.remove(local_path)
    except: pass

# @celery_app.task(
#     bind=True,
#     max_retries=MAX_RETRIES,
#     default_retry_delay=DEFAULT_RETRY_DELAY,
#     name="chunk_and_embed_task"
# )
# def chunk_and_embed_task(self, source_id: str, cdn_url: str, project_id: str):
#     """
#     DEPRECATED: Downloads the document, picks the right loader (PDF/DOCX/EPUB/etc),
#     streams pages → semantic chunks → batches → embed → insert into Supabase.
    
#     Per-document task:
#         - Download → local temp
#         - get_loader_for → stream_documents()
#         - split into semantic chunks
#         - batch up to BATCH_SIZE → embed & insert into Supabase

#     Retries on any error, with MAX_RETRIES.

#     Returns `source_id` on success.
#     """
#     local_path = None
#     try:
#         local_path = download_stream_to_temp(cdn_url)
#         # pick the correct loader (and OCR vs. text-based PDF)
#         loader = get_loader_for(local_path)
#         doc_iter = loader.stream_documents(local_path)

#         # LangChain splitter for semantic chunks
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP
#         )

#         batch_texts: list[str] = []

#         for page_doc in doc_iter:
#             # split each page-level Document into smaller chunks
#             chunk_docs = splitter.split_documents([page_doc])
#             for chunk in chunk_docs:
#                 batch_texts.append(chunk.page_content)

#                 # once we hit BATCH_SIZE, send off to OpenAI & DB
#                 if len(batch_texts) >= BATCH_SIZE:
#                     _embed_and_store(source_id, project_id, batch_texts)
#                     batch_texts.clear()

#         # embed any remaining chunks
#         if batch_texts:
#             _embed_and_store(source_id, project_id, batch_texts)

#         # return source_id so chord callback knows which docs succeeded
#         return source_id
    
#     except MaxRetriesExceededError as mre:
#         logger.critical(
#             f"[CELERY] chunk_and_embed_task for {source_id} "
#             f"failed after {MAX_RETRIES} retries: {mre}"
#         )
#         # mark document as failed in Supabase
#         supabase_client.table("document_sources") \
#             .update({"vector_embed_status": "FAILED_EMBEDDING"}) \
#             .eq("id", source_id).execute()
#         raise

#     except Exception as e:
#         logger.exception(f"[CELERY] error in chunk_and_embed_task for {source_id}, retrying...")
#         # will raise Retry exception
#         raise self.retry(exc=e)
    
#     finally:
#         # ensure we always clean up
#         try:
#             if local_path and os.path.exists(local_path):
#                 os.remove(local_path)
#         except OSError:
#             pass

# ——— Task 3: Embed a batch —————————————————————————————————————————————

# def _embed_and_store(source_id: str, project_id: str, chunks: list):
#     """
#     DEPRECATED: calls OpenAIEmbeddings asynchronously, checks dims,
#     then bulk-inserts into document_vector_store.
    
#     Steps:
#     1) call OpenAIEmbeddings via asyncio.run
#     2) clean text & metadata, count tokens
#     3) bulk-insert into document_vector_store
#     """
#     texts = [doc.page_content for doc in chunks]
#     metas = [doc.metadata for doc in chunks]

#     # 1) Embed via asyncio.run
#     embeddings = asyncio.run(embedding_model.embed_documents(texts))

#     # 2) validate & build rows
#     vector_rows = []
#     for txt, meta, vec in zip(texts, metas, embeddings):
#         if len(vec) != EXPECTED_EMBEDDING_LEN:
#             raise ValueError(
#                 f"Embedding len {len(vec)} != expected {EXPECTED_EMBEDDING_LEN}"
#             )
#         clean_txt = _clean(txt)
#         num_tokens = len(tokenizer.encode(clean_txt))

#         clean_meta = {
#             k: (_clean(v) if isinstance(v, str) else v)
#             for k, v in meta.items()
#         }

#         vector_rows.append({
#             "source_id":   str(source_id),
#             "content":     clean_txt,
#             "metadata":    clean_meta,
#             "embedding":   vec,
#             "project_id":  str(project_id),
#             "num_tokens":  num_tokens,
#         })

#     # 3) insert vectors
#     supabase_client.table("document_vector_store") \
#         .insert(vector_rows).execute()

@celery_app.task(
    bind=True,
    max_retries=MAX_RETRIES,
    default_retry_delay=DEFAULT_RETRY_DELAY
)
def embed_batch_task(self, source_id: str, project_id: str, texts: list[str], metas: list[dict]):
    """
    Celery task: Embed a small batch (20 chunks per task), insert vectors, update on error
    1) call OpenAIEmbeddings via asyncio.run
    2) clean text & metadata, count tokens
    3) bulk-insert into document_vector_store
    """
    try:
        # 1) Embed via OpenAI using asyncio.run
        embeddings = asyncio.run(embedding_model.embed_documents(texts))
        rows = []
        for txt, meta, vec in zip(texts, metas, embeddings):
            if len(vec) != EXPECTED_EMBEDDING_LEN:
                raise ValueError(f"Embedding len {len(vec)} != expected {EXPECTED_EMBEDDING_LEN}")
            
            # Clean rogue chars and blank spaces
            clean_txt = _clean(txt)
            num_tokens = len(tokenizer.encode(clean_txt))
            clean_meta = {
                k: (_clean(v) if isinstance(v, str) else v)
                for k, v in meta.items()
            }

            # Insert into document vector store
            rows.append({
                'source_id': source_id,
                'content': clean_txt,
                'metadata': clean_meta,
                'embedding': vec,
                'project_id': project_id,
                'num_tokens': num_tokens,
            })
        supabase_client.table('document_vector_store').insert(rows).execute()
        return source_id

    except MaxRetriesExceededError:
        # terminal failure: update doc status and error
        supabase_client.table('document_sources') \
            .update({
                'vector_embed_status': 'FAILED_EMBEDDING',
                'error_message': f"Batch failed after {MAX_RETRIES} retries"
            }).eq('id', source_id).execute()
        raise
    except Exception as e:
        logger.exception(f"Error in embed_batch_task for {source_id}, retrying...")
        raise self.retry(exc=e)

# ——— Task 4: callback after all documents’ embeddings finish —————————————————————
@celery_app.task(bind=True)
def finalize_embeddings(self, header_results: list[str], all_source_ids: list[str]):
    """
    Celery chord callback: header_results is a list of returned source_ids
    from chunk_and_embed_task. Mark all those docs in DB as COMPLETE.
    """
    for source_id in all_source_ids:
        successes = [r for r in header_results if r == source_id]
        if len(successes) == len([r for r in header_results if r == source_id]):
            supabase_client.table('document_sources') \
                .update({'vector_embed_status': 'COMPLETE'}) \
                .eq('id', source_id).execute()
        else:
            supabase_client.table('document_sources') \
                .update({
                    'vector_embed_status': 'PARTIAL',
                    'error_message': 'Some batches failed'
                }).eq('id', source_id).execute()
    return {'completed_ids': header_results}
