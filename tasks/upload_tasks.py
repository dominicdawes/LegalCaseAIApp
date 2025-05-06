"""
This file runs Celery tasks, place any logic that needs to run async 
like api calls and calls to other services. 
"""

from celery import Celery, chain
import logging
import os
import json
import psutil
import requests
import tempfile
from tasks.celery_app import celery_app  # Import the Celery app instance (see celery_app.py for LocalHost config)
from utils.audio_utils import generate_audio, generate_only_dialogue_text
from utils.s3_utils import upload_to_s3, generate_presigned_url, s3_client, s3_bucket_name
from utils.supabase_utils import insert_document_supabase_record, insert_mp3_supabase_record, insert_vector_supabase_record, supabase_client
from utils.cloudfront_utils import get_cloudfront_url
from utils.instruction_templates import INSTRUCTION_TEMPLATES
from time import sleep
from datetime import datetime, timezone
import uuid

# langchain dependencies
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

# === PRODUCTION CELERY TASKS === #

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def process_pdf_task(self, files, metadata=None):
    """
    Main Celery task to:
        1) upload PDFs to S3, 
        2) save to Supabase, 
        3) and trigger vector embedding tasks.

    Args:
        files (List): Containing file CDN urls (created by WeWeb's document upload element)
        metadata (json): {
            'user_id': UUID,
            'project_id': UUID,
            'model_type': 'gpt-4o', 'llama3.1'...,
            'note_type': 'case_summary'
        }

    Returns: 
        source_ids (List): list of source ids (for later use in the celery task chord)
        @TODO project_id: Associated project that is used
    """

    # === RENDER RESOURCE LOGGING (MB) === #
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    logger.info(f"Starting {self.name} with files: {files} and metadata: {metadata}")
    logger.info(f"Memory usage before task: {mem_before / (1024 * 1024)} MB")

    if not files:
        return {"error": "Please upload at least one PDF file before processing."}
    
    rows_to_insert = []
    temp_paths = []  # keep track of temps for cleanup

    # === 1) Download & upload each file ===
    for file in files:
        try:
            # download or read locally
            if file.lower().startswith(("http://", "https://")):
                resp = requests.get(file)
                resp.raise_for_status()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(resp.content)
                tmp.close()
                file_path = tmp.name
            else:
                file_path = file

            # upload to S3
            ext = os.path.splitext(file_path)[1]
            key = f"{uuid.uuid4()}{ext}"
            upload_to_s3(s3_client, file_path, key)

            # get CloudFront URL
            url = get_cloudfront_url(key)

            rows_to_insert.append({
                "cdn_url": url,
                "project_id": metadata["project_id"],
                "content_tags": ["Fitness", "Health", "Wearables"],
                "uploaded_by": metadata["user_id"],
            })
            temp_paths.append(file_path)

            logger.info(f"Prepared upload row for {file_path} → {url}")

        except Exception as e:
            logger.error(f"[file loop] Failed processing '{file}': {e}", exc_info=True)
            # skip this file and continue
            continue

    if not rows_to_insert:
        logger.error("No successful uploads; aborting task.")
        raise RuntimeError("All file uploads failed.")

    # === 2) Bulk insert into Supabase ===
    try:
        resp = supabase_client.table("document_sources") \
                              .insert(rows_to_insert) \
                              .execute()
        if resp.error:
            raise RuntimeError(resp.error)

        # Loop thorugh reoponse to get, source_id list for the chaining/vector_embed tasks
        source_ids = [r["id"] for r in resp.data]
        logger.info(f"Bulk insert succeeded, got source_ids={source_ids}")

    except Exception as e:
        logger.error(f"[Supabase] Bulk insert failed: {e}", exc_info=True)
        # cleanup temp files before re-raising
        for p in temp_paths:
            try: os.unlink(p)
            except: pass
        raise

    # === 3) Cleanup temp files ===
    for path in temp_paths:
        try:
            os.unlink(path)
            logger.debug(f"Deleted temp file {path}")
        except Exception:
            logger.warning(f"Could not delete temp file {path}", exc_info=True)

    # === 4) Kick off embedding tasks ===
    for sid, row in zip(source_ids, rows_to_insert):
        try:
            chunk_and_embed_task.delay(row["cdn_url"], sid)
            logger.info(f"Enqueued embed task for source_id={sid}")
        except Exception as e:
            logger.error(f"[Celery] Failed to enqueue embed for {sid}: {e}", exc_info=True)

    logger.info(f"{self.name} completed successfully.")
    return source_ids


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def process_pdf_task_OLD(self, files, metadata=None):
    """
    Main Celery task to:
        1) upload PDFs to S3, 
        2) save to Supabase, 
        3) and trigger vector embedding tasks.

    Args:
        files (List): Containing file CDN urls (created by WeWeb's document upload element)
        metadata (json): {
            'user_id': UUID,
            'project_id': UUID,
            'model_type': 'gpt-4o', 'llama3.1'...,
            'note_type': 'case_summary'
        }

    Returns: 
        source_ids (List): list of source ids (for later use in the celery task chord)
        @TODO project_id: Associated project that is used
    """

    # === RENDER RESOURCE LOGGING (MB) === #
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    logger.info(f"Starting {self.name} with files: {files} and metadata: {metadata}")
    logger.info(f"Memory usage before task: {mem_before / (1024 * 1024)} MB")

    if not files:
        return {"error": "Please upload at least one PDF file before processing."}

    uploaded_documents = []
    source_ids = [] # used in chained task

    
    rows_to_insert = []
    temp_paths = []  # keep track of temps for cleanup

    # 1) Upload to S3 + build CloudFront URLs, but _don’t_ yet insert into Supabase
    for file in files:
        try:
            # Handle URL or local file path
            if file.startswith('http://') or file.startswith('https://'):
                response = requests.get(file)
                response.raise_for_status()
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                temp_file.write(response.content)
                temp_file.close()
                file_path = temp_file.name
            else:
                file_path = file

            # Generate S3 key and upload file
            ext_ending = os.path.splitext(file_path)[1]
            s3_document_key = f"{uuid.uuid4()}{ext_ending}"
            upload_to_s3(s3_client, file_path, s3_document_key)
            
            # Generate CloudFront URL 
            cloudfront_document_url = get_cloudfront_url(s3_document_key)

            rows_to_insert.append({
                "cdn_url": cloudfront_document_url,
                "project_id": metadata["project_id"],
                "content_tags": ["Fitness", "Health", "Wearables"],
                "uploaded_by": metadata["user_id"],
            })
            # (Optionally track temp paths for later cleanup)
            temp_paths.append(file_path)

        # Insert the document source record into Supabase (returns document_sources.id)
            source_id = insert_document_supabase_record(
                client=supabase_client,
                table_name="document_sources",
                cdn_url=cloudfront_document_url,
                project_id=metadata.get("project_id", ""),
                content_tags=["Fitness", "Health", "Wearables"],   # @TODO: Lets make this AI taggedeventually
                uploaded_by=metadata.get("user_id", "")
            )

            # Append to source_ids list
            source_ids.append(source_id)

            uploaded_documents.append({
                "source_id": source_id,
                "pdf_url": cloudfront_document_url,
                "file_path": file_path
            })

        except Exception as e:
            logger.error(f"Failed to upload and record file {file}: {e}", exc_info=True)
        finally:
            # Clean up temporary files if necessary
            if file.startswith('http'):
                os.unlink(file_path)

    # Trigger the embedding task for each document (chained Celery task)
    for doc in uploaded_documents:
        chunk_and_embed_task.delay(doc["pdf_url"], doc["source_id"])      # enqueue to Celery task queue
    
    # Logging message
    logger.info(" 'message' : PDF upload and record creation completed. Embedding tasks started.")

    return source_ids 

# @celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
# def insert_project_documents_task(self, task_results):
#     """
#     Celery task to insert a row in the join TABLE `project_documents`. Project ⇆ Documents join associates each 
#     project with N number of document_sources.

#     Callback task to accept the list of results from the group tasks. The results will be in the 
#     order the tasks were defined in the group.
    
#     Args:
#         task_results: (list) containing results from process_pdf_task and validate_and_generate_audio_task

#     Returns: 
#         None
#     """
#     # Unpack the results
#     source_ids = task_results[0]
#     media_result = task_results[1]
#     project_id = media_result.get('media_id')

#     try:
#         for source_id in source_ids:
#             supabase_client.table('project_documents').insert({
#                 'source_id': source_id,
#                 'media_id': project_id
#             }).execute()
#         logger.info(f"Successfully linked source_ids {source_ids} with media_id {project_id}")
#     except Exception as e:
#         logger.error(f"Failed to insert into TABLE `project_documents`: {e}")
#         raise

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def insert_sources_media_association_task(self, task_results):
    """
    Celery task to insert a row in TABLE `sources_media_association` table. Callback task to accept 
    the list of results from the group tasks. The results will be in the order the tasks were defined in the group.
    
    - task_results: (list) containing results from process_pdf_task and validate_and_generate_audio_task
    """
    # Unpack the results
    source_ids = task_results[0]
    media_result = task_results[1]
    media_id = media_result.get('media_id')

    try:
        for source_id in source_ids:
            supabase_client.table('sources_media_association').insert({
                'source_id': source_id,
                'media_id': media_id
            }).execute()
        logger.info(f"Successfully linked source_ids {source_ids} with media_id {media_id}")
    except Exception as e:
        logger.error(f"Failed to insert into sources_media_association: {e}")
        raise

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def chunk_and_embed_task_DEPR(self, pdf_url, source_id, chunk_size=1000, chunk_overlap=100):
    """
    Celery task to download, chunk, embed, and store embeddings of a PDF in Supabase. This step involved reading 
    the PDF contents with PyPDFLoader, but will need to be expanded to handle (.doc, .txt, .pdf (OCR), and .epub)
    """
    temp_file = None
    try:
        # 1) Download PDF
        response = requests.get(pdf_url)
        response.raise_for_status()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(response.content)
        temp_file.close()

        # 2) Load & split into chunks
        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)

        # 3) Prepare texts and metadata for batch embedding
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                "source": pdf_url,
                "page_number": chunk.metadata.get("page", None)
            }
            for chunk in chunks
        ]

        # 4) Batch embed all chunks in one call
        embedding_model = OpenAIEmbeddings()
        embeddings = embedding_model.embed_documents(texts)

        # 5) Build rows for bulk insert
        vector_rows = []
        for text, meta, vector in zip(texts, metadatas, embeddings):
            vector_rows.append({
                "source_id": source_id,
                "content": text,
                "metadata": meta,
                "embedding": vector
            })

        # 6) Bulk insert into Supabase
        resp = supabase_client.table("document_vector_store") \
                             .insert(vector_rows) \
                             .execute()
        if resp.error:
            logger.error(f"Supabase bulk vector insert failed: {resp.error}")
            raise Exception(resp.error)

        logger.info(f"Inserted {len(vector_rows)} embeddings for source_id={source_id}")

    except Exception as e:
        logger.error(f"Failed chunk/embed for {pdf_url}: {e}", exc_info=True)
        raise
    finally:
        # 7) Cleanup temp file
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def chunk_and_embed_task(self, pdf_url, source_id, chunk_size=1000, chunk_overlap=100):
    """
    Celery task to download, chunk, embed, and store embeddings of a PDF in Supabase. This step involved reading 
    the PDF contents with PyPDFLoader, but will need to be expanded to handle (.doc, .txt, .pdf (OCR), and .epub)
    """
    temp_file = None
    try:
        # 1) Download PDF
        response = requests.get(pdf_url)
        response.raise_for_status()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(response.content)
        temp_file.close()

        # 2) Load & split into chunks
        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)

        # 3) Prepare texts and metadata for batch embedding
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                "source": pdf_url,
                "page_number": chunk.metadata.get("page", None)
            }
            for chunk in chunks
        ]

        # 4) Batch embed all chunks in one call
        embedding_model = OpenAIEmbeddings()
        embeddings = embedding_model.embed_documents(texts)

        # 5) Build rows for bulk insert
        vector_rows = []
        for text, meta, vector in zip(texts, metadatas, embeddings):
            vector_rows.append({
                "source_id": source_id,
                "content": text,
                "metadata": meta,
                "embedding": vector
            })

        # 6) Bulk insert into Supabase
        resp = supabase_client.table("document_vector_store") \
                             .insert(vector_rows) \
                             .execute()
        if resp.error:
            logger.error(f"Supabase bulk vector insert failed: {resp.error}")
            raise Exception(resp.error)

        logger.info(f"Inserted {len(vector_rows)} embeddings for source_id={source_id}")

    except Exception as e:
        logger.error(f"Failed chunk/embed for {pdf_url}: {e}", exc_info=True)
        raise
    finally:
        # 7) Cleanup temp file
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass
    
    # return {"source_id": source_id, "chunks": len(texts)}
    return None