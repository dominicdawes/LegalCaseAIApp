"""
This file runs Celery tasks, for handling RAG document upload, and vector embedding tasks
"""

from celery import Celery, chain, chord
from celery.exceptions import MaxRetriesExceededError
from celery import states
import logging
import os
import json
import psutil
import requests
import tempfile
from typing import List
from tasks.note_tasks import rag_note_task 
from tasks.celery_app import celery_app  # Import the Celery app instance (see celery_app.py for LocalHost config)
from utils.audio_utils import generate_audio, generate_only_dialogue_text
from utils.s3_utils import upload_to_s3, generate_presigned_url, s3_client, s3_bucket_name
from utils.supabase_utils import insert_document_supabase_record, insert_mp3_supabase_record, insert_vector_supabase_record, supabase_client
from utils.cloudfront_utils import get_cloudfront_url
from utils.instruction_templates import INSTRUCTION_TEMPLATES
from time import sleep
from datetime import datetime, timezone
from dotenv import load_dotenv
import uuid

# langchain dependencies
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAIError

# API Keys
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_PROD_KEY")

logger = logging.getLogger(__name__)

# === PRODUCTION CELERY TASKS === #
def update_db_poll_status(status: str, source_id: str, error_message: str = None):
    """Helper function to update the status in document_sources."""
    try:
        payload = {"vector_embed_status": status}
        if error_message:
            payload["error_message"] = error_message[:255]  # if you have that column
            logger.error(f"[DB] Setting status={status} for {source_id} w/ error: {error_message}")

        logger.debug(f"[DB] update document_sources set status={status} where id={source_id}")
        update_resp = (
            supabase_client
                .table("document_sources")
                .update(payload)
                .eq("id", source_id)
                .execute()
        )

        # **New**: inspect the Supabase response
        if hasattr(update_resp, "data"):
            logger.debug(f"[DB] Update returned data: {update_resp.data}")
        if hasattr(update_resp, "status_code"):
            logger.debug(f"[DB] HTTP status code: {update_resp.status_code}")

    except Exception as db_e:
        logger.critical(f"[DB] Failed to update status for {source_id}: {db_e}", exc_info=True)

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def process_pdf_task(self, files, metadata=None):
    """
    Main Celery task to:
        1) upload PDFs to S3, 
        2) save to Supabase, 
        3) and trigger vector embedding tasks.
    Includes enhanced status tracking.

    Args:
    files (List): Containing file CDN urls (created by WeWeb's document upload element)
    metadata (json): {
        'user_id': UUID,
        'project_id': UUID,
        'provider': 'anthropic...', 'meta'...,
        'model_name': 'gpt-4o-mini', 'llama3.1'...,
        'temparature': '0.7', ...,
        'note_type': 'case_summary'

    """
    try:
        # Initialize task-level state that will be accessible via results = AsyncResult(task_id)
        # result.state → "PENDING"
        # result.info → {"start_time": "..."}
        task_start_time = datetime.now(timezone.utc).isoformat()
        self.update_state(
            state='PROCESSING',
            meta={
                'start_time': task_start_time,
                'total_files': len(files),
                'processed_files': 0,
                'stage': 'INITIALIZING',
                'message': 'Starting document processing workflow'
            }
        )

        # === RENDER RESOURCE LOGGING (MB) === #
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        logger.info(f"Starting {self.name} with {len(files)} files and metadata: {metadata}")
        logger.info(f"Memory usage before task: {mem_before / (1024 * 1024)} MB")

        if not files:
            logger.warning("No files provided for processing.")
            raise ValueError("Please upload at least one PDF file before processing.")
        
        rows_to_insert = []
        temp_paths = []  # keep track of temps for cleanup
        successful_files = 0
        failed_files = 0

        # === 1) Download & upload each file ===
        self.update_state(
            state='PROCESSING',
            meta={
                'start_time': task_start_time,
                'stage': 'DOWNLOADING_AND_UPLOADING',
                'message': f'Downloading and uploading {len(files)} files to S3',
                'processed_files': 0,
                'total_files': len(files)
            }
        )

        for idx, file in enumerate(files):
            try:
                # download or read locally
                if file.lower().startswith(("http://", "https://")):
                    self.update_state(
                        state='PROCESSING',
                        meta={
                            'stage': 'DOWNLOADING',
                            'current_file': file.split('/')[-1],
                            'current_file_index': idx + 1,
                            'total_files': len(files),
                            'message': f'Downloading file {idx + 1}/{len(files)}'
                        }
                    )
                    resp = requests.get(file)
                    resp.raise_for_status()
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(resp.content)
                    tmp.close()
                    file_path = tmp.name
                else:
                    file_path = file

                # upload to S3
                self.update_state(
                    state='PROCESSING',
                    meta={
                        'start_time': task_start_time,
                        'stage': 'UPLOADING_TO_S3',
                        'current_file': file.split('/')[-1],
                        'current_file_index': idx + 1,
                        'total_files': len(files),
                        'message': f'Uploading file {idx + 1}/{len(files)} to S3'
                    }
                )
                ext = os.path.splitext(file_path)[1]
                key = f"{uuid.uuid4()}{ext}"
                upload_to_s3(s3_client, file_path, key)

                # get CloudFront URL
                url = get_cloudfront_url(key)

                rows_to_insert.append({
                    "cdn_url": url,
                    "project_id": str(metadata["project_id"]),
                    "content_tags": ["Fitness", "Health", "Wearables"],
                    "uploaded_by": str(metadata["user_id"]), 
                    "vector_embed_status": "INITIALIZING",  # New initial status
                    "filename": file.split('/')[-1],  # Store original filename
                    "file_size_bytes": os.path.getsize(file_path),  # Store file size for reference
                    "created_at": datetime.now(timezone.utc).isoformat()
                })
                temp_paths.append(file_path)
                successful_files += 1

                logger.info(f"Prepared upload row for {file_path} → {url}")

            except Exception as e:
                failed_files += 1
                logger.error(f"[file loop] Failed processing '{file}': {e}", exc_info=True)
                # Update the task state with error information
                self.update_state(
                    state='PROCESSING',
                    meta={
                        'start_time': task_start_time,
                        'stage': 'FILE_PROCESSING_ERROR',
                        'current_file': file.split('/')[-1] if '/' in file else file,
                        'error_message': str(e),
                        'successful_files': successful_files,
                        'failed_files': failed_files,
                        'message': f'Error processing file {idx + 1}/{len(files)}: {str(e)[:100]}...'
                    }
                )
                # skip this file and continue
                continue

        if not rows_to_insert:
            logger.error("No successful uploads; aborting task.")
            raise RuntimeError("All file uploads failed.")

        # === 2) Bulk insert into Supabase ===
        self.update_state(
            state='PROCESSING',
            meta={
                'start_time': task_start_time,
                'stage': 'DATABASE_INSERTION',
                'message': f'Inserting {len(rows_to_insert)} document records into database',
                'successful_files': successful_files,
                'failed_files': failed_files
            }
        )
        
        source_ids = []  # Initialize source_ids list
        try:
            # First, insert with INITIALIZING status
            response = supabase_client.table("document_sources") \
                                .insert(rows_to_insert) \
                                .execute()
            
            # Extract source_ids from the insert response
            source_ids = [r["id"] for r in response.data]
            
            # Update status to PENDING for embedding
            self.update_state(
                state='PROCESSING',
                meta={
                    'start_time': task_start_time,
                    'stage': 'UPDATING_STATUS',
                    'message': f'Setting {len(source_ids)} documents to PENDING status',
                    'successful_files': successful_files,
                    'failed_files': failed_files,
                    'source_ids': source_ids  # Include source_ids in state for reference
                }
            )
            
            # Update all document statuses to PENDING
            update_payload = [{"id": sid, "vector_embed_status": "PENDING"} for sid in source_ids]
            update_resp = supabase_client.table("document_sources") \
                                .upsert(update_payload) \
                                .execute()

        except Exception as e:
            logger.error(f"[Supabase] Bulk insert failed: {e}", exc_info=True)
            # Cleanup temp files before re-raising the exception for Celery retry
            for p in temp_paths:
                try: 
                    os.unlink(p)
                    logger.debug(f"Deleted temp file {p} after Supabase insert failure.")
                except: 
                    logger.warning(f"[CELERY] Could not delete temp file {p} during Supabase failure cleanup.", exc_info=True)
            
            # Update task state
            self.update_state(
                state='FAILURE',
                meta={
                    'exc_type':     type(e).__name__,
                    'exc_module':   e.__class__.__module__,
                    'exc_message':  str(e),
                    'error': f"Initial DB insert failed: {str(e)}",
                    'stage': 'DATABASE_ERROR',
                    'message': f'Database insertion failed: {str(e)[:100]}...'
                }
            )
            raise

        # === 3) Cleanup temp files ===
        self.update_state(
            state='PROCESSING',
            meta={
                'stage': 'CLEANUP',
                'message': 'Cleaning up temporary files',
                'successful_files': successful_files,
                'failed_files': failed_files,
                'source_ids': source_ids
            }
        )
        
        for path in temp_paths:
            try:
                os.unlink(path)
                logger.debug(f"Deleted temp file {path}")
            except Exception:
                logger.warning(f"[CELERY] Could not delete temp file {path}", exc_info=True)

        # === 4) Kick off the Chord (Embedding Tasks + Final Callback) ===
        self.update_state(
            state='PROCESSING',
            meta={
                'stage': 'INITIATING_EMBEDDING',
                'message': f'Starting embedding tasks for {len(source_ids)} documents',
                'successful_files': successful_files,
                'failed_files': failed_files,
                'source_ids': source_ids
            }
        )

        logger.info(f"{self.name} prepared {len(source_ids)} source_ids, launching embedding chord")

        # Create a list of task signatures for the chord header
        embedding_tasks = [
            chunk_and_embed_task.s(
                row["cdn_url"],                 # Pass the CDN URL
                sid,                            # Pass the source_id from the DB insert
                metadata["project_id"]          # Pass the project_id
            )
            for sid, row in zip(source_ids, rows_to_insert)
        ]

        # Define the chord body (the callback task)
        callback = finalize_document_processing_workflow.s(
            metadata["user_id"],
            source_ids,
            metadata["project_id"],
            metadata.get("note_type", "None"),          # default to "None" if missing
            metadata.get("provider", "openai"),
            metadata.get("model_name", "gpt-4o-mini"),   # default to gpt-4o-mini  
            metadata.get("temperature", "0.7"),
            metadata.get("addtl_params", {}),
        )
        chord_result = chord(embedding_tasks)(callback)
        workflow_id = chord_result.id
        logger.info(f"Chord launched; callback task ID = {workflow_id}")

        # Update final state with success and result information
        self.update_state(
            state='SUCCESS',
            meta={
                'stage': 'COMPLETED',
                'message': 'Document upload complete, embedding in progress...',
                'successful_files': successful_files,
                'failed_files': failed_files,
                'source_ids': source_ids
            }
        )

        # Return the ID of the FINAL callback so you can poll it:
        return {
            "source_ids":      source_ids,
            "successful":      successful_files,
            "failed":          failed_files,
            "total":           len(files),
            "workflow_task_id": workflow_id
        }
    
    except Exception as e:
        # this catches anything that bubbled out of your inner logic
        logger.error(f"Overall process_pdf_task failed: {e}", exc_info=True)

        try:
            # Update state before retry
            self.update_state(
                state='RETRY',
                meta={
                    'error': str(e),
                    'retry_count': self.request.retries,
                    'max_retries': self.max_retries,
                    'message': f'Task failed, attempting retry {self.request.retries + 1}/{self.max_retries + 1}'
                }
            )
            # schedule a retry if we haven't hit max_retries yet
            raise self.retry(exc=e)
        except MaxRetriesExceededError:
            # once retries are exhausted, raise a clear runtime error
            logger.critical(
                f"[CELERY] process_pdf_task failed permanently after {self.max_retries} retries: {e}"
            )
            # Update final failure state 
            self.update_state(
                state='FAILURE',
                meta={
                    'exc_type':     type(e).__name__,
                    'exc_module':   e.__class__.__module__,
                    'exc_message':  str(e),
                    'error': str(e),
                    'retry_count': self.max_retries,
                    'max_retries': self.max_retries,
                    'message': f'Task failed after {self.max_retries} retries'
                }
            )
            raise RuntimeError(
                f"[CELERY] process_pdf_task failed permanently after {self.max_retries} retries: {e}"
            ) from e

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def chunk_and_embed_task(self, pdf_url, source_id, project_id, chunk_size=1000, chunk_overlap=100):
    """
    Celery task to download, chunk, embed, and store embeddings of a PDF in Supabase. This step involved reading
    the PDF contents with PyPDFLoader, but will need to be expanded to handle (.doc, .txt, .pdf (OCR), and .epub)
    """
    temp_file = None

    try:
        logger.info(f"Starting chunk_and_embed_task for URL: {pdf_url}, source_id: {str(source_id)}")

        update_db_poll_status("EMBEDDING", source_id)

        # 1) Download PDF from URL using a temp file
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
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=OPENAI_API_KEY,
        )
        embeddings = embedding_model.embed_documents(texts)

        # Guard against any mis-shaped vector dimensions
        expected_embedding_length = 1536   # ada-002 produces 1536 dimensions
        for i, vec in enumerate(embeddings):
            if len(vec) != expected_embedding_length:
                raise ValueError(f"Embedding #{i} has length {len(vec)}; expected {expected_embedding_length}")

        # 5) Build rows for bulk insert
        def _clean(s: str) -> str:
            # remove literal NULLs and other problematic control characters if needed
            # For this specific error, '\x00' is the key.
            return s.replace('\x00', '')
        
        # 5.1) Build rows for bulk insert
        vector_rows = []
        for text, meta, vector in zip(texts, metadatas, embeddings):
            clean_text = _clean(text)
            
            # Clean metadata values that are strings
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, str):
                    clean_meta[k] = _clean(v)
                else:
                    clean_meta[k] = v

            vector_rows.append({
                "source_id":    str(source_id),
                "content":      clean_text,
                "metadata":     clean_meta,  # Use the cleaned metadata
                "embedding":    vector,
                "project_id":   str(project_id)
            })

        # 6) Bulk insert into Supabase
        logger.debug("Attempting Supabase bulk vector insert into public.document_vector_store.")
        response = supabase_client.table("document_vector_store") \
                            .insert(vector_rows) \
                            .execute()
        # Update status to COMPLETE after successful vector insert
        update_db_poll_status("COMPLETE", source_id)

        logger.info(f"Bulk inserted {len(vector_rows)} embeddings into public.document_vector_store for source_id={source_id}")

    except Exception as e:
        logger.error(f"Failed chunk/embed for {pdf_url}: {e}", exc_info=True)
        update_db_poll_status("FAILED", source_id, error_message=str(e))
        raise
    except OpenAIError as e:
        # Catch specific OpenAI errors for more targeted logging/handling if needed
        logger.error(f"OpenAI API error during embedding for {pdf_url}: {e}", exc_info=True)
        update_db_poll_status("FAILED", source_id, error_message=str(e))
        raise self.retry(exc=e)
    finally:
        # 7) Cleanup temp file
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass

    return None

@celery_app.task(bind=True)
def finalize_document_processing_workflow(
    self, 
    results,    # ← Celery chord injects the list of chunk_and_embed_task results
    user_id: str, 
    source_ids: List[str],
    project_id: str,
    note_type: str,
    provider: str,
    model_name: str,
    temperature: float,
    addtl_params: dict,
):
    """
    Celery callback after all chunk_and_embed_task tasks in a chord complete.
    Contains enhanced with detailed reporting, and automatic rag_note_task firing once embeddings are done.
    """
    try:
        logger.info(f"All embedding tasks in the workflow have completed.")
        
        # Calculate statistics about the completed workflow
        completed_tasks = len(results) if isinstance(results, list) else 0
        source_count = len(source_ids) if source_ids else 0
        
        # Query the database to get final statistics
        if source_ids:
            # Get the status of all sources
            sources_data = supabase_client.table("document_sources") \
                                        .select("id, vector_embed_status") \
                                        .in_("id", source_ids) \
                                        .execute()

            # Get the count of vector embeddings created for document (source_id)
            vectors_data = (
                supabase_client
                .table("document_vector_store")
                .select("*", count="exact")  # count flag goes in select() method
                .in_("source_id", source_ids)
                .execute()  # no count parameter here
            )

            # Get vector count
            vector_count = vectors_data.count if hasattr(vectors_data, 'count') else 0
                                        
            # Process status counts
            sources = sources_data.data if hasattr(sources_data, 'data') else []
            status_counts = {}
            for source in sources:
                status = source.get("vector_embed_status", "UNKNOWN")
                status_counts[status] = status_counts.get(status, 0) + 1

            # Log detailed completion information
            logger.info(
                f"Document processing workflow completed with:\n"
                f"- Sources: {source_count}\n"
                f"- Status counts: {status_counts}\n"
                f"- Total vectors: {vector_count}"
            )

            # 4) Determine overall embedding status
            statuses = set(status_counts.keys())
            if "FAILED" in statuses:
                workflow_status = "ERROR"
            elif statuses - {"COMPLETE"}:
                workflow_status = "IN_PROGRESS"
            else:
                workflow_status = "COMPLETE"

            # 5) Fire the note‐generation as soon as everything’s COMPLETE
            # `note_type` is a string, and can be set to "summary", "flashcards", "compare_contrast"
            # depending on that type of note the user wants generated
            if workflow_status == "COMPLETE" and note_type != "None":   # quick_action is a string not a bool
                logger.info(f"All embeddings DONE—queuing rag_note_task(project_id={project_id})")
                rag_note_task.apply_async(args=[
                    user_id,
                    note_type,
                    project_id,
                    provider,
                    model_name,
                    temperature,
                    addtl_params,
                ])
            
            # Update a workflow status record if needed
            # This could be useful for tracking overall workflow status in a separate table
            # if you want to implement that
        
        # Log the final success message
        logger.info(f"Document processing workflow completed successfully.")
        
        return {
            "status": "COMPLETE",
            "source_count": source_count,
            "source_ids": source_ids,
            "vector_count": vector_count if 'vector_count' in locals() else None,
            "status_counts": status_counts if 'status_counts' in locals() else None,
            "note_type": note_type,
            "completion_time": datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in finalize_document_processing_workflow: {e}", exc_info=True)
        return {
            "status": "ERROR",
            "error": str(e),
            "source_ids": source_ids,
            "error_time": datetime.now(timezone.utc).isoformat()
        }

@celery_app.task(bind=True)
def finalize_document_processing_workflow_05112025(self, results, source_ids=None):
    """
    Celery task that runs after all chunk_and_embed_task tasks in a chord complete.
    Logs the final success message for the entire workflow.

    Args:
        results (list): A list containing the return values of each task in the chord header.
                        (In this case, it will be a list of None values from chunk_and_embed_task)
    """
    logger.info(f"All embedding tasks in the workflow have completed.")
    # You could optionally inspect 'results' here if the header tasks returned meaningful data
    # For example, check if any tasks failed based on their return value (if they returned something other than None on failure)

    # Log the final success message for the entire document processing workflow
    logger.info(f"Document processing workflow completed successfully.")