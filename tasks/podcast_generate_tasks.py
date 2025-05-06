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

# === Simple sanity check tasks for Celery functionality === #

@celery_app.task(bind=True)
def addition_task(self, x, y):
    """
    Celery task to validate if celery and redis (message broker) are working.
    """
    print(f"DEBUG: Task received with x={x}, y={y}")
    sleep(8)
    return x + y

@celery_app.task
def reverse(text):
    sleep(18)        # simulates a long api call
    return text[::-1]

@celery_app.task
def concat_task(x, y):
    sleep(9)
    return x + y

# === PRODUCTION CELERY TASKS === #

# @celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
# def process_pdf_task(self, files, metadata=None):
#     """
#     Main Celery task to upload PDFs to S3, save to Supabase, and trigger vector embedding tasks.
#     """

#     # === RENDER RESOURCE LOGGING (MB) === #
#     process = psutil.Process(os.getpid())
#     mem_before = process.memory_info().rss
#     logger.info(f"Starting {self.name} with files: {files} and metadata: {metadata}")
#     logger.info(f"Memory usage before task: {mem_before / (1024 * 1024)} MB")

#     if not files:
#         return {"error": "Please upload at least one PDF file before processing."}

#     uploaded_documents = []
#     source_ids = [] # used in chained task

#     for file in files:
#         try:
#             # Handle URL or local file path
#             if file.startswith('http://') or file.startswith('https://'):
#                 response = requests.get(file)
#                 response.raise_for_status()
#                 temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
#                 temp_file.write(response.content)
#                 temp_file.close()
#                 file_path = temp_file.name
#             else:
#                 file_path = file

#             # Generate S3 key and upload file
#             ext_ending = os.path.splitext(file_path)[1]
#             s3_document_key = f"{uuid.uuid4()}{ext_ending}"
#             upload_to_s3(s3_client, file_path, s3_document_key)
            
#             # Generate CloudFront URL
#             cloudfront_document_url = get_cloudfront_url(s3_document_key)

#             # Insert the document source record into Supabase
#             source_id = insert_document_supabase_record(
#                 client=supabase_client,
#                 table_name="document_sources",
#                 cdn_url=cloudfront_document_url,
#                 content_tags=["Fitness", "Health", "Wearables"],
#                 uploaded_by=metadata.get("uploaded_by", "")
#             )

#             # Append to source_ids list
#             source_ids.append(source_id)

#             uploaded_documents.append({
#                 "source_id": source_id,
#                 "pdf_url": cloudfront_document_url,
#                 "file_path": file_path
#             })

#         except Exception as e:
#             logger.error(f"Failed to upload and record file {file}: {e}", exc_info=True)
#         finally:
#             # Clean up temporary files if necessary
#             if file.startswith('http'):
#                 os.unlink(file_path)

#     # Trigger the embedding task for each document
#     for doc in uploaded_documents:
#         chunk_and_embed_task.delay(doc["pdf_url"], doc["source_id"])      # enqueue to Celery task queue
    
#     # Logging message
#     logger.info(" 'message' : PDF upload and record creation completed. Embedding tasks started.")

#     return source_ids 

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def validate_and_generate_audio_task(self, files, metadata=None, instructions_key='podcast', *args):
    """
    Celery task to validate and generate audio podcast (.mp3) for a list of PDF files.
    
    Args:
        files (List): list of either urls or local paths (see audio_utils.py)
        metadata (Dict): additional metadata for processing {'uploaded_by':, length:, temperature:, model_choice:...}
        *args: openai_api_key, text_model, audio_model, speaker_1_voice...
    """
    # === RENDER RESOURCE LOGGING === #
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    logger.info(f"Starting {self.name} with args: {args}")
    logger.info(f"Memory usage before task: {mem_before / (1024 * 1024)} MB")

    # Store the start time
    self.update_state(meta={'start_time': datetime.now(timezone.utc).isoformat()})

    if not files:
        return {"error": "Please upload at least one PDF file before generating audio."}
    
    # Initialize presigned_url to avoid UnboundLocalError
    presigned_url = None

    try:
        # Extract the instructions from INSTRUCTION_TEMPLATES using the given instructions_key
        llm_instructions = INSTRUCTION_TEMPLATES.get(instructions_key, {})
        intro_instructions = llm_instructions.get("intro", "")
        text_instructions = llm_instructions.get("text_instructions", "")
        scratch_pad_instructions = llm_instructions.get("scratch_pad", "")
        prelude_dialog = llm_instructions.get("prelude", "")
        podcast_dialog_instructions = llm_instructions.get("dialog", "")

        # Call generate_audio with default or provided arguments
        audio_file, transcript, original_text = generate_audio(
            files,
            intro_instructions=intro_instructions,
            text_instructions=text_instructions,
            scratch_pad_instructions=scratch_pad_instructions,
            prelude_dialog=prelude_dialog,
            podcast_dialog_instructions=podcast_dialog_instructions,
            *args,  # Handle any positional arguments passed via the task
        )
    
        ## Add mp3 to data bucket and CDN
        # Generate unique object keyh for mp3 file
        s3_mp3_object_key = f"{uuid.uuid4()}.mp3"

        # Upload to AWS S3 Bucket
        upload_to_s3(
            s3_client, 
            audio_file, 
            s3_mp3_object_key
        )

        # Generate a CloudFront URL for the uploaded file
        cloudfront_podcast_url = get_cloudfront_url(s3_mp3_object_key)

        # Insert podcast into Supabase
        media_id = insert_mp3_supabase_record(
            client=supabase_client,
            table_name="media_uploads",
            podcast_title="My Podcast", 
            cdn_url=cloudfront_podcast_url,                                        
            transcript=transcript,
            content_tags=["Fitness", "Technology"],  # Pass content_tags as an array,
            uploaded_by=metadata['uploaded_by'],
            is_public=metadata['is_public'],
            is_playlist=False,
        )

        # === RENDER RESOURCE LOGGING === #
        mem_after = process.memory_info().rss
        logger.info(f"Finished {self.name}")
        logger.info(f"Memory usage after task: {mem_after / (1024 * 1024)} MB")

        return {
            'media_id': media_id,
            "media_name": "Podcast Uploaded from Celery Worker...",
            "cdn_url": cloudfront_podcast_url,    
            "transcript": transcript,
            "original_text": original_text,
            "error": None
        }
    
    except Exception as e:
        logger.exception(f"Task {self.name} failed with exception: {e}")
        raise # ask gpt how to do this

# @celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
# def insert_sources_media_association_task(self, task_results):
#     """
#     Celery task to insert a row in TABLE `sources_media_association` table. Callback task to accept 
#     the list of results from the group tasks. The results will be in the order the tasks were defined in the group.
    
#     - task_results: (list) containing results from process_pdf_task and validate_and_generate_audio_task
#     """
#     # Unpack the results
#     source_ids = task_results[0]
#     media_result = task_results[1]
#     media_id = media_result.get('media_id')

#     try:
#         for source_id in source_ids:
#             supabase_client.table('sources_media_association').insert({
#                 'source_id': source_id,
#                 'media_id': media_id
#             }).execute()
#         logger.info(f"Successfully linked source_ids {source_ids} with media_id {media_id}")
#     except Exception as e:
#         logger.error(f"Failed to insert into sources_media_association: {e}")
#         raise

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def validate_and_generate_audio_task_deprecated(self, files, metadata=None, instructions_key='podcast', *args):
    """
    Celery task to validate and generate audio podcast (.mp3) for a list of PDF files.
    
    Args:
        files (List): list of either urls or local paths (see audio_utils.py)
        metadata (Dict): additional metadata for processing {'uploaded_by':, length:, temperature:, model_choice:...}
        *args: openai_api_key, text_model, audio_model, speaker_1_voice...
    """
    # === RENDER RESOURCE LOGGING === #
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    logger.info(f"Starting {self.name} with args: {args}")
    logger.info(f"Memory usage before task: {mem_before / (1024 * 1024)} MB")

    # Store the start time
    self.update_state(meta={'start_time': datetime.now(timezone.utc).isoformat()})

    if not files:
        return {"error": "Please upload at least one PDF file before generating audio."}
    
    # Initialize presigned_url to avoid UnboundLocalError
    presigned_url = None

    try:
        # Extract the instructions from INSTRUCTION_TEMPLATES using the given instructions_key
        llm_instructions = INSTRUCTION_TEMPLATES.get(instructions_key, {})
        intro_instructions = llm_instructions.get("intro", "")
        text_instructions = llm_instructions.get("text_instructions", "")
        scratch_pad_instructions = llm_instructions.get("scratch_pad", "")
        prelude_dialog = llm_instructions.get("prelude", "")
        podcast_dialog_instructions = llm_instructions.get("dialog", "")

        # Call generate_audio with default or provided arguments
        audio_file, transcript, original_text = generate_audio(
            files,
            intro_instructions=intro_instructions,
            text_instructions=text_instructions,
            scratch_pad_instructions=scratch_pad_instructions,
            prelude_dialog=prelude_dialog,
            podcast_dialog_instructions=podcast_dialog_instructions,
            *args,  # Handle any positional arguments passed via the task
        )
    
        ## Add sources to data bucket and CDN
        for file in files:
            try:
                # Check if file is a URL and download it
                if file.startswith('http://') or file.startswith('https://'):
                    response = requests.get(file)
                    response.raise_for_status()  # Raise an error for bad responses
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    temp_file.write(response.content)
                    temp_file.close()
                    file_path = temp_file.name
                    print('temp pdf file downloaded :)')
                else:
                    # Treat as a local file
                    file_path = file

                # Split the file name to get the extension
                ext_ending = os.path.splitext(file_path)[1]
                
                # Generate a unique object key for S3 using a UUID and the file extension
                s3_document_object_key = f"{uuid.uuid4()}{ext_ending}"
                
                # Upload file to S3
                upload_to_s3(
                    s3_client, 
                    file_path, 
                    s3_document_object_key
                )

                # Generate CloudFront URL
                cloudfront_document_url = get_cloudfront_url(s3_document_object_key)

                # check uploaded_by is not null]
                print(metadata)
                metadata['uploaded_by'] = metadata.get('uploaded_by') or ""

                # Insert the document source record into Supabase
                insert_document_supabase_record(
                    client=supabase_client,
                    table_name="document_sources",  
                    cdn_url=cloudfront_document_url,                                         
                    content_tags=["Fitness", "Health", "Wearables"],  # Pass content_tags as an array
                    uploaded_by=metadata['uploaded_by'],
                )

            except Exception as e:
                # Log the error, including the file name, for debugging
                logger.error(f"Failed to process file {file_path}: {e}", exc_info=True)
            finally:
                # Clean up temporary file if it was downloaded
                if file.startswith('http://') or file.startswith('https://'):
                    os.unlink(file_path)

        ## Add mp3 to data bucket and CDN
        # Generate unique object keyh for mp3 file
        s3_mp3_object_key = f"{uuid.uuid4()}.mp3"

        # Upload to S3
        upload_to_s3(
            s3_client, 
            audio_file, 
            s3_mp3_object_key
        )

        # Generate a CloudFront URL for the uploaded file
        cloudfront_podcast_url = get_cloudfront_url(s3_mp3_object_key)

        # Insert podcast into Supabase
        insert_mp3_supabase_record(
            client=supabase_client,
            table_name="media_uploads",
            podcast_title="My Podcast", 
            cdn_url=cloudfront_podcast_url,                                        
            transcript=transcript,
            content_tags=["Fitness", "Technology"],  # Pass content_tags as an array,
            uploaded_by=metadata['uploaded_by'],
            is_public=metadata['is_public'],
            is_playlist=False,
        )

        # === RENDER RESOURCE LOGGING === #
        mem_after = process.memory_info().rss
        logger.info(f"Finished {self.name}")
        logger.info(f"Memory usage after task: {mem_after / (1024 * 1024)} MB")

        return {
            "cdn_url": cloudfront_podcast_url,                        # Changed (10/15) from audio_file --> audio-presign-url
            "transcript": transcript,
            "original_text": original_text,
            "error": None
        }
    
    except Exception as e:
        logger.exception(f"Task {self.name} failed with exception: {e}")
        raise
        # return {
        #     "cdn_url": cloudfront_url,
        #     "transcript": None,
        #     "original_text": None,
        #     "error": str(e)
        # }

# @celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
# def chunk_and_embed_task(self, pdf_url, source_id, chunk_size=1000, chunk_overlap=100):
#     """
#     Celery task to download, chunk, embed, and store embeddings of a PDF in Supabase.
#     """
#     try:
#         # Download the PDF file from the URL
#         response = requests.get(pdf_url)
#         response.raise_for_status()
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
#         temp_file.write(response.content)
#         temp_file.close()
        
#         # Load and split the PDF into chunks
#         loader = PyPDFLoader(temp_file.name)
#         documents = loader.load()
#         text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#         chunks = text_splitter.split_documents(documents)

#         # Initialize embedding model
#         embedding_model = OpenAIEmbeddings()

#         # Process each chunk
#         for chunk in chunks:
#             try:
#                 chunk_text = chunk.page_content
#                 embedding = embedding_model.embed_query(chunk_text)
                
#                 # Insert the chunk and embedding into the Supabase 'document_vector_store' table
#                 insert_vector_supabase_record(
#                     client=supabase_client,
#                     table_name="document_vector_store",
#                     source_id=source_id,
#                     content=chunk_text,
#                     metadata={
#                         "source": pdf_url,
#                         "page_number": chunk.metadata.get("page", None)
#                     },
#                     embedding=embedding
#                 )
#             except Exception as e:
#                 logger.error(f"Error processing chunk: {e}", exc_info=True)
#     except Exception as e:
#         logger.error(f"Failed to download and process file {pdf_url}: {e}", exc_info=True)
#     finally:
#         # Clean up temporary file
#         os.unlink(temp_file.name)

#     return {"message": f"Embedding task for {pdf_url} completed successfully."}

# --- More of a debug / sanity task, not really used in prod
@celery_app.task(bind=True, name='tasks.generate_tasks.generate_dialogue_only_task')
def generate_dialogue_only_task(self, files, instructions_key='podcast', *args):
    """
    Celery task to validate and generate ONLY text dialogue for a list of PDF files.

    Args:
        files (List): list of either urls or local paths (see audio_utils.py) 
        *args: openai_api_key, text_model, audio_model, speaker_1_voice...
    """
    # Store the start time
    self.update_state(meta={'start_time': datetime.now(timezone.utc).isoformat()})

    if not files:
        return {"error": "Please upload at least one PDF file before generating dialogue."}

    try:
        # Extract the instructions from INSTRUCTION_TEMPLATES using the given instructions_key
        llm_instructions = INSTRUCTION_TEMPLATES.get(instructions_key, {})
        intro_instructions = llm_instructions.get("intro", "")
        text_instructions = llm_instructions.get("text_instructions", "")
        scratch_pad_instructions = llm_instructions.get("scratch_pad", "")
        prelude_dialog = llm_instructions.get("prelude", "")
        podcast_dialog_instructions = llm_instructions.get("dialog", "")

        # Call the generate_only_dialogue function with the instructions as keyword arguments
        dialogue_text = generate_only_dialogue_text(
            files,
            intro_instructions=intro_instructions,
            text_instructions=text_instructions,
            scratch_pad_instructions=scratch_pad_instructions,
            prelude_dialog=prelude_dialog,
            podcast_dialog_instructions=podcast_dialog_instructions,
            *args
        )

        return {
            "dialogue_text": dialogue_text,
            "error": None
        }
    except Exception as e:
        return {
            "dialogue_text": None,
            "error": str(e)
        }
