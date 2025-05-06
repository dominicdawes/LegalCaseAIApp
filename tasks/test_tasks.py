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

