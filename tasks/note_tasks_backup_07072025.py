'''
This file runs Celery tasks for handling RAG AI note creation tasks (outlines, summaries, compare-contrast)
Note genereation (with RAG) is done without token streaming. Returns full answer in one go.
'''
import multiprocessing as mp
import gc
import traceback
from celery import Celery, Task
from celery.exceptions import MaxRetriesExceededError, TimeoutError
from tasks.celery_app import celery_app  # Import the Celery app instance (see celery_app.py for LocalHost config)
import logging
import os
import redis
from supabase import create_client, Client
from utils.supabase_utils import insert_note_supabase_record, supabase_client
from utils.prompt_utils import load_yaml_prompt, build_prompt_template_from_yaml
from datetime import datetime, timezone
import tiktoken
import requests
import tempfile
import uuid
import json
from dotenv import load_dotenv

# langchain imports
from langchain_core.load import dumpd
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

#### GLOBAL ####
# API Keys 
load_dotenv()
# # OPENAI_API_KEY = os.environ.get("OPENAI_API_PROD_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()    # ← only for embeddings


# Logging
logger = logging.getLogger(__name__)

# Initialize Redis sync client for pub/sub
REDIS_LABS_URL = 'redis://default:' + os.getenv("REDIS_PASSWORD") + '@' + os.getenv("REDIS_PUBLIC_ENDPOINT")
redis_sync = redis.Redis.from_url(REDIS_LABS_URL, decode_responses=True)

class BaseTaskWithRetry(Task):
    """
    Base Celery Task class enabling automatic retries on exceptions.
    """
    autoretry_for = (Exception,)
    retry_backoff = True
    retry_kwargs = {"max_retries": 5}
    retry_jitter = True

def publish_token(chat_session_id: str, token: str):
    """
    Stub function to publish a single token to clients subscribed to a chat session.
    Replace with actual pub/sub (e.g. Redis, Supabase Realtime).
    """
    # Example: redis.publish(f"chat:{chat_session_id}", token)
    pass

class StreamToClientHandler(BaseCallbackHandler):
    """
    LangChain callback handler that emits each new LLM token via publish_token().
    """
    def __init__(self, session_id: str):
        self.session_id = session_id

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Called by LangChain for every new token in streaming mode
        publish_token(self.session_id, token)


@celery_app.task(bind=True, base=BaseTaskWithRetry)
def rag_note_task(
    self, 
    user_id, 
    note_type, 
    project_id,
    note_title, 
    provider: str,  # ← “openai”, “anthropic”, etc.
    model_name: str,
    temperature: float = 0.7,
    addtl_params: dict = None,      # <-- a dict containing optional overrides, like {"num_questions": 10, "document_ids":[...]}
):
    """
    RAG workflow for various note types (exam questions, case briefs, outlines, etc.).

    Steps:
    0) Mark start time in Celery state
    1) Embed a short “retrieval” query (as dim-1536 embedding)
    2) Fetch top-K relevant chunks via Supabase RPC
    3) Load the appropriate YAML prompt based on note_type
    4) Format that YAML template with {context} + any override from meta (e.g. num_questions)
    5) Call the LLM once with the fully formed prompt (Stream LLM response tokens or return all at once)
    6) Persist note output to public.notes
    """
    
    if addtl_params is None:
        addtl_params = {}
    
    try:
        # Step 0) Mark explicit start time in self.metadata
        # This manually sets result = AsyncResult(task_id) when checking on this celery task via job.id
        # result.state → "PENDING"
        # result.info → {"start_time": "..."}
        self.update_state(
            state="STARTED", 
            meta={"start_time": datetime.now(timezone.utc).isoformat()}
        )

        # Step 1) Choose the prompt based on user selection of "note_type"
        if note_type == "outline":
            yaml_file = "case-outline-prompt.yaml"
        elif note_type == "exam_questions":
            yaml_file = "exam-questions-prompt.yaml"
        elif note_type == "case_brief":
            yaml_file = "case-brief-prompt.yaml"
        elif note_type == "compare_contrast":
            yaml_file = "compare-contrast-prompt.yaml"
        elif note_type == "flashcards":
            yaml_file = "flashcards-prompt.yaml"
        else:
            raise ValueError(f"Unknown note_type: {note_type}")
        
        # Load YAML and extract “base_prompt” and “template” 
        yaml_dict = load_yaml_prompt(yaml_file)
        base_query      = yaml_dict.get("base_prompt")
        prompt_template = build_prompt_template_from_yaml(yaml_dict)
        if not base_query:
            raise KeyError(f"`base_prompt` not found in {yaml_file}")

        # Step 2) Embed the short base query using OpenAI Ada embeddings (1536 dims)
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=OPENAI_API_KEY,
        )
        query_embedding = embedding_model.embed_query(base_query)

        # Step 3) Fetch top-K relevant chunks via Supabase RPC && Format for llm context window
        relevant_chunks = fetch_relevant_chunks(query_embedding, project_id)
        chunk_context = "\n\n".join(chunk["content"] for chunk in relevant_chunks)

        # Step 4) Question-type specific adjustments (question number, legal course name, what ever idiosyncracies you can think of)
        if note_type == "exam_questions":
            # Decide how many questions to generate (override via addtl_params)
            num_questions = addtl_params.get("num_questions")
            if num_questions and isinstance(num_questions, int) and num_questions > 0:
                llm_input = prompt_template.format(
                    context=chunk_context,
                    n_questions=num_questions
                )
            else:
                # fallback to YAML default of 15 if no override
                llm_input = prompt_template.format(
                    context=chunk_context,
                    n_questions=addtl_params.get("num_questions", 15)
                )

        elif note_type == "case_brief":
            # YAML template only needs {context}
            llm_input = prompt_template.format(context=chunk_context)

        elif note_type == "outline":
            # Assuming your "case-outline-prompt.yaml" has only {context}
            llm_input = prompt_template.format(context=chunk_context)
        
        elif note_type == "compare_contrast":
            llm_input = prompt_template.format(context=chunk_context)
        else:
            # (We already checked above, but safe‐guard here)
            raise ValueError(f"Unsupported note_type: {note_type}")

        # Step 5) Generate llm client from factory
        from utils.llm_factory import LLMFactory       # Lazy-import heavy modules
        llm_client = LLMFactory.get_client(
            provider=provider,
            model_name=model_name,
            temperature=temperature
        )

        # Step 6) Generate answer for client
        full_answer = llm_client.chat(llm_input)

        # Step 7) Save note to the public.notes table in Supabase (realtime Supabase table)
        save_note(
            project_id=project_id,
            user_id=user_id,
            note_type=note_type,
            note_title=note_title, 
            content=full_answer      # full rag reponse
        )

        # Garbage collect cleanup and return success (proactively release large in-memory buffers)
        try:
            del relevant_chunks, chunk_context, llm_input, full_answer
        except NameError:
            pass
        gc.collect()

        # Return nothing
        return "RAG Note Task suceess"

    except Exception as e:
        logger.error(f"RAG Note Task failed: {e}", exc_info=True)
        try:
            raise self.retry(exc=e)
        except MaxRetriesExceededError:
            raise RuntimeError(
                f"[CELERY] Step 2) RAG Note creattion failed permanently after {self.max_retries} retries: {e}"
            ) from e


def fetch_relevant_chunks(query_embedding, project_id, match_count=10):
    """
    Calls Supabase RPC to retrieve the nearest neighbor chunks using HNSW index.
    """
    try:
        response = supabase_client.rpc("match_document_chunks_hnsw", {
            "p_project_id": project_id,
            "p_query": query_embedding,
            "p_k": match_count
        }).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching relevant chunks: {e}", exc_info=True)
        raise


# def generate_rag_answer(
#         llm_client,
#         query, 
#         relevant_chunks: list,
#         max_chat_history: int = 10,
#     ):
#     """
#     [Replaced by llm_client.chat(llm_input)]
#     Build prompt, invoke streaming LLM, publish tokens in real-time,
#     and return the full generated answer at completion.

#     Q: **Why no `chat_session_id` ?
#     A: because this is going to a project not a chat histoty
#     """
#     # Build conversational context
#     # chat_history = fetch_chat_history(chat_session_id)[-max_chat_history:]
#     # formatted_history = format_chat_history(chat_history) if chat_history else ""
#     chunk_context = "\n\n".join(c["content"] for c in relevant_chunks)
#     full_context = (
#         f"Relevant Context:\n{chunk_context}\n\n"
#         f"User Query: {query}\nAssistant:"
#     )

#     trimmed_context = trim_context_length(
#         full_context=full_context,
#         query=query,
#         relevant_chunks=relevant_chunks,
#         model_name=llm_client.model_name,
#         max_tokens=127999
#     )

#     # 5) Finally, call the LLM client once
#     try:
#         # All of your LLMClient implementations expose `.chat(prompt: str) -> str`
#         answer = llm_client.chat(trimmed_context)
#         # llm_client = get_chat_llm(model_name) # being SUNSET !!! 
#         return answer
#     except Exception as e:
#         logger.error(f"Error in LLM call (model={llm_client.model_name}): {e}", exc_info=True)
#         raise


def save_note(
        project_id, 
        user_id, 
        note_type, 
        note_title,
        content
    ):
    """
    Persists generated summary note into Supabase public.notes table.
    """
    insert_note_supabase_record(
        client=supabase_client,
        table_name="notes",
        user_id=user_id,
        project_id=project_id,
        note_title=note_title,
        content_markdown=content,
        note_type=note_type,
        is_generated=True,
        is_shareable=False,
        created_at=datetime.now(timezone.utc).isoformat()
    )


def trim_context_length(full_context, query, relevant_chunks, model_name, max_tokens):
    model_to_encoding = {
        "o4-mini": "o200k_base",
        "gpt-4o": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
    }
    encoding_name = model_to_encoding.get(model_name, "cl100k_base")
    try:
        tokenizer = tiktoken.get_encoding(encoding_name)
    except Exception:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    history = full_context
    while len(tokenizer.encode(history)) > max_tokens and relevant_chunks:
        relevant_chunks.pop()
        chunk_context = "\n\n".join(c["content"] for c in relevant_chunks)
        history = f"Relevant Context:\n{chunk_context}\n\nUser Query: {query}\nAssistant:"
    return history