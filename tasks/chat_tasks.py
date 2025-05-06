'''
This file runs Celery tasks for handling RAG chat tasks
'''

from celery import Celery, Task
from tasks.celery_app import celery_app  # Import the Celery app instance (see celery_app.py for LocalHost config)
import logging
import os
from langchain_core.load import dumpd, dumps, load, loads
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage
from supabase import create_client, Client
from utils.supabase_utils import insert_conversation_supabase_record, supabase_client
from datetime import datetime, timezone
import uuid
import json

# from celery import Celery, chain
# import logging
# import os
# import json
# import psutil
# import requests
# import tempfile
# from tasks.celery_app import celery_app  # Import the Celery app instance (see celery_app.py for LocalHost config)
# from utils.audio_utils import generate_audio, generate_only_dialogue_text
# from utils.s3_utils import upload_to_s3, s3_client, s3_bucket_name
# from utils.supabase_utils import insert_conversation_supabase_record, supabase_client
# from utils.cloudfront_utils import get_cloudfront_url
# from utils.instruction_templates import INSTRUCTION_TEMPLATES
# from time import sleep
# from datetime import datetime, timezone
# import uuid

# # langchain dependencies
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

class BaseTaskWithRetry(Task):
    autoretry_for = (Exception,)
    retry_backoff = True
    retry_kwargs = {"max_retries": 5}
    retry_jitter = True

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def rag_chat_task(self, user_id, conversation_id, query, document_ids, project_id, model_type):
    """
    MAIN Celery task for handling RAG (Retrieval-Augmented Generation) chatbot logic.

    example json body from the front-end
    {
        "user_id": "user-32kfowdkbf0ouskhw9hs-sk",
        "conversation_id": "conv456",      // Or leave empty for a new conversation
        "query": "What were the results of the researchers' conclusion?",
        "document_ids": ["fjs78dkbf9fksfnuhis-89s", "789s-khw9hdfjksdjhk9sdh"]
        "project_id": "vjkbnoasdf7y892wb", 
        "model_type": "deepseek-v3"
    }
    """
    try:
        # Step 1: Handle first-time chat session (create new conversation if needed)
        if not conversation_id:
            conversation_id = create_new_conversation(user_id, document_ids, media_id)

        # Step 2: Vectorize the query
        embedding_model = OpenAIEmbeddings()
        query_embedding = embedding_model.embed_query(query)

        # Step 3: Fetch relevant document chunks
        relevant_chunks = fetch_relevant_chunks(query_embedding, document_ids)

        # Step 4: Generate the answer using RAG
        response = generate_rag_answer(query, conversation_id, relevant_chunks, model_name='gpt-4o-mini')

        # Step 4.1: Extract the message content from the AIMessage object
        json_dict = dumpd(response)
        try:
            json_content = json_dict['kwargs']['content']  # Adjust keys based on actual nesting
            # print("\nContent:", json_content)
        except KeyError as e:
            print(f"KeyError: {e} - Make sure the key exists in the structure.")

        # Step 5: Save query and response in message history
        save_conversation(conversation_id, user_id, query, answer=json_content)

        # Api call returns the answer and metadata for UI formatting
        json_response = {}
        try:
            json_response = {
            "answer": json_content,
            "message_role": "assistant"
            }
        except KeyError as e:
            print(f"KeyError: {e} - Make sure the key exists in the structure.")
        
        return json_response
    except Exception as e:
        logger.error(f"RAG Chat Task failed: {str(e)}", exc_info=True)
        raise self.retry(exc=e)

def create_new_conversation(user_id, document_ids, media_id):
    """
    Create a new conversation in the `conversations` table.
    """
    new_conversation = {
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document_ids": document_ids,
        "media_id":media_id                     # ...needs to be here i believe
    }
    try:
        response = supabase_client.table("conversations").insert(new_conversation).execute()
        return response.data[0]["id"]
    except Exception as e:
        logger.error(f"Error creating new conversation: {str(e)}", exc_info=True)
        raise

def fetch_relevant_chunks(query_embedding, document_ids, match_count=3):
    """
    Fetch the most relevant document chunks from the Supabase vector store.
    """
    try:
        response = supabase_client.rpc("match_document_chunks", {
            "query_embedding": query_embedding,
            "document_ids": document_ids,
            "match_count": match_count
        }).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching relevant chunks: {str(e)}", exc_info=True)
        raise

def generate_rag_answer(query, conversation_id, relevant_chunks, model_name, max_chat_history=10):
    """
    Generates the RAG answer by combining chat history, query, and relevant chunks.
    """
    try:
        chat_history = fetch_chat_history(conversation_id)
        chat_history = chat_history[-max_chat_history:]
        formatted_history = format_chat_history(chat_history) if chat_history else ""
        chunk_context = " ".join([chunk["content"] for chunk in relevant_chunks])
        full_context = f"{formatted_history}\nRelevant Context:\n{chunk_context}\n\nUser Query: {query}\nAssistant:"
        full_context = trim_context_length(full_context, query, relevant_chunks, model_name, max_tokens=127999)
        
        llm = ChatOpenAI(model=model_name, temperature=0.7)
        prompt_template = PromptTemplate(input_variables=["context"], template="{context}")
        pipeline = prompt_template | llm
        return pipeline.invoke({"context": full_context})
    except Exception as e:
        logger.error(f"Error generating RAG answer: {str(e)}", exc_info=True)
        raise

def save_conversation(conversation_id, user_id, query, answer):
    """
    Save the user's query and the RAG response to the conversation history.
    """
    try:
        insert_conversation_supabase_record(
            supabase_client, 
            table_name="message", 
            user_id=user_id, 
            conversation_id=conversation_id, 
            message_role="user",
            message_content=query,
            created_at=datetime.now(timezone.utc).isoformat()
        )

        insert_conversation_supabase_record(
            supabase_client, 
            table_name="message", 
            user_id=user_id, 
            conversation_id=conversation_id, 
            message_role="assistant",
            message_content=answer,
            created_at=datetime.now(timezone.utc).isoformat()
        )

        # supabase_client.table("message").insert({
        #     "user_id": user_id,
        #     "conversation_id": conversation_id,
        #     "message_role": "user",
        #     "message_content": query,
        #     "created_at": datetime.now(timezone.utc).isoformat(),
        # }).execute()

        # supabase_client.table("message").insert({
        #     "user_id": user_id,
        #     "conversation_id": conversation_id,
        #     "message_role": "assistant",
        #     "message_content": answer,
        #     "created_at": datetime.now(timezone.utc).isoformat(),
        # }).execute()
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}", exc_info=True)
        raise

def fetch_chat_history(conversation_id):
    """
    Fetches the chat history for a given conversation_id.
    Returns a list of messages sorted by created_at.
    """
    try: 
        response = supabase_client.table("message").select("*").eq("conversation_id", conversation_id).order("created_at").execute()
        print("Successful fetching chat history from Supabase !!")
        return response.data
    except Exception as e:
        raise Exception(f"Error fetching chat history: {e}")

def format_chat_history(chat_history):
    """
    Formats the chat history into a conversational string for the LLM prompt.
    """
    formatted_history = ""
    for message in chat_history:
        role = message["message_role"]
        content = message["message_content"]
        formatted_history += f"{role.capitalize()}: {content}\n"
    return formatted_history

def trim_context_length(full_context, query, relevant_chunks, model_name, max_tokens):
    # Ensure formatted_history and chunk_context are initialized
    formatted_history = ""
    chunk_context = " ".join([chunk["content"] for chunk in relevant_chunks])

    # Helper function to estimate token length
    def count_tokens(text, model_name):
        import tiktoken
        tokenizer = tiktoken.encoding_for_model(model_name)
        return len(tokenizer.encode(text))

    # Dynamically trim context to fit within token limit
    while count_tokens(full_context, model_name) > max_tokens:
        if 'chat_history' in locals() and len(chat_history) > 1:
            chat_history = chat_history[1:]  # Remove the oldest message
            formatted_history = format_chat_history(chat_history)
        elif len(relevant_chunks) > 1:
            relevant_chunks.pop()  # Remove the least relevant chunk
            chunk_context = " ".join([chunk["content"] for chunk in relevant_chunks])
        else:
            break  # Cannot reduce further

    # Safeguard: Ensure formatted_history and chunk_context are properly defined
    if not formatted_history:
        formatted_history = "No prior chat history available."

    if not chunk_context:
        chunk_context = "No relevant context found."

    return f"{formatted_history}\nRelevant Context:\n{chunk_context}\n\nUser Query: {query}\nAssistant:"