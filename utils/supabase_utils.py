import os
import json
import uuid
import boto3
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_client: Client = create_client(
    os.getenv('SUPABASE_URL'), 
    os.getenv('SUPABASE_SERVICE_ROLE_KEY')  # Ensure this key is the service role
)

def insert_mp3_supabase_record(
        client, 
        table_name, 
        podcast_title, 
        cdn_url, 
        transcript,
        content_tags,
        uploaded_by, 
        is_public,
        is_playlist
    ):
    """Inserts a record into the Supabase Library table."""
    try:
        data = {
            "podcast_title": podcast_title,
            "cdn_url": cdn_url,
            "uploaded_by":uploaded_by,
            "is_public": is_public,
            "transcript": transcript,
            "description": 'this description could be workshopped just a bit...',
            "is_playlist": False
        }
        # Execute the insert query
        response = client.table(table_name).insert(data).execute()   
        
        # Check if the response contains data
        if response.data:
            # Successful insertion
            print("Successful Supabase 'podcast' row insert")
            return response.data[0]["id"]  # Return podcast ID... prior version is just response.data
        else:
            # If there's no data, check for errors
            raise Exception(f"Supabase error: {response.error}")
    except Exception as e:
        raise Exception(f"Failed to insert into Supabase: {e}")

def insert_document_supabase_record(client, table_name, cdn_url, project_id, content_tags, uploaded_by):
    """Inserts a record into the Supabase Library table."""
    try:
        data = {
            "cdn_url": cdn_url,
            "project_id": project_id,
            "content_tags": content_tags,
            "uploaded_by":uploaded_by,
        }
        # Execute the insert query
        response = client.table(table_name).insert(data).execute()   
        
        # Check if the response contains data
        if response.data:
            # Successful insertion
            print("Successful Supabase row insert")
            return response.data[0]["id"]  # Return document_sources.id
        else:
            # If there's no data, check for errors
            raise Exception(f"Supabase error: {response.error}")
    except Exception as e:
        raise Exception(f"Failed to insert into Supabase: {e}")

def insert_vector_supabase_record(client, table_name, source_id, content, metadata, embedding):
    try:
        response = client.table(table_name).insert({
            "source_id": source_id,
            "content": content,
            "metadata": json.dumps(metadata),
            "embedding": embedding
        }).execute()
    except Exception as e:
        raise Exception(f"Failed to insert into Supabase vector-store: {e}")
    
def insert_conversation_supabase_record(client, table_name, user_id, conversation_id, message_role, message_content, created_at):
    try:
        response = client.table(table_name).insert({
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message_role": message_role,
            "message_content": message_content,
            "created_at": created_at,
        }).execute()
    except Exception as e:
        raise Exception(f"Error saving messages: {e}")
