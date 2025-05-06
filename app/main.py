from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import List, Dict, Any
import uuid
from utils.pdf_utils import extract_text_from_pdf
from tasks.podcast_generate_tasks import validate_and_generate_audio_task, generate_dialogue_only_task
from tasks.upload_tasks import process_pdf_task, insert_sources_media_association_task, insert_project_documents_task
from tasks.test_tasks import addition_task
from tasks.chat_tasks import rag_chat_task
from celery import chain, chord, group
from celery.result import AsyncResult



# Init fast api app
app = FastAPI()
# celery_app = Celery('tasks', broker='redis://localhost:6379/0')


origins = [
    "https://app.weweb.io",  # Replace with the actual WeWeb domain if different
    "https://editor.weweb.io",
    # Add other domains as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======== PYDANTIC MODELS ======== #

class Numbers(BaseModel):
    x: int
    y: int
    
# Define Pydantic models for the responses
class SumResponse(BaseModel):
    sum: int

class PDFCaptureResponse(BaseModel):
    uuid: str
    url: str

# Define Pydantic model for the PDF extraction response
class PDFExtractResponse(BaseModel):
    filename: str
    combined_text: str

class PDFExtractBatchResponse(BaseModel):
    results: List[PDFExtractResponse]

# [SANITY CHECK] Request model for addition sanity
class AdditionRequest(BaseModel):
    x: int
    y: int

# Pydantic model for the request body
class PDFRequest(BaseModel):
    ''' WeWeb specific pydantic struct '''
    files: List[str]  # List of URLs or file paths of the PDFs
    metadata: Dict[str, Any]  # A dictionary for any metadata information

# New RAG pipeline request model
class NewRagRequest(BaseModel):
    ''' WeWeb specific pydantic struct '''
    files: List[str]  # List of URLs or file paths of the PDFs
    metadata: Dict[str, Any]  # A dictionary for any metadata information

# New RAG pipeline response model
class NewRagResponse(BaseModel):
    embedding_task_id: str      # from vector embedding task

# Pydantic model for the response
class PDFResponse(BaseModel):
    audio_task_id: str
    embedding_task_id: str

# RAG Query pydantic model
class RAGRequest(BaseModel):
    user_id: str
    conversation_id: str
    query: str
    document_ids: List[str]
    project_id: str
    model_type: str

# ================================================ #
#               TEST ENDPOINTS
# ================================================ #

# Root endpoint
@app.get("/")
async def root():
    return {"success": "Hello Server LegalNoteAI FastAPI App"}

# Endpoint for sanity check 
@app.post("/sum_two_num/")
async def sum_two_num(numbers: AdditionRequest):
    print("execute 2 sum")
    task = addition_task.delay(numbers.x, numbers.y)
    print(f"Task submitted: {task.id}")
    return {"task_id": task.id}

# Endpoint for capturing PDF info (sanity check endpoint)
@app.get("/pdf_capture", response_model=PDFCaptureResponse)
async def pdf_capture(file: str):
    unique_id = str(uuid.uuid4())  # Generate a unique UUID
    return {"uuid": unique_id, "url": file}

# Endpoint for extracting a single PDF's content
@app.get("/pdf_extract", response_model=PDFExtractResponse)
async def pdf_extract(file: str):
    try:
        # Extract text using the utility function
        combined_text = extract_text_from_pdf(file)
        return {"filename": file.split("/")[-1], "combined_text": combined_text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Endpoint for extracting content from multiple PDFs
@app.post("/pdf_extract_batch", response_model=PDFExtractBatchResponse)
async def pdf_extract_batch(files: List[str]):
    results = []
    
    for file_url in files:
        try:
            # Extract text using the utility function
            combined_text = extract_text_from_pdf(file_url)
            filename = file_url.split("/")[-1]
            results.append(PDFExtractResponse(filename=filename, combined_text=combined_text))
        
        except Exception as e:
            # Append an error message for the current file
            results.append(PDFExtractResponse(filename=file_url, combined_text=f"Error: {str(e)}"))
    
    return {"results": results}

# POST endpoint for addition using Celery
@app.post("/celery_test_addition/")
async def celery_test_addition(request: AdditionRequest):
    try:
        # Enqueue the Celery task
        print(f"API DEBUG: {[request.x, request.y]}")
        task = addition_task.apply_async(args=[request.x, request.y])
        
        # Return the task ID to the client
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================================================ #
#               RAG API ENDPOINTS
# ================================================ #
@app.post("/new-rag-project/", response_model=NewRagResponse)
async def creat_new_rag_project(request: NewRagRequest, background_tasks: BackgroundTasks):
    '''
    Endpoint to create both a RAG pipeline for AI notes in one call:
        - process_pdf_task:
            input: request.files, request.metadata
            returns: source_ids
        
    Request contains:
        request.files (List): list of pdf file links
        request.metadata (json): {
            project_id:
        }
    '''
    try:
        job = process_pdf_task.apply_async(
            args=[request.files, request.metadata]
        )
        return {"task_id": job.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/new-rag-project/", response_model=NewRagResponse)
async def create_new_rag_project_DEPR(request: NewRagRequest, background_tasks: BackgroundTasks):
    '''
    @INVESTIGATE: Should i use this to kill 2 birds with one stone? Using chords
    Endpoint to create both a RAG pipeline for AI notes in one call:
        - process_pdf_task:
            input: request.files, request.metadata
            returns: source_ids
        - process_pdf_task:
            input: source_ids
            returns: None
        
    Request contains
    project_id
        request.files (List): list of pdf file links
        request.metadata (json): {
            project_id:
        }
    '''
    try:
        # Create signatures for the tasks
        process_pdf_task_signature = process_pdf_task.s(request.files, request.metadata)        # Upload PDFs to AWS S3, and Supabase
        # validate_and_generate_audio_task_signature = validate_and_generate_audio_task.s(request.files, request.metadata)

        # Create the group
        task_group = group(
            process_pdf_task_signature,
            # validate_and_generate_audio_task_signature
        )

        # Create the chord with the group and callback
        task_chord = chord(task_group)(insert_project_documents_task.s())

        # The result of the chord is the AsyncResult of the callback task
        chord_result = task_chord

        # Get task IDs
        process_pdf_task_id = chord_result.parent.results[0].id
        # validate_and_generate_audio_task_id = chord_result.parent.results[1].id
        insert_task_id = chord_result.id

        # Return the task IDs to the client
        return {
            # "audio_task_id": validate_and_generate_audio_task_id,
            "embedding_task_id": process_pdf_task_id,
            "insert_task_id": insert_task_id,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/add-to-rag-project/", response_model=NewRagResponse)
async def append_sources_to_project(request: NewRagRequest, background_tasks: BackgroundTasks):
    '''
    @TODO HAVE NOT STARTED !!!
    Endpoint to add new sources to an existing RAG pipeline for AI notes in one call:
        - process_pdf_task:
            input: request.files, request.metadata
            returns: source_ids
        
    Request contains
    project_id
        request.files (List): list of pdf file links
        request.metadata (json): {
            project_id:
        }
    '''
    try:
        # Create signatures for the tasks
        process_pdf_task_signature = process_pdf_task.s(request.files, request.metadata)        # Upload PDFs to AWS S3, and Supabase
        # validate_and_generate_audio_task_signature = validate_and_generate_audio_task.s(request.files, request.metadata)

        # Create the group
        task_group = group(
            process_pdf_task_signature,
            # validate_and_generate_audio_task_signature
        )

        # Create the chord with the group and callback
        task_chord = chord(task_group)(insert_project_documents_task.s())

        # The result of the chord is the AsyncResult of the callback task
        chord_result = task_chord

        # Get task IDs
        process_pdf_task_id = chord_result.parent.results[0].id
        # validate_and_generate_audio_task_id = chord_result.parent.results[1].id
        insert_task_id = chord_result.id

        # Return the task IDs to the client
        return {
            # "audio_task_id": validate_and_generate_audio_task_id,
            "embedding_task_id": process_pdf_task_id,
            "insert_task_id": insert_task_id,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate-rag-note/")
async def rag_chat(request: RAGRequest):
    try:
        # Trigger the RAG task asynchronously and add it to the queue
        task = rag_chat_task.apply_async(args=[
            request.user_id,
            request.conversation_id,
            request.query,
            request.document_ids,
            request.project_id,
            request.model_type
        ])
        
        # Return the task ID to the client
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rag-chat/")
async def rag_chat(request: RAGRequest):
    try:
        # Trigger the RAG task asynchronously and add it to the queue
        task = rag_chat_task.apply_async(args=[
            request.user_id,
            request.conversation_id,
            request.query,
            request.document_ids,
            request.project_id
        ])
        
        # Return the task ID to the client
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/rag-chat-status/{task_id}")
async def get_rag_chat_status(task_id: str):
    """Endpoint to check the status and result of the RAG chat task."""
    result = AsyncResult(task_id)
    if result.state == "SUCCESS":
        return {"status": result.state, "result": result.result}
    elif result.state == "FAILURE":
        return {"status": result.state, "error": str(result.result)}
    else:
        return {"status": result.state}

# ================================================ #
#              PODCAST API ENDPOINTS
# ================================================ #

@app.post("/pdf-to-dialogue/", response_model=PDFResponse)
async def pdf_to_dialogue(request: PDFRequest, background_tasks: BackgroundTasks):
    '''
    Endpoint to create both a pdf/RAG pipeline and create an AI podcast in one call:
        - process_pdf_task 
        - validate_and_generate_audio_task
    '''
    try:
        # Create signatures for the tasks
        process_pdf_task_signature = process_pdf_task.s(request.files, request.metadata)
        validate_and_generate_audio_task_signature = validate_and_generate_audio_task.s(request.files, request.metadata)

        # Create the group
        task_group = group(
            process_pdf_task_signature,
            validate_and_generate_audio_task_signature
        )

        # Create the chord with the group and callback
        task_chord = chord(task_group)(insert_sources_media_association_task.s())

        # The result of the chord is the AsyncResult of the callback task
        chord_result = task_chord

        # Get task IDs
        process_pdf_task_id = chord_result.parent.results[0].id
        validate_and_generate_audio_task_id = chord_result.parent.results[1].id
        insert_task_id = chord_result.id

        # Return the task IDs to the client
        return {
            "audio_task_id": validate_and_generate_audio_task_id,
            "embedding_task_id": process_pdf_task_id,
            "insert_task_id": insert_task_id,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
# Endpoint to process PDFs and generate dialogue transcript
@app.post("/pdf-to-dialogue-transcript/", response_model=PDFResponse)
async def pdf_to_dialogue_transcript(request: PDFRequest, background_tasks: BackgroundTasks):
    try:
        # Enqueue the Celery task for dialogue generation
        print(f"API DEBUG: {request.files}")
        task = generate_dialogue_only_task.apply_async(
            args=[request.files]
        )
        
        # Return the task ID to the client
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Combined endpoint to check the status of any Celery task
@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    try:
        task_result = AsyncResult(task_id)

        # Retrieve task metadata
        task_meta = task_result.info if isinstance(task_result.info, dict) else {}
        start_time_str = task_meta.get('start_time')
        elapsed_time = None

        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str)
            elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        if task_result.state == 'PENDING':
            return {
                "task_id": task_id,
                "status": "PENDING",
                "elapsed_time": elapsed_time
            }
        elif task_result.state == 'SUCCESS':
            return {
                "task_id": task_id,
                "status": "SUCCESS",
                "result": task_result.result,
                "elapsed_time": elapsed_time
            }
        elif task_result.state == 'FAILURE':
            return {
                "task_id": task_id,
                "status": "FAILURE",
                "error": str(task_result.result),
                "elapsed_time": elapsed_time
            }
        else:
            return {
                "task_id": task_id,
                "status": task_result.state,
                "elapsed_time": elapsed_time
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


    except Exception as e:
        # Handle any unexpected exceptions
        raise HTTPException(status_code=500, detail=f"Error retrieving task status: {str(e)}")