# app/ws_handlers/handlers.py

import asyncio
import json
import logging
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from celery import chain
from celery.utils.log import get_task_logger


# Import your tasks and dependencies
from tasks.chat_tasks import persist_user_query, rag_chat_task
from .connection_manager import manager

# ——— Logging & Env Load ———————————————————————————————————————————————————————————

logger = get_task_logger(__name__)
logger.propagate = False

# ——— Main ———————————————————————————————————————————————————————————

async def listen_redis_and_forward(pubsub, session_id: str):
    """Listen to Redis pub/sub and forward to WebSocket with health checks"""
    reconnect_attempts = 0
    max_reconnect_attempts = 5
    
    try:
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    # Parse the Redis message
                    data = json.loads(message['data'])
                    
                    logger.info(f"📡 Redis message for session {session_id}: {data.get('type', 'unknown')}")
                    
                    # Forward to all connections in this session
                    await manager.send_to_session(session_id, data)
                    
                    # Reset reconnect attempts on successful message
                    reconnect_attempts = 0
                    
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Invalid JSON from Redis: {message['data']}")
                except Exception as e:
                    logger.error(f"❌ Error forwarding Redis message: {e}")
                    
            elif message['type'] == 'subscribe':
                logger.info(f"📡 Subscribed to Redis channel for session {session_id}")
                
    except asyncio.CancelledError:
        logger.info(f"🔄 Redis listener cancelled for session {session_id}")
        raise
    except Exception as e:
        logger.error(f"❌ Redis listener error for session {session_id}: {e}")
        reconnect_attempts += 1
        
        if reconnect_attempts < max_reconnect_attempts:
            logger.info(f"🔁 Attempting Redis reconnect {reconnect_attempts}/{max_reconnect_attempts}")
            await asyncio.sleep(min(2 ** reconnect_attempts, 30))  # Exponential backoff
            # Note: In production, you'd want to recreate the pubsub connection here
        else:
            logger.error(f"💥 Max Redis reconnect attempts reached for session {session_id}")
            raise

async def receive_websocket_messages(websocket: WebSocket, session_id: str, user_id: str):
    """Handle incoming WebSocket messages from client"""
    try:
        while True:
            # Receive message from WebSocket
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Invalid JSON from client: {message}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
                continue
            
            # Handle different message types
            message_type = data.get('type')
            logger.info(f"📥 Received message type: {message_type} from session {session_id}")
            
            if message_type == 'ping':
                # Respond to ping with pong
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
                logger.debug(f"🏓 Sent pong to session {session_id}")
                
            elif message_type == 'hello':
                # Client handshake (already handled in connect, but respond anyway)
                await websocket.send_text(json.dumps({
                    "type": "hello_ack", 
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }))
                
            elif message_type == 'chat_message':
                # Handle new chat message (could trigger RAG workflow)
                await handle_chat_message(data, session_id, user_id)
                
            elif message_type == 'subscribe_status':
                # Subscribe to additional channels (e.g., task status)
                await handle_status_subscription(data, session_id, user_id)
                
            else:
                logger.warning(f"⚠️ Unknown message type: {message_type}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }))
                
    except WebSocketDisconnect:
        logger.info(f"🔌 Client disconnected from session {session_id}")
    except asyncio.CancelledError:
        logger.info(f"🔄 Message receiver cancelled for session {session_id}")
    except Exception as e:
        logger.error(f"❌ Error receiving WebSocket message from session {session_id}: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Internal server error"
            }))
        except:
            pass  # Connection might be dead

async def handle_chat_message(data: dict, session_id: str, user_id: str):
    """Handle incoming chat message and trigger RAG workflow"""
    try:
        query = data.get('query')
        project_id = data.get('project_id')
        provider = data.get('provider', 'openai')
        model_name = data.get('model_name', 'gpt-4o-mini')
        temperature = data.get('temperature', 0.7)
        
        if not query or not project_id:
            await manager.send_to_session(session_id, {
                "type": "error",
                "message": "Missing query or project_id"
            })
            return
        
        # Send acknowledgment
        await manager.send_to_session(session_id, {
            "type": "message_received",
            "message": "Processing your query...",
            "timestamp": datetime.now().isoformat()
        })
        
        # Trigger RAG workflow
        job = chain(
            persist_user_query.s(
                user_id, session_id, query, project_id, model_name
            ),
            rag_chat_task.s(
                user_id, session_id, query, project_id, 
                provider, model_name, temperature
            )
        ).apply_async()
        
        # Send task started notification
        await manager.send_to_session(session_id, {
            "type": "task_started",
            "task_id": job.id,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"🚀 Started RAG workflow {job.id} for session {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Error handling chat message: {e}")
        await manager.send_to_session(session_id, {
            "type": "error",
            "message": f"Error processing message: {str(e)}"
        })

async def handle_status_subscription(data: dict, session_id: str, user_id: str):
    """Handle subscription to task status updates"""
    try:
        task_id = data.get('task_id')
        if task_id:
            # Subscribe to task-specific channel
            logger.info(f"📡 Subscribing to task status: {task_id}")
            # Your Celery tasks can publish to channels like: task_status:{task_id}
            await manager.send_to_session(session_id, {
                "type": "status_subscribed",
                "task_id": task_id,
                "timestamp": datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"❌ Error handling status subscription: {e}")