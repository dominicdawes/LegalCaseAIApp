"""
This creates and configures the celery_app instance. There are 2 versions, the localhost version
and the cloud hosted (Render) version.
"""

# celery_app.py
import logging
import sys
import asyncio
import threading
from celery import Celery
from celery.signals import worker_init, worker_shutdown
from celery.signals import after_setup_logger, after_setup_task_logger
from celery.app.log import TaskFormatter
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CLOUD_AMQP_ENDPOINT = os.getenv("CLOUDAMQP_PUBLIC_ENDPOINT", "amqp://guest@localhost//")
REDIS_LABS_ENDPOINT = (
    "redis://default:"
    + (os.getenv("REDIS_PASSWORD") or "")
    + "@"
    + (os.getenv("REDIS_PUBLIC_ENDPOINT") or "localhost:6379")
)

# <--- localhost version --->

# Initialize the Celery app
# celery_app = Celery(
#     'celery_app',
#     broker='amqps://btwzozrv:pcIervFsmCoKgcB2KtOSdNNHMJD7qWRJ@octopus.rmq3.cloudamqp.com/btwzozrv',
#     backend='redis://localhost:6380/0',  # Memurai instance as backend on port 6380
#     task_serializer='json',
#     result_serializer='json',
#     accept_content=['json'],  # Accept only JSON content
# )

# <--- cloud hosted version --->

# Initialize the Celery app
celery_app = Celery(
    "celery_app",
    broker=CLOUD_AMQP_ENDPOINT,  # AMQP instance url and password endpoint
    backend=REDIS_LABS_ENDPOINT,  # RedisLabs instance url and password endpoint
    task_serializer="json",
    result_serializer="pickle",
    accept_content=["json", "pickle"],  # Accept JSON and pickle content
)

# 🎛️ Toggle Controls from Environment Variables
SHOW_CELERY_LOGS = "true"   # os.getenv("SHOW_CELERY_LOGS", "true").lower() == "true"
SHOW_MANUAL_LOGS = "true"  # os.getenv("SHOW_MANUAL_LOGS", "true").lower() == "true"

# ——— Event Loop ———————————————————————————————————————————————————————————

def _run_event_loop(loop):
    """Run the event loop in a dedicated thread"""
    asyncio.set_event_loop(loop)
    loop.run_forever()

@worker_init.connect
def worker_init_handler(sender=None, **kwargs):
    """Initialize persistent event loop when worker starts"""
    global _worker_loop, _worker_thread
    
    # Create new event loop for this worker
    _worker_loop = asyncio.new_event_loop()
    
    # Run it in a dedicated thread
    _worker_thread = threading.Thread(
        target=_run_event_loop, 
        args=(_worker_loop,),
        daemon=True
    )
    _worker_thread.start()
    
    print(f"🔄 Worker {sender} initialized with persistent event loop")

@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Clean up event loop when worker shuts down"""
    global _worker_loop, _worker_thread
    
    if _worker_loop and not _worker_loop.is_closed():
        _worker_loop.call_soon_threadsafe(_worker_loop.stop)
        _worker_loop = None
    
    if _worker_thread and _worker_thread.is_alive():
        _worker_thread.join(timeout=5)
        _worker_thread = None
    
    print(f"🛑 Worker {sender} event loop cleaned up")

def run_async_in_worker(coro):
    """Run coroutine in the persistent worker event loop"""
    global _worker_loop
    
    if _worker_loop is None or _worker_loop.is_closed():
        raise RuntimeError("Worker event loop not initialized")
    
    # Submit coroutine to the persistent loop and wait for result
    future = asyncio.run_coroutine_threadsafe(coro, _worker_loop)
    return future.result()

# ——— Logging & Env Load ———————————————————————————————————————————————————————————

@after_setup_logger.connect
def setup_loggers(logger, **kwargs):
    """Configurable logger setup with toggles for different log types"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # 🎯 Toggle 1: Celery Built-in Logs (task lifecycle, worker events)
    if SHOW_CELERY_LOGS:
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        print("✅ Celery logs enabled")
    else:
        logger.setLevel(logging.WARNING)  # Only show warnings/errors
        print("🔇 Celery logs disabled")
    
    # 🎯 Toggle 2: Manual Application Logs (your emoji logs)
    if SHOW_MANUAL_LOGS:
        # Configure all your task module loggers
        task_modules = [
            'tasks.upload_tasks',
            'tasks.note_tasks', 
            'tasks.chat_tasks',
            'tasks.chat_streaming_tasks',
            'tasks.profile_tasks',
            'tasks.sample_tasks'
        ]
        
        for module_name in task_modules:
            module_logger = logging.getLogger(module_name)
            module_logger.addHandler(handler)
            module_logger.setLevel(logging.INFO)
            module_logger.propagate = True
        
        print("✅ Manual logs enabled")
    else:
        # Disable manual logs by setting high threshold
        for module_name in ['tasks.upload_tasks', 'tasks.note_tasks', 'tasks.chat_tasks', 'tasks.chat_streaming_tasks', 'tasks.sample_tasks']:
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(logging.ERROR)  # Only errors
        
        print("🔇 Manual logs disabled")

# 🎛️ Update Celery config to respect toggles
celery_config_updates = {
    "task_serializer": "json",
    "accept_content": ["json"],
    "result_serializer": "json",
    "timezone": "UTC",
    "enable_utc": True,
    "broker_connection_retry_on_startup": True,
    "task_protocol": 2,
    "worker_prefetch_multiplier": 1,
    "task_acks_late": True,
    "worker_disable_rate_limits": False,
    "task_reject_on_worker_lost": True,
}

# 🎯 Add Celery-specific log control
if not SHOW_CELERY_LOGS:
    celery_config_updates.update({
        "worker_log_format": "",  # Minimize worker logs
        "worker_task_log_format": "",  # Minimize task logs
        "worker_hijack_root_logger": False,
    })

celery_app.conf.update(**celery_config_updates)
# Import tasks to register them with Celery (THIS WORKS!!!)
# import tasks.generate_tasks
import tasks.chat_tasks
import tasks.chat_streaming_tasks
import tasks.upload_tasks_chains_0730
import tasks.upload_tasks
import tasks.note_tasks
import tasks.sample_tasks

# # Optional: ASK GPT ABOUT IT'S PURPOSE: Automatically discover tasks in specified modules
# # This allows Celery to find tasks in modules like `generate_tasks.py` and `other_tasks.py`
celery_app.autodiscover_tasks(["tasks"])

# Sanity check print statement - these should show up in Render logs
print("🚀 Celery app initialized successfully!")
print("📋 Registered tasks:")
print(list(celery_app.tasks.keys()))
print("✅ Ready to process tasks")


# ——— Module Exports ————————————————————————————————————————————————————————————————
__all__ = ['celery_app', 'run_async_in_worker']