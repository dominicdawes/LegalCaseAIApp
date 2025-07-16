"""
This creates and configures the celery_app instance. There are 2 versions, the localhost version
and the cloud hosted (Render) version.
"""

# celery_app.py
import logging
import sys
from celery import Celery
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

@after_setup_logger.connect
def setup_loggers(logger, **kwargs):
    """
    This function is triggered after Celery sets up its logger.
    We can add our own handlers here.
    """
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create stdout handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    # Also configure the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

@after_setup_task_logger.connect
def setup_task_loggers(logger, **kwargs):
    """
    Configure task-specific loggers
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Optional: Load configuration from a separate config file or object
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],  # Specify content types to accept
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,  # Added setting

    # NEW: Async support configuration for Celery 5.4
    task_protocol=2,  # Enable async task protocol
    # worker_pool='gevent',  # Use gevent for async compatibility
    # worker_concurrency=10,  # 2 to 200 Adjust based on your server capacity
    worker_prefetch_multiplier=1,  # Good for async tasks
    
    # Additional async-friendly settings
    task_acks_late=True,  # Acknowledge tasks after completion
    worker_disable_rate_limits=False,  # Keep rate limits for API calls
    task_reject_on_worker_lost=True,  # Reject tasks if worker dies

    # CRITICAL: Logging configuration for Render
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
    worker_hijack_root_logger=False,  # Don't hijack root logger
)

# Import tasks to register them with Celery (THIS WORKS!!!)
# import tasks.generate_tasks
import tasks.chat_tasks
import tasks.chat_streaming_tasks
import tasks.upload_tasks
import tasks.sample_tasks


# # Optional: ASK GPT ABOUT IT'S PURPOSE: Automatically discover tasks in specified modules
# # This allows Celery to find tasks in modules like `generate_tasks.py` and `other_tasks.py`
celery_app.autodiscover_tasks(["tasks"])

# Sanity check print statement - these should show up in Render logs
print("🚀 Celery app initialized successfully!")
print("📋 Registered tasks:")
print(list(celery_app.tasks.keys()))
print("✅ Ready to process tasks")
