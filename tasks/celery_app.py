"""
This creates and configures the celery_app instance. There are 2 versions, the localhost version
and the cloud hosted (Render) version.
"""

# celery_app.py

from celery import Celery
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

# Optional: Load configuration from a separate config file or object
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],  # Specify content types to accept
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,  # Added setting
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

# Sanity check print statement
print("Registered tasks:")
print(celery_app.tasks.keys())
print(" ")
