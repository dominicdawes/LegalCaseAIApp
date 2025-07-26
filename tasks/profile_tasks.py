# tasks/profile_tasks.py

from celery import Celery
from celery.utils.log import get_task_logger
import httpx
import io
import uuid
from utils.s3_utils import upload_to_s3
from utils.cloudfront_utils import get_cloudfront_url
from utils.supabase_utils import supabase_client

from tasks.celery_app import celery_app

logger = get_task_logger(__name__)

@celery_app.task(bind=True)
def upload_profile_picture_task(self, user_id: str, image_url: str) -> dict:
    """Download a profile picture from URL, upload to S3, save CDN link to Supabase."""
    try:
        # Step 1: Download image to memory
        logger.info(f"📥 Downloading image from {image_url}")
        async_client = httpx.Client()
        response = async_client.get(image_url, timeout=15.0)
        response.raise_for_status()
        image_bytes = io.BytesIO(response.content)

        # Step 2: Generate clean filename
        extension = image_url.split(".")[-1].split("?")[0][:5]
        key = f"user_pfp/{user_id}_{uuid.uuid4().hex[:8]}.{extension}"

        # Step 3: Upload to S3 (in-memory)
        upload_to_s3(key, image_bytes, content_type="image/jpeg")  # adjust type as needed

        # Step 4: Generate CloudFront URL
        cdn_url = get_cloudfront_url(key)

        # Step 5: Persist in Supabase
        update_resp = (
            supabase_client
            .table("profiles")
            .update({"user_pfp_url": cdn_url})
            .eq("id", user_id)
            .execute()
        )
        logger.info(f"✅ Updated user_pfp for user_id={user_id}")

        return {
            "status": "success",
            "cdn_url": cdn_url
        }

    except Exception as e:
        logger.error(f"❌ Failed to process image: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }
