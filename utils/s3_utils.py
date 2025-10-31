import os
import uuid
import boto3
from botocore.exceptions import ClientError
from botocore.client import Config
from boto3.s3.transfer import TransferConfig
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone


# Load environment variables
load_dotenv()

# Initialize S3 client
s3_client = boto3.client(
    's3',
    region_name='us-east-2',  # Specify your bucket's region
    config=Config(signature_version='s3v4'),                # S3 client to use Signature Version 4, you align with AWS's required authentication mechanism
    aws_access_key_id=os.getenv('AWS_IAM_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('AWS_IAM_SECRET')
)

# Declare S3 bucket name
s3_bucket_name = os.getenv('AWS_S3_BUCKET_NAME')

def upload_to_s3(client, file_source, s3_object_key, bucket_name=s3_bucket_name, content_type=None):
    """
    [ENHANCED] Uploads a file to the S3 bucket from either (an auto-detected) file path or file-like object.
    
    Args:
        client: S3 client
        file_source: Either a file path (str) or file-like object (BytesIO, etc.)
        s3_object_key: S3 object key
        bucket_name: S3 bucket name
        content_type: Optional content type override
    
    Returns:
        str: S3 URL of uploaded file
    """
    try:
        # Calculate the Expires header (1 year from today)
        expires_date = (datetime.now(timezone.utc) + timedelta(days=365)).strftime('%a, %d %b %Y %H:%M:%S GMT')

        # Determine if we're dealing with a file path or file-like object
        is_file_like = hasattr(file_source, 'read') and hasattr(file_source, 'seek')
        
        # Auto-detect content type if not provided
        if content_type is None:
            # Get the file extension from either the file_source path or s3_object_key
            file_ext = (file_source.lower() if not is_file_like else s3_object_key.lower())
            
            if file_ext.endswith('.pdf'):
                content_type = "application/pdf"
            elif file_ext.endswith(('.txt', '.md')):
                content_type = "text/plain"
            elif file_ext.endswith(('.jpg', '.jpeg')):
                content_type = "image/jpeg"
            elif file_ext.endswith('.png'):
                content_type = "image/png"
            elif file_ext.endswith('.docx'):
                content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif file_ext.endswith('.doc'):
                content_type = "application/msword"
            else:
                content_type = "binary/octet-stream"

        # Extract filename from s3_object_key for Content-Disposition
        filename = s3_object_key.split('/')[-1]
        
        # Common upload arguments
        extra_args = {
            'ContentType': content_type,
            'ContentDisposition': f'inline; filename="{filename}"',  # inline prevents auto-download
            'CacheControl': "public, max-age=31536000",
            'Expires': expires_date
        }

        # Transfer config for multipart uploads
        transfer_config = TransferConfig(multipart_chunksize=8*1024*1024)

        if is_file_like:
            # Handle file-like objects (BytesIO, jpg, etc.)
            file_source.seek(0)  # Ensure we're at the beginning
            client.upload_fileobj(
                file_source,
                bucket_name,
                s3_object_key,
                ExtraArgs=extra_args,
                Config=transfer_config
            )
        else:
            # Handle file paths (original behavior)
            client.upload_file(
                file_source,
                bucket_name,
                s3_object_key,
                ExtraArgs=extra_args,
                Config=transfer_config
            )

        return f"https://{bucket_name}.s3.amazonaws.com/{s3_object_key}"
        
    except Exception as e:
        raise Exception(f"Failed to upload to S3: {e}")
    
    
# def upload_file_to_s3(client, file_path, s3_object_key, bucket_name=s3_bucket_name):
#     """
#     LEGACY: Version of function that uploads a file to the S3 bucket and returns the file URL.
#     """
#     try:

#         # Calculate the Expires header (1 year from today)
#         expires_date = (datetime.now(timezone.utc) + timedelta(days=365)).strftime('%a, %d %b %Y %H:%M:%S GMT')

#         # set pdf content type
#         content_type = "application/pdf" if file_path.endswith(".pdf") else "binary/octet-stream"

#         # Multipart uploads for large PDFs
#         client.upload_file(
#             file_path, 
#             bucket_name, 
#             s3_object_key,
#             ExtraArgs={
#                 'ContentType': content_type,                    # Set Content-Type metadata
#                 "ContentDisposition": "inline",                 # inline display for preview
#                 "CacheControl": "public, max-age=31536000",     # Cache for 1 year
#                 "Expires": expires_date
#             },
#             Config = TransferConfig(multipart_chunksize=8*1024*1024)   
#         )
#         return f"https://{bucket_name}.s3.amazonaws.com/{s3_object_key}"
#     except Exception as e:
#         raise Exception(f"Failed to upload to S3: {e}")

def generate_presigned_url(client, bucket_name, object_key, expiration=7200):
    """
    Generate a presigned URL to share an S3 object

    :param client: Boto3 S3 client
    :param bucket_name: string
    :param object_key: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """
    try:
        response = client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_key},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        print(f"Error generating presigned URL: {e}")
        return None

    # The response contains the presigned URL
    return response