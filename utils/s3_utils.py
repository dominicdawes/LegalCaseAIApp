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
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

# Declare S3 bucket name
s3_bucket_name = os.getenv('AWS_S3_BUCKET_NAME')

def upload_to_s3(client, file_path, s3_object_key, bucket_name=s3_bucket_name):
    """Uploads a file to the S3 bucket and returns the file URL."""
    try:

        # Calculate the Expires header (1 year from today)
        expires_date = (datetime.now(timezone.utc) + timedelta(days=365)).strftime('%a, %d %b %Y %H:%M:%S GMT')

        # set pdf content type
        content_type = "application/pdf" if file_path.endswith(".pdf") else "binary/octet-stream"

        # Multipart uploads for large PDFs
        client.upload_file(
            file_path, 
            bucket_name, 
            s3_object_key,
            ExtraArgs={
                'ContentType': content_type,                    # Set Content-Type metadata
                "ContentDisposition": "inline",                 # inline display for preview
                "CacheControl": "public, max-age=31536000",     # Cache for 1 year
                "Expires": expires_date
            },
            cfg = TransferConfig(multipart_chunksize=8*1024*1024)   
        )
        return f"https://{bucket_name}.s3.amazonaws.com/{s3_object_key}"
    except Exception as e:
        raise Exception(f"Failed to upload to S3: {e}")

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