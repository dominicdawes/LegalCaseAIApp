import os
import uuid
import boto3
from botocore.exceptions import ClientError
from botocore.client import Config
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# CloudFront domain name (replace with actual CloudFront domain or use an environment variable)
# cloudfront_domain = os.getenv('AWS_CLOUDFRONT_DOMAIN_TEST')
cloudfront_domain = os.getenv('AWS_CLOUDFRONT_DOMAIN_PROD')

def get_cloudfront_url(object_key, domain=cloudfront_domain):
    """
    Generates a CloudFront URL for a given S3 object.
    
    Parameters:
        object_key (str): The key (path) of the object in the S3 bucket.
        domain (str): The CloudFront distribution domain name.
    
    Returns:
        str: The CloudFront URL for the object.
    """
    if not domain:
        raise ValueError("CloudFront domain is not configured.")
    
    return f"https://{domain}/{object_key}"