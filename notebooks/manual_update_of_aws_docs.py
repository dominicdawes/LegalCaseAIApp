# $ python /c/Users/domin/Repos/LegalCaseAIApp/notebooks/manual_update_of_aws_docs.py

import boto3
from botocore.config import Config
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize S3 client
s3_client = boto3.client(
    's3',
    region_name='us-east-2',
    config=Config(signature_version='s3v4'),
    aws_access_key_id=os.getenv('AWS_IAM_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('AWS_IAM_SECRET')
)

# Declare S3 bucket name
bucket_name = os.getenv('AWS_S3_BUCKET_NAME')

def update_document_metadata(bucket, prefix='', dry_run=False):
    """
    Update metadata for all .doc and .docx files in the S3 bucket.
    
    Args:
        bucket: S3 bucket name
        prefix: Optional prefix/folder path (e.g., 'documents/')
        dry_run: If True, only shows what would be updated without making changes
    """
    try:
        print(f"{'[DRY RUN] ' if dry_run else ''}Scanning bucket: {bucket}")
        if prefix:
            print(f"Prefix filter: {prefix}")
        print("-" * 80)
        
        # List all objects in bucket
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        updated_count = 0
        skipped_count = 0
        error_count = 0
        
        for page in pages:
            if 'Contents' not in page:
                print("No objects found in bucket.")
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                file_lower = key.lower()
                
                # Check if file is .doc or .docx
                if file_lower.endswith('.docx') or file_lower.endswith('.doc'):
                    
                    # Determine content type based on extension
                    if file_lower.endswith('.docx'):
                        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        file_type = "DOCX"
                    else:  # .doc
                        content_type = "application/msword"
                        file_type = "DOC"
                    
                    # Extract filename for Content-Disposition
                    filename = key.split('/')[-1]
                    
                    print(f"\n📄 Found {file_type}: {key}")
                    
                    # Get current metadata to show what's changing
                    try:
                        current_obj = s3_client.head_object(Bucket=bucket, Key=key)
                        current_content_type = current_obj.get('ContentType', 'Not set')
                        current_disposition = current_obj.get('ContentDisposition', 'Not set')
                        
                        print(f"   Current ContentType: {current_content_type}")
                        print(f"   Current ContentDisposition: {current_disposition}")
                        print(f"   ➜ New ContentType: {content_type}")
                        print(f"   ➜ New ContentDisposition: inline; filename=\"{filename}\"")
                        
                    except Exception as e:
                        print(f"   ⚠️  Could not read current metadata: {e}")
                    
                    if dry_run:
                        print(f"   [DRY RUN] Would update this file")
                        updated_count += 1
                        continue
                    
                    # Update the metadata by copying object to itself
                    try:
                        s3_client.copy_object(
                            Bucket=bucket,
                            Key=key,
                            CopySource={'Bucket': bucket, 'Key': key},
                            ContentType=content_type,
                            ContentDisposition=f'inline; filename="{filename}"',
                            CacheControl="public, max-age=31536000",
                            MetadataDirective='REPLACE'
                        )
                        
                        print(f"   ✅ Successfully updated!")
                        updated_count += 1
                        
                    except Exception as e:
                        print(f"   ❌ Error updating: {e}")
                        error_count += 1
                        
                else:
                    skipped_count += 1
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Would update' if dry_run else 'Updated'}: {updated_count} files")
        print(f"Skipped (not .doc/.docx): {skipped_count} files")
        if error_count > 0:
            print(f"Errors: {error_count} files")
        print("=" * 80)
        
        if dry_run:
            print("\n💡 This was a dry run. Run with dry_run=False to apply changes.")
        else:
            print("\n✅ All done! Remember to invalidate CloudFront cache if needed.")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        raise

def invalidate_cloudfront_cache():
    """
    Helper function to invalidate CloudFront cache.
    Only call this after updating S3 metadata.
    """
    # Try to get distribution ID from environment
    cloudfront_distribution_id = os.getenv('CLOUDFRONT_DISTRIBUTION_ID')
    
    # Alternative: if you only have the domain, you could look it up
    # but it's better to just add the ID to your .env file
    
    if not cloudfront_distribution_id:
        print("\n⚠️  CLOUDFRONT_DISTRIBUTION_ID not found in environment variables")
        print("Add this to your .env file:")
        print("CLOUDFRONT_DISTRIBUTION_ID=your-distribution-id-here")
        print("\nYou'll need to manually invalidate the CloudFront cache via AWS Console:")
        print("1. Go to CloudFront Console")
        print("2. Select your distribution (d3bcbkrkidmsls.cloudfront.net)")
        print("3. Go to Invalidations tab")
        print("4. Create invalidation with path: /*")
        return
    
    try:
        cloudfront_client = boto3.client(
            'cloudfront',
            aws_access_key_id=os.getenv('AWS_IAM_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('AWS_IAM_SECRET')
        )
        
        import time
        caller_reference = f"doc-metadata-update-{int(time.time())}"
        
        response = cloudfront_client.create_invalidation(
            DistributionId=cloudfront_distribution_id,
            InvalidationBatch={
                'Paths': {
                    'Quantity': 1,
                    'Items': ['/*']  # Invalidate all paths
                },
                'CallerReference': caller_reference
            }
        )
        
        print(f"\n✅ CloudFront invalidation created: {response['Invalidation']['Id']}")
        print("   Cache will be cleared shortly (usually takes 5-15 minutes)")
        
    except Exception as e:
        print(f"\n❌ Error creating CloudFront invalidation: {e}")
        print("You may need to manually invalidate via AWS Console")

def remove_content_disposition_single_file(bucket, key, dry_run=False):
    """
    Removes ONLY the Content-Disposition field from a single S3 object
    by copying it over itself and preserving all other metadata.
    
    Args:
        bucket: S3 bucket name
        key: The full key (path) to the specific file
        dry_run: If True, only shows what would be updated
    """
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Attempting to update single file: {key}")
    
    try:
        # 1. Get all current metadata from the object
        current_obj = s3_client.head_object(Bucket=bucket, Key=key)
        
        # Store all existing metadata we want to preserve
        # We must read all of them to copy them back
        metadata_to_preserve = {
            'ContentType': current_obj.get('ContentType'),
            'CacheControl': current_obj.get('CacheControl'),
            'ContentEncoding': current_obj.get('ContentEncoding'),
            'ContentLanguage': current_obj.get('ContentLanguage'),
            'Expires': current_obj.get('Expires'),
            'Metadata': current_obj.get('Metadata', {}) # User-defined metadata
        }
        
        current_disposition = current_obj.get('ContentDisposition')
        print(f"  Current ContentType: {metadata_to_preserve['ContentType']}")
        print(f"  Current ContentDisposition: {current_disposition if current_disposition else 'Not set'}")
        
        if not current_disposition:
            print("  ✅ 'Content-Disposition' is already 'Not set'. No update needed.")
            return

        print(f"  ➜ Will REMOVE 'Content-Disposition'")
        
        if dry_run:
            print("  [DRY RUN] Would have removed 'Content-Disposition'.")
            return

        # 2. Prepare the parameters for the copy
        # We filter out any keys where the value is None, as copy_object API 
        # doesn't accept 'None' as a value.
        copy_params = {
            'Bucket': bucket,
            'Key': key,
            'CopySource': {'Bucket': bucket, 'Key': key},
            'MetadataDirective': 'REPLACE'
        }
        
        # Add all the metadata fields we want to keep
        for meta_key, meta_value in metadata_to_preserve.items():
            if meta_value is not None:
                copy_params[meta_key] = meta_value
        
        # 3. Perform the copy.
        # By *not* including 'ContentDisposition' in the copy_params,
        # it will be removed from the new version of the object.
        s3_client.copy_object(**copy_params)
        
        print(f"  ✅ Successfully removed 'Content-Disposition'!")

    except Exception as e:
        print(f"  ❌ Error updating {key}: {e}")

if __name__ == "__main__":
    print("=" * 80)
    print("S3 Document Metadata Updater for .doc and .docx files")
    print("=" * 80)
    
    # [Step 0] Option 1: DRY RUN FIRST (recommended) - see what would change
    # print("\n🔍 Running in DRY RUN mode first...")
    # update_document_metadata(bucket_name, dry_run=True)
    
    # # [Step 1] Uncomment below to actually run the updates
    # print("\n" + "=" * 80)
    # print("\n🚀 Now running ACTUAL UPDATE...")
    # update_document_metadata(bucket_name, dry_run=False)
    
    # # [Step 2] Uncomment below to invalidate CloudFront cache after updating
    # invalidate_cloudfront_cache()
    
    # If you only want to update files in a specific folder:
    # update_document_metadata(bucket_name, prefix='documents/', dry_run=False)

    # ------------ SINGLE FILE ------------------- #

    file_key_to_test = "964121d3-c6e3-4f8f-9a56-6727e708ffac/54cd4b7f-0fa9-4c0f-84b3-92479e86d4eb.doc"

    # If the dry run looks good, comment it out and run the real update:
    remove_content_disposition_single_file(bucket_name, file_key_to_test, dry_run=False)

    # [Step 2] After the update, invalidate the CloudFront cache
    # You MUST do this for the change to take effect
    invalidate_cloudfront_cache()