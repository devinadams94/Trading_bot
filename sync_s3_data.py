#!/usr/bin/env python3
"""
Sync flat files to/from S3 (Massive.com)

This script helps you:
1. Upload local flat files to S3 for backup
2. Download flat files from S3 to local storage
3. Sync data between local and S3

Usage:
    # Upload local data to S3
    python3 sync_s3_data.py upload
    
    # Download data from S3 to local
    python3 sync_s3_data.py download
    
    # Sync (upload newer files)
    python3 sync_s3_data.py sync
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("❌ ERROR: boto3 not installed")
    print("   Install with: pip install boto3")
    sys.exit(1)


class S3DataSync:
    """Sync flat files with S3 storage"""
    
    def __init__(self):
        # Get S3 credentials from environment
        self.access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.endpoint_url = os.getenv('S3_ENDPOINT_URL')
        self.bucket_name = os.getenv('S3_BUCKET')
        
        if not all([self.access_key, self.secret_key, self.endpoint_url, self.bucket_name]):
            print("❌ ERROR: Missing S3 credentials in .env file")
            print("   Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT_URL, S3_BUCKET")
            sys.exit(1)
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        
        # Local data directory
        self.data_dir = Path('data/flat_files')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ S3 Client initialized")
        print(f"   Endpoint: {self.endpoint_url}")
        print(f"   Bucket: {self.bucket_name}")
        print(f"   Local dir: {self.data_dir}")
        print()
    
    def upload_file(self, local_path: Path, s3_key: str):
        """Upload a single file to S3"""
        try:
            self.s3_client.upload_file(
                str(local_path),
                self.bucket_name,
                s3_key
            )
            file_size = local_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✅ Uploaded: {s3_key} ({file_size:.2f} MB)")
            return True
        except ClientError as e:
            print(f"  ❌ Failed to upload {s3_key}: {e}")
            return False
    
    def download_file(self, s3_key: str, local_path: Path):
        """Download a single file from S3"""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                str(local_path)
            )
            file_size = local_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✅ Downloaded: {s3_key} ({file_size:.2f} MB)")
            return True
        except ClientError as e:
            print(f"  ❌ Failed to download {s3_key}: {e}")
            return False
    
    def list_s3_files(self, prefix: str = ''):
        """List all files in S3 bucket"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except ClientError as e:
            print(f"❌ Failed to list S3 files: {e}")
            return []
    
    def upload_all(self):
        """Upload all local flat files to S3"""
        print("=" * 80)
        print("📤 UPLOADING LOCAL FILES TO S3")
        print("=" * 80)
        print()
        
        # Find all parquet and csv files
        files = list(self.data_dir.rglob('*.parquet')) + list(self.data_dir.rglob('*.csv'))
        
        if not files:
            print("⚠️  No files found in local directory")
            return
        
        print(f"Found {len(files)} files to upload")
        print()
        
        success_count = 0
        for file_path in files:
            # Create S3 key (relative path from data_dir)
            s3_key = str(file_path.relative_to(self.data_dir))
            if self.upload_file(file_path, s3_key):
                success_count += 1
        
        print()
        print(f"✅ Uploaded {success_count}/{len(files)} files successfully")
    
    def download_all(self):
        """Download all files from S3 to local"""
        print("=" * 80)
        print("📥 DOWNLOADING FILES FROM S3")
        print("=" * 80)
        print()
        
        # List all files in S3
        s3_files = self.list_s3_files()
        
        if not s3_files:
            print("⚠️  No files found in S3 bucket")
            return
        
        print(f"Found {len(s3_files)} files in S3")
        print()
        
        success_count = 0
        for s3_key in s3_files:
            local_path = self.data_dir / s3_key
            if self.download_file(s3_key, local_path):
                success_count += 1
        
        print()
        print(f"✅ Downloaded {success_count}/{len(s3_files)} files successfully")


def main():
    parser = argparse.ArgumentParser(description='Sync flat files with S3')
    parser.add_argument(
        'action',
        choices=['upload', 'download', 'list'],
        help='Action to perform: upload (local→S3), download (S3→local), or list (show S3 files)'
    )
    
    args = parser.parse_args()
    
    # Initialize sync client
    sync = S3DataSync()
    
    # Perform action
    if args.action == 'upload':
        sync.upload_all()
    elif args.action == 'download':
        sync.download_all()
    elif args.action == 'list':
        print("=" * 80)
        print("📋 FILES IN S3 BUCKET")
        print("=" * 80)
        print()
        files = sync.list_s3_files()
        if files:
            for f in files:
                print(f"  • {f}")
            print()
            print(f"Total: {len(files)} files")
        else:
            print("  (empty)")


if __name__ == '__main__':
    main()

