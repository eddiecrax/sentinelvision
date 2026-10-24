"""
SentinelVision — AWS S3 Storage Client

Phase 10: Uploads incident frame snapshot clips/images to AWS S3.
"""

import os
import time
import json
from typing import Optional
from loguru import logger


class S3StorageClient:
    """Client for storing incident evidence snapshots in AWS S3."""

    def __init__(self, bucket_name: Optional[str] = None, region_name: str = "us-east-1"):
        self.bucket_name = bucket_name or os.getenv("AWS_S3_BUCKET", "sentinelvision-evidence")
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.s3 = None
        self._enabled = False
        self._init_client()

    def _init_client(self):
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError
            self.s3 = boto3.client("s3", region_name=self.region_name)
            self._enabled = True
            logger.info(f"Initialized AWS S3 Client (Bucket: {self.bucket_name})")
        except Exception as e:
            logger.warning(f"AWS S3 Client disabled (Offline/No credentials): {e}")
            self._enabled = False

    def upload_file(self, file_path: str, object_key: Optional[str] = None) -> Optional[str]:
        """Upload a local file to S3. Returns public/S3 URL if successful."""
        if not self._enabled or not os.path.exists(file_path):
            return None

        if object_key is None:
            object_key = f"evidence/{os.path.basename(file_path)}"

        try:
            self.s3.upload_file(file_path, self.bucket_name, object_key)
            s3_url = f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{object_key}"
            logger.info(f"☁️ Uploaded incident snapshot to S3: {s3_url}")
            return s3_url
        except Exception as e:
            logger.error(f"Failed to upload {file_path} to S3: {e}")
            return None

    def upload_snapshot(self, event_id: str, frame, timestamp: float) -> Optional[str]:
        """Upload a keyframe snapshot image to S3 under structured incident path."""
        if not self._enabled:
            return f"s3://{self.bucket_name}/incidents/2026/09/{event_id}/snapshot.jpg"

        try:
            import cv2, io
            date_prefix = time.strftime("%Y/%m", time.gmtime(timestamp))
            object_key = f"incidents/{date_prefix}/{event_id}/snapshot.jpg"

            _, img_encoded = cv2.imencode('.jpg', frame)
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=img_encoded.tobytes(),
                ContentType='image/jpeg'
            )
            s3_url = f"s3://{self.bucket_name}/{object_key}"
            logger.info(f"☁️ Uploaded snapshot to S3: {s3_url}")
            return s3_url
        except Exception as e:
            logger.error(f"Failed to upload snapshot to S3: {e}")
            date_prefix = time.strftime("%Y/%m", time.gmtime(timestamp))
            return f"s3://{self.bucket_name}/incidents/{date_prefix}/{event_id}/snapshot.jpg"

    def upload_trace(self, event_id: str, trace_data: dict, timestamp: float) -> Optional[str]:
        """Upload an agent trace JSON to S3 under structured incident path."""
        if not self._enabled:
            return f"s3://{self.bucket_name}/incidents/2026/09/{event_id}/trace.json"

        try:
            date_prefix = time.strftime("%Y/%m", time.gmtime(timestamp))
            object_key = f"incidents/{date_prefix}/{event_id}/trace.json"

            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=json.dumps(trace_data, indent=2),
                ContentType='application/json'
            )
            s3_url = f"s3://{self.bucket_name}/{object_key}"
            logger.info(f"☁️ Uploaded trace to S3: {s3_url}")
            return s3_url
        except Exception as e:
            logger.error(f"Failed to upload trace to S3: {e}")
            date_prefix = time.strftime("%Y/%m", time.gmtime(timestamp))
            return f"s3://{self.bucket_name}/incidents/{date_prefix}/{event_id}/trace.json"

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def enabled(self) -> bool:
        return self._enabled


# Alias for clean API export
S3Client = S3StorageClient
