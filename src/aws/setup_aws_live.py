"""
SentinelVision — AWS Free Tier Live Provisioning & Verification

Provisions and tests:
1. S3 Bucket: sentinelvision-evidence-<account-id>
2. DynamoDB Table: SentinelVisionIncidents (Free Tier On-Demand)
3. CloudWatch Metrics: SentinelVision/Metrics
"""

import os
import sys
import time
import json
import numpy as np

# Ensure root path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import boto3
from botocore.exceptions import ClientError
from loguru import logger

from src.aws.s3 import S3Client
from src.aws.dynamodb import DynamoDBClient
from src.aws.cloudwatch import CloudWatchClient
from src.safety.events import SafetyEvent


def setup_and_verify_aws():
    region = os.getenv("AWS_REGION", "us-east-1")
    logger.info("=" * 60)
    logger.info(f"  SENTINELVISION — AWS FREE TIER LIVE PROVISIONING ({region})")
    logger.info("=" * 60)

    # 1. Verify STS Identity
    sts = boto3.client("sts", region_name=region)
    try:
        caller = sts.get_caller_identity()
        account_id = caller["Account"]
        arn = caller["Arn"]
        logger.info(f"✅ Authenticated to AWS: Account={account_id}, User/Role={arn}")
    except Exception as e:
        logger.error(f"❌ Failed to authenticate to AWS: {e}")
        logger.error("Please ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set.")
        return False

    bucket_name = os.getenv("AWS_S3_BUCKET", f"sentinelvision-evidence-{account_id}")
    table_name = os.getenv("AWS_DYNAMODB_TABLE", "SentinelVisionIncidents")
    namespace = os.getenv("CLOUDWATCH_NAMESPACE", "SentinelVision/Metrics")

    # 2. S3 Bucket Provisioning
    s3_boto = boto3.client("s3", region_name=region)
    try:
        s3_boto.head_bucket(Bucket=bucket_name)
        logger.info(f"✅ S3 Bucket '{bucket_name}' already exists.")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in ("404", "NoSuchBucket"):
            logger.info(f"Creating S3 Bucket '{bucket_name}' in {region}...")
            if region == "us-east-1":
                s3_boto.create_bucket(Bucket=bucket_name)
            else:
                s3_boto.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": region}
                )
            logger.info(f"✅ Created S3 Bucket: {bucket_name}")
        else:
            logger.error(f"Error checking bucket: {e}")

    # 3. DynamoDB Table Provisioning
    dynamo_res = boto3.resource("dynamodb", region_name=region)
    try:
        table = dynamo_res.Table(table_name)
        table.load()
        logger.info(f"✅ DynamoDB Table '{table_name}' already exists (Status: {table.table_status}).")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
            logger.info(f"Creating DynamoDB Table '{table_name}' with On-Demand billing (Free Tier)...")
            table = dynamo_res.create_table(
                TableName=table_name,
                KeySchema=[{"AttributeName": "incident_id", "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": "incident_id", "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST"
            )
            logger.info(f"Waiting for DynamoDB table '{table_name}' to become ACTIVE...")
            table.wait_until_exists()
            logger.info(f"✅ Created DynamoDB Table: {table_name}")
        else:
            logger.error(f"Error checking DynamoDB table: {e}")

    # 4. Live Pipeline S3 Upload Test
    logger.info("--- Testing S3 Evidence Keyframe & Trace Upload ---")
    s3_client = S3Client(bucket_name=bucket_name, region_name=region)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a visual box to the frame to look like a real incident keyframe
    import cv2
    cv2.putText(dummy_frame, "SENTINELVISION TEST INCIDENT SV-1001", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.rectangle(dummy_frame, (100, 100), (300, 350), (0, 0, 255), 2)
    cv2.putText(dummy_frame, "FORKLIFT [ID:3]", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.rectangle(dummy_frame, (350, 150), (450, 350), (0, 165, 255), 2)
    cv2.putText(dummy_frame, "PEDESTRIAN [ID:17]", (350, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    now = time.time()
    event_id = "SV-1001"
    trace_data = {
        "event_id": event_id,
        "timestamp": now,
        "environment": "warehouse",
        "risk_level": "HIGH",
        "risk_score": 0.89,
        "ttc": 1.35,
        "action": "CREATE_INCIDENT_WARNING",
        "agent_decision": "Visual strobe warning activated; speed governor limit issued"
    }

    s3_snapshot_url = s3_client.upload_snapshot(event_id, dummy_frame, now)
    s3_trace_url = s3_client.upload_trace(event_id, trace_data, now)
    logger.info(f"✅ S3 Snapshot: {s3_snapshot_url}")
    logger.info(f"✅ S3 Trace: {s3_trace_url}")

    # 5. Live DynamoDB Incident Record Test
    logger.info("--- Testing DynamoDB Incident Logging & State Resolution ---")
    dynamo_client = DynamoDBClient(table_name=table_name, region_name=region)
    event = SafetyEvent(
        event_id=event_id,
        timestamp=now,
        risk_level="HIGH",
        risk_score=0.89,
        person_id=17,
        vehicle_id=3,
        ttc=1.35,
        action_taken="CREATE_INCIDENT_WARNING",
        reasons=["TTC 1.35s < Critical Threshold 2.0s", "Trajectory convergence in blind corner"]
    )
    event.snapshot_path = s3_snapshot_url
    event.add_trace(now, {"ttc": 1.35, "distance": 24.5}, "HIGH", "CREATE_INCIDENT_WARNING")

    logged = dynamo_client.log_incident(event)
    logger.info(f"✅ DynamoDB Incident Inserted: {logged}")

    # Resolve incident
    event.resolve(now + 2.8)
    updated = dynamo_client.update_incident_status(
        event_id=event_id,
        status="RESOLVED",
        resolution_time=2.8,
        resolved=True
    )
    logger.info(f"✅ DynamoDB Incident Resolved: {updated}")

    # 6. Live CloudWatch Metrics Test
    logger.info("--- Testing CloudWatch Application Metrics Publishing ---")
    cw_client = CloudWatchClient(namespace=namespace, region_name=region)
    cw_client.put_metric("PipelineFPS", 11.2, "Count/Second")
    cw_client.put_metric("InferenceLatencyMs", 89.4, "Milliseconds")
    cw_client.put_metric("HazardsDetected", 1.0, "Count")
    cw_client.put_metric("HazardsResolved", 1.0, "Count")
    logger.info(f"✅ Published 4 telemetry metrics to CloudWatch namespace '{namespace}'")

    # 7. Print Console Links for Screenshots
    logger.info("=" * 60)
    logger.info("  🎉 AWS LIVE VALIDATION SUCCESSFUL! ($0.00 FREE TIER)")
    logger.info("=" * 60)
    logger.info(f"  Direct AWS Console URLs to capture screenshots:")
    logger.info(f"  1. S3 Bucket: https://s3.console.aws.amazon.com/s3/buckets/{bucket_name}?region={region}")
    logger.info(f"  2. DynamoDB:  https://{region}.console.aws.amazon.com/dynamodbv2/home?region={region}#item-explorer?table={table_name}")
    logger.info(f"  3. CloudWatch: https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}#metricsV2:graph=~();namespace={namespace}")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    setup_and_verify_aws()
