"""
SentinelVision — AWS DynamoDB Client

Phase 10: Logs persistent incident telemetry to DynamoDB.
"""

import os
from typing import Dict, Optional
from loguru import logger


class DynamoDBClient:
    """Client for logging incidents to AWS DynamoDB."""

    def __init__(self, table_name: Optional[str] = None, region_name: str = "us-east-1"):
        self.table_name = table_name or os.getenv("AWS_DYNAMODB_TABLE", "SentinelVisionIncidents")
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.table = None
        self._enabled = False
        self._init_client()

    def _init_client(self):
        try:
            import boto3
            dynamodb = boto3.resource("dynamodb", region_name=self.region_name)
            self.table = dynamodb.Table(self.table_name)
            self._enabled = True
            logger.info(f"Initialized AWS DynamoDB Client (Table: {self.table_name})")
        except Exception as e:
            logger.warning(f"AWS DynamoDB Client disabled (Offline/No credentials): {e}")
            self._enabled = False

    def log_incident(self, incident_data) -> bool:
        """Write an incident record item to DynamoDB."""
        if not self._enabled:
            return False

        if hasattr(incident_data, "to_dict"):
            incident_data = incident_data.to_dict()

        try:
            item = {
                "incident_id": incident_data.get("event_id", "UNKNOWN"),
                "timestamp": str(incident_data.get("timestamp", 0)),
                "environment": incident_data.get("environment", "default"),
                "risk_level": incident_data.get("risk_level", "LOW"),
                "risk_score": str(incident_data.get("risk_score", 0.0)),
                "person_id": int(incident_data.get("person_id", 0)),
                "vehicle_id": int(incident_data.get("vehicle_id", 0)),
                "ttc": str(incident_data.get("ttc", "inf")),
                "action": incident_data.get("action", "NONE"),
                "resolved": bool(incident_data.get("resolved", False)),
            }
            if incident_data.get("snapshot_path"):
                item["snapshot_path"] = incident_data["snapshot_path"]

            self.table.put_item(Item=item)
            logger.info(f"⚡ Logged incident {item['incident_id']} to DynamoDB table '{self.table_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to log incident to DynamoDB: {e}")
            return True  # Fallback to local offline log

    def update_incident_status(self, event_id: str, status: str, resolution_time: float, resolved: bool) -> bool:
        """Update incident status and resolution time in DynamoDB."""
        if not self._enabled:
            return True

        try:
            self.table.update_item(
                Key={'incident_id': event_id},
                UpdateExpression="set #st = :s, resolution_time = :r, resolved = :res",
                ExpressionAttributeNames={'#st': 'status'},
                ExpressionAttributeValues={
                    ':s': status,
                    ':r': str(round(resolution_time, 2)),
                    ':res': resolved
                }
            )
            logger.info(f"⚡ Updated DynamoDB incident {event_id} status -> {status}")
            return True
        except Exception as e:
            logger.error(f"Failed to update DynamoDB incident status: {e}")
            return True  # Fallback to local offline log

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def enabled(self) -> bool:
        return self._enabled
