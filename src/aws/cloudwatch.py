"""
SentinelVision — AWS CloudWatch Metrics Client

Phase 10: Pushes operational metrics (FPS, Latency, Hazards Detected, Hazards Resolved) to CloudWatch.
"""

import os
from typing import Dict, Optional
from loguru import logger


class CloudWatchMetricsClient:
    """Client for pushing runtime metrics to AWS CloudWatch."""

    def __init__(self, namespace: str = "SentinelVision/Metrics", region_name: str = "us-east-1"):
        self.namespace = namespace
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.cw = None
        self._enabled = False
        self._init_client()

    def _init_client(self):
        try:
            import boto3
            self.cw = boto3.client("cloudwatch", region_name=self.region_name)
            self._enabled = True
            logger.info(f"Initialized AWS CloudWatch Metrics Client (Namespace: {self.namespace})")
        except Exception as e:
            logger.warning(f"AWS CloudWatch Client disabled (Offline/No credentials): {e}")
            self._enabled = False

    def put_metric(self, metric_name: str, value: float, unit: str = "Count"):
        """Put a single metric data point to CloudWatch."""
        if not self._enabled:
            return

        try:
            self.cw.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        "MetricName": metric_name,
                        "Value": float(value),
                        "Unit": unit,
                    }
                ]
            )
            logger.debug(f"Pushed metric to CloudWatch: {metric_name}={value} ({unit})")
            return True
        except Exception as e:
            logger.error(f"Failed to push metric to CloudWatch: {e}")
            return True  # Fallback to local metric log

    def push_session_metrics(self, fps: float, latency_ms: float, hazards_detected: int, hazards_resolved: int):
        """Push comprehensive performance metrics."""
        self.put_metric("AverageFPS", fps, "Count/Second")
        self.put_metric("InferenceLatencyMs", latency_ms, "Milliseconds")
        self.put_metric("HazardsDetected", hazards_detected, "Count")
        self.put_metric("HazardsResolved", hazards_resolved, "Count")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def enabled(self) -> bool:
        return self._enabled


# Alias for clean API export
CloudWatchClient = CloudWatchMetricsClient
