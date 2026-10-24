"""
SentinelVision — Incident Persistence & Snapshot System

Phase 8: Stores persistent incident logs to JSON/DB and saves frame image snapshots
for high-risk events for audit trails and cloud sync.
"""

import os
import json
import cv2
import time
from typing import List, Dict, Optional
from dataclasses import asdict
from loguru import logger

from src.safety.events import SafetyEvent


class IncidentStorage:
    """Handles saving incident metadata and video/image evidence snapshots."""

    def __init__(self, incidents_file: str = "data/incidents.json",
                 snapshots_dir: str = "data/clips"):
        self.incidents_file = incidents_file
        self.snapshots_dir = snapshots_dir
        os.makedirs(os.path.dirname(self.incidents_file), exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        self._records = self._load_records()

    def _load_records(self) -> List[dict]:
        if os.path.exists(self.incidents_file):
            try:
                with open(self.incidents_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load incidents file: {e}")
                return []
        return []

    def record_incident(self, event: SafetyEvent, frame: Optional[cv2.Mat] = None,
                        environment: str = "default") -> dict:
        """
        Record a safety incident with optional frame snapshot saving.
        """
        snapshot_path = None
        if frame is not None:
            filename = f"snapshot_{event.event_id}_{int(event.timestamp * 1000)}.jpg"
            snapshot_path = os.path.join(self.snapshots_dir, filename)
            try:
                cv2.imwrite(snapshot_path, frame)
                logger.info(f"📸 Saved incident evidence snapshot: {snapshot_path}")
            except Exception as e:
                logger.error(f"Failed to save snapshot: {e}")

        data = event.to_dict()
        data["environment"] = environment
        data["snapshot_path"] = snapshot_path

        # Update existing record or append new
        existing_idx = next((i for i, r in enumerate(self._records)
                             if r["event_id"] == event.event_id), None)
        if existing_idx is not None:
            self._records[existing_idx] = data
        else:
            self._records.append(data)

        self._save()
        return data

    def _save(self):
        try:
            with open(self.incidents_file, 'w') as f:
                json.dump(self._records, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write incidents file: {e}")

    @property
    def all_incidents(self) -> List[dict]:
        return self._records
