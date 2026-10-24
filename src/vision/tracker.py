"""
SentinelVision — Object Tracker

Phase 3: Persistent object tracking across frames.
Assigns consistent IDs so we can build motion history.
Uses IoU-based assignment (no deep features needed for MVP).
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from loguru import logger
from src.vision.detector import Detection


@dataclass
class TrackedObject:
    """An object being tracked across frames."""
    track_id: int
    class_name: str
    bbox: List[float]               # current [x1,y1,x2,y2]
    center: List[float]             # current [cx,cy]
    confidence: float
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0])
    position_history: List[List[float]] = field(default_factory=list)
    timestamp_history: List[float] = field(default_factory=list)
    age: int = 0                    # frames since creation
    hits: int = 1                   # frames with a matched detection
    misses: int = 0                 # consecutive frames without match
    first_seen: float = 0.0
    last_seen: float = 0.0
    is_confirmed: bool = False

    def update_position(self, center: List[float], bbox: List[float],
                        confidence: float, timestamp: float):
        """Update with new detection."""
        # Calculate velocity from last position
        if len(self.position_history) > 0:
            last = self.position_history[-1]
            dt = timestamp - self.last_seen if self.last_seen > 0 else 1/30
            if dt > 0:
                self.velocity = [
                    (center[0] - last[0]) / dt,
                    (center[1] - last[1]) / dt,
                ]

        self.center = center
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = timestamp
        self.hits += 1
        self.misses = 0
        self.age += 1

        # Store history (capped)
        self.position_history.append(center.copy())
        self.timestamp_history.append(timestamp)
        if len(self.position_history) > 120:
            self.position_history.pop(0)
            self.timestamp_history.pop(0)

    def mark_missed(self):
        """Called when no detection matched this track."""
        self.misses += 1
        self.age += 1

    @property
    def speed(self) -> float:
        """Magnitude of velocity in pixels/second."""
        return np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)

    def to_dict(self) -> dict:
        return {
            "id": self.track_id,
            "class": self.class_name,
            "position": self.center,
            "velocity": self.velocity,
            "speed": round(self.speed, 2),
            "confidence": round(self.confidence, 3),
            "age": self.age,
        }


def _iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class Tracker:
    """
    Simple IoU-based multi-object tracker.

    Matches detections to existing tracks using IoU.
    Creates new tracks for unmatched detections.
    Removes tracks that have been lost for too long.
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: Dict[int, TrackedObject] = {}
        self._next_id = 1

    def update(self, detections: List[Detection], timestamp: float = 0.0) -> List[TrackedObject]:
        """
        Update tracks with new detections.
        Returns list of confirmed tracked objects.
        """
        if not detections and not self._tracks:
            return []

        # Build IoU cost matrix
        track_ids = list(self._tracks.keys())
        tracks = [self._tracks[tid] for tid in track_ids]

        if tracks and detections:
            iou_matrix = np.zeros((len(tracks), len(detections)))
            for t_idx, track in enumerate(tracks):
                for d_idx, det in enumerate(detections):
                    iou_matrix[t_idx, d_idx] = _iou(track.bbox, det.bbox)

            # Greedy assignment (simple but effective for our use case)
            matched_tracks = set()
            matched_dets = set()
            assignments = []

            # Sort by IoU descending
            indices = np.unravel_index(
                np.argsort(iou_matrix, axis=None)[::-1], iou_matrix.shape
            )
            for t_idx, d_idx in zip(indices[0], indices[1]):
                if t_idx in matched_tracks or d_idx in matched_dets:
                    continue
                if iou_matrix[t_idx, d_idx] < self.iou_threshold:
                    break
                assignments.append((t_idx, d_idx))
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)

            # Update matched tracks
            for t_idx, d_idx in assignments:
                tid = track_ids[t_idx]
                det = detections[d_idx]
                self._tracks[tid].update_position(
                    det.center, det.bbox, det.confidence, timestamp
                )
                if self._tracks[tid].hits >= self.min_hits:
                    self._tracks[tid].is_confirmed = True

            # Mark unmatched tracks
            for t_idx in range(len(tracks)):
                if t_idx not in matched_tracks:
                    tid = track_ids[t_idx]
                    self._tracks[tid].mark_missed()

            # Create new tracks for unmatched detections
            for d_idx in range(len(detections)):
                if d_idx not in matched_dets:
                    self._create_track(detections[d_idx], timestamp)
        elif detections:
            # No existing tracks — create all
            for det in detections:
                self._create_track(det, timestamp)
        else:
            # No detections — mark all missed
            for tid in track_ids:
                self._tracks[tid].mark_missed()

        # Remove dead tracks
        dead = [tid for tid, t in self._tracks.items() if t.misses > self.max_age]
        for tid in dead:
            del self._tracks[tid]

        # Return confirmed tracks
        return [t for t in self._tracks.values() if t.is_confirmed]

    def _create_track(self, det: Detection, timestamp: float):
        """Create a new track from a detection."""
        track = TrackedObject(
            track_id=self._next_id,
            class_name=det.class_name,
            bbox=det.bbox,
            center=det.center,
            confidence=det.confidence,
            position_history=[det.center.copy()],
            timestamp_history=[timestamp],
            first_seen=timestamp,
            last_seen=timestamp,
        )
        self._tracks[self._next_id] = track
        self._next_id += 1

    @property
    def active_count(self) -> int:
        return len([t for t in self._tracks.values() if t.is_confirmed])

    @property
    def all_tracks(self) -> List[TrackedObject]:
        return list(self._tracks.values())

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        return self._tracks.get(track_id)

    def reset(self):
        self._tracks.clear()
        self._next_id = 1
