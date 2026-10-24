"""
SentinelVision — Motion History

Phase 4: Stores and manages position/velocity history for tracked objects.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class MotionSnapshot:
    """Single point in motion history."""
    position: List[float]    # [x, y]
    velocity: List[float]    # [vx, vy] pixels/sec
    speed: float             # magnitude
    timestamp: float


@dataclass
class ObjectMotion:
    """Complete motion record for one tracked object."""
    track_id: int
    class_name: str
    history: List[MotionSnapshot] = field(default_factory=list)

    def add(self, position, velocity, speed, timestamp):
        self.history.append(MotionSnapshot(
            position=list(position), velocity=list(velocity),
            speed=speed, timestamp=timestamp,
        ))
        # Keep last 120 snapshots
        if len(self.history) > 120:
            self.history.pop(0)

    @property
    def latest(self) -> Optional[MotionSnapshot]:
        return self.history[-1] if self.history else None

    @property
    def duration(self) -> float:
        if len(self.history) < 2:
            return 0.0
        return self.history[-1].timestamp - self.history[0].timestamp

    @property
    def positions(self) -> List[List[float]]:
        return [s.position for s in self.history]


class MotionHistory:
    """Manages motion history for all tracked objects."""

    def __init__(self):
        self._objects: Dict[int, ObjectMotion] = {}

    def update(self, track_id: int, class_name: str,
               position, velocity, speed: float, timestamp: float):
        if track_id not in self._objects:
            self._objects[track_id] = ObjectMotion(
                track_id=track_id, class_name=class_name
            )
        self._objects[track_id].add(position, velocity, speed, timestamp)

    def get(self, track_id: int) -> Optional[ObjectMotion]:
        return self._objects.get(track_id)

    def remove(self, track_id: int):
        self._objects.pop(track_id, None)

    def cleanup(self, active_ids: set):
        """Remove entries for tracks that are no longer active."""
        dead = [tid for tid in self._objects if tid not in active_ids]
        for tid in dead:
            del self._objects[tid]

    @property
    def all_objects(self) -> Dict[int, ObjectMotion]:
        return self._objects
