"""
SentinelVision — Collision Analysis

Phase 5: Evaluate all pairwise person-vehicle interactions.
"""

import itertools
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from loguru import logger

from src.vision.tracker import TrackedObject
from src.prediction.ttc import compute_ttc, classify_ttc, TTCResult
from src.motion.trajectory import predict_trajectory, trajectories_intersect


@dataclass
class CollisionRisk:
    """Risk assessment for a specific object pair."""
    person_id: int
    vehicle_id: int
    person_class: str
    vehicle_class: str
    ttc: float
    distance: float
    closing_velocity: float
    is_converging: bool
    ttc_category: str           # LOW, MEDIUM, HIGH, CRITICAL
    trajectory_intersects: bool
    min_trajectory_distance: float
    time_of_closest: float
    person_center: List[float] = field(default_factory=list)
    vehicle_center: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "vehicle_id": self.vehicle_id,
            "ttc": round(self.ttc, 2) if self.ttc != float('inf') else None,
            "distance": round(self.distance, 1),
            "closing_velocity": round(self.closing_velocity, 1),
            "is_converging": self.is_converging,
            "category": self.ttc_category,
            "trajectory_intersects": self.trajectory_intersects,
        }


class CollisionAnalyzer:
    """Analyzes all person-vehicle pairs for collision risk."""

    def __init__(self, ttc_thresholds=None, proximity_threshold=80.0,
                 prediction_horizon=2.0):
        self.ttc_thresholds = ttc_thresholds or {"low": 5.0, "medium": 3.0, "high": 1.0}
        self.proximity_threshold = proximity_threshold
        self.prediction_horizon = prediction_horizon

    def analyze(self, tracked_objects: List[TrackedObject]) -> List[CollisionRisk]:
        """
        Evaluate all person-vehicle pairs.
        Returns list of CollisionRisk, sorted by TTC ascending.
        """
        persons = [t for t in tracked_objects if t.class_name == "person"]
        vehicles = [t for t in tracked_objects
                    if t.class_name in {"car","truck","bus","motorcycle","bicycle"}]

        risks = []
        for person in persons:
            for vehicle in vehicles:
                risk = self._evaluate_pair(person, vehicle)
                if risk is not None:
                    risks.append(risk)

        # Sort by TTC (most urgent first)
        risks.sort(key=lambda r: r.ttc if r.ttc != float('inf') else 9999)
        return risks

    def _evaluate_pair(self, person: TrackedObject,
                       vehicle: TrackedObject) -> CollisionRisk:
        """Evaluate collision risk for one person-vehicle pair."""
        ttc_result = compute_ttc(
            person.center, person.velocity,
            vehicle.center, vehicle.velocity,
        )
        ttc_result.track_id_a = person.track_id
        ttc_result.track_id_b = vehicle.track_id

        # Trajectory intersection check
        traj_p = predict_trajectory(
            person.center, person.velocity, horizon=self.prediction_horizon
        )
        traj_v = predict_trajectory(
            vehicle.center, vehicle.velocity, horizon=self.prediction_horizon
        )
        intersects, min_dist, t_closest = trajectories_intersect(
            traj_p, traj_v, proximity_threshold=self.proximity_threshold
        )

        category = classify_ttc(ttc_result.ttc, self.ttc_thresholds)

        return CollisionRisk(
            person_id=person.track_id,
            vehicle_id=vehicle.track_id,
            person_class=person.class_name,
            vehicle_class=vehicle.class_name,
            ttc=ttc_result.ttc,
            distance=ttc_result.distance,
            closing_velocity=ttc_result.closing_velocity,
            is_converging=ttc_result.is_converging,
            ttc_category=category,
            trajectory_intersects=intersects,
            min_trajectory_distance=min_dist,
            time_of_closest=t_closest,
            person_center=list(person.center),
            vehicle_center=list(vehicle.center),
        )
