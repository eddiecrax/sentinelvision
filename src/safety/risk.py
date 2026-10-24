"""
SentinelVision — Risk Engine

Phase 6: Multi-factor risk scoring.
Combines TTC, distance, velocity, trajectory, and persistence.
"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from loguru import logger

from src.prediction.collision import CollisionRisk


@dataclass
class RiskAssessment:
    """Complete risk assessment for a collision pair."""
    person_id: int
    vehicle_id: int
    level: str                  # LOW, MEDIUM, HIGH, CRITICAL
    score: float                # 0.0 to 1.0
    reasons: List[str] = field(default_factory=list)
    ttc: float = float('inf')
    distance: float = 0.0
    persistence: float = 0.0    # seconds this risk has persisted
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "vehicle_id": self.vehicle_id,
            "level": self.level,
            "score": round(self.score, 3),
            "reasons": self.reasons,
            "ttc": round(self.ttc, 2) if self.ttc != float('inf') else None,
            "distance": round(self.distance, 1),
            "persistence": round(self.persistence, 2),
        }


class RiskEngine:
    """
    Multi-factor risk scoring engine.

    Risk = weighted combination of:
        - TTC score
        - Distance score
        - Closing velocity score
        - Trajectory convergence
        - Persistence (how long risk has been elevated)
    """

    def __init__(self, weights=None, thresholds=None):
        self.weights = weights or {
            "ttc": 0.35, "distance": 0.20, "velocity": 0.15,
            "trajectory": 0.15, "persistence": 0.10, "zone": 0.05,
        }
        self.thresholds = thresholds or {
            "low": 0.25, "medium": 0.50, "high": 0.75,
        }
        # Track persistence: key = (person_id, vehicle_id)
        self._persistence: Dict[tuple, float] = {}
        self._last_update: Dict[tuple, float] = {}

    def evaluate(self, collision_risks: List[CollisionRisk],
                 timestamp: float = 0.0) -> List[RiskAssessment]:
        """Evaluate risk for all collision pairs."""
        assessments = []
        active_pairs = set()

        for cr in collision_risks:
            pair_key = (cr.person_id, cr.vehicle_id)
            active_pairs.add(pair_key)

            # Update persistence
            if pair_key in self._last_update:
                self._persistence[pair_key] = (
                    self._persistence.get(pair_key, 0.0)
                    + (timestamp - self._last_update[pair_key])
                )
            else:
                self._persistence[pair_key] = 0.0
            self._last_update[pair_key] = timestamp

            assessment = self._score(cr, timestamp)
            assessments.append(assessment)

        # Clean up old pairs
        dead = [k for k in self._persistence if k not in active_pairs]
        for k in dead:
            del self._persistence[k]
            del self._last_update[k]

        assessments.sort(key=lambda a: a.score, reverse=True)
        return assessments

    def _score(self, cr: CollisionRisk, timestamp: float) -> RiskAssessment:
        """Compute multi-factor risk score for a single pair."""
        reasons = []
        scores = {}

        # TTC score (0-1, higher = more dangerous)
        if cr.ttc == float('inf') or cr.ttc > 10:
            scores["ttc"] = 0.0
        elif cr.ttc <= 0.5:
            scores["ttc"] = 1.0
            reasons.append("critical_ttc")
        else:
            scores["ttc"] = max(0, 1.0 - (cr.ttc / 10.0))
            if cr.ttc < 3.0:
                reasons.append("short_ttc")

        # Distance score (closer = higher risk)
        if cr.distance < 30:
            scores["distance"] = 1.0
            reasons.append("very_close_proximity")
        elif cr.distance < 100:
            scores["distance"] = 1.0 - (cr.distance - 30) / 70
            reasons.append("close_proximity")
        elif cr.distance < 300:
            scores["distance"] = max(0, 0.3 - (cr.distance - 100) / 666)
        else:
            scores["distance"] = 0.0

        # Closing velocity score
        if cr.closing_velocity > 100:
            scores["velocity"] = 1.0
            reasons.append("high_closing_speed")
        elif cr.closing_velocity > 10:
            scores["velocity"] = cr.closing_velocity / 100
        else:
            scores["velocity"] = 0.0

        # Trajectory convergence
        if cr.trajectory_intersects:
            scores["trajectory"] = min(1.0, 80 / max(cr.min_trajectory_distance, 1))
            reasons.append("trajectory_convergence")
        else:
            scores["trajectory"] = 0.0

        # Persistence
        pair_key = (cr.person_id, cr.vehicle_id)
        persistence = self._persistence.get(pair_key, 0.0)
        if persistence > 3.0:
            scores["persistence"] = min(1.0, persistence / 10.0)
            reasons.append("sustained_risk")
        else:
            scores["persistence"] = 0.0

        # Zone (placeholder — will integrate with profile zones later)
        scores["zone"] = 0.0

        # Weighted sum
        total = sum(
            scores.get(k, 0.0) * self.weights.get(k, 0.0)
            for k in self.weights
        )
        total = min(1.0, total)

        # Classify level
        if total >= self.thresholds["high"]:
            level = "CRITICAL"
        elif total >= self.thresholds["medium"]:
            level = "HIGH"
        elif total >= self.thresholds["low"]:
            level = "MEDIUM"
        else:
            level = "LOW"

        return RiskAssessment(
            person_id=cr.person_id,
            vehicle_id=cr.vehicle_id,
            level=level,
            score=total,
            reasons=reasons,
            ttc=cr.ttc,
            distance=cr.distance,
            persistence=persistence,
            timestamp=timestamp,
        )

    def reset(self):
        self._persistence.clear()
        self._last_update.clear()
