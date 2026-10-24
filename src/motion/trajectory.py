"""
SentinelVision — Trajectory Prediction

Phase 4: Linear trajectory extrapolation from current position + velocity.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass
class PredictedTrajectory:
    """Predicted future positions for a tracked object."""
    track_id: int
    points: List[List[float]] = field(default_factory=list)  # [[x,y], ...]
    timestamps: List[float] = field(default_factory=list)
    confidence: float = 1.0


def predict_trajectory(position: List[float],
                       velocity: List[float],
                       horizon: float = 2.0,
                       timestep: float = 0.1) -> PredictedTrajectory:
    """
    Predict future positions using linear extrapolation.

    p(t) = p0 + v * t

    Args:
        position: Current [x, y]
        velocity: Current [vx, vy] in pixels/second
        horizon: How far ahead to predict (seconds)
        timestep: Resolution of prediction (seconds)

    Returns:
        PredictedTrajectory with future positions
    """
    points = []
    timestamps = []

    steps = int(horizon / timestep)
    for i in range(1, steps + 1):
        t = i * timestep
        px = position[0] + velocity[0] * t
        py = position[1] + velocity[1] * t
        points.append([px, py])
        timestamps.append(t)

    return PredictedTrajectory(
        track_id=0,
        points=points,
        timestamps=timestamps,
    )


def trajectories_intersect(traj_a: PredictedTrajectory,
                           traj_b: PredictedTrajectory,
                           proximity_threshold: float = 50.0
                           ) -> Tuple[bool, float, float]:
    """
    Check if two predicted trajectories come within proximity_threshold.

    Returns:
        (intersects, min_distance, time_of_closest_approach)
    """
    min_dist = float('inf')
    min_time = 0.0

    n = min(len(traj_a.points), len(traj_b.points))
    for i in range(n):
        pa = traj_a.points[i]
        pb = traj_b.points[i]
        dist = np.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        if dist < min_dist:
            min_dist = dist
            min_time = traj_a.timestamps[i] if i < len(traj_a.timestamps) else 0.0

    return (min_dist < proximity_threshold, float(min_dist), float(min_time))
