"""
SentinelVision — Velocity Estimation

Phase 4: Smoothed velocity from position history.
"""

import numpy as np
from typing import List, Optional, Tuple


def compute_velocity(positions: List[List[float]],
                     timestamps: List[float],
                     window: int = 5) -> List[float]:
    """
    Compute smoothed velocity from recent position history.

    Uses a sliding window average to reduce noise.
    Returns [vx, vy] in pixels/second.
    """
    if len(positions) < 2 or len(timestamps) < 2:
        return [0.0, 0.0]

    n = min(window, len(positions))
    recent_pos = positions[-n:]
    recent_ts = timestamps[-n:]

    dt = recent_ts[-1] - recent_ts[0]
    if dt <= 0:
        return [0.0, 0.0]

    dx = recent_pos[-1][0] - recent_pos[0][0]
    dy = recent_pos[-1][1] - recent_pos[0][1]

    return [dx / dt, dy / dt]


def compute_speed(velocity: List[float]) -> float:
    """Magnitude of velocity vector."""
    return float(np.sqrt(velocity[0]**2 + velocity[1]**2))


def compute_direction(velocity: List[float]) -> float:
    """Direction angle in radians (0 = right, pi/2 = down)."""
    return float(np.arctan2(velocity[1], velocity[0]))


def is_stationary(speed: float, threshold: float = 0.5) -> bool:
    """Check if an object is effectively stationary."""
    return speed < threshold
