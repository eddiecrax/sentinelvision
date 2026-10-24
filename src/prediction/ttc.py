"""
SentinelVision — Time-to-Collision (TTC)

Phase 5: Estimate when two objects will collide.
TTC = distance / closing_velocity
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TTCResult:
    """Time-to-collision result for an object pair."""
    track_id_a: int
    track_id_b: int
    ttc: float                  # seconds (inf = not converging)
    distance: float             # current distance in pixels
    closing_velocity: float     # pixels/second (positive = approaching)
    is_converging: bool         # are they getting closer?
    min_predicted_distance: float = float('inf')
    time_of_closest: float = 0.0


def compute_ttc(pos_a: List[float], vel_a: List[float],
                pos_b: List[float], vel_b: List[float]) -> TTCResult:
    """
    Compute time-to-collision between two objects.

    Uses relative position and velocity:
        TTC ≈ distance / closing_velocity

    Returns TTCResult with ttc=inf if objects are diverging.
    """
    # Relative position: B relative to A
    rel_pos = [pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]]
    distance = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)

    # Relative velocity: how fast they're approaching each other
    rel_vel = [vel_b[0] - vel_a[0], vel_b[1] - vel_a[1]]

    # Closing velocity = component of relative velocity along the line connecting them
    if distance > 0:
        # Unit vector from A to B
        unit = [rel_pos[0] / distance, rel_pos[1] / distance]
        # Dot product: negative = approaching
        closing = -(rel_vel[0] * unit[0] + rel_vel[1] * unit[1])
    else:
        closing = 0.0

    is_converging = closing > 0.5  # small threshold to avoid noise

    if is_converging and closing > 0:
        ttc = distance / closing
    else:
        ttc = float('inf')

    return TTCResult(
        track_id_a=0,
        track_id_b=0,
        ttc=ttc,
        distance=distance,
        closing_velocity=closing,
        is_converging=is_converging,
    )


def classify_ttc(ttc: float, thresholds: dict = None) -> str:
    """
    Classify TTC into risk category.

    Default thresholds:
        > 5s  → LOW
        3-5s  → MEDIUM
        1-3s  → HIGH
        < 1s  → CRITICAL
    """
    if thresholds is None:
        thresholds = {"low": 5.0, "medium": 3.0, "high": 1.0}

    if ttc == float('inf') or ttc > thresholds["low"]:
        return "LOW"
    elif ttc > thresholds["medium"]:
        return "MEDIUM"
    elif ttc > thresholds["high"]:
        return "HIGH"
    else:
        return "CRITICAL"
