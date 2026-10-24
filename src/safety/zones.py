"""
SentinelVision — Safety Zones

Phase 6: Define restricted/monitored areas in the camera view.
Placeholder for profile-based zone configuration.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Zone:
    """A defined safety zone in camera coordinates."""
    name: str
    zone_type: str              # "restricted", "vehicle_only", "pedestrian_only"
    polygon: List[List[int]]    # list of [x,y] vertices
    description: str = ""
    risk_boost: float = 0.0

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this zone using ray casting."""
        n = len(self.polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi)+xi):
                inside = not inside
            j = i
        return inside


class ZoneManager:
    """Manages safety zones for the current environment."""

    def __init__(self):
        self._zones: List[Zone] = []

    def add_zone(self, zone: Zone):
        self._zones.append(zone)

    def check_violations(self, track_id: int, class_name: str,
                         x: float, y: float) -> List[dict]:
        """Check if an object violates any zone rules."""
        violations = []
        for zone in self._zones:
            if not zone.contains_point(x, y):
                continue
            # Check if object type violates zone type
            is_person = class_name == "person"
            is_vehicle = class_name in {"car","truck","bus","motorcycle","bicycle"}

            if zone.zone_type == "vehicle_only" and is_person:
                violations.append({
                    "zone": zone.name,
                    "violation": "person_in_vehicle_zone",
                    "risk_boost": zone.risk_boost,
                })
            elif zone.zone_type == "pedestrian_only" and is_vehicle:
                violations.append({
                    "zone": zone.name,
                    "violation": "vehicle_in_pedestrian_zone",
                    "risk_boost": zone.risk_boost,
                })
            elif zone.zone_type == "restricted":
                violations.append({
                    "zone": zone.name,
                    "violation": f"{class_name}_in_restricted_zone",
                    "risk_boost": zone.risk_boost,
                })
        return violations

    @property
    def zones(self) -> List[Zone]:
        return self._zones
