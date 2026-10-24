"""
SentinelVision — Safety Events

Phase 6: Incident creation, tracking, and resolution.
"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from loguru import logger

from src.safety.risk import RiskAssessment


@dataclass
class SafetyEvent:
    """A safety event/incident with auditable Agentic Vision timeline trace."""
    event_id: str
    timestamp: float
    risk_level: str
    risk_score: float
    person_id: int
    vehicle_id: int
    ttc: float
    action_taken: str
    reasons: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[float] = None
    escalated: bool = False
    timeline: List[dict] = field(default_factory=list)

    def resolve(self, timestamp: float):
        self.resolved = True
        self.resolution_time = timestamp - self.timestamp
        self.add_trace(timestamp, {"ttc": self.ttc, "resolved": True}, "RESOLVE", "Incident resolved")

    def add_trace(self, timestamp: float, vision: dict, decision: str, action: str):
        self.timeline.append({
            "timestamp": round(timestamp, 2),
            "vision": vision,
            "decision": decision,
            "action": action
        })

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 3),
            "person_id": self.person_id,
            "vehicle_id": self.vehicle_id,
            "ttc": round(self.ttc, 2) if self.ttc != float('inf') else None,
            "action": self.action_taken,
            "reasons": self.reasons,
            "resolved": self.resolved,
            "resolution_time": (round(self.resolution_time, 2)
                                if self.resolution_time else None),
            "escalated": self.escalated,
            "timeline": self.timeline,
        }


class EventManager:
    """Manages safety events lifecycle."""

    def __init__(self):
        self._events: List[SafetyEvent] = []
        self._active: Dict[tuple, SafetyEvent] = {}  # (person_id, vehicle_id)
        self._counter = 0

    def create_event(self, assessment: RiskAssessment,
                     action: str) -> SafetyEvent:
        """Create a new safety event."""
        self._counter += 1
        event_id = f"SV-{self._counter:04d}"

        event = SafetyEvent(
            event_id=event_id,
            timestamp=assessment.timestamp,
            risk_level=assessment.level,
            risk_score=assessment.score,
            person_id=assessment.person_id,
            vehicle_id=assessment.vehicle_id,
            ttc=assessment.ttc,
            action_taken=action,
            reasons=assessment.reasons.copy(),
        )

        pair_key = (assessment.person_id, assessment.vehicle_id)
        self._active[pair_key] = event
        self._events.append(event)

        # Initial trace record
        event.add_trace(
            timestamp=assessment.timestamp,
            vision={"ttc": assessment.ttc, "distance": assessment.distance, "risk": assessment.level},
            decision=assessment.level,
            action=action
        )

        logger.warning(
            f"🚨 EVENT {event_id}: {assessment.level} | "
            f"Person #{assessment.person_id} ↔ Vehicle #{assessment.vehicle_id} | "
            f"TTC={assessment.ttc:.1f}s | Action={action}"
        )
        return event

    def resolve_event(self, person_id: int, vehicle_id: int,
                      timestamp: float) -> Optional[SafetyEvent]:
        """Resolve an active event."""
        pair_key = (person_id, vehicle_id)
        event = self._active.pop(pair_key, None)
        if event and not event.resolved:
            event.resolve(timestamp)
            logger.info(
                f"✅ RESOLVED {event.event_id} in {event.resolution_time:.1f}s"
            )
        return event

    def escalate_event(self, person_id: int, vehicle_id: int) -> Optional[SafetyEvent]:
        pair_key = (person_id, vehicle_id)
        event = self._active.get(pair_key)
        if event:
            event.escalated = True
            logger.critical(f"🔴 ESCALATED {event.event_id}")
        return event

    def get_active(self, person_id: int, vehicle_id: int) -> Optional[SafetyEvent]:
        return self._active.get((person_id, vehicle_id))

    def has_active(self, person_id: int, vehicle_id: int) -> bool:
        return (person_id, vehicle_id) in self._active

    @property
    def active_events(self) -> List[SafetyEvent]:
        return list(self._active.values())

    @property
    def all_events(self) -> List[SafetyEvent]:
        return self._events

    @property
    def stats(self) -> dict:
        total = len(self._events)
        resolved = sum(1 for e in self._events if e.resolved)
        escalated = sum(1 for e in self._events if e.escalated)
        return {
            "total": total,
            "active": len(self._active),
            "resolved": resolved,
            "escalated": escalated,
        }
