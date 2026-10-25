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
    person_class: str = "person"
    vehicle_class: str = "vehicle"
    source: str = ""
    person_center: List[float] = field(default_factory=list)
    vehicle_center: List[float] = field(default_factory=list)
    last_seen_timestamp: float = 0.0
    low_risk_start: Optional[float] = None

    def resolve(self, timestamp: float):
        self.resolved = True
        self.resolution_time = timestamp - self.timestamp
        self.add_trace(timestamp, {"ttc": self.ttc, "resolved": True}, "RESOLVE", "Incident resolved")

    def update_state(self, timestamp: float, assessment: RiskAssessment):
        """Update active event state with latest perception."""
        self.last_seen_timestamp = timestamp
        if assessment.level != "LOW":
            self.low_risk_start = None  # Reset low risk counter only when elevated risk resumes
        old_level = self.risk_level
        self.risk_level = assessment.level
        self.risk_score = max(self.risk_score, assessment.score)
        self.ttc = assessment.ttc
        if assessment.person_center:
            self.person_center = assessment.person_center
        if assessment.vehicle_center:
            self.vehicle_center = assessment.vehicle_center
        if assessment.level != old_level:
            self.add_trace(
                timestamp=timestamp,
                vision={"ttc": assessment.ttc, "distance": assessment.distance, "risk": assessment.level},
                decision=assessment.level,
                action=f"Risk transitioned {old_level} -> {assessment.level}"
            )

    def handover_track(self, new_person_id: int, new_center: List[float], timestamp: float, distance_px: float):
        """Reassociate track ID following tracker flicker / brief occlusion."""
        old_id = self.person_id
        self.person_id = new_person_id
        if new_center:
            self.person_center = new_center
        self.last_seen_timestamp = timestamp
        self.add_trace(
            timestamp=timestamp,
            vision={"ttc": self.ttc, "spatial_delta": round(distance_px, 1)},
            decision="TRACK_HANDOVER",
            action=f"Reassociated person track P#{old_id} -> P#{new_person_id} (offset {distance_px:.1f}px)"
        )
        logger.info(
            f"🔄 TRACK HANDOVER in {self.event_id}: P#{old_id} -> P#{new_person_id} (offset {distance_px:.1f}px)"
        )

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
            "person_class": self.person_class,
            "vehicle_class": self.vehicle_class,
            "source": self.source,
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
            person_class=getattr(assessment, "person_class", "person"),
            vehicle_class=getattr(assessment, "vehicle_class", "vehicle"),
            person_center=getattr(assessment, "person_center", []),
            vehicle_center=getattr(assessment, "vehicle_center", []),
            last_seen_timestamp=assessment.timestamp,
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

    def handover_event(self, old_pair: tuple, new_pair: tuple,
                       new_center: List[float], timestamp: float,
                       distance_px: float) -> Optional[SafetyEvent]:
        """Transfer an active incident tracking record to a new track ID upon flicker/occlusion."""
        event = self._active.pop(old_pair, None)
        if event:
            event.handover_track(new_pair[0], new_center, timestamp, distance_px)
            self._active[new_pair] = event
            return event
        return None

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
