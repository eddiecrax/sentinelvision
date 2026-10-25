"""
SentinelVision — Agentic Action Loop

Phase 7: The agent that perceives, decides, acts, observes, and resolves/escalates.
This is the core differentiator — the agentic feedback loop.
"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from src.safety.risk import RiskAssessment
from src.safety.events import EventManager, SafetyEvent


class ActionType(Enum):
    MONITOR = "monitor"
    INCREASE_MONITORING = "increase_monitoring"
    VISUAL_WARNING = "visual_warning"
    AUDIO_WARNING = "audio_warning"
    CREATE_INCIDENT = "create_incident"
    ESCALATE = "escalate"
    RESOLVE = "resolve"


@dataclass
class AgentAction:
    """An action decided by the agent."""
    action_type: ActionType
    risk_level: str
    person_id: int
    vehicle_id: int
    reason: str
    timestamp: float


class AgentPolicy:
    """
    Defines what the agent does at each risk level.

    LOW      → continue monitoring
    MEDIUM   → increase monitoring frequency
    HIGH     → visual warning + create incident
    CRITICAL → escalate + notify operator
    """

    def decide(self, assessment: RiskAssessment,
               has_active_event: bool,
               event_age: float = 0.0) -> List[ActionType]:
        """Decide actions based on risk assessment."""
        actions = []

    def decide(self, assessment: RiskAssessment,
               has_active_event: bool,
               event_age: float = 0.0) -> List[ActionType]:
        """Decide actions based on risk assessment."""
        actions = []

        if assessment.level == "LOW":
            actions.append(ActionType.MONITOR)

        elif assessment.level == "MEDIUM":
            actions.append(ActionType.INCREASE_MONITORING)
            if not has_active_event and assessment.score > 0.4:
                actions.append(ActionType.CREATE_INCIDENT)

        elif assessment.level == "HIGH":
            actions.append(ActionType.VISUAL_WARNING)
            if not has_active_event:
                actions.append(ActionType.CREATE_INCIDENT)
            # Auto-escalate after sustained HIGH
            if has_active_event and event_age > 3.0:
                actions.append(ActionType.ESCALATE)

        elif assessment.level == "CRITICAL":
            actions.append(ActionType.VISUAL_WARNING)
            actions.append(ActionType.AUDIO_WARNING)
            if not has_active_event:
                actions.append(ActionType.CREATE_INCIDENT)
            actions.append(ActionType.ESCALATE)

        return actions


class SentinelAgent:
    """
    The agentic core of SentinelVision.

    Loop:
        PERCEIVE → PREDICT → DECIDE → ACT → OBSERVE → RESOLVE/ESCALATE

    Includes industrial debouncing, temporal hysteresis, and spatial track handover
    to prevent alarm fatigue and eliminate fragmented duplicate incidents.
    """

    def __init__(self, cooldown: float = 5.0,
                 confirmation_frames: int = 3,
                 confirmation_time: float = 0.25,
                 grace_period: float = 1.2,
                 spatial_handover_distance: float = 120.0):
        self.policy = AgentPolicy()
        self.event_manager = EventManager()
        self.cooldown = cooldown
        self.confirmation_frames = confirmation_frames
        self.confirmation_time = confirmation_time
        self.grace_period = grace_period
        self.spatial_handover_distance = spatial_handover_distance

        self._cooldowns: Dict[tuple, float] = {}
        self._action_log: List[AgentAction] = []
        self._warnings_active: Dict[tuple, str] = {}
        self._candidate_risks: Dict[tuple, dict] = {}

    def process(self, assessments: List[RiskAssessment],
                timestamp: float) -> List[AgentAction]:
        """
        Process risk assessments and decide actions with temporal debouncing
        and spatial track continuity.
        """
        actions = []
        active_pairs = set()

        active_person_ids = {a.person_id for a in assessments}

        for assessment in assessments:
            pair_key = (assessment.person_id, assessment.vehicle_id)
            active_pairs.add(pair_key)

            # 1. Check if this pair directly matches an active event
            has_active = self.event_manager.has_active(
                assessment.person_id, assessment.vehicle_id
            )
            event = self.event_manager.get_active(
                assessment.person_id, assessment.vehicle_id
            )

            # 2. If not active, check for spatial track handover / danger clustering
            if not has_active and getattr(assessment, "person_center", None):
                p_c = assessment.person_center
                for active_ev in self.event_manager.active_events:
                    if active_ev.vehicle_id == assessment.vehicle_id and active_ev.person_center:
                        time_since_seen = timestamp - active_ev.last_seen_timestamp
                        if time_since_seen <= self.grace_period:
                            ev_c = active_ev.person_center
                            dist = ((p_c[0] - ev_c[0]) ** 2 + (p_c[1] - ev_c[1]) ** 2) ** 0.5
                            if dist <= self.spatial_handover_distance:
                                if active_ev.person_id not in active_person_ids:
                                    # True track drop handover
                                    old_pair = (active_ev.person_id, active_ev.vehicle_id)
                                    event = self.event_manager.handover_event(
                                        old_pair, pair_key, p_c, timestamp, dist
                                    )
                                    if old_pair in self._warnings_active:
                                        self._warnings_active[pair_key] = self._warnings_active.pop(old_pair)
                                else:
                                    # Overlapping / duplicate bounding box on same physical target
                                    event = active_ev
                                has_active = True
                                break

            # 3. If event is active, update live perception and execute dynamic response
            if has_active and event:
                event.update_state(timestamp, assessment)
                event_age = timestamp - event.timestamp

                action_types = self.policy.decide(
                    assessment, has_active_event=True, event_age=event_age
                )
                for action_type in action_types:
                    action = self._execute(action_type, assessment, timestamp)
                    if action:
                        actions.append(action)
                continue

            # 4. Check cooldown for inactive pair
            if pair_key in self._cooldowns:
                if timestamp - self._cooldowns[pair_key] < self.cooldown:
                    continue

            # 5. Debouncing for newly elevated candidates
            if assessment.level in ("MEDIUM", "HIGH", "CRITICAL"):
                if pair_key not in self._candidate_risks:
                    self._candidate_risks[pair_key] = {
                        "frames": 1,
                        "first_seen": timestamp,
                        "last_seen": timestamp,
                        "assessment": assessment,
                    }
                else:
                    c = self._candidate_risks[pair_key]
                    c["frames"] += 1
                    c["last_seen"] = timestamp
                    c["assessment"] = assessment

                c = self._candidate_risks[pair_key]
                is_imminent_critical = (assessment.level == "CRITICAL" and assessment.ttc <= 0.8)
                is_confirmed = (
                    is_imminent_critical or
                    (c["frames"] >= self.confirmation_frames) or
                    ((timestamp - c["first_seen"]) >= self.confirmation_time) or
                    (self.confirmation_frames <= 1)
                )

                if is_confirmed:
                    # Sustained risk confirmed: promote to incident
                    self._candidate_risks.pop(pair_key, None)
                    action_types = self.policy.decide(assessment, has_active_event=False, event_age=0.0)
                    for action_type in action_types:
                        action = self._execute(action_type, assessment, timestamp)
                        if action:
                            actions.append(action)
                else:
                    # Provide visual warning during confirmation without minting premature incident
                    if assessment.level in ("HIGH", "CRITICAL"):
                        self._warnings_active[pair_key] = assessment.level

            elif assessment.level == "LOW":
                # Clear unconfirmed candidate if risk drops back before confirmation
                self._candidate_risks.pop(pair_key, None)

        # 6. Cleanup stale candidates
        stale = [
            k for k, v in self._candidate_risks.items()
            if k not in active_pairs and (timestamp - v["last_seen"]) > 0.5
        ]
        for k in stale:
            self._candidate_risks.pop(k, None)

        # 7. Resolution Hysteresis Grace Period
        for event in list(self.event_manager.active_events):
            pair_key = (event.person_id, event.vehicle_id)

            if pair_key in active_pairs:
                if event.risk_level == "LOW":
                    if event.low_risk_start is None:
                        event.low_risk_start = timestamp
                    elif (timestamp - event.low_risk_start) >= self.grace_period:
                        self.event_manager.resolve_event(
                            event.person_id, event.vehicle_id, timestamp
                        )
                        self._cooldowns[pair_key] = timestamp
                        self._warnings_active.pop(pair_key, None)
                        actions.append(AgentAction(
                            action_type=ActionType.RESOLVE,
                            risk_level="LOW",
                            person_id=event.person_id,
                            vehicle_id=event.vehicle_id,
                            reason="risk_dissipated_sustained_low",
                            timestamp=timestamp,
                        ))
                else:
                    event.low_risk_start = None
            else:
                # Pair absent or occluded
                time_missing = timestamp - event.last_seen_timestamp
                if time_missing >= self.grace_period:
                    self.event_manager.resolve_event(
                        event.person_id, event.vehicle_id, timestamp
                    )
                    self._cooldowns[pair_key] = timestamp
                    self._warnings_active.pop(pair_key, None)
                    actions.append(AgentAction(
                        action_type=ActionType.RESOLVE,
                        risk_level="LOW",
                        person_id=event.person_id,
                        vehicle_id=event.vehicle_id,
                        reason="risk_dissipated_target_departed",
                        timestamp=timestamp,
                    ))

        return actions

    def _execute(self, action_type: ActionType,
                 assessment: RiskAssessment,
                 timestamp: float) -> Optional[AgentAction]:
        """Execute a single action."""
        pair_key = (assessment.person_id, assessment.vehicle_id)

        if action_type == ActionType.CREATE_INCIDENT:
            self.event_manager.create_event(assessment, "warning")

        elif action_type == ActionType.ESCALATE:
            self.event_manager.escalate_event(
                assessment.person_id, assessment.vehicle_id
            )

        elif action_type == ActionType.RESOLVE:
            self.event_manager.resolve_event(
                assessment.person_id, assessment.vehicle_id, timestamp
            )
            self._cooldowns[pair_key] = timestamp
            self._warnings_active.pop(pair_key, None)

        elif action_type == ActionType.VISUAL_WARNING:
            self._warnings_active[pair_key] = assessment.level

        elif action_type == ActionType.MONITOR:
            return None  # No explicit action needed

        action = AgentAction(
            action_type=action_type,
            risk_level=assessment.level,
            person_id=assessment.person_id,
            vehicle_id=assessment.vehicle_id,
            reason=", ".join(assessment.reasons) if assessment.reasons else "monitoring",
            timestamp=timestamp,
        )
        self._action_log.append(action)
        return action

    @property
    def active_warnings(self) -> Dict[tuple, str]:
        return self._warnings_active

    @property
    def stats(self) -> dict:
        event_stats = self.event_manager.stats
        return {
            **event_stats,
            "total_actions": len(self._action_log),
            "active_warnings": len(self._warnings_active),
        }

    def reset(self):
        self._cooldowns.clear()
        self._action_log.clear()
        self._warnings_active.clear()
        self._candidate_risks.clear()
        self.event_manager = EventManager()
