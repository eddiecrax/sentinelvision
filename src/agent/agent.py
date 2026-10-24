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

        if assessment.level == "LOW":
            if has_active_event:
                actions.append(ActionType.RESOLVE)
            else:
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
    """

    def __init__(self, cooldown: float = 5.0):
        self.policy = AgentPolicy()
        self.event_manager = EventManager()
        self.cooldown = cooldown
        self._cooldowns: Dict[tuple, float] = {}
        self._action_log: List[AgentAction] = []
        self._warnings_active: Dict[tuple, str] = {}

    def process(self, assessments: List[RiskAssessment],
                timestamp: float) -> List[AgentAction]:
        """
        Process risk assessments and decide actions.
        This is called every frame (or at agent poll interval).
        """
        actions = []
        active_pairs = set()

        for assessment in assessments:
            pair_key = (assessment.person_id, assessment.vehicle_id)
            active_pairs.add(pair_key)

            # Check cooldown
            if pair_key in self._cooldowns:
                if timestamp - self._cooldowns[pair_key] < self.cooldown:
                    continue

            has_active = self.event_manager.has_active(
                assessment.person_id, assessment.vehicle_id
            )
            event = self.event_manager.get_active(
                assessment.person_id, assessment.vehicle_id
            )
            event_age = (timestamp - event.timestamp) if event else 0.0

            # Get policy decisions
            action_types = self.policy.decide(
                assessment, has_active, event_age
            )

            for action_type in action_types:
                action = self._execute(action_type, assessment, timestamp)
                if action:
                    actions.append(action)

        # Check for resolved situations (active events with no current risk)
        for event in list(self.event_manager.active_events):
            pair_key = (event.person_id, event.vehicle_id)
            if pair_key not in active_pairs:
                # Risk has dissipated — resolve
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
                    reason="risk_dissipated",
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
