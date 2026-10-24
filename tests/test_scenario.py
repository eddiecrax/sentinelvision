"""
SentinelVision — Scenario Simulation Test

Tests the full perception → prediction → risk → agent loop using synthetic detections
simulating a person-vehicle collision course and resolution.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision.detector import Detection
from src.vision.tracker import Tracker
from src.motion.history import MotionHistory
from src.prediction.collision import CollisionAnalyzer
from src.safety.risk import RiskEngine
from src.agent.agent import SentinelAgent, ActionType
from loguru import logger


def run_simulated_scenario():
    logger.info("============================================================")
    logger.info("  SENTINELVISION — Simulated Scenario Test")
    logger.info("============================================================")

    tracker = Tracker(max_age=10, min_hits=1, iou_threshold=0.2)
    motion_history = MotionHistory()
    collision_analyzer = CollisionAnalyzer(proximity_threshold=150.0, prediction_horizon=3.0)
    risk_engine = RiskEngine()
    agent = SentinelAgent(cooldown=1.0)

    # Initial positions
    # Vehicle moving right towards (400, 200)
    v_x, v_y = 100.0, 200.0
    v_vx, v_vy = 30.0, 0.0  # 30 px/sec right

    # Person moving left/up towards (400, 200)
    p_x, p_y = 500.0, 200.0
    p_vx, p_vy = -30.0, 0.0 # 30 px/sec left

    fps = 10
    dt = 1.0 / fps

    events_summary = []

    for step in range(60): # 6 seconds simulation
        timestamp = step * dt

        # Phase 1: approaching (steps 0..35)
        # Phase 2: evasion at step 35 -> Person stops/moves away
        if step >= 35:
            p_vx = 0.0 # Person stopped
            p_vy = 40.0 # Moving down away from vehicle path

        v_x += v_vx * dt
        v_y += v_vy * dt
        p_x += p_vx * dt
        p_y += p_vy * dt

        # Generate synthetic detections
        det_vehicle = Detection(
            class_id=2, class_name="car", confidence=0.9,
            bbox=[v_x - 30, v_y - 20, v_x + 30, v_y + 20],
            center=[v_x, v_y], area=2400.0, timestamp=timestamp
        )
        det_person = Detection(
            class_id=0, class_name="person", confidence=0.95,
            bbox=[p_x - 10, p_y - 20, p_x + 10, p_y + 20],
            center=[p_x, p_y], area=800.0, timestamp=timestamp
        )

        # Update pipeline
        tracked = tracker.update([det_vehicle, det_person], timestamp)

        active_ids = set()
        for t in tracked:
            motion_history.update(t.track_id, t.class_name, t.center, t.velocity, t.speed, timestamp)
            active_ids.add(t.track_id)
        motion_history.cleanup(active_ids)

        collision_risks = collision_analyzer.analyze(tracked)
        risk_assessments = risk_engine.evaluate(collision_risks, timestamp)
        actions = agent.process(risk_assessments, timestamp)

        # Log frame details
        top_risk = risk_assessments[0] if risk_assessments else None
        risk_level = top_risk.level if top_risk else "NONE"
        ttc_val = f"{top_risk.ttc:.2f}s" if top_risk and top_risk.ttc != float('inf') else "N/A"
        dist_val = f"{top_risk.distance:.1f}px" if top_risk else "N/A"

        action_names = [a.action_type.name for a in actions] if actions else []

        if action_names or step % 5 == 0:
            logger.info(
                f"t={timestamp:4.1f}s | Step {step:2d} | Risk: {risk_level:<8s} | "
                f"TTC: {ttc_val:<6s} | Dist: {dist_val:<7s} | Actions: {action_names}"
            )

        for a in actions:
            events_summary.append((timestamp, a.action_type.name, risk_level))

    logger.info("-" * 50)
    logger.info("Scenario Summary:")
    logger.info(f"Total Agent Actions Triggered: {len(events_summary)}")
    for t, action, lvl in events_summary:
        logger.info(f"  [{t:4.1f}s] {action:<20s} (Risk Level: {lvl})")

    agent_stats = agent.stats
    logger.info(f"Agent Stats: {agent_stats}")
    logger.info("============================================================")

    assert agent_stats["total"] > 0, "Agent failed to create any incident events!"
    assert agent_stats["resolved"] > 0, "Agent failed to resolve incident upon evasion!"
    logger.info("✅ SIMULATION TEST PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_simulated_scenario()
