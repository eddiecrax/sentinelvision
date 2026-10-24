"""
SentinelVision — Test for Phase 8 (Incident System) & Phase 9 (Open-World Profiles)
"""

import os
import sys
import numpy as np
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.profiles.loader import ProfileLoader
from src.safety.events import SafetyEvent
from src.safety.storage import IncidentStorage


def test_phase_8_and_9():
    logger.info("============================================================")
    logger.info("  Testing Phase 8 (Incident Storage) & Phase 9 (Profiles)")
    logger.info("============================================================")

    # 1. Test Profile Loader across all environments
    loader = ProfileLoader(profiles_dir="config/environments")
    profiles = loader.available
    logger.info(f"Available profiles: {profiles}")

    assert len(profiles) >= 4, "Expected at least 4 environment profiles!"

    for p_name in ["warehouse", "construction", "factory", "logistics"]:
        p_data = loader.load(p_name)
        assert p_data is not None, f"Failed to load profile {p_name}"
        logger.info(f"  ✅ Loaded profile '{p_name}': {p_data.get('description')}")
        assert "vehicles" in p_data or "name" in p_data, f"Invalid schema in {p_name}"

    # 2. Test Incident Storage & Image Evidence Snapshots
    storage = IncidentStorage(incidents_file="data/incidents_test.json", snapshots_dir="data/clips/test_snapshots")

    # Create mock safety event
    event = SafetyEvent(
        event_id="SV-TEST-0001",
        timestamp=10.5,
        risk_level="HIGH",
        risk_score=0.88,
        person_id=7,
        vehicle_id=2,
        ttc=1.42,
        action_taken="VISUAL_WARNING",
        reasons=["trajectory_convergence", "short_ttc"]
    )

    # Mock frame (360, 640, 3) BGR image
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    dummy_frame[:, :] = (0, 0, 255) # Red warning image

    rec = storage.record_incident(event, frame=dummy_frame, environment="construction")

    assert os.path.exists("data/incidents_test.json"), "Incidents JSON file not created!"
    assert rec["snapshot_path"] is not None, "Snapshot path missing!"
    assert os.path.exists(rec["snapshot_path"]), "Snapshot image file not created on disk!"

    logger.info(f"  ✅ Recorded Incident {rec['event_id']} in environment '{rec['environment']}'")
    logger.info(f"  📸 Snapshot saved: {rec['snapshot_path']}")

    # Clean up test artifacts
    if os.path.exists(rec["snapshot_path"]):
        os.remove(rec["snapshot_path"])
    if os.path.exists("data/incidents_test.json"):
        os.remove("data/incidents_test.json")

    logger.info("============================================================")
    logger.info("  ✅ PHASE 8 & 9 VERIFICATION PASSED SUCCESSFULLY!")
    logger.info("============================================================")


if __name__ == "__main__":
    test_phase_8_and_9()
