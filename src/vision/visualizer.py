"""
SentinelVision — Visualization

Draws bounding boxes, tracks, trajectories, risk indicators,
and HUD overlay on frames for display.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from src.vision.tracker import TrackedObject
from src.prediction.collision import CollisionRisk
from src.safety.risk import RiskAssessment


# Color palette (BGR for OpenCV)
COLORS = {
    "person": (0, 220, 0),         # green
    "car": (0, 165, 255),          # orange
    "truck": (0, 140, 255),        # dark orange
    "bus": (0, 100, 255),          # red-orange
    "motorcycle": (255, 200, 0),   # cyan-ish
    "bicycle": (255, 255, 0),      # cyan
    "LOW": (0, 200, 0),            # green
    "MEDIUM": (0, 255, 255),       # yellow
    "HIGH": (0, 140, 255),         # orange
    "CRITICAL": (0, 0, 255),       # red
}


class Visualizer:
    """Draws detection, tracking, and risk overlays on frames."""

    def __init__(self, show_bboxes=True, show_ids=True,
                 show_trajectories=True, show_risk=True, show_fps=True):
        self.show_bboxes = show_bboxes
        self.show_ids = show_ids
        self.show_trajectories = show_trajectories
        self.show_risk = show_risk
        self.show_fps = show_fps

    def draw(self, frame: np.ndarray,
             tracks: List[TrackedObject],
             risks: List[RiskAssessment] = None,
             fps: float = 0.0,
             inference_ms: float = 0.0,
             agent_stats: dict = None,
             warnings: Dict[tuple, str] = None) -> np.ndarray:
        """Draw all overlays on frame. Returns annotated frame."""
        overlay = frame.copy()

        # Draw tracked objects
        if self.show_bboxes:
            for track in tracks:
                self._draw_track(overlay, track)

        # Draw trajectories
        if self.show_trajectories:
            for track in tracks:
                self._draw_trajectory(overlay, track)

        # Draw risk connections
        if self.show_risk and risks:
            for risk in risks:
                self._draw_risk(overlay, risk, tracks)

        # Draw warning flash for active warnings
        if warnings:
            for (pid, vid), level in warnings.items():
                self._draw_warning_flash(overlay, level)

        # Draw HUD
        self._draw_hud(overlay, fps, inference_ms,
                       len(tracks), risks, agent_stats)

        return overlay

    def _draw_track(self, frame, track: TrackedObject):
        """Draw bounding box and label for a tracked object."""
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        color = COLORS.get(track.class_name, (200, 200, 200))

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        if self.show_ids:
            label = f"#{track.track_id} {track.class_name} {track.confidence:.0%}"
            speed_label = f"spd:{track.speed:.0f}px/s"

            # Background for label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+4, y1), color, -1)
            cv2.putText(frame, label, (x1+2, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Speed below box
            cv2.putText(frame, speed_label, (x1, y2+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _draw_trajectory(self, frame, track: TrackedObject):
        """Draw position history as a trail."""
        if len(track.position_history) < 2:
            return

        color = COLORS.get(track.class_name, (200, 200, 200))
        points = track.position_history[-30:]  # last 30 positions

        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(alpha * 3))
            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            fade_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pt1, pt2, fade_color, thickness)

    def _draw_risk(self, frame, risk: RiskAssessment,
                   tracks: List[TrackedObject]):
        """Draw risk connection between person and vehicle."""
        if risk.level == "LOW":
            return

        person = next((t for t in tracks if t.track_id == risk.person_id), None)
        vehicle = next((t for t in tracks if t.track_id == risk.vehicle_id), None)
        if not person or not vehicle:
            return

        color = COLORS.get(risk.level, (200, 200, 200))
        p_center = (int(person.center[0]), int(person.center[1]))
        v_center = (int(vehicle.center[0]), int(vehicle.center[1]))

        # Dashed line between objects
        thickness = 2 if risk.level in ("HIGH", "CRITICAL") else 1
        cv2.line(frame, p_center, v_center, color, thickness)

        # Risk label at midpoint
        mx = (p_center[0] + v_center[0]) // 2
        my = (p_center[1] + v_center[1]) // 2

        ttc_str = f"{risk.ttc:.1f}s" if risk.ttc != float('inf') else "∞"
        label = f"⚠ {risk.level} TTC:{ttc_str}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (mx-tw//2-4, my-th-8),
                      (mx+tw//2+4, my+4), color, -1)
        cv2.putText(frame, label, (mx-tw//2, my-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def _draw_warning_flash(self, frame, level: str):
        """Flash frame border for active warnings."""
        color = COLORS.get(level, (0, 0, 255))
        h, w = frame.shape[:2]
        thickness = 6 if level == "CRITICAL" else 4
        cv2.rectangle(frame, (0, 0), (w-1, h-1), color, thickness)

    def _draw_hud(self, frame, fps, inference_ms, track_count,
                  risks, agent_stats):
        """Draw heads-up display with stats."""
        h, w = frame.shape[:2]

        # Semi-transparent background for HUD
        hud_h = 100
        overlay_region = frame[0:hud_h, 0:w].copy()
        cv2.rectangle(frame, (0, 0), (w, hud_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay_region, 0.3, frame[0:hud_h, 0:w], 0.7, 0,
                        frame[0:hud_h, 0:w])

        # Title
        cv2.putText(frame, "SENTINELVISION", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # FPS & Latency
        if self.show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Latency: {inference_ms:.0f}ms", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Tracks: {track_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Risk summary
        if risks:
            high_count = sum(1 for r in risks
                             if r.level in ("HIGH", "CRITICAL"))
            x_pos = w - 250
            cv2.putText(frame, f"Active Risks: {len(risks)}",
                        (x_pos, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            if high_count > 0:
                cv2.putText(frame, f"HIGH/CRITICAL: {high_count}",
                            (x_pos, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)

        # Agent stats
        if agent_stats:
            x_pos = w // 2 - 80
            cv2.putText(frame, f"Events: {agent_stats.get('total', 0)}",
                        (x_pos, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            cv2.putText(frame,
                        f"Resolved: {agent_stats.get('resolved', 0)}",
                        (x_pos, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 0), 1)
            cv2.putText(frame,
                        f"Escalated: {agent_stats.get('escalated', 0)}",
                        (x_pos, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 100, 255), 1)
