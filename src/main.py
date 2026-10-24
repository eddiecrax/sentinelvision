"""
SentinelVision — Main Pipeline

The complete camera → detection → tracking → prediction → risk → agent loop.

Usage:
    python src/main.py --source webcam
    python src/main.py --source file --path data/videos/test.mp4
    python src/main.py --source file --path data/videos/test.mp4 --no-display
"""

import sys
import os
import time
import argparse
import yaml
import cv2
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision.camera import create_source, VideoSource
from src.vision.preprocessing import FramePreprocessor
from src.vision.detector import Detector
from src.vision.tracker import Tracker
from src.vision.visualizer import Visualizer
from src.motion.history import MotionHistory
from src.prediction.collision import CollisionAnalyzer
from src.safety.risk import RiskEngine
from src.agent.agent import SentinelAgent
from src.profiles.loader import ProfileLoader
from src.safety.storage import IncidentStorage


def load_config(path: str = "config/system.yaml") -> dict:
    """Load system configuration."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    logger.warning(f"Config not found at {path}, using defaults")
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="SentinelVision — Agentic Safety Vision")
    parser.add_argument("--source", type=str, default="webcam",
                        choices=["webcam", "file", "rtsp"],
                        help="Video source type")
    parser.add_argument("--path", type=str, default=None,
                        help="Path to video file (when source=file)")
    parser.add_argument("--device-index", type=int, default=0,
                        help="Webcam device index")
    parser.add_argument("--rtsp-url", type=str, default=None,
                        help="RTSP stream URL")
    parser.add_argument("--model", type=str, default="models/yolov8n.pt",
                        help="YOLO model to use")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Inference device (cpu/cuda)")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without display window")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save annotated output video")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames to process (0=all)")
    parser.add_argument("--profile", type=str, default="warehouse",
                        help="Environment profile (warehouse, construction, factory, logistics)")
    parser.add_argument("--config", type=str, default="config/system.yaml",
                        help="Path to system config")
    return parser.parse_args()


class SentinelVisionPipeline:
    """
    The complete SentinelVision processing pipeline.

    CAMERA → PREPROCESS → DETECT → TRACK → MOTION → PREDICT → RISK → AGENT → DISPLAY
    """

    def __init__(self, source: VideoSource, config: dict,
                 show_display: bool = True, output_path: str = None,
                 max_frames: int = 0, profile_name: str = "warehouse"):
        self.source = source
        self.config = config
        self.show_display = show_display
        self.output_path = output_path
        self.max_frames = max_frames
        self.profile_name = profile_name

        # Load environment profile (Phase 9)
        self.profile_loader = ProfileLoader()
        self.profile = self.profile_loader.load(profile_name) or {}

        # Incident storage (Phase 8)
        self.incident_storage = IncidentStorage()

        # Pipeline components
        det_cfg = config.get("detection", {})
        track_cfg = config.get("tracking", {})

        self.preprocessor = FramePreprocessor()
        self.detector = Detector(
            model_path=det_cfg.get("model", "models/yolov8n.pt"),
            confidence=det_cfg.get("confidence", 0.5),
            iou_threshold=det_cfg.get("iou_threshold", 0.45),
            device=det_cfg.get("device", "cpu"),
        )
        self.tracker = Tracker(
            max_age=track_cfg.get("max_age", 30),
            min_hits=track_cfg.get("min_hits", 3),
            iou_threshold=track_cfg.get("iou_threshold", 0.3),
        )
        self.motion_history = MotionHistory()
        self.collision_analyzer = CollisionAnalyzer(
            ttc_thresholds=config.get("risk", {}).get("thresholds"),
            proximity_threshold=80.0,
            prediction_horizon=config.get("prediction", {}).get("horizon", 2.0),
        )
        self.risk_engine = RiskEngine(
            weights=config.get("risk", {}).get("weights"),
        )
        self.agent = SentinelAgent(
            cooldown=config.get("agent", {}).get("cooldown", 5.0),
        )
        self.visualizer = Visualizer()

        # Output video writer
        self._writer = None

        # Performance tracking
        self._fps_counter = 0
        self._fps_start = 0.0
        self._current_fps = 0.0
        self._log_interval = 30  # log stats every N frames in headless mode

    def run(self):
        """Run the main processing loop."""
        logger.info("=" * 60)
        logger.info("  SENTINELVISION — Starting Pipeline")
        logger.info("=" * 60)

        if not self.source.open():
            logger.error("Failed to open video source")
            return

        logger.info("Loading detection model...")
        self.detector.load()
        logger.info("Pipeline ready. Press 'q' to quit.")

        self._fps_start = time.time()

        try:
            while True:
                # === PERCEIVE ===
                frame_data = self.source.read()
                if frame_data is None:
                    logger.info("End of video stream")
                    break

                # Check frame limit
                if self.max_frames > 0 and frame_data.frame_id > self.max_frames:
                    logger.info(f"Reached max frames ({self.max_frames})")
                    break

                frame = frame_data.frame
                timestamp = frame_data.timestamp

                # Preprocess
                processed = self.preprocessor.process(frame)

                # === DETECT ===
                det_result = self.detector.detect(
                    processed, frame_data.frame_id, timestamp
                )

                # === TRACK ===
                tracked = self.tracker.update(det_result.detections, timestamp)

                # Update motion history
                active_ids = set()
                for t in tracked:
                    self.motion_history.update(
                        t.track_id, t.class_name,
                        t.center, t.velocity, t.speed, timestamp
                    )
                    active_ids.add(t.track_id)
                self.motion_history.cleanup(active_ids)

                # === PREDICT ===
                collision_risks = self.collision_analyzer.analyze(tracked)

                # === RISK ===
                risk_assessments = self.risk_engine.evaluate(
                    collision_risks, timestamp
                )

                # === AGENT ===
                actions = self.agent.process(risk_assessments, timestamp)

                # Log significant actions and persist incidents with evidence snapshot (Phase 8)
                for action in actions:
                    if action.action_type.value not in ("monitor", "increase_monitoring"):
                        logger.info(
                            f"AGENT → {action.action_type.value.upper()} | "
                            f"P#{action.person_id} ↔ V#{action.vehicle_id} | "
                            f"{action.reason}"
                        )
                        # Save incident metadata + evidence frame snapshot
                        evt = self.agent.event_manager.get_active(action.person_id, action.vehicle_id)
                        if evt:
                            self.incident_storage.record_incident(
                                event=evt,
                                frame=frame,
                                environment=self.profile_name
                            )

                # === UPDATE FPS ===
                self._update_fps()

                # === RENDER ANNOTATED FRAME ===
                annotated = self.visualizer.draw(
                    frame,
                    tracks=tracked,
                    risks=risk_assessments,
                    fps=self._current_fps,
                    inference_ms=det_result.inference_time_ms,
                    agent_stats=self.agent.stats,
                    warnings=self.agent.active_warnings,
                )

                # === WRITE OUTPUT VIDEO ===
                if self.output_path and self._writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self._writer = cv2.VideoWriter(
                        self.output_path, fourcc, 30.0, (w, h)
                    )
                    logger.info(f"Writing output to: {self.output_path}")
                if self._writer is not None:
                    self._writer.write(annotated)

                # === DISPLAY ===
                if self.show_display:
                    cv2.imshow("SentinelVision", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User quit")
                        break
                    elif key == ord('p'):
                        logger.info("Paused. Press any key to continue.")
                        cv2.waitKey(0)

                # === HEADLESS LOGGING ===
                if not self.show_display and frame_data.frame_id % self._log_interval == 0:
                    persons = sum(1 for t in tracked if t.class_name == "person")
                    vehicles = sum(1 for t in tracked if t.class_name != "person")
                    high_risks = sum(1 for r in risk_assessments
                                     if r.level in ("HIGH", "CRITICAL"))
                    logger.info(
                        f"Frame {frame_data.frame_id:>5d} | "
                        f"FPS: {self._current_fps:>5.1f} | "
                        f"Det: {det_result.count:>2d} | "
                        f"Tracks: {len(tracked):>2d} (P:{persons} V:{vehicles}) | "
                        f"Risks: {len(risk_assessments):>2d} (⚠{high_risks}) | "
                        f"Latency: {det_result.inference_time_ms:>5.0f}ms"
                    )

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._shutdown()

    def _update_fps(self):
        """Track FPS with rolling window."""
        self._fps_counter += 1
        elapsed = time.time() - self._fps_start
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_start = time.time()

    def _shutdown(self):
        """Clean shutdown."""
        logger.info("-" * 40)
        logger.info("SentinelVision — Session Summary")
        logger.info(f"  Frames processed: {self.source.frame_count}")
        logger.info(f"  Avg detection:    {self.detector.avg_inference_ms:.1f}ms")
        logger.info(f"  Avg FPS:          {self._current_fps:.1f}")
        stats = self.agent.stats
        logger.info(f"  Events created:   {stats['total']}")
        logger.info(f"  Events resolved:  {stats['resolved']}")
        logger.info(f"  Events escalated: {stats['escalated']}")
        logger.info("-" * 40)

        if self._writer is not None:
            self._writer.release()
            logger.info(f"Output video saved: {self.output_path}")

        self.source.release()
        if self.show_display:
            cv2.destroyAllWindows()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Override config with CLI args
    if "detection" not in config:
        config["detection"] = {}
    config["detection"]["model"] = args.model
    config["detection"]["confidence"] = args.confidence
    config["detection"]["device"] = args.device

    # Create video source
    source_kwargs = {}
    if args.source == "webcam":
        source_kwargs["device_index"] = args.device_index
    elif args.source == "file":
        if not args.path:
            logger.error("--path required when source=file")
            sys.exit(1)
        source_kwargs["path"] = args.path
        source_kwargs["loop"] = False  # Don't loop for testing
    elif args.source == "rtsp":
        if not args.rtsp_url:
            logger.error("--rtsp-url required when source=rtsp")
            sys.exit(1)
        source_kwargs["url"] = args.rtsp_url

    source = create_source(args.source, **source_kwargs)

    # Run pipeline
    pipeline = SentinelVisionPipeline(
        source=source,
        config=config,
        show_display=not args.no_display,
        output_path=args.output,
        max_frames=args.max_frames,
        profile_name=args.profile,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
