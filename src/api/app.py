"""
SentinelVision — FastAPI Web Server & API Layer

Phase 11: Provides MJPEG video streaming, incident APIs, telemetry stats,
and environment profile switching for the web dashboard.
"""

import os
import sys
import time
import cv2
from typing import Optional
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.vision.camera import create_source
from src.main import SentinelVisionPipeline, load_config
from src.safety.storage import IncidentStorage
from src.profiles.loader import ProfileLoader

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="SentinelVision API", version="0.1.0")

# Global pipeline instance state
video_sources = {
    "warehouse": os.path.join(BASE_DIR, "data", "videos", "warehouse_forklift.mp4"),
    "factory": os.path.join(BASE_DIR, "data", "videos", "factory_forklift.mp4"),
    "pedestrian": os.path.join(BASE_DIR, "data", "videos", "pedestrian_traffic.mp4"),
    "traffic": os.path.join(BASE_DIR, "data", "videos", "test_traffic.mp4"),
}
current_video_key = "warehouse"
current_profile = "factory"
global_pipeline: Optional[SentinelVisionPipeline] = None

# Static dashboard files directory
dashboard_dir = os.path.join(BASE_DIR, "dashboard")
os.makedirs(dashboard_dir, exist_ok=True)


class ProfileSwitchRequest(BaseModel):
    profile: str


class VideoSwitchRequest(BaseModel):
    video_key: str


class IncidentOverrideRequest(BaseModel):
    event_id: str
    action: str


@app.get("/api/health")
@app.get("/health")
def health_check():
    return {"status": "online", "system": "SentinelVision", "timestamp": time.time()}


@app.get("/api/incidents")
def get_incidents():
    incidents_path = os.path.join(BASE_DIR, "data", "incidents.json")
    storage = IncidentStorage(incidents_file=incidents_path)
    records = list(storage.all_incidents)

    if global_pipeline is not None:
        live_dicts = [e.to_dict() for e in global_pipeline.agent.event_manager.all_events]
        seen_ids = set()
        merged = []
        for item in live_dicts + records:
            if item["event_id"] not in seen_ids:
                seen_ids.add(item["event_id"])
                merged.append(item)
        return JSONResponse(content=merged)

    return JSONResponse(content=records)


@app.post("/api/incident/override")
def human_override_incident(req: IncidentOverrideRequest):
    incidents_path = os.path.join(BASE_DIR, "data", "incidents.json")
    storage = IncidentStorage(incidents_file=incidents_path)
    logger.info(f"👨‍✈️ HUMAN OVERRIDE: Event {req.event_id} -> Action {req.action}")

    if global_pipeline is not None:
        for evt in global_pipeline.agent.event_manager.all_events:
            if evt.event_id == req.event_id:
                if req.action == "ACKNOWLEDGE":
                    evt.action_taken = "OPERATOR_ACKNOWLEDGED"
                elif req.action == "ESCALATE":
                    evt.escalated = True
                    evt.action_taken = "OPERATOR_ESCALATED"
                elif req.action == "DISMISS":
                    evt.resolve(time.time())
                    evt.action_taken = "OPERATOR_DISMISSED"
                storage.record_incident(evt, environment=current_profile)
                break

    return {"status": "success", "event_id": req.event_id, "action": req.action}


@app.get("/api/profiles")
def get_profiles():
    profiles_dir = os.path.join(BASE_DIR, "config", "environments")
    loader = ProfileLoader(profiles_dir=profiles_dir)
    return {"available": loader.available, "current": current_profile}


@app.post("/api/profile")
def set_profile(req: ProfileSwitchRequest):
    global current_profile
    profiles_dir = os.path.join(BASE_DIR, "config", "environments")
    loader = ProfileLoader(profiles_dir=profiles_dir)
    if req.profile in loader.available:
        current_profile = req.profile
        logger.info(f"Switched profile to: {current_profile}")
        return {"status": "success", "profile": current_profile}
    return JSONResponse(status_code=400, content={"error": f"Invalid profile: {req.profile}"})


@app.post("/api/video_source")
def set_video_source(req: VideoSwitchRequest):
    global current_video_key
    if req.video_key in video_sources:
        current_video_key = req.video_key
        logger.info(f"Switched video feed to: {current_video_key}")
        return {"status": "success", "video_key": current_video_key}
    return JSONResponse(status_code=400, content={"error": f"Invalid video key: {req.video_key}"})


def generate_frames():
    """Video streaming generator function (MJPEG format)."""
    global current_video_key, global_pipeline
    active_key = current_video_key
    video_path = video_sources.get(active_key, video_sources["warehouse"])
    config_path = os.path.join(BASE_DIR, "config", "system.yaml")
    config = load_config(config_path)
    source = create_source("file", path=video_path, loop=True)
    pipeline = SentinelVisionPipeline(
        source=source,
        config=config,
        show_display=False,
        profile_name=current_profile
    )
    global_pipeline = pipeline

    if not source.open():
        logger.error("Failed to open video stream for web server")
        return

    pipeline.detector.load()
    fps_start = time.time()
    fps_counter = 0
    current_fps = 0.0

    try:
        while True:
            frame_data = source.read()
            if frame_data is None:
                break

            frame = frame_data.frame
            timestamp = frame_data.timestamp

            processed = pipeline.preprocessor.process(frame)
            det_result = pipeline.detector.detect(processed, frame_data.frame_id, timestamp)
            tracked = pipeline.tracker.update(det_result.detections, timestamp)

            active_ids = set()
            for t in tracked:
                pipeline.motion_history.update(
                    t.track_id, t.class_name, t.center, t.velocity, t.speed, timestamp
                )
                active_ids.add(t.track_id)
            pipeline.motion_history.cleanup(active_ids)

            collision_risks = pipeline.collision_analyzer.analyze(tracked)
            risk_assessments = pipeline.risk_engine.evaluate(collision_risks, timestamp)
            actions = pipeline.agent.process(risk_assessments, timestamp)

            # Persist significant safety incidents to storage
            for action in actions:
                if action.action_type.value not in ("monitor", "increase_monitoring"):
                    evt = pipeline.agent.event_manager.get_active(action.person_id, action.vehicle_id)
                    if not evt:
                        for e in pipeline.agent.event_manager.all_events:
                            if e.person_id == action.person_id and e.vehicle_id == action.vehicle_id:
                                evt = e
                                break
                    if evt:
                        pipeline.incident_storage.record_incident(
                            event=evt,
                            frame=frame,
                            environment=current_profile
                        )

            # Update FPS
            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

            annotated = pipeline.visualizer.draw(
                frame,
                tracks=tracked,
                risks=risk_assessments,
                fps=current_fps,
                inference_ms=det_result.inference_time_ms,
                agent_stats=pipeline.agent.stats,
                warnings=pipeline.agent.active_warnings,
            )

            # Encode BGR image to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.02) # ~30-50 FPS stream cap

    finally:
        source.release()


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/", response_class=HTMLResponse)
def index():
    html_file = os.path.join(dashboard_dir, "index.html")
    if os.path.exists(html_file):
        with open(html_file, 'r') as f:
            return f.read()
    return "<h1>SentinelVision Web Dashboard</h1><p>Dashboard template generating...</p>"
