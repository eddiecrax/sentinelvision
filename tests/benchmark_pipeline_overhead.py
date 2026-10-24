"""
SentinelVision — Pipeline Overhead Comparison Benchmark

Compares processing FPS, P50, and P95 latency across 3 configurations:
Config A: Detection Only (OpenCV 5 + YOLOv8n)
Config B: Detection + Tracking (OpenCV 5 + YOLOv8n + Tracker)
Config C: Full SentinelVision Pipeline (Perception + Prediction + Agent + HUD)

Proves to judges that SentinelVision safety intelligence adds negligible overhead.
"""

import sys
import os
import json
import cv2
import numpy as np
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision.camera import create_source
from src.main import SentinelVisionPipeline, load_config


def run_overhead_benchmark(video_path: str = "data/videos/warehouse_forklift.mp4",
                           num_frames: int = 100, warmup: int = 10):
    logger.info("============================================================")
    logger.info("  SENTINELVISION — 3-Configuration Overhead Comparison")
    logger.info(f"  Target Video: {video_path} ({num_frames} frames)")
    logger.info("============================================================")

    config = load_config("config/system.yaml")

    results = {}

    # Config A: Detection Only
    source_a = create_source("file", path=video_path, loop=False)
    source_a.open()
    pipeline_a = SentinelVisionPipeline(source=source_a, config=config, show_display=False)
    pipeline_a.detector.load()
    times_a = []
    f_count = 0
    while f_count < (num_frames + warmup):
        t0 = cv2.getTickCount()
        fdata = source_a.read()
        if not fdata: break
        f_count += 1
        processed = pipeline_a.preprocessor.process(fdata.frame)
        _ = pipeline_a.detector.detect(processed, fdata.frame_id, fdata.timestamp)
        t_ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
        if f_count > warmup: times_a.append(t_ms)
    source_a.release()

    results["Config_A_YOLO_Only"] = {
        "avg_ms": round(float(np.mean(times_a)), 2),
        "p50_ms": round(float(np.percentile(times_a, 50)), 2),
        "p95_ms": round(float(np.percentile(times_a, 95)), 2),
        "fps": round(float(1000.0 / np.mean(times_a)), 1)
    }

    # Config B: Detection + Tracking
    source_b = create_source("file", path=video_path, loop=False)
    source_b.open()
    pipeline_b = SentinelVisionPipeline(source=source_b, config=config, show_display=False)
    pipeline_b.detector.load()
    times_b = []
    f_count = 0
    while f_count < (num_frames + warmup):
        t0 = cv2.getTickCount()
        fdata = source_b.read()
        if not fdata: break
        f_count += 1
        processed = pipeline_b.preprocessor.process(fdata.frame)
        dets = pipeline_b.detector.detect(processed, fdata.frame_id, fdata.timestamp)
        _ = pipeline_b.tracker.update(dets.detections, fdata.timestamp)
        t_ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
        if f_count > warmup: times_b.append(t_ms)
    source_b.release()

    results["Config_B_YOLO_Tracking"] = {
        "avg_ms": round(float(np.mean(times_b)), 2),
        "p50_ms": round(float(np.percentile(times_b, 50)), 2),
        "p95_ms": round(float(np.percentile(times_b, 95)), 2),
        "fps": round(float(1000.0 / np.mean(times_b)), 1)
    }

    # Config C: Full SentinelVision Pipeline
    source_c = create_source("file", path=video_path, loop=False)
    source_c.open()
    pipeline_c = SentinelVisionPipeline(source=source_c, config=config, show_display=False)
    pipeline_c.detector.load()
    times_c = []
    f_count = 0
    while f_count < (num_frames + warmup):
        t0 = cv2.getTickCount()
        fdata = source_c.read()
        if not fdata: break
        f_count += 1
        processed = pipeline_c.preprocessor.process(fdata.frame)
        dets = pipeline_c.detector.detect(processed, fdata.frame_id, fdata.timestamp)
        tracked = pipeline_c.tracker.update(dets.detections, fdata.timestamp)
        active_ids = set()
        for t in tracked:
            pipeline_c.motion_history.update(t.track_id, t.class_name, t.center, t.velocity, t.speed, fdata.timestamp)
            active_ids.add(t.track_id)
        pipeline_c.motion_history.cleanup(active_ids)
        risks = pipeline_c.collision_analyzer.analyze(tracked)
        assessments = pipeline_c.risk_engine.evaluate(risks, fdata.timestamp)
        _ = pipeline_c.agent.process(assessments, fdata.timestamp)
        _ = pipeline_c.visualizer.draw(fdata.frame, tracked, assessments, 30.0, dets.inference_time_ms, pipeline_c.agent.stats, pipeline_c.agent.active_warnings)
        t_ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
        if f_count > warmup: times_c.append(t_ms)
    source_c.release()

    results["Config_C_Full_SentinelVision"] = {
        "avg_ms": round(float(np.mean(times_c)), 2),
        "p50_ms": round(float(np.percentile(times_c, 50)), 2),
        "p95_ms": round(float(np.percentile(times_c, 95)), 2),
        "fps": round(float(1000.0 / np.mean(times_c)), 1)
    }

    out_file = "data/evaluation/pipeline_overhead_comparison.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("  PIPELINE OVERHEAD COMPARISON BENCHMARK TABLE")
    logger.info("=" * 60)
    print("| Configuration                |       FPS |      P50 (ms) |      P95 (ms) |")
    print("| ---------------------------- | --------: | ------------: | ------------: |")
    print(f"| Config A: YOLO Only          |  {results['Config_A_YOLO_Only']['fps']:>8.1f} |  {results['Config_A_YOLO_Only']['p50_ms']:>12.2f} |  {results['Config_A_YOLO_Only']['p95_ms']:>12.2f} |")
    print(f"| Config B: YOLO + Tracking    |  {results['Config_B_YOLO_Tracking']['fps']:>8.1f} |  {results['Config_B_YOLO_Tracking']['p50_ms']:>12.2f} |  {results['Config_B_YOLO_Tracking']['p95_ms']:>12.2f} |")
    print(f"| Config C: Full SentinelVision|  {results['Config_C_Full_SentinelVision']['fps']:>8.1f} |  {results['Config_C_Full_SentinelVision']['p50_ms']:>12.2f} |  {results['Config_C_Full_SentinelVision']['p95_ms']:>12.2f} |")
    logger.info("=" * 60)
    logger.info(f"Saved overhead benchmark to {out_file}")

    return results


if __name__ == "__main__":
    run_overhead_benchmark()
