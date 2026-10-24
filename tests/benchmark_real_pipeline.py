"""
SentinelVision — Real End-to-End Pipeline Stage-by-Stage Benchmark

Priority 2: Benchmarks actual video decode, OpenCV 5 preprocessing, YOLOv8 inference,
tracking, motion, TTC, risk engine, agent decision, and visualization stages using cv2.TickMeter.
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


def run_real_pipeline_benchmark(video_path: str = "data/videos/warehouse_forklift.mp4",
                                num_frames: int = 150, warmup_frames: int = 10):
    logger.info("============================================================")
    logger.info("  SENTINELVISION — Real Stage-by-Stage Pipeline Benchmark")
    logger.info(f"  Target Video: {video_path} ({num_frames} frames)")
    logger.info("============================================================")

    config_path = "config/system.yaml"
    config = load_config(config_path)

    source = create_source("file", path=video_path, loop=False)
    if not source.open():
        logger.error(f"Failed to open video: {video_path}")
        return {}

    pipeline = SentinelVisionPipeline(
        source=source,
        config=config,
        show_display=False,
        profile_name="factory"
    )
    pipeline.detector.load()

    # OpenCV TickMeter timers for each stage
    tm_decode = cv2.TickMeter()
    tm_preprocess = cv2.TickMeter()
    tm_yolo = cv2.TickMeter()
    tm_tracker = cv2.TickMeter()
    tm_motion = cv2.TickMeter()
    tm_ttc = cv2.TickMeter()
    tm_risk = cv2.TickMeter()
    tm_agent = cv2.TickMeter()
    tm_render = cv2.TickMeter()
    tm_total = cv2.TickMeter()

    stage_timings = {
        "decode": [],
        "preprocess": [],
        "yolo": [],
        "tracker": [],
        "motion": [],
        "ttc": [],
        "risk": [],
        "agent": [],
        "render": [],
        "total": [],
    }

    frame_count = 0

    while frame_count < (num_frames + warmup_frames):
        tm_total.reset(); tm_total.start()

        # 1. Video Decode Stage
        tm_decode.reset(); tm_decode.start()
        frame_data = source.read()
        tm_decode.stop()

        if frame_data is None:
            break

        frame_count += 1
        is_warmup = frame_count <= warmup_frames

        frame = frame_data.frame
        timestamp = frame_data.timestamp

        # 2. OpenCV Preprocessing Stage
        tm_preprocess.reset(); tm_preprocess.start()
        processed = pipeline.preprocessor.process(frame)
        tm_preprocess.stop()

        # 3. YOLO Inference Stage
        tm_yolo.reset(); tm_yolo.start()
        det_result = pipeline.detector.detect(processed, frame_data.frame_id, timestamp)
        tm_yolo.stop()

        # 4. Multi-Object Tracking Stage
        tm_tracker.reset(); tm_tracker.start()
        tracked = pipeline.tracker.update(det_result.detections, timestamp)
        tm_tracker.stop()

        # 5. Motion Analysis Stage
        tm_motion.reset(); tm_motion.start()
        active_ids = set()
        for t in tracked:
            pipeline.motion_history.update(
                t.track_id, t.class_name, t.center, t.velocity, t.speed, timestamp
            )
            active_ids.add(t.track_id)
        pipeline.motion_history.cleanup(active_ids)
        tm_motion.stop()

        # 6. Collision & TTC Prediction Stage
        tm_ttc.reset(); tm_ttc.start()
        collision_risks = pipeline.collision_analyzer.analyze(tracked)
        tm_ttc.stop()

        # 7. Risk Engine Stage
        tm_risk.reset(); tm_risk.start()
        risk_assessments = pipeline.risk_engine.evaluate(collision_risks, timestamp)
        tm_risk.stop()

        # 8. Agent Decision Stage
        tm_agent.reset(); tm_agent.start()
        actions = pipeline.agent.process(risk_assessments, timestamp)
        tm_agent.stop()

        # 9. Visualizer Rendering Stage
        tm_render.reset(); tm_render.start()
        annotated = pipeline.visualizer.draw(
            frame,
            tracks=tracked,
            risks=risk_assessments,
            fps=30.0,
            inference_ms=det_result.inference_time_ms,
            agent_stats=pipeline.agent.stats,
            warnings=pipeline.agent.active_warnings,
        )
        tm_render.stop()

        tm_total.stop()

        if not is_warmup:
            stage_timings["decode"].append(tm_decode.getTimeMilli())
            stage_timings["preprocess"].append(tm_preprocess.getTimeMilli())
            stage_timings["yolo"].append(tm_yolo.getTimeMilli())
            stage_timings["tracker"].append(tm_tracker.getTimeMilli())
            stage_timings["motion"].append(tm_motion.getTimeMilli())
            stage_timings["ttc"].append(tm_ttc.getTimeMilli())
            stage_timings["risk"].append(tm_risk.getTimeMilli())
            stage_timings["agent"].append(tm_agent.getTimeMilli())
            stage_timings["render"].append(tm_render.getTimeMilli())
            stage_timings["total"].append(tm_total.getTimeMilli())

    source.release()

    total_avg_ms = float(np.mean(stage_timings["total"]))
    end_to_end_fps = float(1000.0 / total_avg_ms) if total_avg_ms > 0 else 0.0

    summary = {
        "video_evaluated": video_path,
        "frames_benchmarked": len(stage_timings["total"]),
        "stage_averages_ms": {
            "video_decode": round(float(np.mean(stage_timings["decode"])), 2),
            "opencv_preprocessing": round(float(np.mean(stage_timings["preprocess"])), 2),
            "yolov8_inference": round(float(np.mean(stage_timings["yolo"])), 2),
            "tracking": round(float(np.mean(stage_timings["tracker"])), 2),
            "motion_estimation": round(float(np.mean(stage_timings["motion"])), 2),
            "ttc_prediction": round(float(np.mean(stage_timings["ttc"])), 2),
            "risk_engine": round(float(np.mean(stage_timings["risk"])), 2),
            "agent_decision": round(float(np.mean(stage_timings["agent"])), 2),
            "visualization_rendering": round(float(np.mean(stage_timings["render"])), 2),
        },
        "total_pipeline_avg_ms": round(total_avg_ms, 2),
        "total_pipeline_p50_ms": round(float(np.percentile(stage_timings["total"], 50)), 2),
        "total_pipeline_p95_ms": round(float(np.percentile(stage_timings["total"], 95)), 2),
        "end_to_end_fps": round(end_to_end_fps, 1)
    }

    out_file = "data/evaluation/benchmark_real_pipeline.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("  REAL END-TO-END PIPELINE BENCHMARK RESULTS")
    logger.info("=" * 60)
    print("| Stage                      |   Latency (ms) |")
    print("| -------------------------- | -------------: |")
    print(f"| Video decode               |  {summary['stage_averages_ms']['video_decode']:>11.2f} |")
    print(f"| OpenCV preprocessing       |  {summary['stage_averages_ms']['opencv_preprocessing']:>11.2f} |")
    print(f"| YOLOv8 inference           |  {summary['stage_averages_ms']['yolov8_inference']:>11.2f} |")
    print(f"| Tracking                   |  {summary['stage_averages_ms']['tracking']:>11.2f} |")
    print(f"| Motion estimation          |  {summary['stage_averages_ms']['motion_estimation']:>11.2f} |")
    print(f"| TTC prediction             |  {summary['stage_averages_ms']['ttc_prediction']:>11.2f} |")
    print(f"| Risk engine                |  {summary['stage_averages_ms']['risk_engine']:>11.2f} |")
    print(f"| Agent decision             |  {summary['stage_averages_ms']['agent_decision']:>11.2f} |")
    print(f"| Visualization rendering    |  {summary['stage_averages_ms']['visualization_rendering']:>11.2f} |")
    print("| -------------------------- | -------------: |")
    print(f"| Total Pipeline (Avg)       |  {summary['total_pipeline_avg_ms']:>11.2f} |")
    print(f"| Total Pipeline (P50)       |  {summary['total_pipeline_p50_ms']:>11.2f} |")
    print(f"| Total Pipeline (P95)       |  {summary['total_pipeline_p95_ms']:>11.2f} |")
    print(f"| End-to-End Real FPS        |  {summary['end_to_end_fps']:>11.1f} |")
    logger.info("=" * 60)
    logger.info(f"Saved real pipeline benchmark to {out_file}")

    return summary


if __name__ == "__main__":
    run_real_pipeline_benchmark()
