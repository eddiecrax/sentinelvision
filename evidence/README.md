# 🛡️ SentinelVision — Competition Evidence Directory

This directory contains reproducible, defensible evidence artifacts supporting our submission for the **OpenCV AI Competition 2026** (specifically targeting **Technical Execution**, **OpenCV 5 Substantive Integration**, and the **Agentic Vision Special Award**).

---

## 📁 Evidence Directory Index

```text
evidence/
├── benchmarks/
│   ├── real_pipeline_stage_breakdown.json
│   ├── pipeline_overhead_comparison.json
│   └── decision_loop_scenarios.json
├── failures/
│   └── failure_test_results.json
├── agent/
│   └── sample_incident_traces.json
├── aws/
│   ├── architecture.md
│   └── deployment_config.json
└── README.md
```

---

## 📊 Summary of Evidence Artifacts

### 1. Stage-by-Stage Latency (`benchmarks/real_pipeline_stage_breakdown.json`)
* Measured using `cv2.TickMeter` on CPU (`720x404 @ 59.9 FPS`).
* Demonstrates exact stage-by-stage timings: Decode (`0.89 ms`), Preprocessing (`0.09 ms`), YOLOv8 (`93.31 ms`), Tracking/Motion/TTC/Agent (`< 0.05 ms`), Rendering (`0.39 ms`).

### 2. 3-Configuration Overhead Comparison (`benchmarks/pipeline_overhead_comparison.json`)
* Proves that adding the full predictive agent engine adds `< 4.5%` overhead over raw YOLO detection.
* **Config A (YOLO Only)**: `9.9 FPS` (`100.71 ms` avg)
* **Config B (YOLO + Tracking)**: `9.0 FPS` (`111.30 ms` avg)
* **Config C (Full SentinelVision)**: `9.5 FPS` (`105.24 ms` avg)

### 3. Failure & Edge Case Matrix (`failures/failure_test_results.json`)
* 6 verified edge case tests: Occlusion retention (F1), Noise suppression (F2), High object density (F3), Near miss (F4), Direct collision trajectory (F5), Degraded camera confidence (F6).

### 4. Auditable Agentic Vision Traces (`agent/sample_incident_traces.json`)
* Complete JSON timeline traces proving perception $\rightarrow$ prediction $\rightarrow$ action $\rightarrow$ re-observation loop.

### 5. Defensible AWS Architecture (`aws/architecture.md`)
* Edge-first architecture pushing safety incidents to S3, DynamoDB, and CloudWatch.
