# 🛡️ SentinelVision

**Autonomous Vision-Based Industrial Hazard Perception & Agentic Incident Mitigation**  
*OpenCV AI Competition 2026 powered by AWS*

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![OpenCV 5.0.0](https://img.shields.io/badge/OpenCV-5.0.0-green.svg)](https://opencv.org/)
[![PyTorch 2.14 CPU](https://img.shields.io/badge/PyTorch-2.14%2Bcpu-red.svg)](https://pytorch.org/)
[![AWS Serverless](https://img.shields.io/badge/AWS-Serverless-orange.svg)](https://aws.amazon.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Industrial warehouses, factories, and construction sites experience frequent collision hazards between mobile machinery and pedestrians. Most facility cameras are passive: they record incidents after they occur or trigger unvarying, unranked sirens that lead to operator alarm fatigue.

**SentinelVision** is an edge-first **Agentic Vision** safety system. Powered by OpenCV 5 and lightweight CPU-optimized YOLOv8, SentinelVision observes dynamic interactions between vehicles and pedestrians, estimates real-time relative closing velocities and Time-to-Collision (TTC), evaluates multi-factor risk, and executes an autonomous closed-loop decision workflow:

$$\text{Perceive} \longrightarrow \text{Predict} \longrightarrow \text{Decide} \longrightarrow \text{Act} \longrightarrow \text{Observe} \longrightarrow \text{Resolve / Escalate}$$

---

## Key Differentiators

- **Substantive OpenCV 5 Pipeline**: Low-lux denoising (`fastNlMeansDenoisingColored`), adaptive contrast normalization (`createCLAHE` in LAB space), multi-source video capture (`cv2.VideoCapture`), dynamic vector HUD rendering (`cv2.line`, `cv2.rectangle`, `cv2.putText`), and multipart JPEG streaming (`cv2.imencode`).
- **Autonomous Agentic Decision Loop**: 28 autonomous actions logged across a closed-loop incident cycle (`INCREASE_MONITORING` $\rightarrow$ `CREATE_INCIDENT` $\rightarrow$ `VISUAL_WARNING` $\rightarrow$ `RESOLVE`). The agent verifies whether conditions improve and automatically resolves incidents once hazards clear.
- **Serverless AWS Cloud Telemetry**: Live S3 incident keyframe snapshot and audit trace storage, DynamoDB persistent incident ledger, and CloudWatch operational metrics publishing.
- **Zero-Downtime Edge Fault Tolerance**: Local safety loop operates with 100% autonomy and zero crashes during complete cloud network outages.

---

## System Architecture

```text
[Camera / RTSP / MP4]
       │
       ▼
[OpenCV 5 Ingestion & Denoising/CLAHE] ──> [YOLOv8 CPU Detection]
                                                     │
                                                     ▼
                                            [IoU Multi-Object Tracker]
                                                     │
                                                     ▼
                                            [Motion & Velocity Extrapolation]
                                                     │
                                                     ▼
                                            [TTC & Spatial Risk Engine]
                                                     │
                                                     ▼
                                            [Sentinel Safety Agent]
                                            ┌────────┴────────┐
                                            ▼                 ▼
                                    [Local Delivery]   [AWS Cloud Tier]
                                    - FastAPI MJPEG    - Amazon S3 (Snapshots)
                                    - OpenCV HUD       - DynamoDB (Ledger)
                                    - Operator Panel   - CloudWatch (Metrics)
```

*(Full architectural specification available in [`submission/architecture.mmd`](submission/architecture.mmd)).*

---

## Model Scope & Important Limitations

1. **Pretrained Model Classes vs. Forklifts**:
   The baseline `yolov8n.pt` detector uses the standard 80 COCO classes (`person`, `car`, `truck`, `bus`, `motorcycle`, `bicycle`). **Standard COCO does not possess a native `forklift` class.** In industrial profiles, vehicles are mapped using vehicle proxies (`truck`/`car`) or synthetic scenarios. Production deployment requires fine-tuning on domain-specific forklift datasets.
2. **CPU Inference Latency**:
   YOLOv8 forward passes average ~98 ms on CPU (~10 FPS throughput on 4-thread CPU). High-speed closing collisions (>15 m/s) require GPU acceleration or AWS Graviton NEON optimization.
3. **Situational Awareness Advisory**:
   SentinelVision is an algorithmic situational-awareness prototype and is **not** certified under ISO 13849 as a physical safety-rated emergency stop interlock.
4. **Synthetic Decision Benchmark vs. Real-World Video**:
   The 100% precision and recall metrics reported in `tests/test_evidence_suite.py` evaluate algorithmic incident classification decisions across 100 controlled synthetic trajectory scenarios. They validate the decision state-machine logic in isolation, not bounding-box detection accuracy or real-world industrial safety performance. End-to-end processing of real video (`benchmark_real_pipeline.py`) currently runs at ~8 FPS on a 4-core CPU.

---

## Quickstart

### Prerequisites
- Python `3.12` (recommended) or Docker.
- Minimum 2 CPU cores, 4 GB RAM. (No GPU required).

### Option A: Local Python Environment

```bash
# 1. Clone repository
git clone <YOUR_REPOSITORY_URL> sentinelvision
cd sentinelvision

# 2. Create and activate Python 3.12 environment
conda create -n sentinelvision python=3.12 -y
conda activate sentinelvision
# Or: python3 -m venv venv && source venv/bin/activate

# 3. Install CPU-optimized PyTorch first
pip install torch==2.14.0+cpu torchvision==0.29.0+cpu --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Model Weights
# The default model is yolov8n.pt. It will auto-download on first execution, or pre-fetch with:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 6. Configure Environment (Optional AWS Cloud Sync)
cp .env.example .env
# Leave AWS keys blank to run in 100% offline local mode.

# 7. Start the Web Dashboard & Live Stream
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000/`** (or `http://localhost:8000/`) in your browser.  
*(In the top-right Video Feed dropdown, select **Pedestrian Walkway** or **Traffic Flow** to observe live multi-person/vehicle detection and dynamic vector HUD).*

#### Running via CLI
```bash
python src/main.py --source file --path data/videos/test_traffic.mp4 --no-display
```

---

### Option B: Clean Docker Container

The container provides an isolated, verified CPU-only runtime:

```bash
# 1. Ensure yolov8n.pt is in the project root
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 2. Build production Docker image
docker build -t sentinelvision:latest -f deployment/Dockerfile .

# 3. Run container locally
docker run -d --name sentinelvision -p 8000:8000 sentinelvision:latest

# 4. Verify health probes
curl -s http://localhost:8000/health
# {"status":"online","system":"SentinelVision"}

# Open http://localhost:8000/ in your browser
```

---

## Automated Test & Benchmark Suite

SentinelVision includes a full suite of reproducible tests and benchmarks:

```bash
# 1. Industrial Failure-Case Matrix (6 scenarios: occlusion, noise, density, near miss, collision, degradation)
python tests/test_failures.py

# 2. Closed-Loop Agentic Perception-Decision-Action Scenario (28 actions logged)
python tests/test_scenario.py

# 3. Environment Profiles & Evidence Storage Test
python tests/test_phase8_9.py

# 4. AWS Integration & Offline Network Fault Tolerance Test
python tests/test_aws_integration.py

# 5. Real End-to-End Pipeline Latency Benchmark (150 Frames via cv2.TickMeter)
python tests/benchmark_real_pipeline.py

# 6. Full Algorithmic Core Benchmark Suite (100 runs)
python tests/test_evidence_suite.py
```

---

## Project Structure

```
sentinelvision/
├── config/                 # Environment profiles (warehouse, factory, etc.) & system config
├── dashboard/              # HTML5/Canvas real-time safety HUD dashboard
├── data/
│   ├── evaluation/         # Traceable JSON benchmark & evaluation results
│   └── videos/             # Sample video clips (traffic, pedestrians, warehouse)
├── deployment/
│   ├── Dockerfile          # Clean CPU-only production container definition
│   └── aws/                # ECS Fargate task definitions & deployment scripts
├── evidence/               # Detailed test audit logs & integrity reports
├── models/                 # Model weight directory (.gitkeep)
├── src/
│   ├── agent/              # Autonomous safety agent & state machine
│   ├── api/                # FastAPI application & MJPEG streaming endpoints
│   ├── aws/                # AWS S3, DynamoDB, and CloudWatch client modules
│   ├── motion/             # Velocity calculation, history, & trajectory rays
│   ├── prediction/         # Time-to-Collision (TTC) & collision analyzer
│   ├── profiles/           # Dynamic industrial environment loader
│   ├── safety/             # Multi-factor risk engine & incident storage
│   ├── vision/             # OpenCV 5 preprocessor, detector, tracker, HUD visualizer
│   └── main.py             # CLI pipeline entrypoint
├── submission/             # Competition deliverables (Report, Architecture, Checklist, Demo)
└── tests/                  # Automated test & benchmark suite
```

---

## Competition Deliverables Index

For competition evaluators and judges:
- 📄 **[Technical Report](submission/technical_report.md)**: Full engineering report detailing OpenCV 5 substantive role, mathematical formulations, agentic state machine, and evaluation.
- 📐 **[System Architecture](submission/architecture.mmd)**: Mermaid system architecture diagram.
- 📊 **[Evaluation & Evidence Report](submission/evaluation.md)**: Stage-by-stage latency breakdowns and controlled failure evaluations.
- 🎬 **[Video Demonstration Script](submission/demo_script.md)**: 4-minute 30-second timed demo walkthrough script.
- 📋 **[Final Submission Checklist](submission/FINAL_SUBMISSION_CHECKLIST.md)**: Competition requirements checklist and submission field mappings.
- 🔍 **[Final Pre-Submission Audit](submission/FINAL_AUDIT.md)**: Pre-submission verification audit report.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
