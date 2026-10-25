# 🛡️ SentinelVision — Quickstart & Reproducibility Guide

> **OpenCV AI Competition 2026 powered by AWS**  
> Autonomous Vision-Based Industrial Hazard Perception & Agentic Incident Mitigation

---

## 1. System Overview & Core Capabilities

Industrial warehouses, factories, and construction sites experience frequent collision hazards between mobile machinery and pedestrians. SentinelVision is an edge-first **Agentic Vision** safety system powered by OpenCV 5 and lightweight CPU-optimized YOLOv8. It observes dynamic interactions between vehicles and pedestrians, estimates real-time relative closing velocities and Time-to-Collision (TTC), evaluates multi-factor risk, and executes an autonomous closed-loop decision workflow:

$$\text{Perceive} \longrightarrow \text{Predict} \longrightarrow \text{Decide} \longrightarrow \text{Act} \longrightarrow \text{Observe} \longrightarrow \text{Resolve / Escalate}$$

### Substantive OpenCV 5 Integration
- **Low-lux Denoising**: `fastNlMeansDenoisingColored` for low-light camera streams.
- **Adaptive Contrast**: `createCLAHE` in LAB color space for glare and shadow normalization.
- **Multi-Source Capture**: High-throughput `cv2.VideoCapture` with buffer management.
- **Dynamic HUD**: Real-time vector rendering (`cv2.line`, `cv2.rectangle`, `cv2.putText`) displaying closing velocity and TTC.
- **Multipart MJPEG Streaming**: Efficient frame encoding via `cv2.imencode` to FastAPI endpoints.

---

## 2. System Requirements

- **Operating System**: Linux (Ubuntu 22.04+ / Debian 12+) or macOS / Windows with Docker.
- **Python**: Version `3.12` (pinned dependencies).
- **Compute**: Minimum 2 CPU cores, 4 GB RAM. (No GPU required; CPU-optimized PyTorch 2.14).
- **Storage**: ~3 GB disk space for Docker image and models.

---

## 3. Quickstart Option A: Local Python Environment

### Step 1: Clone & Create Environment
```bash
git clone <YOUR_REPOSITORY_URL> sentinelvision
cd sentinelvision

conda create -n sentinelvision python=3.12 -y
conda activate sentinelvision
# Or via venv: python3 -m venv venv && source venv/bin/activate
```

### Step 2: Install Pinned Dependencies
```bash
# Install CPU-optimized PyTorch first
pip install torch==2.14.0+cpu torchvision==0.29.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install remaining competition requirements
pip install -r requirements.txt
```

### Step 3: Obtain Model Weights
Model binaries are kept out of Git version control. Acquire the baseline COCO model using the Ultralytics CLI or Python:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```
*Note: SentinelVision automatically detects model weights in either the project root (`yolov8n.pt`) or the `models/` directory (`models/yolov8n.pt`).*

### Step 4: Configure Environment (Optional AWS Cloud Sync)
Copy the example environment configuration:
```bash
cp .env.example .env
```
- **Offline Mode (Default)**: Leave AWS credentials in `.env` blank. SentinelVision automatically falls back to local storage, logging all incident evidence snapshots to `data/clips/snapshots/` with zero cloud dependencies.
- **Live AWS Cloud Mode**: Provide `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION` to stream operational metrics to CloudWatch, store incident audit items in DynamoDB, and upload keyframe snapshots to S3.

### Step 5: Run the Application
Start the real-time FastAPI service and visual dashboard:
```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```
Open your browser to:
- **Dashboard HUD**: `http://127.0.0.1:8000/` (or `http://localhost:8000/`)  
  *(Use the top-right Video Feed selector to switch between Pedestrian Walkway, Traffic Flow, or Factory feeds).*
- **Live MJPEG Stream**: `http://localhost:8000/video_feed`
- **System Health Probes**: `http://localhost:8000/health` and `http://localhost:8000/api/health`

#### Running via CLI Pipeline
```bash
python src/main.py --source file --path data/videos/test_traffic.mp4 --no-display
```

---

## 4. Quickstart Option B: Clean Docker Container

The container runs a fully self-contained CPU environment with OpenCV 5, PyTorch CPU, and YOLOv8:

### Step 1: Ensure Model Weights Exist
The container build requires `yolov8n.pt` in the project root:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Step 2: Build the Production Container
```bash
docker build -t sentinelvision:latest -f deployment/Dockerfile .
```

### Step 3: Run the Container Locally (Offline Mode)
```bash
docker run -d --name sentinelvision -p 8000:8000 sentinelvision:latest
```

### Step 4: Verify Health Probe & Access
```bash
curl -s http://localhost:8000/health
# Response: {"status":"online","system":"SentinelVision"}
```
Open `http://localhost:8000/` in your web browser.

### Optional: Running Docker with Live AWS Cloud Credentials
```bash
docker run -d --name sentinelvision -p 8000:8000 \
  --env-file .env \
  sentinelvision:latest
```

---

## 5. Reproducing Tests & Benchmarks

Run the complete test and evaluation suite locally:

```bash
# 1. Edge Case & Failure Mode Matrix (Occlusion, Noise, Density, Near Miss, Collision, Degradation)
python tests/test_failures.py

# 2. Agentic Closed-Loop Perception-Action Scenario (28 Autonomous Actions)
python tests/test_scenario.py

# 3. Environment Profiles & Incident Evidence Storage Test
python tests/test_phase8_9.py

# 4. AWS Integration & Offline Network Fault Tolerance Test
python tests/test_aws_integration.py

# 5. Real End-to-End Pipeline Latency Benchmark (150 Frames via cv2.TickMeter)
python tests/benchmark_real_pipeline.py

# 6. Full Evidence Benchmark Suite (100 runs)
python tests/test_evidence_suite.py
```

---

## 6. Competition Deliverables Index

All competition artifacts and technical documentation are available in the repository:
- 📄 **[Technical Report](technical_report.md)**: Full engineering paper detailing OpenCV 5 substantive role, mathematical formulations, agentic state machine, and evaluation.
- 📐 **[System Architecture](architecture.mmd)**: Mermaid architectural diagram.
- 📊 **[Evaluation & Evidence Report](evaluation.md)**: Stage-by-stage latency breakdowns, real-video performance, and controlled failure evaluations.
- 🎬 **[Video Demonstration Script](demo_script.md)**: 4-minute 30-second timed demo walkthrough script.
- 📋 **[Final Submission Checklist](FINAL_SUBMISSION_CHECKLIST.md)**: Competition requirements checklist and submission field mappings.
- 🔍 **[Final Pre-Submission Audit](FINAL_AUDIT.md)**: Pre-submission verification audit report.

---

## 7. Security & Secret Management

- **Zero Baked Secrets**: The Docker image and Git repository contain no embedded API keys or credentials.
- **Environment Isolation**: Cloud credentials are read strictly from runtime environment variables or `~/.aws/credentials`.
- **Offline Resilience**: When credentials are missing or network connectivity is severed, SentinelVision automatically operates in local fallback mode, writing evidence snapshots to `data/clips/snapshots/` with zero crash risk.

---

## 8. Known Model Scope & Limitations

1. **Pretrained COCO Classes**: Uses standard COCO-pretrained `yolov8n.pt` with classes restricted to `person`, `car`, `truck`, `bus`, `motorcycle`, and `bicycle`.
2. **Industrial Vehicle Proxy**: Standard COCO does not possess a native `forklift` class. In factory/warehouse environments, industrial vehicles are mapped using vehicle proxies (`truck`/`car`). Production deployment requires fine-tuning on domain-specific forklift datasets.
3. **Safety Rating**: Prototype safety system for situational awareness. Not certified as an ISO 13849 physical safety-rated emergency stop interlock.
4. **Synthetic Decision Benchmark vs. Real Video**: The 100% precision and recall metrics reported in `tests/test_evidence_suite.py` evaluate algorithmic incident classification decisions across 100 controlled synthetic trajectory scenarios. They validate the decision state-machine logic in isolation, not bounding-box detection accuracy or real-world safety performance. Real-video throughput currently runs at ~8 FPS on a 4-core CPU.
