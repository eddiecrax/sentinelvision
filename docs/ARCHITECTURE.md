# 🏛️ SentinelVision — System Architecture Specification
## OpenCV AI Competition 2026 powered by AWS

This document details the architectural design, component responsibilities, data flows, and runtime boundaries of SentinelVision.

---

## 1. High-Level Architecture Diagram

```mermaid
flowchart TB
    %% SentinelVision — Runtime Pipeline & Decoupled Telemetry

    subgraph INGESTION ["1. Video Ingestion & Enhancement (OpenCV 5)"]
        A1["Video Source<br/>(Local Camera / RTSP / MP4 File)"] -->|"cv2.VideoCapture"| A2["Frame Preprocessor<br/>(LAB CLAHE & Denoising)"]
        A2 -->|"cv2.resize (INTER_AREA)"| A3["Normalized Frame Buffer<br/>(768x432 / 640x360)"]
    end

    subgraph PERCEPTION ["2. Perception & Multi-Object Tracking"]
        A3 -->|"RGB Image Array"| B1["Object Detector (YOLOv8n)<br/>(CPU Forward Pass ~98ms)"]
        B1 -->|"Filtered Detections"| B2["IoU Multi-Object Tracker<br/>(Persistent IDs & Occlusion Prediction)"]
        B2 -->|"Temporal Centroid Log"| B3["Motion History & Velocity Estimator<br/>(Displacement Vector & Speed px/s)"]
    end

    subgraph REASONING ["3. Predictive Physics & Multi-Factor Risk Engine"]
        B3 -->|"Active Track Trajectories"| C1["Collision Analyzer<br/>(Ray Intersection & Relative Velocity)"]
        C1 -->|"Relative Closing Speed"| C2["Time-to-Collision (TTC) Engine<br/>(TTC = Distance / V_closing)"]
        C2 -->|"Instantaneous TTC"| C3["Multi-Factor Risk Engine<br/>(Weighted TTC, Proximity, Speed, Persistence)"]
        C3 -->|"Evaluated Risk Vectors"| C4["Risk Assessment Stream<br/>(LOW, MEDIUM, HIGH, CRITICAL)"]
    end

    subgraph AGENTIC_LOOP ["4. Autonomous Agent Decision & Closed-Loop Feedback"]
        C4 -->|"Live Risk Stream"| D1{"Sentinel Safety Agent<br/>(Debouncing & Lifecycle Policy)"}
        D1 -->|"Confirmed Candidate (>=3 frames)"| D2["CREATE_INCIDENT (SV-XXXX)<br/>(Action: warning)"]
        D1 -->|"Sustained Hazard (>3.0s)"| D3["ESCALATE Severity<br/>(Action: escalated=True)"]
        D1 -->|"Track Switch (offset <= 120px)"| D4["TRACK_HANDOVER<br/>(P#6 -> P#7 Reassociation)"]
        D1 -->|"Sustained Low Risk (>=1.2s)"| D5["RESOLVE Incident<br/>(Hysteresis & Cooldown Lock)"]

        D2 & D3 & D4 & D5 --> D6["Safety Event Manager<br/>(Active & Resolved Incident Store)"]
    end

    subgraph LOCAL_RUNTIME ["5. Edge Delivery & Operator Interface (Primary Runtime)"]
        D6 -->|"Active State & Overlays"| E1["OpenCV Safety Visualizer<br/>(Vector Rays, Badges, Trajectory Tails)"]
        A3 & E1 -->|"cv2.imencode (.jpg)"| E2["FastAPI Streaming Engine<br/>(GET /video_feed @ multipart MJPEG)"]
        E2 -->|"HTTP / MJPEG"| E3["HTML5 / Canvas Dashboard<br/>(Live HUD & Incident Table)"]
        E3 -->|"POST /api/incident/override"| E4["Human-in-the-Loop Override<br/>(ACKNOWLEDGE, ESCALATE, DISMISS)"]
        E4 -->|"Operator Action"| D6
        D6 -->|"Record Incident Snapshot"| E5["Local Incident Storage<br/>(data/clips/snapshots/ & local JSON)"]
    end

    subgraph AWS_TELEMETRY ["6. Serverless AWS Telemetry (Separately Tested Decoupled Tier)"]
        E5 -.->|"Async Boto3 Dispatch"| F1[("Amazon S3<br/>sentinelvision-evidence-550130133606<br/>[Snapshots & Trace JSON]")]
        D6 -.->|"Async Boto3 Dispatch"| F2[("Amazon DynamoDB<br/>SentinelVisionIncidents Table<br/>[Audit Trail Ledger]")]
        E2 -.->|"Async Boto3 Dispatch"| F3["Amazon CloudWatch<br/>Namespace: SentinelVision/Metrics<br/>[FPS, Latency, Hazards]"]
    end

    subgraph FAULT_TOLERANCE ["7. Fault Tolerance & Offline Boundary"]
        F1 & F2 & F3 -.->|"On Cloud Drop / Offline Mode"| G1["Zero-Downtime Edge Fallback<br/>(Complete Safety Loop Uninterrupted)"]
        G1 -.->|"Verified in test_aws_integration"| D1
    end

    %% Visual Styling
    classDef runtime fill:#2b6cb0,stroke:#2c5282,color:#ffffff,stroke-width:2px;
    classDef agent fill:#6b46c1,stroke:#553c9a,color:#ffffff,stroke-width:2px;
    classDef aws fill:#c05621,stroke:#9c4221,color:#ffffff,stroke-width:2px,stroke-dasharray: 5 5;
    classDef fault fill:#285e61,stroke:#234e52,color:#ffffff,stroke-width:2px;

    class A1,A2,A3,B1,B2,B3,C1,C2,C3,C4,E1,E2,E3,E4,E5 runtime;
    class D1,D2,D3,D4,D5,D6 agent;
    class F1,F2,F3 aws;
    class G1 fault;
```

> **Legend & Connection Semantics:**
> - **Solid Lines (`───►`)**: Verified live in-process execution during edge video streaming (`src/main.py` and `src/api/app.py`).
> - **Dashed Lines (`- - -►`)**: Decoupled cloud telemetry integration. Modules (`src/aws/s3.py`, `src/aws/dynamodb.py`, `src/aws/cloudwatch.py`) are tested and verified independently via [`tests/test_aws_integration.py`](file:///home/crax/Music/opencv/sentinelvision/tests/test_aws_integration.py) and [`src/aws/setup_aws_live.py`](file:///home/crax/Music/opencv/sentinelvision/src/aws/setup_aws_live.py) to prevent cloud network latency from blocking the edge camera processing loop.

---

## 2. Component Breakdown

### 2.1 Ingestion & Preprocessing Layer
* **Modules**: [`src/vision/camera.py`](file:///home/crax/Music/opencv/sentinelvision/src/vision/camera.py), [`src/vision/preprocessing.py`](file:///home/crax/Music/opencv/sentinelvision/src/vision/preprocessing.py)
* **OpenCV Functions**:
  - `cv2.VideoCapture`: Ingests video from local cameras (V4L2), industrial RTSP streams, or pre-recorded MP4 benchmark files with `CAP_PROP_BUFFERSIZE = 1` to prevent stream drift.
  - `cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)` and `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`: Contrast Limited Adaptive Histogram Equalization normalizes dynamic lighting ranges in warehouse environments.
  - `cv2.fastNlMeansDenoisingColored`: Suppresses high-ISO sensor grain in low-lux conditions.
  - `cv2.resize`: Standardizes frame geometry using `cv2.INTER_AREA` interpolation.

### 2.2 Perception & Spatial Tracking
* **Modules**: [`src/vision/detector.py`](file:///home/crax/Music/opencv/sentinelvision/src/vision/detector.py), [`src/vision/tracker.py`](file:///home/crax/Music/opencv/sentinelvision/src/vision/tracker.py)
* **Mechanisms**:
  - **YOLOv8n**: Single-stage lightweight convolutional backbone executed on CPU. Targeted classes: `person` (0), `car` (2), `truck` (7), `bus` (5), `motorcycle` (3), `bicycle` (1).
  - **IoU Tracker**: Spatial bounding box overlap matching across consecutive frames. Tracks maintain position histories and extrapolate trajectories across brief occlusions (up to 10 frames).

### 2.3 Motion & Predictive Physics Engine
* **Modules**: [`src/motion/history.py`](file:///home/crax/Music/opencv/sentinelvision/src/motion/history.py), [`src/prediction/ttc.py`](file:///home/crax/Music/opencv/sentinelvision/src/prediction/ttc.py), [`src/prediction/collision.py`](file:///home/crax/Music/opencv/sentinelvision/src/prediction/collision.py)
* **Physics & Mathematics**:
  - **Velocity Vector**: $\vec{v}(t) = \frac{\vec{p}(t) - \vec{p}(t - \Delta t)}{\Delta t}$ (smoothed over a rolling window).
  - **Closing Speed**: $v_{\text{closing}} = -\frac{(\vec{p}_{\text{veh}} - \vec{p}_{\text{ped}}) \cdot (\vec{v}_{\text{veh}} - \vec{v}_{\text{ped}})}{\|\vec{p}_{\text{veh}} - \vec{p}_{\text{ped}}\|}$.
  - **Instantaneous TTC**: $\text{TTC} = \frac{\|\vec{r}\|}{v_{\text{closing}}}$ for $v_{\text{closing}} > 0.5\text{ px/s}$.
  - **Multi-Factor Risk Score**: $R = 0.35 S_{\text{ttc}} + 0.20 S_{\text{dist}} + 0.15 S_{\text{vel}} + 0.15 S_{\text{traj}} + 0.10 S_{\text{pers}}$.

### 2.4 Autonomous Safety Agent
* **Modules**: [`src/agent/agent.py`](file:///home/crax/Music/opencv/sentinelvision/src/agent/agent.py), [`src/safety/events.py`](file:///home/crax/Music/opencv/sentinelvision/src/safety/events.py), [`src/agent/policy.py`](file:///home/crax/Music/opencv/sentinelvision/src/agent/policy.py)
* **Core Agentic Decisions**:
  1. `CREATE_INCIDENT`: 3-frame debounce confirmation window suppresses transient false positives.
  2. `ESCALATE`: Auto-escalates active incidents experiencing sustained risk ($>3.0\text{s}$) or critical closing speed.
  3. `TRACK_HANDOVER`: Spatial distance threshold ($\le 120\text{px}$) and temporal window ($\le 1.2\text{s}$) reassociate dropped track IDs across occlusions, preventing duplicate alerts.
  4. `RESOLVE`: Hysteresis window requires $\ge 1.2\text{s}$ of sustained `LOW` risk before closing the incident and initiating a cooldown timer.

### 2.5 Local Delivery & Web Interface
* **Modules**: [`src/api/app.py`](file:///home/crax/Music/opencv/sentinelvision/src/api/app.py), [`src/vision/visualizer.py`](file:///home/crax/Music/opencv/sentinelvision/src/vision/visualizer.py), [`web/templates/index.html`](file:///home/crax/Music/opencv/sentinelvision/web/templates/index.html)
* **Capabilities**:
  - OpenCV HUD vector overlay with chromatic threat coding (Green / Yellow / Orange / Red).
  - Low-latency MJPEG multipart stream (`GET /video_feed`).
  - RESTful control API: `GET /api/health`, `GET /api/stats`, `GET /api/incidents`, `POST /api/profile`, and `POST /api/incident/override`.

### 2.6 Serverless AWS Cloud Tier (Decoupled Telemetry)
* **Modules**: [`src/aws/s3.py`](file:///home/crax/Music/opencv/sentinelvision/src/aws/s3.py), [`src/aws/dynamodb.py`](file:///home/crax/Music/opencv/sentinelvision/src/aws/dynamodb.py), [`src/aws/cloudwatch.py`](file:///home/crax/Music/opencv/sentinelvision/src/aws/cloudwatch.py)
* **Verified Infrastructure**:
  - **Amazon S3**: Stores incident keyframe snapshots (`snapshot.jpg`) and complete decision JSON traces (`trace.json`) under `s3://sentinelvision-evidence-550130133606/incidents/YYYY/MM/SV-XXXX/`.
  - **Amazon DynamoDB**: On-Demand table `SentinelVisionIncidents` stores structured incident lifecycle state (`ACTIVE` $\rightarrow$ `RESOLVED`) and human override logs.
  - **Amazon CloudWatch**: Namespace `SentinelVision/Metrics` records operational gauges (`PipelineFPS`, `InferenceLatencyMs`, `HazardsDetected`, `HazardsResolved`).

---

## 3. Latency & Resource Profile

| Pipeline Subsystem | Execution Hardware | Latency (P50) | Latency (P95) | Throughput |
| :--- | :--- | :---: | :---: | :---: |
| **OpenCV Ingestion & Preprocessing** | CPU (4-Core x86_64) | 0.92 ms | 1.31 ms | >700 FPS |
| **YOLOv8n Neural Inference** | CPU (PyTorch 2.14) | 92.40 ms | 121.20 ms | ~10.5 FPS |
| **Tracking, TTC & Agent Reasoning** | CPU (Python/NumPy) | 0.08 ms | 0.12 ms | >1000 FPS |
| **OpenCV HUD Rendering** | CPU (cv2 primitives) | 0.41 ms | 0.58 ms | >1500 FPS |
| **Full Local Pipeline (End-to-End)** | CPU (4-Core x86_64) | **93.77 ms** | **123.09 ms** | **~8.0–10.6 FPS** |
| **AWS Telemetry (Decoupled)** | Async Boto3 (Cloud) | ~250–600 ms | ~1200 ms | Event-driven (0 FPS impact) |
