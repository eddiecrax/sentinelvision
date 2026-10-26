# SentinelVision — Technical Report
## Autonomous Vision-Based Industrial Hazard Perception & Agentic Incident Mitigation
### OpenCV AI Competition 2026 powered by AWS

**Author**: Eduardo Kagyezaho
**Date**: October 26, 2026
**Repository**: `sentinelvision`
**OpenCV Version**: `5.0.0.93` (Python)
**PyTorch Version**: `2.14.0+cpu`
**AWS Region**: `us-east-1` (Account ID: `550130133606`)
**Demonstration Video**: [`submission/sentinelvision_official_demo.mp4`](file:///home/crax/Music/opencv/sentinelvision/submission/sentinelvision_official_demo.mp4) (101.5 seconds, 1080p, H.264/AAC)

---

### Abstract

Industrial facilities, manufacturing plants, logistics hubs, and construction zones experience severe collision hazards involving heavy mobile machinery and walking personnel. Existing industrial vision deployments typically operate as static, passive detectors: they draw bounding boxes around pedestrians or forklifts and sound unvarying alarms, leading to chronic alarm fatigue and operator disengagement.

**SentinelVision** introduces an autonomous **Agentic Vision** safety system. Powered by OpenCV 5 and lightweight YOLOv8 on CPU, SentinelVision observes dynamic interactions between vehicles and pedestrians, estimates real-time relative closing velocities and Time-to-Collision (TTC), evaluates multi-factor risk, and executes an autonomous closed-loop decision workflow: **Perceive → Predict → Decide → Act → Observe → Resolve/Escalate**.

The system operates locally on edge hardware with a FastAPI MJPEG streaming HUD and integrates with a serverless AWS Cloud Data Tier (Amazon S3, Amazon DynamoDB, and Amazon CloudWatch) with zero idle-compute overhead. SentinelVision includes built-in fault tolerance, continuing edge safety monitoring without interruption even during complete cloud network outages.

---

### 1. Problem Statement & Target Users

According to OSHA and international workplace safety registries, struck-by incidents involving forklifts and mobile equipment remain among the top causes of occupational fatalities. In warehouses and factory loading bays:
- Sightlines are obstructed by high shelving, blind corners, and machinery pillars.
- Workers and mobile equipment share tight thoroughfares.
- Standard passive cameras record incidents for post-hoc forensic review but fail to intervene proactively.
- Conventional threshold-based motion detectors generate high false-positive rates when vehicles and pedestrians travel safely in parallel, causing operators to ignore or disable safety warnings.

**Target Users**: Safety managers, warehouse operations leads, factory floor supervisors, and equipment operators in warehousing, automotive manufacturing, logistics sorting centers, and construction sites.

---

### 2. Core Technical Differentiation

| Capability | Conventional Safety Vision | SentinelVision Agentic Pipeline |
| :--- | :--- | :--- |
| **Detection Paradigm** | Single-frame object tagging (isolated boxes) | Temporal trajectory & persistent multi-object tracking |
| **Hazard Metric** | Static proximity distance (bounding box overlap) | Dynamic Time-to-Collision (TTC) & closing velocity vectors |
| **Action Capability** | Passive recording or fixed unvarying siren | Autonomous tiered agentic intervention loop (Debounce, Warn, Escalate, Resolve) |
| **Feedback Mechanism** | None (open loop) | Closed-loop re-observation: verifies if hazard cleared or escalates |
| **Occlusion Handling** | Drops track, restarts counter, causes duplicate alert | Spatial track handover ($\le 120\text{px}$, $\le 1.2\text{s}$) maintains incident continuity |
| **Cloud Dependency** | Tight coupling; video fails if cloud drops | Edge-first autonomy; cloud telemetry with zero-downtime offline fallback |
| **Cloud Cost** | Expensive continuous cloud video streaming | Serverless event-driven data tier (S3 snapshots + DynamoDB On-Demand telemetry) |

---

### 3. OpenCV 5 Substantive Role

OpenCV 5 serves as the core computational foundation for image enhancement, video decoding, spatial visualization, and streaming:

1. **Adaptive Image Filtering (`src/vision/preprocessing.py`)**:
   - `cv2.fastNlMeansDenoisingColored`: Suppresses sensor grain and sensor thermal noise in low-lux factory environments.
   - `cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)` + `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`: Contrast Limited Adaptive Histogram Equalization normalizes uneven illumination caused by skylights, overhead high-bay lamps, and shadows.
2. **Video Capture & Multi-Source Ingestion (`src/vision/camera.py`)**:
   - Unified `cv2.VideoCapture` architecture supporting local webcams, industrial MP4 benchmark clips, and RTSP video streams via `cv2.CAP_FFMPEG` with minimal input buffer latency (`CAP_PROP_BUFFERSIZE = 1`).
3. **Safety HUD & Spatial Trajectory Rendering (`src/vision/visualizer.py`)**:
   - Dynamic collision vector rays (`cv2.line`) rendered between converging entities with chromatic risk coding.
   - Spatial risk badges and TTC overlays rendered with `cv2.putText`, `cv2.getTextSize`, and `cv2.rectangle`.
   - Velocity history trails rendered with temporal alpha transparency.
4. **Encoding & Multipart Streaming (`src/api/app.py`, `src/aws/s3.py`)**:
   - In-memory `cv2.imencode('.jpg', ...)` produces multipart MJPEG frames for browser streaming and direct byte-buffer payloads for S3 keyframe uploads.

---

### 4. Perception, Tracking, Motion & Risk Logic

#### A. Detection & Tracking
- **Detector**: YOLOv8n running in CPU mode. Inference is restricted to relevant target classes: `person` (0), `car` (2), `truck` (7), `bus` (5), `motorcycle` (3), and `bicycle` (1).
- **Tracker**: Intersection-over-Union (IoU) tracker maintaining persistent identity across frame sequences with a velocity-extrapolating bounding box predictor during temporary occlusions (up to 10 frames).

#### B. Motion & Velocity Estimation
For each tracked entity $i$, velocity vector $\vec{v}_i = [v_x, v_y]$ is computed over a temporal smoothing window $\Delta t$:
$$\vec{v}_i(t) = \frac{\text{center}_i(t) - \text{center}_i(t - \Delta t)}{\Delta t}$$

#### C. Time-to-Collision (TTC) Formulation
Between person $A$ and vehicle $B$, relative position is $\vec{r} = \vec{p}_B - \vec{p}_A$ and relative velocity is $\vec{v}_{\text{rel}} = \vec{v}_B - \vec{v}_A$.
Closing velocity $v_{\text{closing}}$ is the projection of relative velocity along the line connecting the centers:
$$v_{\text{closing}} = -\frac{\vec{r} \cdot \vec{v}_{\text{rel}}}{\|\vec{r}\|}$$
If $v_{\text{closing}} > 0.5\text{ px/s}$, the objects are converging and Instantaneous Time-to-Collision is:
$$\text{TTC} = \frac{\|\vec{r}\|}{v_{\text{closing}}}$$
If $v_{\text{closing}} \le 0.5\text{ px/s}$, $\text{TTC} = \infty$ (diverging or stationary).

#### D. Instantaneous TTC vs. Advance Warning Lead Time
A critical engineering distinction verified in real video:
- **Instantaneous Geometric TTC ($0.61\text{s} - 0.67\text{s}$)**: The dynamic kinematic calculation at the moment of detection (Frame 330) based on immediate closing speed ($45\text{ px/s}$) and separation distance ($29\text{ px}$).
- **Advance Warning Lead Time ($1.58\text{s}$)**: The actual real-world elapsed time between initial warning dispatch at Frame 330 and the physical moment of closest approach at Frame 349 ($19\text{ frames} / 12\text{ fps} = 1.58\text{s}$).

#### E. Multi-Factor Risk Engine
Risk score $R \in [0.0, 1.0]$ is computed as a weighted sum of normalized factors:
$$R = w_{\text{ttc}} S_{\text{ttc}} + w_{\text{dist}} S_{\text{dist}} + w_{\text{vel}} S_{\text{vel}} + w_{\text{traj}} S_{\text{traj}} + w_{\text{pers}} S_{\text{pers}}$$
- Weights: $w_{\text{ttc}} = 0.35, w_{\text{dist}} = 0.20, w_{\text{vel}} = 0.15, w_{\text{traj}} = 0.15, w_{\text{pers}} = 0.10$.
- Thresholds: $R < 0.25$ (LOW), $0.25 \le R < 0.50$ (MEDIUM), $0.50 \le R < 0.75$ (HIGH), $R \ge 0.75$ (CRITICAL).

---

### 5. The Agentic Perception-Decision-Action Feedback Loop

Visual evidence directly alters subsequent incident actions across the encounter lifecycle:

1. **Debounced Creation (`CREATE_INCIDENT`)**: Candidate pairs must persist for $\ge 3$ consecutive frames or $0.25\text{s}$ (unless critical $TTC \le 0.8\text{s}$ triggers immediate dispatch). In `factory_forklift.mp4`, candidate registered at Frame 328 ($t=30.08\text{s}$), confirmed at Frame 330 ($t=30.25\text{s}$) creating `SV-0001`.
2. **Autonomous Escalation (`ESCALATE`)**: Active incidents with sustained high risk or age $>3.0\text{s}$ transition to `escalated = True` ($t=30.85\text{s}$).
3. **Spatial Track Handover (`TRACK_HANDOVER`)**: When Person #6 occluded and tracker minted Person #7, the agent detected spatial offset ($60.3\text{px} \le 120\text{px}$) and handed over ownership of `SV-0001` at $t=33.15\text{s}$ rather than triggering a false duplicate incident.
4. **Hysteresis Auto-Resolution (`RESOLVE`)**: When worker and vehicle diverged, risk fell to `LOW`. The agent enforced a 1.2s grace period of sustained low risk before closing `SV-0001` at $t=34.95\text{s}$ (`resolution_time = 4.71s`).

Ground-truth reproducible artifact: [`evidence/agent/verified_real_video_trace_SV-0001.json`](file:///home/crax/Music/opencv/sentinelvision/evidence/agent/verified_real_video_trace_SV-0001.json) (62 timeline steps).

---

### 6. Human Oversight & Operational Control

Through [`POST /api/incident/override`](file:///home/crax/Music/opencv/sentinelvision/src/api/app.py#L118-L145), human operators maintain supervisory control over the autonomous agent:
* `ACKNOWLEDGE`: Operator marks incident seen (`action_taken = "OPERATOR_ACKNOWLEDGED"`, audit trace added).
* `ESCALATE`: Operator overrides agent priority (`escalated = True`, `action_taken = "OPERATOR_ESCALATED"`).
* `DISMISS`: Operator resolves benign near-miss (`resolved = True`, `action_taken = "OPERATOR_DISMISSED"`).

All actions are logged to local storage and mirrored to AWS DynamoDB audit trails.

---

### 7. AWS Cloud Architecture: Verified Integration vs. Runtime Decoupling

SentinelVision implements an edge-first architecture where safety-critical decisions never depend on cloud latency.

| AWS Service | Configured Resource | Verified Functionality | Status |
| :--- | :--- | :--- | :---: |
| **Amazon S3** | `sentinelvision-evidence-550130133606` | Live upload of JPEG incident keyframes and JSON audit traces | **🟢 VERIFIED LIVE** |
| **Amazon DynamoDB** | `SentinelVisionIncidents` (On-Demand) | Real-time incident item creation and status updates (`ACTIVE` → `RESOLVED`) | **🟢 VERIFIED LIVE** |
| **Amazon CloudWatch**| `SentinelVision/Metrics` | Telemetry publishing (`PipelineFPS`, `InferenceLatencyMs`, `HazardsDetected`) | **🟢 VERIFIED LIVE** |

#### Runtime Separation Note:
The primary video streaming and detection loop in `src/main.py` and `src/api/app.py` writes to local storage (`data/clips/snapshots/`) to guarantee deterministic edge frame rates. AWS S3, DynamoDB, and CloudWatch clients are fully implemented in `src/aws/` and independently verified via [`tests/test_aws_integration.py`](file:///home/crax/Music/opencv/sentinelvision/tests/test_aws_integration.py) and [`src/aws/setup_aws_live.py`](file:///home/crax/Music/opencv/sentinelvision/src/aws/setup_aws_live.py).

#### Edge Fault Tolerance:
In [`tests/test_aws_integration.py:87-113`](file:///home/crax/Music/opencv/sentinelvision/tests/test_aws_integration.py#L87-L113), when cloud network disconnection was simulated, the local safety agent continued evaluating hazards and creating incidents with **100% test success and zero unhandled exceptions**.

---

### 8. Evaluation & Benchmarking Summary

1. **Real-Video Pipeline Benchmark (`benchmark_real_pipeline.py`)**:
   - Video: `data/videos/warehouse_forklift.mp4` (720x404)
   - Total End-to-End Latency: **P50 = 93.77 ms, P95 = 123.09 ms (~8.0–10.6 FPS on CPU)**
   - YOLOv8n forward pass represents 98.6% of pipeline time; OpenCV and agent reasoning account for <1.5 ms.
2. **Controlled Failure Scenarios (`test_failures.py`)**:
   - `F1` (Occlusion retention): **PASS**
   - `F2` (1-frame noise suppression): **PASS**
   - `F3` (Dense multi-object tracking): **PASS** (5/5 preserved)
   - `F4` (Parallel non-hazard trajectory discrimination): **PASS** (LOW risk)
   - `F5` (Head-on collision escalation): **PASS** (CREATE_INCIDENT)
   - `F6` (Sensor blackout diagnostic): **PASS** (CAMERA_DEGRADED)
3. **Tangible Safety Benchmark (`benchmark_tangible_safety.py`)**:
   - Encounter Recall: **100.0% (2/2 ground-truth encounters detected)**
   - Alarm Fatigue Reduction: **50.0% reduction in fragmented alerts**
   - Advance Warning Lead Time: **2.04 seconds average**
   - OSHA Required Stopping Window: **1.55 seconds** (0.75s reaction + 0.80s mechanical braking @ 2.0 m/s)
   - Safety Margin: **+0.49 seconds (+0.98 meters buffer before impact)**

---

### 9. Limitations & Boundary Conditions

1. **COCO Class Proxies**: Standard `yolov8n.pt` weights do not contain a native `forklift` class. Industrial vehicles are mapped using `truck` or `car` proxies. Production deployments require fine-tuning on industrial machinery datasets.
2. **CPU Throughput**: Forward passes average ~98 ms on CPU (~10 FPS). High-speed vehicular applications (>15 m/s) require GPU acceleration or AWS Graviton NEON optimization.
3. **Prototype Scope**: SentinelVision is an AI-assisted situational awareness advisory system, not a safety-certified machinery emergency-stop interlock under ISO 13849.
4. **Privacy Protection**: Zero facial recognition or biometric profiling is performed. All tracking is based on bounding box centroids.
