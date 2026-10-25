# SentinelVision — Technical Report
## Autonomous Vision-Based Industrial Hazard Perception & Agentic Incident Mitigation

**Competition**: OpenCV AI Competition 2026 powered by AWS  
**Author**: Eduardo Kagyezaho  
**Date**: October 24, 2026  
**Repository**: `sentinelvision`  
**OpenCV Version**: 5.0.0 (Python)  
**Cloud Infrastructure**: Amazon Web Services (Serverless Event-Driven Data Tier, Region: `us-east-1`, AWS account: `550130133606`)  

---

### Abstract

Industrial facilities, manufacturing plants, logistics hubs, and construction zones experience severe collision hazards involving heavy mobile machinery and walking personnel. Existing industrial vision deployments typically operate as static, passive detectors: they draw bounding boxes around pedestrians or forklifts and sound unranked alarms, leading to chronic alarm fatigue and operator disengagement.

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
| **Action Capability** | Passive recording or fixed unvarying siren | Autonomous tiered agentic intervention loop (28 distinct actions) |
| **Feedback Mechanism** | None (open loop) | Closed-loop re-observation: verifies if hazard cleared or escalates |
| **Cloud Dependency** | Tight coupling; video fails if cloud drops | Edge-first autonomy; cloud telemetry with zero-downtime offline fallback |
| **Cloud Cost** | Expensive continuous cloud video streaming | Serverless event-driven data tier (S3 snapshots + DynamoDB On-Demand telemetry) |

---

### 3. System Architecture & Data Flow

SentinelVision is structured into four decoupled layers: Ingestion/Perception, Spatial Reasoning, Agentic Control, and Delivery/Telemetry.

```text
[Camera/RTSP/MP4]
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

The detailed diagram is specified in `submission/architecture.mmd`.

---

### 4. OpenCV 5 Substantive Role

OpenCV 5 serves as the computational foundation for image enhancement, video decoding, spatial visualization, and streaming:

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

### 5. Perception, Tracking, Motion & Risk Logic

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
If $v_{\text{closing}} > 0.5\text{ px/s}$, the objects are converging and Time-to-Collision is:
$$\text{TTC} = \frac{\|\vec{r}\|}{v_{\text{closing}}}$$
If $v_{\text{closing}} \le 0.5\text{ px/s}$, $\text{TTC} = \infty$ (diverging or stationary).

#### D. Multi-Factor Risk Engine
Risk score $R \in [0.0, 1.0]$ is computed as a weighted sum of normalized factors:
$$R = w_{\text{ttc}} S_{\text{ttc}} + w_{\text{dist}} S_{\text{dist}} + w_{\text{vel}} S_{\text{vel}} + w_{\text{traj}} S_{\text{traj}} + w_{\text{pers}} S_{\text{pers}}$$
- Weights: $w_{\text{ttc}} = 0.35, w_{\text{dist}} = 0.20, w_{\text{vel}} = 0.15, w_{\text{traj}} = 0.15, w_{\text{pers}} = 0.10$.
- Thresholds: $R < 0.25$ (LOW), $0.25 \le R < 0.50$ (MEDIUM), $0.50 \le R < 0.75$ (HIGH), $R \ge 0.75$ (CRITICAL).

---

### 6. The Agentic Perception-Decision-Action Feedback Loop

SentinelVision is designed to fulfill the **Agentic Vision Special Award** criteria: visual perception actively alters downstream actions across multiple successive time steps.

```text
[Time Step t0]
Perception: Worker #2 and Forklift #1 converging. TTC = 4.7s.
Risk Level: MEDIUM (Score 0.42).
Agent Action: INCREASE_MONITORING + CREATE_INCIDENT SV-0002.
State: Incident SV-0002 marked ACTIVE. Snapshot queued.

[Time Step t1 (+1.3s)]
Perception: Convergence accelerates. Distance closes to 190px. TTC drops to 3.17s.
Risk Level: HIGH (Score 0.68).
Agent Action: VISUAL_WARNING dispatched to HUD; alert strobe triggered.

[Time Step t2 (+0.3s)]
Perception: Worker steps aside into pedestrian lane. Distance stabilizes at 187px. TTC relaxes to 6.42s.
Risk Level: LOW (Score 0.18).
Agent Action: RESOLVE Incident SV-0002.
State: Incident SV-0002 closed; resolution duration logged as 1.6s; cooldown timer started.
```

The agent maintains an explicit audit trace of every decision step, recording timestamp, observed metric values, policy triggers, and execution status.

---

### 7. Human Oversight & Incident Handling

Autonomous safety systems must never be black boxes. SentinelVision provides multi-tiered human operator integration:
1. **Interactive Safety Dashboard**: Real-time canvas HUD showing active track IDs, speed readouts, TTC countdown badges, and recent incident logs.
2. **Operator Acknowledgement & Override**: Operators can manually acknowledge alerts, trigger an emergency stop override, or adjust sensitivity profiles in real time.
3. **Environment Profiles (`config/profiles/`)**: Pre-calibrated configuration sets for **Warehouse**, **Factory**, **Logistics Hub**, and **Construction Site**, adapting TTC cutoffs, speed limits, and restricted safety zones.

---

### 8. AWS Architecture & Verified Live Integrations

To ensure strict cost control without incurring unpredictable charges, SentinelVision implements a cost-optimized, serverless AWS Cloud Data Tier with on-demand event ingestion. Live end-to-end functionality was verified on AWS account (`550130133606`, Region: `us-east-1`):

| AWS Service | Configured Resource | Verified Functionality | Status |
| :--- | :--- | :--- | :---: |
| **Amazon S3** | `sentinelvision-evidence-550130133606` | Live upload of JPEG incident keyframes and JSON audit traces | **🟢 VERIFIED LIVE** |
| **Amazon DynamoDB** | `SentinelVisionIncidents` (On-Demand) | Real-time incident item creation and status updates (`ACTIVE` → `RESOLVED`) | **🟢 VERIFIED LIVE** |
| **Amazon CloudWatch**| `SentinelVision/Metrics` | Telemetry publishing (`PipelineFPS`, `InferenceLatencyMs`, `HazardsDetected`) | **🟢 VERIFIED LIVE** |
| **Amazon ECS / Fargate**| `sentinelvision-task` | Task definitions authored (`deployment/aws/task-definition.json`); remote compute deferred | **🟡 PREPARED** |

#### Verified AWS Console Artifacts
The live operation of all three AWS services is documented through authentic AWS Management Console artifacts:
- **S3 Audit Trail**: Structured event folder storage under `incidents/2026/10/SV-1001/` with both `trace.json` and keyframe `snapshot.jpg` ([Evidence](file:///home/crax/Music/opencv/sentinelvision/evidence/aws/screenshots/aws_s3_snapshot_console.png)).
- **DynamoDB State Ledger**: Table `SentinelVisionIncidents` scan capturing complete state transitions from detection to resolution ([Evidence](file:///home/crax/Music/opencv/sentinelvision/evidence/aws/screenshots/aws_dynamodb_items_console.png)).
- **CloudWatch Telemetry**: Namespace `SentinelVision/Metrics` live tracking pipeline execution FPS, model latency, and hazard alerts ([Evidence](file:///home/crax/Music/opencv/sentinelvision/evidence/aws/screenshots/aws_cloudwatch_metrics_console.png)).

#### Edge Fault Tolerance
When simulated cloud connection drops were induced in `tests/test_aws_integration.py`, the edge vision pipeline completed all safety evaluations, logged incidents to local disk (`data/clips/snapshots/`), and maintained 100% test passing rate without raising unhandled network exceptions.

---

### 9. Evaluation Results & Controlled Failure Scenarios

#### A. End-to-End Latency Benchmark (150 Frames on Real Video)
Measured on Intel Core i5-6300HQ CPU @ 2.30 GHz (4 cores, 4 threads):
- **Video Decode (OpenCV)**: 0.86 ms
- **Image Preprocessing (OpenCV CLAHE/Resize)**: 0.11 ms
- **YOLOv8n Inference (CPU)**: 98.43 ms
- **Spatial Tracking & Motion**: <0.05 ms
- **TTC & Risk Engine**: <0.05 ms
- **Agent Decision & Dispatch**: <0.05 ms
- **HUD Visualization (OpenCV)**: 0.43 ms
- **Total Pipeline Latency**: **P50 = 93.77 ms, P95 = 123.09 ms (10.0 FPS)**

#### B. Controlled Failure Evaluation Suite (`tests/test_failures.py`)
| Test ID | Edge Condition | Stress Verification | Result |
| :---: | :--- | :--- | :---: |
| **F1** | Temporary Occlusion | Target occluded for 3 frames behind obstacle; trajectory extrapolated | **🟢 PASS** |
| **F2** | Transient Noise | Single-frame false detection spike rejected by persistence filter | **🟢 PASS** |
| **F3** | High Crowd Density | 5 overlapping people in shared aisle; distinct track IDs preserved | **🟢 PASS** |
| **F4** | Non-Converging Trajectories | Parallel transit with 400px lateral offset correctly categorized as LOW risk | **🟢 PASS** |
| **F5** | Direct Collision Course | Head-on closing trajectory triggers immediate escalation and warning | **🟢 PASS** |
| **F6** | Sensor Degradation | Occluded/blacked-out camera input triggers camera degraded diagnostic alert | **🟢 PASS** |

---

### 10. Limitations & Known Failure Modes

1. **Pretrained Model Classes vs. Industrial Machinery**:
   The default `yolov8n.pt` model is trained on the standard 80 COCO classes. It natively detects `person`, `car`, `truck`, and `bus`, but **does not possess a dedicated `forklift` class**. Industrial vehicles are mapped using vehicle proxies (`truck`/`car`). Production deployment requires transfer learning on warehouse datasets (e.g., Forklift-COCO).
2. **CPU Inference Latency Bounding**:
   At ~98 ms per forward pass on CPU, real-time throughput is bounded at ~10 FPS. While adequate for typical warehouse forklift transit speeds (5–12 km/h), high-speed vehicular environments require GPU acceleration (NVIDIA TensorRT) or AWS Graviton NEON optimization.
3. **No Industrial Safety Certification**:
   SentinelVision is an advanced algorithmic situational-awareness prototype. It is **not** certified as a safety interlock under ISO 13849 (PLd/PLe) or IEC 61508.

---

### 11. Privacy, Responsible AI & Safety Ethics

- **Zero Facial Recognition**: SentinelVision tracks bounding box spatial centroids and velocities; it does not perform facial recognition, identity matching, or biometrics.
- **Edge-First Data Minimization**: Continuous video is processed in volatile memory on the edge and never streamed to the cloud in bulk. Only low-resolution keyframes associated with confirmed critical incidents are stored in S3.
- **Auditability**: All agent interventions are stored as auditable JSON traces in DynamoDB and S3 for safety retrospective reviews.

---

### 12. Conclusion & Future Roadmap

SentinelVision establishes that an intelligent safety system does not require expensive GPU servers or proprietary hardware. By coupling **OpenCV 5**'s image processing and spatial rendering with an **autonomous agentic feedback loop** and **serverless AWS telemetry**, the project demonstrates proactive collision avoidance that reduces alarm fatigue, responds adaptively to changing conditions, and remains resilient in industrial field deployments.

**Future Work**:
- Supervised fine-tuning of YOLOv8 on specialized industrial vehicle datasets (forklifts, reach trucks, AGVs).
- AWS Graviton ARM64 deployment with NEON/AVX-512 tensor optimizations.
- Multi-camera spatial fusion with homography ground-plane projection.
