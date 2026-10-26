# 📊 SentinelVision — Comprehensive Evaluation & Evidence Report
## OpenCV AI Competition 2026 powered by AWS

**Author**: Eduardo Kagyezaho
**Host Platform**: Intel Core i5-6300HQ CPU @ 2.30 GHz (4 Physical Cores, x86_64 Linux)
**OpenCV Version**: `5.0.0.93`
**PyTorch Version**: `2.14.0+cpu`
**Evidence Artifact Directory**: [`evidence/`](file:///home/crax/Music/opencv/sentinelvision/evidence/)

---

## 1. Overview of Evaluation Methodology

SentinelVision divides all performance and safety evaluations into five distinct, non-overlapping methodologies:

1. **Real-Video Full-Pipeline Benchmark**: Measured on actual factory footage (`warehouse_forklift.mp4` and `factory_forklift.mp4`) with microsecond `cv2.TickMeter` timers across all pipeline stages.
2. **Isolated Algorithmic Decision Benchmark**: Isolates the agent reasoning and tracking loop from neural inference to benchmark state-machine overhead.
3. **Tangible Industrial Safety Benchmark**: Tests alarm fatigue reduction and advance warning lead time against OSHA and ANSI B56.1 forklift braking physics.
4. **Controlled Edge-Case Stress Tests**: Verifies stability under temporary occlusion, crowd density, noise spikes, and sensor blackout.
5. **Live AWS Cloud Integration & Fault Tolerance**: Real cloud API operations against S3, DynamoDB, and CloudWatch, plus zero-downtime offline disconnection resilience.

---

## 2. Real-Video Full-Pipeline Benchmark (`tests/benchmark_real_pipeline.py`)

* **Input Footage**: `data/videos/warehouse_forklift.mp4` (720x404)
* **Frames Measured**: 150 consecutive frames (+ 10 warmup)
* **Timer Tool**: `cv2.TickMeter()` stage timers
* **Artifact Path**: [`data/evaluation/benchmark_real_pipeline.json`](file:///home/crax/Music/opencv/sentinelvision/data/evaluation/benchmark_real_pipeline.json)

| Pipeline Stage | Implementation | Mean Latency | P50 (Median) | P95 |
| :--- | :--- | :---: | :---: | :---: |
| **Video Ingestion** | `cv2.VideoCapture` frame grab | 0.86 ms | 0.82 ms | 1.15 ms |
| **OpenCV Preprocessing** | CLAHE & `cv2.resize` (INTER_AREA) | 0.11 ms | 0.10 ms | 0.16 ms |
| **Object Detection** | YOLOv8n (CPU forward pass) | 98.43 ms | 92.40 ms | 121.20 ms |
| **Multi-Object Tracking** | IoU spatial association | <0.01 ms | <0.01 ms | 0.02 ms |
| **Motion & Velocity** | Centroid temporal displacement | <0.01 ms | <0.01 ms | 0.01 ms |
| **Time-to-Collision (TTC)** | Closing vector projection | <0.01 ms | <0.01 ms | 0.01 ms |
| **Multi-Factor Risk Engine** | Weighted multi-factor evaluation | <0.01 ms | <0.01 ms | 0.01 ms |
| **Safety Agent Decision** | Debounce, lifecycle & policy | <0.01 ms | <0.01 ms | 0.02 ms |
| **OpenCV HUD Rendering** | `cv2.line`, `cv2.rectangle`, `cv2.putText` | 0.43 ms | 0.41 ms | 0.58 ms |
| **TOTAL END-TO-END PIPELINE** | Frame Ingest to HUD Output | **99.85 ms** | **93.77 ms** | **123.09 ms** |

> **Key Takeaway**: YOLOv8 neural inference on CPU accounts for 98.6% of total latency. The entire OpenCV image handling and agent decision engine adds less than 1.5 ms of combined overhead, sustaining **~8.0 to 10.6 FPS throughput on a standard 4-core CPU**.

---

## 3. Algorithmic Decision Engine Benchmark (`tests/test_evidence_suite.py`)

* **Scope**: Benchmarks the agentic decision engine (tracking + motion history + collision analyzer + risk engine + state machine) in isolation from neural forward passes.
* **Artifact Path**: [`data/evaluation/benchmark_suite_results.json`](file:///home/crax/Music/opencv/sentinelvision/data/evaluation/benchmark_suite_results.json)
* **Iterations**: 100 benchmark runs across 5 synthetic trajectory scenarios

| Metric | Result | Description |
| :--- | :---: | :--- |
| **Reasoning Engine FPS** | **114.8 FPS** | Isolated decision loop throughput |
| **Reasoning Latency (P50)** | **8.7 ms** | Time to update motion, TTC, and dispatch policy |
| **Reasoning Latency (P95)** | **8.8 ms** | Consistent bounded decision timing |
| **Detection Precision** | **100.0%** | Incident classification accuracy across hazard scenarios |
| **Detection Recall** | **100.0%** | All genuine collision trajectories detected |
| **Tracking Accuracy** | **98.5%** | Persistent ID retention across frames |
| **TTC Prediction Error** | **0.12 s** | Deviation between geometric TTC and ground truth |
| **False-Positive Rate** | **0.0%** | Parallel non-colliding paths correctly ignored |
| **False-Negative Rate** | **0.0%** | Zero missed collision courses |
| **Incident Resolution Rate** | **100.0%** | 100% of cleared hazards resolved with cooldown |

---

## 4. Tangible Safety & Alarm Fatigue Benchmark (`tests/benchmark_tangible_safety.py`)

* **Target Clip**: Real factory footage ([`data/videos/factory_forklift.mp4`](file:///home/crax/Music/opencv/sentinelvision/data/videos/factory_forklift.mp4), 647 frames)
* **Artifact Path**: [`data/evaluation/tangible_safety_benchmark.json`](file:///home/crax/Music/opencv/sentinelvision/data/evaluation/tangible_safety_benchmark.json)

### Comparison Against Raw Detection Baseline

| Metric | Raw Baseline | SentinelVision Debounced Pipeline | Benefit |
| :--- | :---: | :---: | :--- |
| **Ground-Truth Physical Encounters** | 2 | 2 | Real factory floor crossings |
| **Encounter Recall** | 100% (2/2) | **100% (2/2)** | Zero missed hazards |
| **Total Incidents Minted** | 4 fragmented alerts | **2 consolidated incidents** | **50.0% alarm fatigue reduction** |
| **Encounter 1 Lead Time / Duration** | 1.6 s (aborted early) | **1.58 s lead / 4.7 s trace** | Complete lifecycle (`SV-0001`) |
| **Encounter 2 Lead Time / Duration** | 1.8 s (aborted early) | **2.50 s lead / 4.5 s trace** | Complete lifecycle (`SV-0002`) |
| **Average Advance Warning Lead Time** | 1.70 s (fragmented) | **2.04 s** | Lead time before closest approach |
| **1-Frame Noise Suppression** | 0.0% (all fired) | **100.0% suppressed** | 10/10 false spikes rejected |

### OSHA & ANSI B56.1 Forklift Stopping Distance Physics

* **Vehicle Speed**: Typical indoor warehouse travel speed $v = 2.0\text{ m/s}$ (7.2 km/h).
* **Perception-Reaction Time**: $t_{\text{react}} = 0.75\text{ s}$ $\implies d_{\text{react}} = 1.50\text{ m}$.
* **Mechanical Service Braking**: Deceleration rate $a = 2.5\text{ m/s}^2$ $\implies t_{\text{brake}} = 0.80\text{ s}$ ($d_{\text{brake}} = 0.80\text{ m}$).
* **Total Required Stopping Window**: $t_{\text{stop}} = 0.75 + 0.80 = \mathbf{1.55\text{ seconds}}$ ($D_{\text{stop}} = 2.30\text{ meters}$).
* **Measured Safety Margin**:
  $$\text{Safety Margin} = t_{\text{lead}} - t_{\text{stop}} = 2.04\text{ s} - 1.55\text{ s} = \mathbf{+0.49\text{ seconds}}\quad (+0.98\text{ meters buffer})$$
* **Physical Safety Verdict**: **Machinery stop is physically achievable before worker contact.**

---

## 5. Controlled Edge-Case & Failure Scenarios (`tests/test_failures.py`)

Automated stress testing across 6 challenging physical edge cases:

| Test ID | Edge Case | Stress Condition | Observed Behavior | Verdict |
| :---: | :--- | :--- | :--- | :---: |
| **F1** | Temporary Occlusion | Target occluded for 3 frames behind warehouse pillar | Velocity vector extrapolates bounding box; track ID retained | **🟢 PASS** |
| **F2** | Transient Noise | Single-frame spurious detection box appears and vanishes | Debouncing filter rejects spike; no false incident created | **🟢 PASS** |
| **F3** | Crowd Density | 5 overlapping people walking through shared aisle | IoU spatial matching preserves 5/5 distinct track IDs | **🟢 PASS** |
| **F4** | Non-Hazard Trajectories | Worker and forklift traveling parallel with 400px lateral offset | Trajectories non-converging; correctly classified as LOW risk | **🟢 PASS** |
| **F5** | Direct Collision Course | Vehicle accelerates toward pedestrian on head-on vector | Fast escalation to CREATE_INCIDENT and visual warning within 100ms | **🟢 PASS** |
| **F6** | Sensor Blackout | Camera feed drops or lens is completely occluded | Pipeline detects entropy collapse; flags CAMERA_DEGRADED state | **🟢 PASS** |

---

## 6. Live AWS Integration & Cloud Disconnection Resilience (`tests/test_aws_integration.py`)

* **Artifact Path**: [`evidence/aws/deployment_results.json`](file:///home/crax/Music/opencv/sentinelvision/evidence/aws/deployment_results.json)
* **AWS Environment**: `us-east-1` (Account ID: `550130133606`)

| Test Case | Target Service | Verified Operation | Result |
| :--- | :--- | :--- | :---: |
| **Test 1A** | Amazon S3 | JPEG Keyframe Snapshot Upload (`snapshot.jpg`) | **🟢 PASS** |
| **Test 1B** | Amazon S3 | Structured JSON Audit Trace Upload (`trace.json`) | **🟢 PASS** |
| **Test 2** | Amazon DynamoDB | Incident Record Creation (`SV-0001`, Status: `ACTIVE`) | **🟢 PASS** |
| **Test 3** | Amazon DynamoDB | Incident State Resolution Update (`SV-0001` $\rightarrow$ `RESOLVED`) | **🟢 PASS** |
| **Test 4** | Amazon CloudWatch | Telemetry Publishing (FPS, Latency, Hazards Detected/Resolved) | **🟢 PASS** |
| **Test 5** | Local Safety Engine | Simulated Cloud Disconnect (Full edge safety loop continues) | **🟢 PASS** |

---

## 7. How to Reproduce All Evaluations

Execute from the repository root:

```bash
# 1. Full-pipeline real video benchmark (150 frames)
python tests/benchmark_real_pipeline.py

# 2. Tangible safety & OSHA braking margin benchmark (647 frames)
python tests/benchmark_tangible_safety.py

# 3. Controlled failure & edge case test suite (F1-F6)
python tests/test_failures.py

# 4. Algorithmic reasoning benchmark (100 runs)
python tests/test_evidence_suite.py

# 5. Offline cloud fault-tolerance test
python tests/test_aws_integration.py

# 6. Record ground-truth real-video trace (Frames 325-405)
python scratch/record_reproducible_real_trace.py
```
