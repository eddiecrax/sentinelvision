# 🎬 SentinelVision — Competition Video Demonstration Script
## Target Duration: 4 Minutes 30 Seconds (Max Allowed: 5 Minutes)

> **Competition**: OpenCV AI Competition 2026 powered by AWS  
> **Speaker**: Eduardo Kagyezaho  
> **Screen Resolution**: 1920x1080 (Full HD)  
> **Safety Notice**: Ensure terminal history, browser bookmarks, and `.env` files with secret keys are closed before recording.

---

### [0:00 – 0:20] 1. Problem & Value Proposition (20s)
- **Visual**: Slide or title card: *SentinelVision — Autonomous Industrial Hazard Perception & Agentic Incident Mitigation*.
- **Spoken Audio**:
  > *"Hello, I'm Eduardo Kagyezaho, presenting SentinelVision for the OpenCV AI Competition 2026 powered by AWS. In industrial warehouses and manufacturing facilities, forklift and machinery collisions with pedestrians remain a leading cause of severe workplace injuries. Most industrial cameras are completely passive — they record accidents after they happen, or sound unranked alarms that cause operator fatigue. SentinelVision introduces an autonomous agentic vision pipeline that actively calculates collision risk, intervenes proactively, verifies whether conditions improve, and logs tamper-evident audit records."*

---

### [0:20 – 1:15] 2. Live Application & Vision Pipeline (55s)
- **Visual**: Browser window showing the SentinelVision Dashboard at `http://localhost:8000/`. The MJPEG video stream is running live with OpenCV HUD overlays.
- **Actions on Screen**:
  1. In the top-right "Video Feed" dropdown, select **Pedestrian Walkway** (`pedestrian_traffic.mp4`) or **Traffic Flow** (`test_traffic.mp4`) to show live multi-person/vehicle detection.
  2. Highlight the HUD: bounding boxes around entities, persistent track IDs, dynamic velocity vectors, and the active FPS counter (~10 FPS CPU).
  3. Point out the OpenCV 5 preprocessing indicators (adaptive CLAHE contrast normalization for uneven factory lighting).
- **Spoken Audio**:
  > *"Here is SentinelVision running locally on CPU. Ingested through OpenCV 5, each video frame undergoes adaptive CLAHE contrast enhancement and noise suppression. YOLOv8 detects workers and machinery, while our IoU tracker maintains temporal track identities. OpenCV dynamically renders velocity vectors and spatial collision trajectories directly on the frame in real time."*

---

### [1:15 – 2:15] 3. Agentic Loop: Perception Influencing Action (60s)
- **Visual**: Terminal showing `python tests/test_scenario.py` or the live dashboard incident panel triggering an event.
- **Actions on Screen**:
  1. Show a vehicle and pedestrian converging.
  2. Highlight the Time-to-Collision (TTC) dropping below 3.5 seconds.
  3. Show the Risk Engine upgrading the threat from MEDIUM to HIGH.
  4. Show the Safety Agent autonomously executing `CREATE_INCIDENT` (`SV-0002`) and flashing a visual warning on the HUD.
- **Spoken Audio**:
  > *"This is the core differentiator: the Agentic Vision loop. SentinelVision does not just detect objects; it projects their movement into the future. When relative velocity indicates a closing trajectory, the system computes Time-to-Collision. Here, as the TTC drops below critical thresholds, our autonomous agent makes a policy decision: it creates Incident SV-0002, elevates monitoring frequency, and triggers an immediate visual warning."*

---

### [2:15 – 2:50] 4. Re-observation & Autonomous Resolution (35s)
- **Visual**: The pedestrian steps out of the trajectory path; TTC relaxes; incident updates.
- **Actions on Screen**:
  1. Show the entities diverging.
  2. The risk level drops to LOW.
  3. The agent observes the hazard has cleared and autonomously executes `RESOLVE`.
  4. Show the incident status change to `RESOLVED` in 1.6 seconds with full audit trace.
- **Spoken Audio**:
  > *"Crucially, the agent continues observing subsequent frames to evaluate the impact of the intervention. When the worker steps into the safe pedestrian walkway and trajectories diverge, the agent detects the hazard has dissipated. It automatically transitions the incident state to RESOLVED, logs the resolution duration, and enforces a cooldown to prevent alarm fatigue."*

---

### [2:50 – 3:40] 5. Genuine AWS Cloud Evidence ($0.00 Free Tier) (50s)
- **Visual**: Switch to AWS Management Console in browser (AWS account redacted, Region: `us-east-1`).
- **Actions on Screen**:
  1. **Amazon S3**: Open bucket `sentinelvision-evidence-(redacted)`. Show the `incidents/2026/10/SV-1001/` folder containing `snapshot.jpg` and `trace.json`.
  2. **Amazon DynamoDB**: Open table `SentinelVisionIncidents`. Click **Explore items** and show the incident row with `incident_id`, `risk_score`, `ttc`, and `resolved: true`.
  3. **Amazon CloudWatch**: Open `SentinelVision/Metrics`. Show the live telemetry metrics: `PipelineFPS`, `InferenceLatencyMs`, and `HazardsDetected`.
- **Spoken Audio**:
  > *"Every critical event is backed by real AWS cloud infrastructure operating under the AWS Free Tier. Here in Amazon S3, you can see the incident keyframe snapshot and the complete decision trace uploaded automatically. In Amazon DynamoDB, our persistent incident ledger records the event lifecycle and resolution timestamps. And in Amazon CloudWatch, real-time pipeline latency and hazard metrics provide centralized operational visibility across facilities."*

---

### [3:40 – 4:15] 6. Controlled Failure & Edge Resilience (35s)
- **Visual**: Terminal running `python tests/test_failures.py`.
- **Actions on Screen**:
  1. Show the 6 controlled failure cases passing (F1 Occlusion, F2 Noise, F3 Density, F4 Parallel Non-Collision, F5 Head-on Escalation, F6 Camera Degradation).
  2. Mention cloud disconnect test: edge safety continues with 100% autonomy if internet fails.
- **Spoken Audio**:
  > *"We tested SentinelVision against six industrial edge cases. As shown here, the tracker survives temporary occlusions behind warehouse pillars, filters out single-frame false detections, and distinguishes harmless parallel walking from true collision courses. Furthermore, if the internet connection to AWS is severed, SentinelVision's edge loop continues uninterrupted, logging evidence locally without crashing."*

---

### [4:15 – 4:30] 7. Limitations & Conclusion (15s)
- **Visual**: Final summary slide or return to live dashboard.
- **Spoken Audio**:
  > *"As an honest engineering note, our baseline model uses COCO vehicle classes and achieves 10 FPS on CPU; production scaling will include fine-tuned forklift datasets and GPU acceleration. SentinelVision demonstrates that proactive, closed-loop vision agents can meaningfully protect human life in industrial workplaces. Thank you."*

---

## 8. Exact Steps for Recording Your Screen

1. **Clean Screen Setup**:
   - Close all browser tabs except the SentinelVision Dashboard (`http://localhost:8000/`) and the AWS Console tabs (S3, DynamoDB, CloudWatch).
   - Ensure terminal windows do not show secret keys or `.env` files.
2. **Start Local App**:
   ```bash
   conda activate sentinelvision
   python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
   ```
3. **Recording Software**:
   - Use **OBS Studio** or **SimpleScreenRecorder** (Linux) to capture full 1080p desktop video with microphone audio.
4. **Export**: Save as `sentinelvision_demo.mp4` (H.264 / AAC, target length: 4:25).
