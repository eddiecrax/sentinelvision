# 🛡️ SentinelVision — Independent Technical Audit of Claimed Innovations & IP

> **Audit Date**: October 26, 2026
> **Auditor Role**: Senior Systems Engineer & Skeptical Technical Auditor
> **Target System**: SentinelVision (OpenCV AI Competition 2026 Entry)
> **Repository Baseline**: Git Branch `main` (Commit `d9c622f`)
> **Audit Directive**: Verify whether the four claimed technical innovations are genuinely implemented in source code, supported by automated tests, and accurately represented. Strict read-only inspection of application code.

---

## 1. Executive Summary

An exhaustive, read-only architectural audit of the SentinelVision codebase was conducted to evaluate four technical innovations previously characterized as patentable Intellectual Property (IP). The audit cross-referenced production source modules (`src/`), evaluation and simulation benchmarks (`tests/`), benchmark JSON data artifacts (`data/evaluation/`), and cloud telemetry handlers (`src/aws/`).

### Primary Audit Findings:
1. **Core Engineering is Functional**: SentinelVision possesses a working end-to-end edge pipeline combining OpenCV 5 image ingestion, YOLOv8 inference, temporal IoU tracking, geometric Time-to-Collision (TTC) projection, and an event-driven state machine.
2. **Substantial Gap Between Claims and Implementation**: Several capabilities previously described as proprietary algorithms (such as *momentum-matched track handover*, *runtime braking deceleration interlocks*, *TTC numerical derivatives*, *write-ahead circular buffering*, and *sub-30ms CPU processing*) are either **partially implemented heuristics**, **offline evaluation scripts**, or **contradicted by measured repository benchmarks**.
3. **Synthetic vs. Real-World Disconnect**: The widely cited "8.6 ms latency" and "100% precision/recall" metrics derive strictly from idealized synthetic geometric simulations without camera noise or vision inference. Real-video benchmark data recorded in `data/evaluation/benchmark_real_pipeline.json` reveals an actual end-to-end CPU pipeline latency of **124.86 ms (~8.0 FPS)**.
4. **Patentability Assessment**: None of the four mechanisms in their current state constitute defensible, standalone patentable claims without significant formalization. Claimed innovations consist largely of standard heuristics (Euclidean clustering, linear extrapolation, try/except error catching) coupled with post-hoc evaluation scripts.

---

## 2. Innovation Status Matrix

| Innovation Claim | Status | Source Location | Test Support | Auditor Confidence |
| :--- | :---: | :--- | :--- | :---: |
| **A. Spatial Track-to-Incident Handover** | **PARTIALLY IMPLEMENTED** | `src/agent/agent.py:L139-162`<br>`src/safety/events.py:L65-81, 182-191` | **NONE** (No unit tests verify handover logic) | **Medium** |
| **B. Physics-Grounded Braking Feasibility Engine** | **NOT IMPLEMENTED (Runtime)**<br>*Evaluation Only* | `src/prediction/ttc.py:L26-67`<br>`tests/benchmark_tangible_safety.py:L97-128` | Synthetic trajectory tests only | **High (Confirmed Absence)** |
| **C. Closed-Loop Agentic Mitigation & Auto-Resolution** | **PARTIALLY IMPLEMENTED** | `src/agent/agent.py:L163-273`<br>`src/safety/events.py:L39-64` | `tests/test_scenario.py`<br>`tests/test_evidence_suite.py` | **High** |
| **D. Fault-Tolerant Edge/Cloud Reconciliation** | **PARTIALLY IMPLEMENTED** | `src/aws/s3.py:L24-76`<br>`src/aws/dynamodb.py:L22-85`<br>`src/safety/storage.py:L19-69` | `tests/test_aws_integration.py` (Mocked fallbacks only) | **High** |

---

## 3. Deep-Dive Audit of the Four Claimed Innovations

### A. Spatial Track-to-Incident Handover

#### 1. Claimed Behavior vs. Actual Source Implementation
* **The Claim**: An algorithm that preserves incident continuity across dropped tracks using a 1.2-second temporal grace window, a 120-pixel spatial threshold, *and trajectory-momentum matching ($\Delta \theta$)* to eliminate duplicate alarms.
* **Source Evidence**:
  - `src/agent/agent.py` lines 101–102, 108–109: Parameterized with `grace_period = 1.2` and `spatial_handover_distance = 120.0`.
  - `src/agent/agent.py` lines 140–161:
    ```python
    if not has_active and getattr(assessment, "person_center", None):
        p_c = assessment.person_center
        for active_ev in self.event_manager.active_events:
            if active_ev.vehicle_id == assessment.vehicle_id and active_ev.person_center:
                time_since_seen = timestamp - active_ev.last_seen_timestamp
                if time_since_seen <= self.grace_period:
                    ev_c = active_ev.person_center
                    dist = ((p_c[0] - ev_c[0]) ** 2 + (p_c[1] - ev_c[1]) ** 2) ** 0.5
                    if dist <= self.spatial_handover_distance:
                        if active_ev.person_id not in active_person_ids:
                            old_pair = (active_ev.person_id, active_ev.vehicle_id)
                            event = self.event_manager.handover_event(
                                old_pair, pair_key, p_c, timestamp, dist
                            )
    ```
  - `src/safety/events.py` lines 65–80 (`handover_track`) and lines 182–191 (`handover_event`): Transfers ownership of `SafetyEvent` from `old_pair` to `new_pair` in the active dictionary, preserving `event.timeline` and `event.event_id`.

#### 2. Discrepancies & Overstatements
- **No Momentum / Velocity Angle Matching**: The implementation evaluates **strictly 2D Euclidean distance** between the new person center and the last seen center. There is **zero trajectory-momentum matching**, dot-product directional alignment, or velocity vector continuity check.
- **Unidirectional Association**: The logic only associates lost *pedestrians* while keeping the *vehicle* ID strictly fixed (`active_ev.vehicle_id == assessment.vehicle_id`). If the vehicle tracking ID flickers, the handover fails completely and triggers a duplicate incident.
- **Test Evidence**: **0 tests**. Ripgrep search for `handover` across the entire `tests/` directory returns **0 matches**. The handover mechanism was never exercised in automated regression or unit testing.

---

### B. Physics-Grounded Collision and Braking Feasibility Engine

#### 1. Claimed Behavior vs. Actual Source Implementation
* **The Claim**: A closed-form kinematic engine that dynamically couples relative velocity, closest point of approach ($d_{cpa}$), and ANSI B56.1 mechanical braking physics ($a = 2.5\,\text{m/s}^2$, $t_{react} = 0.75\,\text{s}$), triggering alerts *only when the physical stopping envelope is violated*.
* **Source Evidence**:
  - `src/prediction/ttc.py` lines 36–58 (`compute_ttc`):
    ```python
    rel_pos = [pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]]
    distance = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
    rel_vel = [vel_b[0] - vel_a[0], vel_b[1] - vel_a[1]]
    unit = [rel_pos[0] / distance, rel_pos[1] / distance]
    closing = -(rel_vel[0] * unit[0] + rel_vel[1] * unit[1])
    is_converging = closing > 0.5
    if is_converging and closing > 0:
        ttc = distance / closing
    ```
  - `src/motion/trajectory.py` lines 21–54 (`predict_trajectory`): Linear position extrapolation ($p(t) = p_0 + v \cdot t$) up to `horizon = 2.0s`.
  - `src/safety/risk.py` lines 58–65, 102–174: Linear heuristic scoring:
    $$\text{Risk Score} = 0.35 \cdot S_{ttc} + 0.20 \cdot S_{dist} + 0.15 \cdot S_{vel} + 0.15 \cdot S_{traj} + 0.10 \cdot S_{persist}$$

#### 2. Discrepancies & Overstatements
- **Braking Physics is Absent from Runtime Code**: Ripgrep search for `ANSI`, `B56`, `braking`, `deceleration`, or `2.5` across all production code in `src/` returns **0 matches**.
- **Post-Hoc Analysis Only**: The ANSI B56.1 stopping distance formula ($d_{stop} = 1.50\,\text{m} + 0.80\,\text{m} = 2.30\,\text{m}$) exists **only** in `tests/benchmark_tangible_safety.py:L97-128` as an offline spreadsheet-style comparison against observed warning lead times.
- **Pixel Space vs. Physical Metric Space**: All runtime calculations in `ttc.py`, `trajectory.py`, and `risk.py` operate strictly in **uncalibrated 2D pixel coordinates**. There is no homography transformation matrix, ground-plane projection, or camera extrinsic calibration implemented to translate pixel velocities to physical meters per second.
- **Handling of Trajectories**:
  - Converging: **VERIFIED**. Correctly flagged when closing speed $> 0.5$ px/s.
  - Diverging: **VERIFIED**. Sets `ttc = inf`, closing velocity $\le 0$.
  - Parallel: **VERIFIED**. Tested in `tests/test_failures.py:F4` and `tests/test_evidence_suite.py:Scenario_1`. Parallel non-converging entities remain `LOW` risk.
  - Zero False Alarms: **OVERSTATED**. The "0% false alarm rate" was measured on 20 identical runs of a clean synthetic straight-line script, not on unconstrained real-world industrial footage.

---

### C. Closed-Loop Agentic Mitigation & Automatic Resolution

#### 1. Claimed Behavior vs. Actual Source Implementation
* **The Claim**: A closed-loop perception-action-observation agent that tracks the *first derivative of Time-to-Collision ($\frac{d}{dt}TTC > 0$)*, validates corrective divergence, automatically de-escalates to `RESOLVED` in 4.6 seconds, and enforces an *exponential cooldown*.
* **Source Evidence**:
  - `src/agent/agent.py` lines 39–85 (`AgentPolicy.decide`): Defines transitions based on risk level and `event_age`.
  - `src/agent/agent.py` lines 182–218: Implements debouncing confirmation window (`confirmation_frames = 3`, `confirmation_time = 0.25s`) before elevating to an active incident.
  - `src/agent/agent.py` lines 231–272: Evaluates active events. If `event.risk_level == "LOW"` sustains for `grace_period` (1.2s), it issues `ActionType.RESOLVE`.
  - `src/safety/events.py` lines 39–43: Marks `resolved = True`, records `resolution_time = timestamp - self.timestamp`.

#### 2. Discrepancies & Overstatements
- **No TTC Derivative ($\frac{d}{dt}TTC$)**: The code contains no differential calculus or derivative calculation. Resolution is triggered strictly by a categorical state check: `if event.risk_level == "LOW"` sustained for 1.2 seconds.
- **No Exponential Cooldown**: Cooldown is enforced via a flat, linear floating-point timestamp check (`agent.py:179`): `if timestamp - self._cooldowns[pair_key] < self.cooldown` (`cooldown = 5.0s`). There is no exponential backoff or dynamic window scaling.
- **4.6-Second Resolution is Video-Specific Ground Truth, Not a System Parameter**:
  - In `tests/test_scenario.py`, the resolution time was measured as **2.8 seconds**.
  - In `src/aws/setup_aws_live.py`, the mock resolution time was **2.8 seconds**.
  - On the real video `factory_forklift.mp4`, the encounter occurred across frames 330 to 376 (~46 frames @ 10 FPS = 4.6 seconds). Quoting "4.6 seconds" as an intrinsic feature of the algorithm conflates a single observational measurement with an architectural property.

---

### D. Fault-Tolerant Edge/Cloud Reconciliation

#### 1. Claimed Behavior vs. Actual Source Implementation
* **The Claim**: Sub-30ms local edge processing with a write-ahead circular buffer, retry queue, and idempotent batch reconciliation ensuring zero downtime and complete data recovery when cloud connectivity drops.
* **Source Evidence**:
  - `src/aws/s3.py` lines 24–34, 54–56: Initial client connection failure logs a warning and sets `self._enabled = False`. When disabled, `upload_snapshot` returns a mocked URI string `s3://...` without throwing an exception.
  - `src/aws/dynamodb.py` lines 60–62, 83–84: Network/client exceptions inside `log_incident` and `update_incident_status` are caught:
    ```python
    except Exception as e:
        logger.error(f"Failed to log incident to DynamoDB: {e}")
        return True  # Fallback to local offline log
    ```
  - `src/safety/storage.py` lines 40–68: Local persistence writes incident records to `data/incidents.json` and snapshots to `data/clips/`.

#### 2. Discrepancies & Overstatements
- **Contradiction on "Sub-30ms Edge Processing"**:
  - Verified Benchmark Artifact: `data/evaluation/benchmark_real_pipeline.json`
  - Total Pipeline Average: **124.86 ms**
  - YOLOv8 CPU Inference Stage: **123.10 ms**
  - End-to-End Throughput: **8.0 FPS**
  - *Auditor Note*: The claim of "under 30 ms" or "sub-30ms" appears to conflate the synthetic state-machine test (8.6 ms without vision) with real end-to-end vision processing. On standard multi-core CPU, the system runs at ~125 ms.
- **No Reconciliation Worker or Outbound Queue**:
  - When AWS is unreachable, failed cloud payloads are **discarded**.
  - There is no write-ahead log, no in-memory FIFO retry queue, and no background sync thread that reads `data/incidents.json` upon network restoration to replay un-synced events to DynamoDB or S3.
  - "Zero downtime" is accurate in the narrow sense that the local perception loop does not crash when the network drops; however, claims of "reconciliation", "retries", and "eventual cloud consistency" are **NOT IMPLEMENTED**.
- **Mocked Testing Trap in Test Suite**:
  - In `tests/test_aws_integration.py` lines 65, 75, 84:
    `results["Test_2_DynamoDB_Incident_Created"] = db_create_success or (not dynamo.enabled)`
  - When AWS credentials are missing, `(not dynamo.enabled)` evaluates to `True`, masking the fact that no actual cloud transmission or queuing took place.

---

## 4. Unsupported Numerical & Commercial Claims

The following claims found in recent draft scripts, promotional notes, or IP analyses must be corrected or excised:

| Claimed Statement | Ground Truth in Code & Artifacts | Audit Verdict |
| :--- | :--- | :---: |
| *"Sub-30ms edge processing latency"* | Measured real pipeline average is **124.86 ms** (`benchmark_real_pipeline.json`). | **CONTRADICTED** |
| *"100% precision and 0% false alarm rate"* | Evaluated on 20 runs of a single synthetic geometric straight-line path (`test_evidence_suite.py`). Not verified on unconstrained real video. | **OVERSTATED** |
| *"Coupled with ANSI B56.1 runtime braking deceleration"* | Braking formulas exist only in an offline evaluation script (`benchmark_tangible_safety.py`). No physics formulas exist in `src/`. | **NOT IMPLEMENTED** |
| *"Calculates first derivative of TTC ($\frac{d}{dt}TTC > 0$)"* | No derivative calculation exists. Code checks a boolean categorical string `if risk_level == 'LOW'`. | **NOT IMPLEMENTED** |
| *"Trajectory momentum and angle matching ($\Delta \theta$)"* | Code evaluates basic 2D Euclidean distance ($r \le 120$ px). No angle or momentum checks exist. | **NOT IMPLEMENTED** |
| *"Write-ahead buffer and automated cloud reconciliation"* | Cloud failures are caught and logged; failed items are not queued or re-synced on reconnect. | **NOT IMPLEMENTED** |
| *"Autonomous resolution in 4.6 seconds"* | 4.6s is the observed duration of one specific encounter in `factory_forklift.mp4`. Simulation tests resolve in 2.8s. | **MISREPRESENTED** |

---

## 5. Prioritized Remediation: Missing Tests & Measurements

To bring technical credibility in line with rigorous software engineering standards, the following tests should be implemented:

1. **Unit Test for Spatial Track Handover (`tests/test_track_handover.py`)** *(High Priority)*:
   - Create a deterministic test where a track `P#1` triggers an incident, drops for 3 frames, and reappears as `P#2` at a distance of 50px ($<120$px). Assert that `event_id` is retained and no second incident is minted.
   - Assert that if `P#2` appears at 200px ($>120$px) or after 2.0s ($>1.2$s), a separate incident is minted.
2. **Real-Video Latency Profiling Table** *(High Priority)*:
   - Replace all references to "sub-30ms" with the actual measured numbers from `data/evaluation/benchmark_real_pipeline.json`:
     - Average End-to-End Latency: **124.9 ms**
     - YOLOv8 CPU Inference: **123.1 ms**
     - OpenCV Preprocessing + HUD: **0.6 ms**
     - Tracking + Motion + Risk + Agent: **0.1 ms**
     - Sustained Framerate: **8.0 to 10.6 FPS**
3. **Offline Queue Persistence Unit Test** *(Medium Priority)*:
   - Test verifying whether incidents written to `data/incidents.json` during network failure can be manually re-dispatched to DynamoDB via an administrative CLI command.
4. **Camera Calibration Metric Transform** *(Low Priority / Future Work)*:
   - Formally document that velocity and distance calculations are currently in **pixel units** and identify camera homography ground-plane calibration as a planned production feature.

---

## 6. Recommended Wording for Competition Deliverables & Video

To avoid credibility damage during judging, replace speculative or exaggerated phrases with precise, defensible language:

### Suggested Text Replacements:

* **Instead of**: *"SentinelVision features sub-30ms edge processing independent of cloud connectivity."*
  **Use**: *"SentinelVision executes the complete perception and safety loop locally on CPU in ~125 ms (~8 to 10 FPS), streaming incident telemetry to a serverless AWS cloud tier."*

* **Instead of**: *"A physics-grounded collision engine that enforces ANSI B56.1 forklift braking distance interlocks."*
  **Use**: *"A collision prediction pipeline evaluated against OSHA and ANSI B56.1 forklift braking standards, demonstrating an average advance lead time of 2.04 seconds against a 1.55-second required stopping window."*

* **Instead of**: *"Spatial track handover with momentum-matching trajectory continuity."*
  **Use**: *"Spatio-temporal track handover that binds re-acquired entities within a 120-pixel, 1.2-second window to active incidents, reducing duplicate alarm fatigue."*

* **Instead of**: *"Write-ahead buffering with automated cloud reconciliation."*
  **Use**: *"Resilient local fallback that preserves continuous safety monitoring and writes local evidence logs without crashing if AWS connectivity drops."*

* **Instead of**: *"Autonomous hazard resolution in 4.6 seconds via TTC derivative monitoring."*
  **Use**: *"Autonomous hazard resolution that de-escalates incidents after a verified 1.2-second low-risk hysteresis window, resolving within 4.6 seconds on real factory test footage."*

---

## 7. Preliminary Novelty Assessment & Patent Counsel Considerations

### Technical Novelty Evaluation
* **Invention A (Track Handover)**:
  - *Novelty Assessment*: **Low to Moderate**. Spatial proximity association across brief occlusions is standard in Multi-Object Tracking (MOT) literature (e.g. Hungarian algorithm distance gating, tracklet stitching). The specific linkage of re-associated tracks to a *stateful incident management object* is good application engineering, but unlikely to overcome non-obviousness hurdles (35 U.S.C. § 103) on its own without advanced trajectory modeling.
* **Invention B (Braking Feasibility)**:
  - *Novelty Assessment*: **Low**. Calculating stopping distance ($v \cdot t + v^2 / 2a$) is basic Newtonian physics and cannot be patented per se (*Alice Corp. v. CLS Bank*). Furthermore, because it is not implemented in the runtime engine, it cannot be claimed as an operable invention in this codebase.
* **Invention C (Closed-Loop Agentic Mitigation)**:
  - *Novelty Assessment*: **Moderate**. The closed-loop state machine with hysteresis debouncing and automated de-escalation is well-engineered. However, extensive prior art exists in automotive ADAS (Mobileye, Bosch Forward Collision Warning patents) covering automatic alert termination upon trajectory divergence.
* **Invention D (Dual-Tier Edge/Cloud)**:
  - *Novelty Assessment*: **Low**. Local execution with try/except cloud fallback is a standard software architecture pattern (AWS IoT Greengrass, edge computing best practices). Without an implemented novel transactional synchronization protocol, it is not patentable.

### Critical Legal Flags for the Inventor:
1. **The 1-Year US Grace Period (35 U.S.C. § 102(b))**:
   - Public release of the repository or presentation in the OpenCV AI Competition triggers the 1-year US statutory clock. A US Provisional Patent Application must be filed within 12 months of disclosure to preserve US rights.
2. **Absolute Novelty in International Jurisdictions**:
   - In Europe (EPO), China (CNIPA), and PCT countries, public disclosure prior to filing a patent application destroys patentability immediately.
3. **Recommendation**:
   - Focus your submission on **software excellence, open-source reproducibility, and competition impact**.
   - If commercialization or patenting is pursued post-competition, engage a registered patent attorney to conduct a formal prior-art search on *Track-to-Incident Affinity Binding* and implement true ground-plane metric calibration before filing.
