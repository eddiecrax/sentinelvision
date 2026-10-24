# SentinelVision — Offline Docker Integrity Audit & Resolution Report

**Date**: September 16, 2026  
**Project**: SentinelVision (OpenCV AI Competition 2026)  
**Host Environment**: Linux (Debian 6.1.128-1 x86_64), Intel Core i5-6300HQ CPU @ 2.30GHz  
**Docker Engine**: Version `29.7.2` (build `a7dcaa6`)  
**Base Image**: `python:3.12-slim`  

---

## 1. Original Docker Problem

The existing Docker image tagged `sentinelvision:latest` (ID `cd3c16d15a24`) successfully responds to HTTP probes (`GET /api/health` and `GET /health`), but crashes immediately when computer vision inference is triggered via `GET /video_feed` or CLI pipeline:

```text
ValueError: libcublasLt.so.*[0-9] not found in the system path ['', '/usr/local/lib/python312.zip', '/usr/local/lib/python3.12', '/usr/local/lib/python3.12/lib-dynload', '/usr/local/lib/python3.12/site-packages']
```

Additionally, the image size is bloated to **6.54 GB** on disk, far exceeding typical CPU container footprints (~1.2–1.8 GB).

---

## 2. Root Cause Analysis

1. **Host Package Staging Contamination**:  
   During a previous attempt to perform a zero-data build without internet, the host environment's Python `site-packages` was copied into a staging directory inside the Docker build context (`deployment/.site-packages*`).
2. **CUDA vs. CPU Mismatch**:  
   The host conda environment (`sentinelvision`) had `torch==2.14.0+cu130` installed with CUDA 13.0 bindings. While the host has NVIDIA libraries in conda's `lib/` directory, the Docker image is based on `python:3.12-slim` (Debian trixie minimal) without NVIDIA drivers or CUDA libraries (`libcudart.so.13`, `libcublasLt.so`).
3. **Layer Duplication Bloat**:  
   The staging directory was copied into a temporary container path (`/tmp/local-packages`, **2.13 GB**) and then copied again into `/usr/local/lib/python3.12/site-packages` (**2.13 GB**), inflating the image to **6.54 GB**.
4. **Internal Pip Corruption**:  
   The mixing of conda host dist-info with Debian Python 3.12 broke container pip with:
   `ImportError: cannot import name 'open_rich_spinner' from 'pip._internal.cli.spinners'`.

---

## 3. Wheel & Cache Inspection (Read-Only)

A thorough inspection of local storage and package caches was executed across the machine:

- **Pip Cache (`pip cache list`, `~/.cache/pip`)**:
  - `promise-2.3-py3-none-any.whl` (21 kB)
  - `python_pptx-0.6.21-py3-none-any.whl` (471 kB)
  - `sqlalchemy-1.4.54-cp313-cp313-linux_x86_64.whl` (1.6 MB)
  - *Result*: Zero PyTorch or OpenCV wheels in pip cache.
- **System Wheel Search (`find ~ /tmp /var/tmp -name "*.whl"`)**:
  - Found only standard `pip` and `setuptools` bundled wheels inside conda/Android toolchains. Zero PyTorch wheels.
- **Conda Package Cache (`miniconda3/pkgs`)**:
  - `pytorch-2.9.1-cpu_mkl_py313hc7c4928_101.conda` (Python 3.13, Conda format, incompatible with Python 3.12 Debian base).
- **Other Conda Environments**:
  - `amaaso311`: Python 3.11 with `torch 2.2.2+cpu` (site-packages only, no wheels; incompatible with Python 3.12).
  - `sentinelvision`: Python 3.12 with `torch 2.14.0+cu130` (CUDA-linked).
- **Docker Layer Cache**:
  - Only contains the existing contaminated layer from `sentinelvision:latest`.

---

## 4. Current Status: Offline Clean Build Blocked

In accordance with Phase 2 instructions:
> *"If the correct CPU Torch wheel is NOT locally available: STOP. Do not download it. Report: 'Offline build blocked: CPU Torch package is not available locally.' Also list the exact package/version/platform that is missing. Do NOT replace CPU Torch with CUDA Torch just to make the build succeed."*

**Verdict**: **Offline build blocked: CPU Torch package is not available locally.**

### Exact Missing Packages for Clean Offline Build:
1. `torch==2.14.0+cpu` (Platform: `cp312-cp312-linux_x86_64`, Source: PyTorch CPU wheel index)
2. `torchvision==0.29.0+cpu` (Platform: `cp312-cp312-linux_x86_64`, Required by Ultralytics YOLOv8)

---

## 5. Required Code & Configuration Changes Prepared

### A. [`deployment/aws/deploy.sh`](deployment/aws/deploy.sh)
Pass `$DOCKER_PLATFORM_FLAG` into `docker build`:
```diff
--- deployment/aws/deploy.sh
+++ deployment/aws/deploy.sh
@@ -26,3 +26,3 @@
 echo "[1/4] Building Docker Image: $ECR_REPO:$IMAGE_TAG..."
-docker build -t $ECR_REPO:$IMAGE_TAG -f deployment/Dockerfile .
+docker build $DOCKER_PLATFORM_FLAG -t $ECR_REPO:$IMAGE_TAG -f deployment/Dockerfile .
```

### B. [`requirements.txt`](requirements.txt)
Disambiguate CPU PyTorch from standard PyPI CUDA wheels:
```diff
--- requirements.txt
+++ requirements.txt
@@ -1,5 +1,6 @@
+--extra-index-url https://download.pytorch.org/whl/cpu
 # SentinelVision — Pinned Competition Dependencies (OpenCV 5 & AWS Verified)
-opencv-python-headless==5.0.0.93
-torch==2.14.0
+opencv-python-headless==5.0.0.93
+torch==2.14.0+cpu
 ultralytics==8.4.138
```

### C. [`deployment/Dockerfile`](deployment/Dockerfile)
Ensure clean single-pass installation without contradictory overriding:
```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/
COPY config/ /app/config/
COPY dashboard/ /app/dashboard/
COPY yolov8n.pt /app/yolov8n.pt
COPY data/ /app/data/
ENV PYTHONUNBUFFERED=1 PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 6. Measurements & Verification Summary

| Metric / Parameter | Host Local Runtime | Existing Docker Container (`cd3c16d15a24`) |
| :--- | :--- | :--- |
| **Python Version** | `3.12.14` | `3.12.14` |
| **Torch Version** | `2.14.0+cu130` | `2.14.0` |
| **Torch CUDA Version** | `13.0` | Attempted (Failed to load) |
| **OpenCV Version** | `5.0.0` | `5.0.0` |
| **Ultralytics Version**| `8.4.138` | `8.4.138` |
| **Inference Status** | **🟢 PASS** (Video processed, 115 FPS) | **❌ FAIL** (`libcublasLt.so missing`) |
| **Health API (`/health`)** | **🟢 PASS** (HTTP 200) | **🟢 PASS** (HTTP 200) |
| **Video Feed (`/video_feed`)**| **🟢 PASS** (Streaming JPEG frames) | **❌ FAIL** (Crashes on model load) |
| **Total Size on Disk** | 67 MB (Repo) | **6.54 GB** (Contaminated layer bloat) |
| **AWS Touched** | **NO** | **NO** |

---

## 7. Next Action When Network Is Re-enabled

Once internet connectivity / mobile data is connected:
1. Run `pip download --dest /tmp/wheels torch==2.14.0+cpu torchvision==0.29.0+cpu --index-url https://download.pytorch.org/whl/cpu` (or allow Docker build to fetch it directly via `--extra-index-url`).
2. Run clean `docker build -t sentinelvision:latest -f deployment/Dockerfile .`.
3. Verify image size drops from 6.54 GB to ~1.2 GB.
4. Verify `torch.version.cuda == None` and `/video_feed` inference passes inside the fresh container.

---

## 8. Resolution — Clean CPU-Only Rebuild Verified (October 24, 2026)

The clean rebuild (`docker build -t sentinelvision:latest -f deployment/Dockerfile .`) completed using `requirements.txt` with `torch==2.14.0+cpu`. The build took ~45 minutes on a slow mobile connection. No AWS resources were touched.

| Check | Result |
| :--- | :--- |
| Image ID | `45f9be20827a` (replaces contaminated `cd3c16d15a24`) |
| Image size | 2.71 GB on disk / 639 MB content (previously 6.54 GB) |
| In-container versions | `torch 2.14.0+cpu`, `torch.version.cuda = None`, `cv2 5.0.0` |
| `GET /health` | HTTP 200 `{"status":"online","system":"SentinelVision"}` |
| `GET /video_feed` | HTTP 200, ~23.1 MB of MJPEG streamed in 40 s window |
| Model load in container | `yolov8n.pt` loaded on `device=cpu` in 59 ms; no `libcublasLt` error |
| Docker HEALTHCHECK | `Up (healthy)` |

**Scope of this claim:** YOLOv8 inference runs on CPU inside a clean container and streams frames from the bundled sample video. This is a local test only. It has not been run on AWS ECS/Fargate, and it is not a real-world industrial validation. Inference throughput inside the container was not benchmarked.

### 8.1 Frame-level inference proof (October 24, 2026)

Run inside a fresh container from image `45f9be20827a` (`docker run --rm`, CPU only): YOLOv8n was applied directly to the first 60 frames of `data/videos/warehouse_forklift.mp4`.

```text
cuda=None frames=60 detections=562 classes=['bowl', 'cup', 'refrigerator'] fps=10.4
```

- This confirms real YOLOv8 forward passes on CPU inside the container, not just an HTTP 200 stream.
- **Caveats:** the raw COCO-pretrained `yolov8n.pt` labels the bundled sample video with unrelated classes (`bowl`, `cup`, `refrigerator`) for these frames; this is a raw-detector check and does not show forklift/pedestrian detection accuracy. ~10 FPS is a rough 60-frame figure from a single run on an i5-6300HQ host, not a formal benchmark.
