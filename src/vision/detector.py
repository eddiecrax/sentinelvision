"""
SentinelVision — Object Detection

Phase 2: Detect persons and vehicles using YOLOv8.
"""

import os
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger


@dataclass
class Detection:
    """A single detected object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]           # [x1, y1, x2, y2]
    center: List[float]         # [cx, cy]
    area: float
    timestamp: float = 0.0

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class DetectionResult:
    """Result of detection on a single frame."""
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    frame_id: int = 0
    timestamp: float = 0.0

    @property
    def persons(self) -> List[Detection]:
        return [d for d in self.detections if d.class_name == "person"]

    @property
    def vehicles(self) -> List[Detection]:
        vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle"}
        return [d for d in self.detections if d.class_name in vehicle_classes]

    @property
    def count(self) -> int:
        return len(self.detections)


# COCO class IDs we care about
PERSON_CLASS = 0
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}
RELEVANT_CLASSES = {PERSON_CLASS: "person", **VEHICLE_CLASSES}


class Detector:
    """YOLOv8-based object detector for persons and vehicles."""

    def __init__(self, model_path="yolov8n.pt", confidence=0.5,
                 iou_threshold=0.45, device="cpu", classes=None):
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes or list(RELEVANT_CLASSES.keys())
        self._model = None
        self._loaded = False
        self._total_inferences = 0
        self._total_time_ms = 0.0

    @staticmethod
    def _resolve_model_path(path: str) -> str:
        """Resolve model path across models/ directory, repository root, and current directory."""
        if os.path.exists(path):
            return path
        # Check inside models/ relative to current directory
        candidate = os.path.join("models", os.path.basename(path))
        if os.path.exists(candidate):
            return candidate
        # Check in current working directory root
        candidate = os.path.basename(path)
        if os.path.exists(candidate):
            return candidate
        # Check relative to repository root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for candidate in [
            os.path.join(base_dir, "models", os.path.basename(path)),
            os.path.join(base_dir, os.path.basename(path)),
            os.path.join(base_dir, path),
        ]:
            if os.path.exists(candidate):
                return candidate
        return path

    def load(self):
        """Load the YOLO model."""
        from ultralytics import YOLO
        resolved_path = self._resolve_model_path(self.model_path)
        logger.info(f"Loading model: {resolved_path} (device={self.device})")
        start = time.time()
        self._model = YOLO(resolved_path)
        logger.info(f"Model loaded in {(time.time()-start)*1000:.0f}ms")
        self._loaded = True

    def detect(self, frame: np.ndarray, frame_id=0, timestamp=0.0) -> DetectionResult:
        """Run detection on a single frame."""
        if not self._loaded:
            self.load()

        start = time.time()
        results = self._model(
            frame, conf=self.confidence, iou=self.iou_threshold,
            classes=self.classes, device=self.device, verbose=False,
        )
        inference_ms = (time.time() - start) * 1000
        self._total_inferences += 1
        self._total_time_ms += inference_ms

        detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                if class_id not in RELEVANT_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append(Detection(
                    class_id=class_id,
                    class_name=RELEVANT_CLASSES[class_id],
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                    center=[(x1+x2)/2, (y1+y2)/2],
                    area=(x2-x1)*(y2-y1),
                    timestamp=timestamp,
                ))

        return DetectionResult(
            detections=detections, inference_time_ms=inference_ms,
            frame_id=frame_id, timestamp=timestamp,
        )

    @property
    def avg_inference_ms(self) -> float:
        if self._total_inferences == 0:
            return 0.0
        return self._total_time_ms / self._total_inferences

    @property
    def is_loaded(self) -> bool:
        return self._loaded
