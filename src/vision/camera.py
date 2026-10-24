"""
SentinelVision — Video Source Abstraction

Phase 1: Video ingestion from webcam, file, or RTSP stream.
Provides a unified interface for all video sources.
"""

import cv2
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple
from loguru import logger


@dataclass
class FrameData:
    """Container for a single video frame with metadata."""
    frame: any                    # numpy array (BGR)
    frame_id: int = 0
    timestamp: float = 0.0       # seconds since start
    source_fps: float = 0.0
    width: int = 0
    height: int = 0

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)


class VideoSource(ABC):
    """Abstract base class for all video sources."""

    def __init__(self, max_width: int = 1280, max_height: int = 720):
        self.max_width = max_width
        self.max_height = max_height
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count: int = 0
        self._start_time: float = 0.0
        self._is_open: bool = False

    @abstractmethod
    def open(self) -> bool:
        """Open the video source. Returns True if successful."""
        pass

    def read(self) -> Optional[FrameData]:
        """Read the next frame from the source."""
        if not self._is_open or self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        # Resize if needed
        h, w = frame.shape[:2]
        if w > self.max_width or h > self.max_height:
            scale = min(self.max_width / w, self.max_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        self._frame_count += 1
        timestamp = time.time() - self._start_time

        return FrameData(
            frame=frame,
            frame_id=self._frame_count,
            timestamp=timestamp,
            source_fps=self._cap.get(cv2.CAP_PROP_FPS) or 0.0,
            width=w,
            height=h
        )

    def release(self):
        """Release the video source."""
        if self._cap is not None:
            self._cap.release()
            self._is_open = False
            logger.info("Video source released")

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()


class WebcamSource(VideoSource):
    """Video source from a local webcam."""

    def __init__(self, device_index: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.device_index = device_index

    def open(self) -> bool:
        logger.info(f"Opening webcam (device {self.device_index})")
        self._cap = cv2.VideoCapture(self.device_index)

        if not self._cap.isOpened():
            logger.error(f"Failed to open webcam at index {self.device_index}")
            return False

        # Set buffer size to minimize latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._is_open = True
        self._start_time = time.time()

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Webcam opened: {w}x{h} @ {fps:.1f} FPS")

        return True


class FileSource(VideoSource):
    """Video source from a local file."""

    def __init__(self, path: str, loop: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.loop = loop
        self._total_frames: int = 0

    def open(self) -> bool:
        logger.info(f"Opening video file: {self.path}")
        self._cap = cv2.VideoCapture(self.path)

        if not self._cap.isOpened():
            logger.error(f"Failed to open video file: {self.path}")
            return False

        self._is_open = True
        self._start_time = time.time()
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        duration = self._total_frames / fps if fps > 0 else 0
        logger.info(f"Video opened: {w}x{h} @ {fps:.1f} FPS, {self._total_frames} frames, {duration:.1f}s")

        return True

    def read(self) -> Optional[FrameData]:
        frame_data = super().read()

        if frame_data is None and self.loop and self._is_open:
            # Loop back to start
            logger.debug("Video looping back to start")
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._frame_count = 0
            self._start_time = time.time()
            frame_data = super().read()

        return frame_data

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def progress(self) -> float:
        """Return progress as fraction 0.0 to 1.0."""
        if self._total_frames == 0:
            return 0.0
        return min(self._frame_count / self._total_frames, 1.0)


class RTSPSource(VideoSource):
    """Video source from an RTSP/IP camera stream."""

    def __init__(self, url: str, **kwargs):
        super().__init__(**kwargs)
        self.url = url

    def open(self) -> bool:
        logger.info(f"Opening RTSP stream: {self.url}")

        # Use FFMPEG backend for RTSP
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        if not self._cap.isOpened():
            logger.error(f"Failed to open RTSP stream: {self.url}")
            return False

        # Minimize latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._is_open = True
        self._start_time = time.time()

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"RTSP stream opened: {w}x{h}")

        return True


def create_source(source_type: str, **kwargs) -> VideoSource:
    """Factory function to create a video source."""
    sources = {
        "webcam": WebcamSource,
        "file": FileSource,
        "rtsp": RTSPSource,
    }

    if source_type not in sources:
        raise ValueError(f"Unknown source type: {source_type}. Options: {list(sources.keys())}")

    return sources[source_type](**kwargs)
