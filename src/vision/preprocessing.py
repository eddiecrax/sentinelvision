"""
SentinelVision — Frame Preprocessing

Phase 2: Normalize frames before detection.
Handles resolution, color space, and optional enhancements.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from loguru import logger


@dataclass
class PreprocessConfig:
    """Configuration for frame preprocessing."""
    target_width: int = 640
    target_height: int = 640
    normalize: bool = False
    enhance_contrast: bool = False
    denoise: bool = False


class FramePreprocessor:
    """Preprocesses video frames for detection."""

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to a frame.

        The frame passed to detection is the original (or lightly enhanced).
        YOLO handles its own resizing internally, so we don't resize here
        unless explicitly needed for other purposes.
        """
        result = frame.copy()

        if self.config.denoise:
            result = cv2.fastNlMeansDenoisingColored(result, None, 6, 6, 7, 21)

        if self.config.enhance_contrast:
            # CLAHE on the L channel of LAB color space
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(l_channel)
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return result

    def letterbox(self, frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Resize frame with letterboxing (preserving aspect ratio).
        Useful for manual model input if needed.
        """
        h, w = frame.shape[:2]
        tw, th = target_size

        scale = min(tw / w, th / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        x_offset = (tw - new_w) // 2
        y_offset = (th - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas
