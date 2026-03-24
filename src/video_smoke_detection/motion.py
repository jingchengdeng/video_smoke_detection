from dataclasses import dataclass, field

import cv2
import numpy as np

from video_smoke_detection.config import DetectionConfig
from video_smoke_detection.image_ops import resize_image, to_gray


def motion_difference(frame_a: np.ndarray, frame_b: np.ndarray, max_size: int) -> np.ndarray:
    return cv2.absdiff(to_gray(frame_a, max_size), to_gray(frame_b, max_size))


@dataclass
class MotionHistory:
    height: int
    width: int
    config: DetectionConfig = field(default_factory=DetectionConfig)
    timestamp: int = 0
    motion_history: np.ndarray = field(init=False)
    last_frame: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.motion_history = np.zeros((self.height, self.width), np.float32)
        self.last_frame = np.zeros((self.height, self.width), np.float32)

    def update(self, frame: np.ndarray) -> tuple[int, np.ndarray]:
        resized = resize_image(frame, self.config.max_size)
        if self.timestamp == 0:
            self.last_frame = resized
            self.timestamp += 1
            return self.timestamp, np.uint8(self.motion_history)

        difference = motion_difference(self.last_frame, resized, self.config.max_size)
        _, motion_mask = cv2.threshold(
            difference,
            self.config.default_threshold,
            1,
            cv2.THRESH_BINARY,
        )
        self.timestamp += 1
        self.last_frame = resized
        self.motion_history[motion_mask == 1] = self.timestamp
        stale = self.motion_history < (self.timestamp - self.config.mhi_duration)
        self.motion_history[stale] = 0
        visualization = np.uint8(
            np.clip(
                (self.motion_history - (self.timestamp - self.config.mhi_duration))
                / self.config.mhi_duration,
                0,
                1,
            )
            * 255
        )
        return self.timestamp, visualization
