from dataclasses import dataclass

import cv2
import numpy as np

from video_smoke_detection.color_analysis import color_analysis_mask
from video_smoke_detection.config import DetectionConfig
from video_smoke_detection.dark_channel import dark_prior_mask
from video_smoke_detection.motion import MotionHistory


@dataclass
class FrameAnalysis:
    color_mask: np.ndarray
    motion_mask: np.ndarray
    dark_mask: np.ndarray
    combined_mask: np.ndarray
    count: int


def combine_masks(
    color_mask: np.ndarray,
    motion_mask: np.ndarray,
    dark_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    combined = ((color_mask > 75) & (motion_mask > 1) & (dark_mask > 75)).astype(np.uint8) * 255
    return combined, int(np.count_nonzero(combined))


def draw_mask(image: np.ndarray, mask: np.ndarray, block_size: int = 3) -> np.ndarray:
    output = image.copy()
    overlay = image.copy()
    height, width = mask.shape
    for row in range(2, height - 2, 2 * block_size):
        for column in range(2, width - 2, 2 * block_size):
            if mask[row][column] == 255:
                cv2.rectangle(
                    overlay,
                    (column - block_size, row - block_size),
                    (column + block_size, row + block_size),
                    (0, 0, 255),
                    -1,
                )
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    return output


def analyze_frame(
    frame: np.ndarray,
    motion_history: MotionHistory,
    config: DetectionConfig,
    use_guided_filter: bool,
) -> FrameAnalysis:
    color_mask = color_analysis_mask(frame, config.color_threshold, config.max_size)
    _, motion_mask = motion_history.update(frame)
    dark_mask = dark_prior_mask(frame, config.max_size, use_guided_filter)
    combined_mask, count = combine_masks(color_mask, motion_mask, dark_mask)
    return FrameAnalysis(
        color_mask=color_mask,
        motion_mask=motion_mask,
        dark_mask=dark_mask,
        combined_mask=combined_mask,
        count=count,
    )
