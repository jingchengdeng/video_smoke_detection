import cv2
import numpy as np

from video_smoke_detection.image_ops import resize_image


def color_analysis_mask(image: np.ndarray, alpha: int, max_size: int) -> np.ndarray:
    resized = resize_image(image, max_size)
    blurred = cv2.GaussianBlur(resized, (5, 5), 5)

    red = blurred[:, :, 0].astype(np.int16)
    green = blurred[:, :, 1].astype(np.int16)
    blue = blurred[:, :, 2].astype(np.int16)

    mean_intensity = (red + green + blue) / 3
    is_neutral = (
        (np.abs(red - green) < alpha)
        & (np.abs(red - blue) < alpha)
        & (np.abs(green - blue) < alpha)
        & (mean_intensity > 100)
    )

    return np.where(is_neutral, 255, 0).astype(np.uint8)
