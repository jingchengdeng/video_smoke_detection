import cv2
import numpy as np


def resize_image(image: np.ndarray, max_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height > max_size or width > max_size:
        scale = min(float(max_size) / float(height), float(max_size) / float(width))
        return cv2.resize(image, (int(width * scale), int(height * scale)))
    return image


def to_gray(image: np.ndarray, max_size: int) -> np.ndarray:
    resized = resize_image(image, max_size)
    if len(resized.shape) > 2:
        return cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    return resized


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    return (image * 255).astype(np.uint8)
