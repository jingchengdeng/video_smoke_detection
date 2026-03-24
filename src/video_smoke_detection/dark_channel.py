import cv2
import numpy as np

from video_smoke_detection.guidedfilter import guided_filter
from video_smoke_detection.image_ops import resize_image


def get_dark_channel(image: np.ndarray, block_size: int) -> np.ndarray:
    blue, green, red = cv2.split(image)
    min_rgb = cv2.min(cv2.min(red, green), blue)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (block_size, block_size))
    return cv2.erode(min_rgb, kernel)


def get_atmospheric_light(image: np.ndarray, dark_channel: np.ndarray) -> np.ndarray:
    height, width = dark_channel.shape[:2]
    pixel_count = int(max(np.floor((height * width) / 1000), 1))
    dark_reshape = dark_channel.reshape(height * width, 1)
    image_reshape = image.reshape(height * width, 3)
    indices = dark_reshape.argsort()[height * width - pixel_count :]

    atmosphere = np.zeros((1, 3))
    for index in range(1, pixel_count):
        atmosphere = atmosphere + image_reshape[index]
    return atmosphere / pixel_count


def transmission(
    image: np.ndarray,
    atmosphere: np.ndarray,
    block_size: int,
    use_guided_filter: bool,
) -> np.ndarray:
    omega = 0.95
    normalized = np.empty(image.shape, image.dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        for channel in range(3):
            normalized[:, :, channel] = image[:, :, channel] / atmosphere[0, channel]

    result = 1 - omega * get_dark_channel(normalized, block_size)
    result[result < 0.1] = 0.1

    if use_guided_filter:
        norm_image = (image - image.min()) / (image.max() - image.min())
        result = guided_filter(norm_image, result, 40, 0.0001)

    return result


def dark_prior_mask(image: np.ndarray, max_size: int, use_guided_filter: bool) -> np.ndarray:
    resized = resize_image(image, max_size)
    normalized = resized.astype("float64") / 255
    dark_channel = get_dark_channel(normalized, 15)
    atmosphere = get_atmospheric_light(normalized, dark_channel)
    transmission_map = transmission(normalized, atmosphere, 15, use_guided_filter)

    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    grayscale[transmission_map < 0.4] = 255
    grayscale[transmission_map >= 0.4] = 0
    return grayscale
