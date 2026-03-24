import numpy as np


def test_resize_image_keeps_small_image_shape() -> None:
    from video_smoke_detection.image_ops import resize_image

    image = np.zeros((10, 8, 3), dtype=np.uint8)

    result = resize_image(image, 50)

    assert result.shape == (10, 8, 3)


def test_to_gray_returns_two_dimensional_image() -> None:
    from video_smoke_detection.image_ops import to_gray

    image = np.zeros((2, 2, 3), dtype=np.uint8)

    result = to_gray(image, 500)

    assert result.shape == (2, 2)


def test_float_to_uint8_matches_legacy_scaling() -> None:
    from video_smoke_detection.image_ops import float_to_uint8

    image = np.array([[0.5, 1.0]], dtype=np.float32)

    result = float_to_uint8(image)

    assert result.tolist() == [[127, 255]]
