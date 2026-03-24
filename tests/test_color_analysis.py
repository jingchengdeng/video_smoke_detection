import numpy as np


def test_color_mask_marks_bright_neutral_pixels_as_smoke() -> None:
    from video_smoke_detection.color_analysis import color_analysis_mask

    image = np.full((1, 1, 3), 200, dtype=np.uint8)

    mask = color_analysis_mask(image, alpha=85, max_size=500)

    assert mask.tolist() == [[255]]


def test_color_mask_rejects_black_pixels() -> None:
    from video_smoke_detection.color_analysis import color_analysis_mask

    image = np.zeros((1, 1, 3), dtype=np.uint8)

    mask = color_analysis_mask(image, alpha=85, max_size=500)

    assert mask.tolist() == [[0]]


def test_color_mask_rejects_strongly_colored_pixels() -> None:
    from video_smoke_detection.color_analysis import color_analysis_mask

    image = np.array([[[200, 50, 50]]], dtype=np.uint8)

    mask = color_analysis_mask(image, alpha=85, max_size=500)

    assert mask.tolist() == [[0]]
