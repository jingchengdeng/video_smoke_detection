import numpy as np


def test_combine_masks_only_marks_pixels_present_in_all_masks() -> None:
    from video_smoke_detection.pipeline import combine_masks

    mask_a = np.array([[76, 80], [0, 100]], dtype=np.uint8)
    mask_b = np.array([[2, 1], [2, 2]], dtype=np.uint8)
    mask_c = np.array([[80, 80], [80, 0]], dtype=np.uint8)

    mask, count = combine_masks(mask_a, mask_b, mask_c)

    assert mask.tolist() == [[255, 0], [0, 0]]
    assert count == 1


def test_draw_mask_preserves_image_shape() -> None:
    from video_smoke_detection.pipeline import draw_mask

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[2, 2] = 255

    overlay = draw_mask(image, mask, block_size=1)

    assert overlay.shape == image.shape
