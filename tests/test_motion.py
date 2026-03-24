import numpy as np


def test_motion_difference_returns_zero_for_identical_frames() -> None:
    from video_smoke_detection.motion import motion_difference

    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    diff = motion_difference(frame, frame, max_size=500)

    assert np.count_nonzero(diff) == 0


def test_motion_history_first_update_returns_empty_history() -> None:
    from video_smoke_detection.motion import MotionHistory

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    history = MotionHistory(height=4, width=4)

    timestamp, mask = history.update(frame)

    assert timestamp == 1
    assert mask.shape == (4, 4)
    assert np.count_nonzero(mask) == 0


def test_motion_history_second_update_returns_visualization_without_motempl() -> None:
    from video_smoke_detection.motion import MotionHistory

    first = np.zeros((4, 4, 3), dtype=np.uint8)
    second = np.full((4, 4, 3), 64, dtype=np.uint8)
    history = MotionHistory(height=4, width=4)

    history.update(first)
    timestamp, mask = history.update(second)

    assert timestamp == 2
    assert mask.shape == (4, 4)
