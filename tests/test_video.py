from pathlib import Path

import cv2
import numpy as np


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (8, 8),
        True,
    )
    for value in (0, 32, 64, 96, 128):
        frame = np.full((8, 8, 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_process_video_writes_output_file(tmp_path: Path) -> None:
    from video_smoke_detection.video import process_video

    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "output.avi"
    _write_test_video(input_path)

    result = process_video(input_path, output_path=output_path)

    assert result.exists()
    assert result.stat().st_size > 0
