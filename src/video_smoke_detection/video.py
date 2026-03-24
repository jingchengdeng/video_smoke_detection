from pathlib import Path

import cv2

from video_smoke_detection.config import DetectionConfig
from video_smoke_detection.image_ops import resize_image
from video_smoke_detection.motion import MotionHistory
from video_smoke_detection.pipeline import analyze_frame, draw_mask


def _video_fourcc(path: Path) -> int:
    return cv2.VideoWriter_fourcc(*("MJPG" if path.suffix.lower() == ".avi" else "mp4v"))


def process_video(
    input_path: Path | str,
    *,
    output_path: Path | str,
    use_guided_filter: bool = False,
    config: DetectionConfig | None = None,
) -> Path:
    config = config or DetectionConfig()
    input_path = Path(input_path)
    output_path = Path(output_path)

    capture = cv2.VideoCapture(str(input_path))
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise ValueError(f"Unable to read video: {input_path}")

    resized = resize_image(frame, config.max_size)
    height, width = resized.shape[:2]
    history = MotionHistory(height=height, width=width, config=config)
    writer = cv2.VideoWriter(str(output_path), _video_fourcc(output_path), 5.0, (width, height), True)

    try:
        current = frame
        while current is not None:
            resized = resize_image(current, config.max_size)
            analysis = analyze_frame(resized, history, config, use_guided_filter)
            writer.write(draw_mask(resized, analysis.combined_mask))
            ok, current = capture.read()
            if not ok:
                current = None
    finally:
        writer.release()
        capture.release()

    return output_path


def extract_frames(
    input_path: Path | str,
    *,
    output_dir: Path | str,
    every_n_frames: int = 5,
) -> list[Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    frame_count = 1
    written: list[Path] = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_count % every_n_frames == 0:
                path = output_dir / f"frame{frame_count}.jpg"
                cv2.imwrite(str(path), frame)
                written.append(path)
            frame_count += 1
    finally:
        capture.release()

    return written
