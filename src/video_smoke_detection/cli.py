import argparse
from pathlib import Path

from video_smoke_detection.video import extract_frames, process_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-smoke-detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Process a video and write an overlay output.")
    process_parser.add_argument("input_path")
    process_parser.add_argument("--output", required=True)
    process_parser.add_argument("--guided-filter", action="store_true")

    extract_parser = subparsers.add_parser("extract-frames", help="Extract frames from a video.")
    extract_parser.add_argument("input_path")
    extract_parser.add_argument("--output-dir", required=True)
    extract_parser.add_argument("--every", type=int, default=5)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "process":
        process_video(
            Path(args.input_path),
            output_path=Path(args.output),
            use_guided_filter=args.guided_filter,
        )
        return 0

    extract_frames(
        Path(args.input_path),
        output_dir=Path(args.output_dir),
        every_n_frames=args.every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
