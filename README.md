# Video Smoke Detection

Original smoke-detection coursework project completed in November 2018, now being modernized into a small Python package with a CLI, tests, and a cleaner repository structure.

## Project History

The original project version lives on the legacy baseline in the `master` branch. That version reflects the 2018 coursework submission and repo layout.

This `dev` branch modernizes the codebase because some of the original package/runtime assumptions are no longer reliably available in a current environment. The goal of the modernization work is to preserve the original project intent and core pipeline while making the repository easier to run, test, and maintain today.

## What It Does

The pipeline combines three signals on each frame:

- color-based smoke candidate detection
- motion history analysis
- dark-channel-prior smoke masking

Those masks are combined into a final smoke mask, which can then be rendered back onto output video frames.

## Project Layout

```text
src/video_smoke_detection/   package code
tests/                       automated tests
data/                        tracked sample still images
videos/                      tracked sample source videos
docs/superpowers/            design and implementation docs
```

## Setup

Create a local environment and install the package dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -e . pytest opencv-python-headless scikit-video
```

## CLI Usage

Process a video and write an overlay result:

```bash
.venv/bin/python -m video_smoke_detection.cli process videos/simple_smoke.mp4 --output outputs/output.avi
```

You can also invoke the installed command directly:

```bash
video-smoke-detection process videos/simple_smoke.mp4 --output outputs/output.avi
```

Extract frames from a video:

```bash
.venv/bin/python -m video_smoke_detection.cli extract-frames videos/simple_smoke.mp4 --output-dir extracted_frames --every 5
```

## Tests

Run the full test suite with:

```bash
.venv/bin/pytest -v
```

## Media Policy

Source and sample media remain tracked in git so the project stays reproducible. Generated outputs are ignored and should not be committed. That includes derived overlay videos, extracted-frame dumps, local test artifacts, and worktree/session files.

## Legacy Entry Point

`smoke.py` remains as a compatibility entrypoint and now forwards to the package CLI.
