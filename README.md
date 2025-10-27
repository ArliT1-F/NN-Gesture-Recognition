# Python Starter Project

A minimal Python project with a simple CLI and sensible defaults. Includes OpenCV as a headless dependency and treats MediaPipe as optional due to runtime compatibility.

## Requirements
- Python 3.11+ recommended
- On Python 3.12/3.13, MediaPipe wheels may be unavailable; this project excludes MediaPipe automatically on those versions. If you need MediaPipe, use Python 3.11.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Usage
- Greet someone:
```bash
python3 main.py hello "World"
```
- Show runtime and library versions:
```bash
python3 main.py versions
```
- List available cameras (indices):
```bash
python3 main.py list-cameras --max-index 10
```
- Run gesture/pose analysis from a camera or file:
```bash
# From default camera 0, hands mode, with display window
python3 main.py run --source 0 --mode hands --display --mirror

# From a video file, pose mode, no display (prints detections)
python3 main.py run --source ./sample.mp4 --mode pose --max-frames 300
```

## Dependencies
- `opencv-python-headless`: avoids GUI/GL dependencies for server/CI environments; use `--display` only if a GUI is available. If you need on-screen windows locally, you may switch to `opencv-python`.
- `mediapipe` (optional): installed only on Python < 3.12 via environment marker. On Python 3.12/3.13, gesture commands will inform you that MediaPipe is unavailable.

## Project Structure
```
.
├── main.py              # CLI entrypoint
├── example.py           # trivial example script
├── requirements.txt     # pinned, environment-aware dependencies
└── README.md            # this file
```

## Notes
- If you run on a headless Linux server without a display (`$DISPLAY` unset), `--display` will be ignored and results will print to stdout.
- If you need GUI features broadly, switch to `opencv-python` locally.
- Consider adding a LICENSE file before publishing.
