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
python3 main.py --hello "World"
```
- Show runtime and library versions:
```bash
python3 main.py --show-versions
```

## Dependencies
- `opencv-python-headless`: avoids GUI/GL dependencies for server/CI environments
- `mediapipe` (optional): installed only on Python < 3.12 via environment marker

## Project Structure
```
.
├── main.py              # CLI entrypoint
├── example.py           # trivial example script
├── requirements.txt     # pinned, environment-aware dependencies
└── README.md            # this file
```

## Notes
- If you need GUI features, switch to `opencv-python` locally.
- Consider adding a LICENSE file before publishing.
