import argparse
import os
import json
import time
from pathlib import Path
import sys
from typing import List, Optional


def greet(name: str) -> None:
    print(f"Hello, {name}")


def show_versions() -> None:
    python_version = sys.version.split()[0]
    print(f"Python: {python_version}")

    # Try OpenCV
    try:
        import cv2  # type: ignore

        print(f"OpenCV: {cv2.__version__}")
    except Exception:
        print("OpenCV: not installed")

    # Try MediaPipe
    try:
        import mediapipe as mp  # type: ignore

        print(f"MediaPipe: {mp.__version__}")
    except Exception:
        print("MediaPipe: not installed")


def is_display_available() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY"))
    # On macOS/Windows we optimistically assume a windowing system exists
    return True


def list_available_cameras(max_index: int = 10) -> List[int]:
    try:
        import cv2  # type: ignore
    except Exception:
        print("OpenCV is not installed. Install requirements first.")
        return []

    available: List[int] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        if ok:
            available.append(index)
        cap.release()
    return available


def run_gesture_stream(
    source: str,
    mode: str,
    mirror: bool,
    display: bool,
    width: Optional[int],
    height: Optional[int],
    max_frames: Optional[int],
    draw_overlays: bool,
    record_path: Optional[str],
    record_fps: Optional[float],
    snapshot_dir: Optional[str],
    snapshot_interval: Optional[int],
    json_path: Optional[str],
    json_landmarks: bool,
    quiet: bool,
) -> int:
    try:
        import cv2  # type: ignore
    except Exception:
        print("OpenCV is not installed. Install requirements first.")
        return 2

    try:
        import mediapipe as mp  # type: ignore
    except Exception:
        print(
            "MediaPipe is not installed or unsupported on this Python version. "
            "Use Python 3.11 and reinstall, or run without gesture mode."
        )
        return 2

    is_camera = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_camera else source)
    if not cap.isOpened():
        print(f"Failed to open source: {source}")
        return 2

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    def draw_hands(frame_bgr, results_hands) -> None:
        if not results_hands or not getattr(results_hands, "multi_hand_landmarks", None):
            return
        for hand_landmarks in results_hands.multi_hand_landmarks:
            drawing_utils.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )

    def draw_pose(frame_bgr, results_pose) -> None:
        if not results_pose or not getattr(results_pose, "pose_landmarks", None):
            return
        drawing_utils.draw_landmarks(
            frame_bgr,
            results_pose.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
        )

    display_ok = display and is_display_available()

    # Recording setup (lazy-init after first frame)
    writer = None
    target_writer_fps: float | None = record_fps
    record_enabled = bool(record_path)
    record_path_obj: Path | None = Path(record_path) if record_path else None
    fourcc_code = None

    # Snapshot setup
    snapshot_enabled = bool(snapshot_dir)
    snapshot_dir_path: Path | None = Path(snapshot_dir) if snapshot_dir else None
    if snapshot_dir_path:
        snapshot_dir_path.mkdir(parents=True, exist_ok=True)

    # JSON lines setup
    json_enabled = bool(json_path)
    json_fp = None
    if json_enabled:
        if json_path == "-":
            json_fp = sys.stdout
        else:
            json_fp = open(json_path, "w", encoding="utf-8")

    frame_counter = 0

    if mode == "hands":
        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                # Prepare output info
                num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
                if not quiet:
                    print(f"hands: {num_hands}")

                # Drawing
                frame_to_show = frame
                if draw_overlays:
                    draw_hands(frame_to_show, results)

                # JSON
                if json_enabled:
                    payload = {
                        "frame_index": frame_counter,
                        "timestamp_ms": int(time.time() * 1000),
                        "mode": "hands",
                        "hands": {
                            "count": num_hands,
                        },
                    }
                    if json_landmarks and results.multi_hand_landmarks:
                        lm = []
                        for hand in results.multi_hand_landmarks:
                            lm.append([[p.x, p.y, getattr(p, "z", 0.0)] for p in hand.landmark])
                        payload["hands"]["landmarks"] = lm
                    json_fp.write(json.dumps(payload) + "\n")
                    if json_fp is not sys.stdout:
                        json_fp.flush()

                # Display
                if display_ok:
                    try:
                        cv2.imshow("Gesture - Hands", frame_to_show)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break
                    except Exception:
                        display_ok = False

                # Record
                if record_enabled:
                    if writer is None:
                        h, w = frame_to_show.shape[:2]
                        if target_writer_fps is None or target_writer_fps <= 0:
                            cap_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                            target_writer_fps = cap_fps if cap_fps and cap_fps > 0 else 30.0
                        fourcc_code = cv2.VideoWriter_fourcc(*("mp4v" if str(record_path_obj).lower().endswith(".mp4") else "XVID"))
                        writer = cv2.VideoWriter(str(record_path_obj), fourcc_code, float(target_writer_fps), (w, h))
                    writer.write(frame_to_show)

                # Snapshots
                if snapshot_enabled and snapshot_interval and snapshot_interval > 0:
                    if frame_counter % snapshot_interval == 0:
                        assert snapshot_dir_path is not None
                        out_path = snapshot_dir_path / f"frame_{frame_counter:06d}.jpg"
                        cv2.imwrite(str(out_path), frame_to_show)

                frame_counter += 1
                if max_frames is not None and frame_counter >= max_frames:
                    break

    elif mode == "pose":
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                has_pose = bool(results.pose_landmarks)
                if not quiet:
                    print(f"pose: {int(has_pose)}")

                # Drawing
                frame_to_show = frame
                if draw_overlays:
                    draw_pose(frame_to_show, results)

                # JSON
                if json_enabled:
                    payload = {
                        "frame_index": frame_counter,
                        "timestamp_ms": int(time.time() * 1000),
                        "mode": "pose",
                        "pose": {"present": int(has_pose)},
                    }
                    json_fp.write(json.dumps(payload) + "\n")
                    if json_fp is not sys.stdout:
                        json_fp.flush()

                # Display
                if display_ok:
                    try:
                        cv2.imshow("Gesture - Pose", frame_to_show)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break
                    except Exception:
                        display_ok = False

                # Record
                if record_enabled:
                    if writer is None:
                        h, w = frame_to_show.shape[:2]
                        if target_writer_fps is None or target_writer_fps <= 0:
                            cap_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                            target_writer_fps = cap_fps if cap_fps and cap_fps > 0 else 30.0
                        fourcc_code = cv2.VideoWriter_fourcc(*("mp4v" if str(record_path_obj).lower().endswith(".mp4") else "XVID"))
                        writer = cv2.VideoWriter(str(record_path_obj), fourcc_code, float(target_writer_fps), (w, h))
                    writer.write(frame_to_show)

                # Snapshots
                if snapshot_enabled and snapshot_interval and snapshot_interval > 0:
                    if frame_counter % snapshot_interval == 0:
                        assert snapshot_dir_path is not None
                        out_path = snapshot_dir_path / f"frame_{frame_counter:06d}.jpg"
                        cv2.imwrite(str(out_path), frame_to_show)

                frame_counter += 1
                if max_frames is not None and frame_counter >= max_frames:
                    break

    elif mode == "holistic":
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                left = 1 if results.left_hand_landmarks else 0
                right = 1 if results.right_hand_landmarks else 0
                pose_found = 1 if results.pose_landmarks else 0
                if not quiet:
                    print(f"holistic: hands_left={left} hands_right={right} pose={pose_found}")

                # Drawing
                frame_to_show = frame
                if draw_overlays:
                    draw_hands(frame_to_show, type("R", (), {"multi_hand_landmarks": results.left_hand_landmarks and [results.left_hand_landmarks] or []}))
                    draw_hands(frame_to_show, type("R", (), {"multi_hand_landmarks": results.right_hand_landmarks and [results.right_hand_landmarks] or []}))
                    draw_pose(frame_to_show, results)

                # JSON
                if json_enabled:
                    payload = {
                        "frame_index": frame_counter,
                        "timestamp_ms": int(time.time() * 1000),
                        "mode": "holistic",
                        "hands": {"left": left, "right": right},
                        "pose": {"present": pose_found},
                    }
                    if json_landmarks:
                        if results.left_hand_landmarks:
                            payload.setdefault("hands", {})["left_landmarks"] = [
                                [p.x, p.y, getattr(p, "z", 0.0)] for p in results.left_hand_landmarks.landmark
                            ]
                        if results.right_hand_landmarks:
                            payload.setdefault("hands", {})["right_landmarks"] = [
                                [p.x, p.y, getattr(p, "z", 0.0)] for p in results.right_hand_landmarks.landmark
                            ]
                    json_fp.write(json.dumps(payload) + "\n")
                    if json_fp is not sys.stdout:
                        json_fp.flush()

                # Display
                if display_ok:
                    try:
                        cv2.imshow("Gesture - Holistic", frame_to_show)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break
                    except Exception:
                        display_ok = False

                # Record
                if record_enabled:
                    if writer is None:
                        h, w = frame_to_show.shape[:2]
                        if target_writer_fps is None or target_writer_fps <= 0:
                            cap_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                            target_writer_fps = cap_fps if cap_fps and cap_fps > 0 else 30.0
                        fourcc_code = cv2.VideoWriter_fourcc(*("mp4v" if str(record_path_obj).lower().endswith(".mp4") else "XVID"))
                        writer = cv2.VideoWriter(str(record_path_obj), fourcc_code, float(target_writer_fps), (w, h))
                    writer.write(frame_to_show)

                # Snapshots
                if snapshot_enabled and snapshot_interval and snapshot_interval > 0:
                    if frame_counter % snapshot_interval == 0:
                        assert snapshot_dir_path is not None
                        out_path = snapshot_dir_path / f"frame_{frame_counter:06d}.jpg"
                        cv2.imwrite(str(out_path), frame_to_show)

                frame_counter += 1
                if max_frames is not None and frame_counter >= max_frames:
                    break
    else:
        print(f"Unknown mode: {mode}")
        cap.release()
        return 2

    cap.release()
    try:
        import cv2  # type: ignore  # re-import for linters
        if display_ok:
            cv2.destroyAllWindows()
    except Exception:
        pass
    if writer is not None:
        try:
            writer.release()
        except Exception:
            pass
    if json_enabled and json_fp is not None and json_fp is not sys.stdout:
        try:
            json_fp.close()
        except Exception:
            pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gesture recognition CLI")

    # Backward-compatible simple flags
    parser.add_argument("--hello", metavar="NAME", help="Print a greeting to NAME")
    parser.add_argument(
        "--show-versions",
        action="store_true",
        help="Display Python and library versions",
    )

    subparsers = parser.add_subparsers(dest="command")

    # hello subcommand
    hello_p = subparsers.add_parser("hello", help="Print a greeting")
    hello_p.add_argument("name", metavar="NAME")

    # versions subcommand
    subparsers.add_parser("versions", help="Show Python and library versions")

    # list-cameras subcommand
    list_p = subparsers.add_parser("list-cameras", help="List available camera device indices")
    list_p.add_argument("--max-index", type=int, default=10, help="Probe camera indices up to this value (default: 10)")

    # run subcommand
    run_p = subparsers.add_parser("run", help="Run gesture/pose analysis")
    run_p.add_argument("--source", default="0", help="Camera index (e.g. 0) or video file path")
    run_p.add_argument("--mode", choices=["hands", "pose", "holistic"], default="hands", help="Analysis mode")
    run_p.add_argument("--mirror", action="store_true", help="Mirror frames horizontally (selfie view)")
    run_p.add_argument("--display", action="store_true", help="Display a window with overlays if available")
    run_p.add_argument("--width", type=int, help="Requested capture width in pixels")
    run_p.add_argument("--height", type=int, help="Requested capture height in pixels")
    run_p.add_argument("--max-frames", type=int, help="Stop after this many frames (for testing)")
    run_p.add_argument("--no-draw", action="store_true", help="Disable drawing overlays on frames")
    run_p.add_argument("--record", help="Path to output video file (.mp4 or .avi)")
    run_p.add_argument("--record-fps", type=float, help="Override output video FPS")
    run_p.add_argument("--snapshot-dir", help="Directory to save snapshot images")
    run_p.add_argument("--snapshot-interval", type=int, help="Save a snapshot every N frames")
    run_p.add_argument("--json", dest="json_path", help="Write JSON Lines to path, or '-' for stdout")
    run_p.add_argument("--json-landmarks", action="store_true", help="Include hand landmarks in JSON output")
    run_p.add_argument("--quiet", action="store_true", help="Suppress per-frame prints")

    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)

    # Backward-compatible flags
    if getattr(args, "hello", None):
        greet(args.hello)
        return 0
    if getattr(args, "show_versions", False):
        show_versions()
        return 0

    # Subcommands
    if args.command == "hello":
        greet(args.name)
        return 0
    if args.command == "versions":
        show_versions()
        return 0
    if args.command == "list-cameras":
        indices = list_available_cameras(max_index=args.max_index)
        if indices:
            print("Available cameras:", ", ".join(str(i) for i in indices))
            return 0
        print("No cameras detected.")
        return 1
    if args.command == "run":
        return run_gesture_stream(
            source=args.source,
            mode=args.mode,
            mirror=args.mirror,
            display=args.display,
            width=args.width,
            height=args.height,
            max_frames=args.max_frames,
            draw_overlays=not args.no_draw,
            record_path=args.record,
            record_fps=args.record_fps,
            snapshot_dir=args.snapshot_dir,
            snapshot_interval=args.snapshot_interval,
            json_path=args.json_path,
            json_landmarks=args.json_landmarks,
            quiet=args.quiet,
        )

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
