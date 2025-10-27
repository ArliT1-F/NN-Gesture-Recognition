import argparse
import sys


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal project CLI")
    parser.add_argument(
        "--hello",
        metavar="NAME",
        help="Print a greeting to NAME",
    )
    parser.add_argument(
        "--show-versions",
        action="store_true",
        help="Display Python and library versions",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)

    did_anything = False

    if args.hello:
        greet(args.hello)
        did_anything = True

    if args.show_versions:
        show_versions()
        did_anything = True

    if not did_anything:
        parser.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
