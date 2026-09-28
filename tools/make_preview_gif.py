"""Make a short preview GIF from an annotated run mp4.

Trims a range of frames from ``runs/<clip>/<clip>.mp4`` and writes a downscaled,
looping GIF suitable for embedding in the README.

Usage (from the project root, inside the ``person_car`` env):

    python tools/make_preview_gif.py --clip clip2 --start 0 --frames 120

Options:
    --clip     Clip stem whose run mp4 to read (default: clip2).
    --start    Start frame index (default: 0).
    --frames   Number of source frames to capture (default: 120).
    --stride   Keep every Nth source frame; overrides the fps-derived step.
    --width    Output width in px; height auto-scales (default: 480).
    --fps      Output GIF frame rate (default: 12).
    --out      Output path (default: docs/assets/<clip>_preview.gif).
"""

import argparse
from pathlib import Path

import cv2
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", default="clip2")
    ap.add_argument("--start", type=int, default=0, help="Start frame index.")
    ap.add_argument("--frames", type=int, default=120, help="Number of source frames to capture.")
    ap.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Keep every Nth source frame. Overrides the fps-derived step when set.",
    )
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    src = root / "runs" / args.clip / f"{args.clip}.mp4"
    if not src.exists():
        raise SystemExit(f"Run mp4 not found: {src}")

    out = Path(args.out) if args.out else root / "docs" / "assets" / f"{args.clip}_preview.gif"
    out.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Keep every Nth source frame. An explicit --stride wins; otherwise pick a
    # step that makes the GIF play at ~args.fps.
    step = args.stride if args.stride else max(1, round(src_fps / args.fps))
    step = max(1, step)

    start_frame = int(args.start)
    end_frame = start_frame + int(args.frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    idx = start_frame
    while idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start_frame) % step == 0:
            h, w = frame.shape[:2]
            new_w = args.width
            new_h = int(h * new_w / w)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        idx += 1
    cap.release()

    if not frames:
        raise SystemExit("No frames captured — check --start/--frames.")

    duration_ms = int(1000 / args.fps)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_kb = out.stat().st_size / 1024
    print(f"Wrote {out} ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()








