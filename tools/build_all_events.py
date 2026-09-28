"""Consolidate all per-clip events into a single all_events.json.

For each interaction it records clip_id, type, frame range + time span (seconds),
person_id / car_id, and a short plain-language description of the person and the
vehicle derived from the actual pixels (dominant colour + relative size +
position) sampled at a representative frame of the track.

Run:  python tools/build_all_events.py
Writes: all_events.json in the repo root.
"""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
CACHE = ROOT / "track_cache"
VIDEO = ROOT / "data" / "given"


def color_name(bgr):
    """Map a mean BGR triple to a coarse colour name via HSV."""
    b, g, r = [float(x) for x in bgr]
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if v < 50:
        return "black"
    if s < 40:
        return "white" if v > 200 else ("light-grey" if v > 120 else "grey")
    if h < 10 or h >= 160:
        return "red"
    if h < 20:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 100:
        return "cyan"
    if h < 130:
        return "blue"
    return "purple"


def size_word(frac):
    """Describe box area as a fraction of the frame."""
    if frac < 0.01:
        return "small"
    if frac < 0.05:
        return "medium-sized"
    return "large"


def pos_word(cx, width):
    """Describe horizontal position of a box centre."""
    r = cx / max(width, 1)
    if r < 0.33:
        return "left"
    if r < 0.66:
        return "centre"
    return "right"


def describe(frame, box, kind, W, H):
    """Build a short description of a person/car from one cropped box."""
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return f"{kind} (not visible in sampled frame)"
    crop = frame[y1:y2, x1:x2]
    mean_bgr = crop.reshape(-1, 3).mean(axis=0)
    col = color_name(mean_bgr)
    frac = ((x2 - x1) * (y2 - y1)) / float(W * H)
    pos = pos_word((x1 + x2) / 2.0, W)
    if kind == "person":
        return f"a {size_word(frac)} person in mostly {col} clothing, {pos} of frame"
    return f"a {size_word(frac)} {col} vehicle, {pos} of frame"


def sample_frame(cap, idx):
    """Grab a single frame by index."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def rep_box(track, f0, f1):
    """Return (frame_idx, box) nearest the middle of [f0, f1] present in track."""
    keys = sorted(int(f) for f in track)
    span = [f for f in keys if f0 <= f <= f1] or keys
    mid = span[len(span) // 2]
    return mid, track[str(mid)] if str(mid) in track else track[mid]


def main():
    out = []
    for run_dir in sorted(RUNS.iterdir()):
        clip = run_dir.name
        ev_path = run_dir / "event.json"
        if not ev_path.exists():
            continue
        events = json.loads(ev_path.read_text())
        if not events:
            continue
        cache_path = CACHE / f"{clip}__sam3seg_s0.5.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        ptracks = cache.get("person_tracks", {})
        ctracks = cache.get("car_tracks", {})

        cap = cv2.VideoCapture(str(VIDEO / f"{clip}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for e in events:
            f0, f1 = e.get("frame_start", e["frame"]), e.get("frame_end", e["frame"])
            pid, cid = e["person_id"], e["car_id"]

            person_desc = f"person {pid}"
            car_desc = f"vehicle {cid}"
            ptrack = ptracks.get(str(pid))
            ctrack = ctracks.get(str(cid))
            if ptrack:
                fi, pbox = rep_box(ptrack, f0, f1)
                frame = sample_frame(cap, fi)
                if frame is not None:
                    person_desc = describe(frame, pbox, "person", W, H)
                    if ctrack and (str(fi) in ctrack or fi in ctrack):
                        cbox = ctrack.get(str(fi), ctrack.get(fi))
                        car_desc = describe(frame, cbox, "car", W, H)

            out.append({
                "clip_id": clip,
                "type": e["type"],
                "frame_range": [int(f0), int(f1)],
                "time_span_s": [round(f0 / fps, 2), round(f1 / fps, 2)],
                "person_id": int(pid),
                "person_description": person_desc,
                "car_id": int(cid),
                "car_description": car_desc,
            })
        cap.release()

    (ROOT / "all_events.json").write_text(json.dumps(out, indent=2))
    print(f"wrote all_events.json with {len(out)} interactions")


if __name__ == "__main__":
    main()
