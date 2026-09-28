"""Causal, real-time runner: one frame loop, provisional signals, JSON outputs.

At frame `f` only information from frames `<= f` is ever used:

  * SAM3 detection/segmentation is causal by construction; its per-frame output is
    revealed one frame at a time here (mask descriptors are withheld until their
    frame arrives, mirroring a live model).
  * camera ego-motion is fitted from just the previous and current frame.
  * the :class:`ProvisionalDriver` emits loiter/exit as low-latency provisional
    signals and enter at its irreducible causal latency.

Writes ``event.json`` (all emitted signals) and ``diagnostics.json`` (per-track
record of the states each track emitted) per clip, and optionally an annotated
``mp4``.
"""

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2

from config import Config

from .camera import CamMotion
from .detector import Detector
from .driver import ProvisionalDriver
from .visualizer import Visualizer


def _boxes_by_frame(tracks):
    """Invert per-track boxes into a per-frame index.

    Args:
        tracks: {track_id: {frame: box}}.
    Returns:
        {frame: [(track_id, box), ...]} listing every box present in that frame.
    """
    idx = defaultdict(list)
    for tid, tr in tracks.items():
        for f, box in tr.items():
            idx[f].append((tid, box))
    return idx


def run(stem="clip1", cfg=None, display=True, save=True, max_frames=None,
        write_json=True):
    """Run the causal provisional system on one clip; return the emitted events.

    Args:
        stem: clip file stem in cfg.data_dir (e.g. "clip1").
        cfg: optional Config; a default Config() is built when None.
        display: open a live cv2 window (falls back to headless if no GUI).
        save: write an annotated mp4 to cfg.output_dir/stem.
        max_frames: optional cap on the number of frames processed.
        write_json: write event.json and diagnostics.json for the clip.
    Returns:
        list[Event] of every emitted signal, sorted by (frame, person_id).
    """
    cfg = cfg or Config()

    video = cfg.data_dir / f"{stem}.mp4"
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    # Resolve seconds-based thresholds to this clip's fps (exactly like offline).
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    cfg = cfg.resolve_for_fps(fps)

    detector = Detector(cfg)

    # SAM3 forward propagation, run/load it, then reveal it per frame.
    person_tracks, car_tracks, total_frames = detector.track_video(video)
    persons_by_frame = _boxes_by_frame(person_tracks)
    cars_by_frame = _boxes_by_frame(car_tracks)

    # Snapshot the mask descriptors, then clear the detector's live copies so we
    # can feed them back one frame at a time.
    oracle = {
        "pm": {pid: dict(fr) for pid, fr in detector.person_masks.items()},
        "cm": {cid: dict(fr) for cid, fr in detector.car_masks.items()},
        "pp": {pid: dict(fr) for pid, fr in detector.person_polys.items()},
        "cp": {cid: dict(fr) for cid, fr in detector.car_polys.items()},
    }
    detector.person_masks = defaultdict(dict)
    detector.car_masks = defaultdict(dict)
    detector.person_polys = defaultdict(dict)
    detector.car_polys = defaultdict(dict)

    cam = CamMotion(cfg)
    drv = ProvisionalDriver(detector)
    viz = Visualizer(cfg, detector)

    car_hl, person_hl = {}, {}   # id -> event type, set on emission
    shown = {}                   # (pid, type) -> banner entry

    def handle(e, at):
        """Record an emitted event for the banner/highlights and log it.

        Args:
            e: the emitted Event.
            at: the frame index at which it was emitted.
        Returns:
            None.
        """
        key = (e.person_id, e.type)
        shown[key] = {"person_id": e.person_id, "car_id": e.car_id, "type": e.type}
        car_hl[e.car_id] = e.type
        person_hl[e.person_id] = e.type
        print(f"[frame {at:4d}] Person {e.person_id} {e.type.upper()} "
              f"car {e.car_id}  (event frame {e.frame})")

    writer = None
    out_dir = cfg.output_dir / stem
    out_path = out_dir / f"{stem}.mp4"
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    wait = max(1, int(1000 / fps))
    f_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or f_idx >= total_frames:
                break
            if max_frames is not None and f_idx >= max_frames:
                break

            persons = persons_by_frame.get(f_idx, [])
            cars = cars_by_frame.get(f_idx, [])

            # reveal the mask descriptors for the ids visible this frame
            for pid, _b in persons:
                if f_idx in oracle["pm"].get(pid, {}):
                    detector.person_masks[pid][f_idx] = oracle["pm"][pid][f_idx]
                if f_idx in oracle["pp"].get(pid, {}):
                    detector.person_polys[pid][f_idx] = oracle["pp"][pid][f_idx]
            for cid, _b in cars:
                if f_idx in oracle["cm"].get(cid, {}):
                    detector.car_masks[cid][f_idx] = oracle["cm"][cid][f_idx]
                if f_idx in oracle["cp"].get(cid, {}):
                    detector.car_polys[cid][f_idx] = oracle["cp"][cid][f_idx]

            # camera ego-motion (prev + current only)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cur_boxes = [b for _i, b in persons] + [b for _i, b in cars]
            cam_T = cam.step(gray, cur_boxes) #Return the affine (f-1 -> f) for this frame, or None (identity)

            # interaction classifier: one frame in, provisional signals out
            # persons and cars are lists of (id, box) for this frame; cam_T is the affine (f-1 -> f)
            for e in drv.push_frame(f_idx, persons, cars, cam_transform=cam_T):
                handle(e, f_idx)

            if save or display:
                hud = f"frame {f_idx}/{total_frames - 1}"
                viz.draw(frame, f_idx, persons, cars, car_hl, person_hl,
                         shown.values(), total_frames, hud)

            if save:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                    )
                writer.write(frame)

            if display:
                try:
                    cv2.imshow("causal stream", frame)
                    if (cv2.waitKey(wait) & 0xFF) == ord("q"):
                        break
                except cv2.error:
                    print("  [display] no GUI available; continuing headless "
                          "(use --save to write an mp4)")
                    display = False

            f_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    # end of stream: finalize the tail (the driver)
    events = drv.flush()

    print(f"\n{stem}: {len(events)} event(s) "
          f"(causal, provisional, {drv.finalize_delay}-frame latency)")
    for e in sorted(events, key=lambda e: (e.frame, e.person_id)):
        print(f"  Person {e.person_id} {e.type.upper()} car {e.car_id} @ frame {e.frame}")

    if write_json:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "event.json").write_text(
            json.dumps([asdict(e) for e in events], indent=2)
        )
        (out_dir / "diagnostics.json").write_text(
            json.dumps(detector.last_diagnostics, indent=2)
        )
        print(f"  saved {out_dir / 'event.json'}")
        print(f"  saved {out_dir / 'diagnostics.json'}")
    if save:
        print(f"  saved {out_path}")
    return events
