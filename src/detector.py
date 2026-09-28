"""Segmentation-aware person-car interaction detector (SAM3 masks).

This class owns everything that is *per finished track* or *per stream*:

  * proximity / motion / continuity primitives (closeness, DIoU, car speed, ...)
  * SAM3 tracking with mask capture, plus a JSON seg-cache reader/writer so a run
    needs no GPU when a cache already exists
  * ``axis_gate`` -- the segmentation major-axis gate that drops enter/exit
    events off the car's body, used by the streaming driver's provisional gates.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .geometry import (
    box_center,
    box_diag,
    center_distance_in_diagonals,
    closeness,
    diou,
    mask_ellipse,
    mask_polygons,
    on_major_axis,
)


class Detector:
    """Tracks people and cars (SAM3) and classifies enter/exit/loiter per track."""

    def __init__(self, cfg):
        """Initialize the detector state (config, SAM3 handle, mask stores).

        Args:
            cfg: the resolved Config dataclass holding every threshold/path.
        Returns:
            None.
        """
        self.cfg = cfg
        self._sam3 = None            # lazily built SAM3 video predictor
        self.last_diagnostics = []   # per-track record of emitted states, last run

        # {track_id: {frame: [cx, cy, area, a, b, theta]}}
        self.person_masks = defaultdict(dict)
        self.car_masks = defaultdict(dict)
        # {track_id: {frame: [[[x, y], ...], ...]}}  (mask silhouette polygons)
        self.person_polys = defaultdict(dict)
        self.car_polys = defaultdict(dict)
        # True stream frame size (known from the first frame -> causal).
        self.frame_w = 0
        self.frame_h = 0

    # ---------------------------------------------------------------- #
    # Proximity primitives
    # ---------------------------------------------------------------- #
    def closeness(self, person, car) -> float:
        """Closeness in [0, 1] of a person box to a car box (uses cfg.near_dist).

        Args:
            person: person box [x1, y1, x2, y2(, conf)].
            car: car box [x1, y1, x2, y2(, conf)].
        Returns:
            float in [0, 1]; 1 = on/inside the car, ~0 = far away.
        """
        return closeness(person, car, self.cfg.near_dist)

    def window_closeness(self, person_track, frames, car_frames) -> float:
        """Mean closeness of a person to one car over a set of frames.

        Args:
            person_track: {frame: person_box} for one person.
            frames: iterable of frame indices to average over.
            car_frames: {frame: car_box} for one car.
        Returns:
            float mean closeness in [0, 1] (0 if `frames` is empty).
        """
        if not frames:
            return 0.0
        total = 0.0
        for f in frames:
            if f in person_track and f in car_frames:
                total += self.closeness(person_track[f], car_frames[f])
        return total / len(frames)

    def window_overlap(self, person_track, frames, car_frames) -> float:
        """Mean fraction of the person box overlapping one car over `frames`.
        Measures, on average across a set of frames, how much of the person's 
        bounding box sits inside the car's bounding box
        (0 = never overlapping, 1 = person box fully inside the car).
        Args:
            person_track: {frame: person_box} for one person.
            frames: iterable of frame indices to average over.
            car_frames: {frame: car_box} for one car.
        Returns:
            float in [0, 1]; mean (intersection / person-area) over `frames`.
        """
        if not frames:
            return 0.0
        total = 0.0
        for f in frames:
            if f in person_track and f in car_frames:
                p, c = person_track[f], car_frames[f]
                ix1, iy1 = max(p[0], c[0]), max(p[1], c[1])
                ix2, iy2 = min(p[2], c[2]), min(p[3], c[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                person_area = max(1e-6, (p[2] - p[0]) * (p[3] - p[1]))
                total += inter / person_area
        return total / len(frames)

    def mean_car_diag(self, frames, car_frames) -> float:
        """Mean car box diagonal over `frames` (0 when the car is absent).

        Args:
            frames: iterable of frame indices to average over.
            car_frames: {frame: car_box} for one car.
        Returns:
            float mean box diagonal in pixels (0 if the car is in no frame).
        """
        diags = [box_diag(car_frames[f]) for f in frames if f in car_frames]
        return sum(diags) / len(diags) if diags else 0.0

    def window_diou(self, person_track, frames, car_frames) -> float:
        """Mean DIoU of a person to one car over `frames` (0 when a box absent).

        Args:
            person_track: {frame: person_box} for one person.
            frames: iterable of frame indices to average over.
            car_frames: {frame: car_box} for one car.
        Returns:
            float mean distance-IoU in [-1, 1] (0 if no shared frame).
        """
        vals = [diou(person_track[f], car_frames[f])
                for f in frames if f in person_track and f in car_frames]
        return sum(vals) / len(vals) if vals else 0.0

    def best_car(self, person_track, frames, car_tracks):
        """Car id with the highest mean closeness to the person over `frames`.

        Args:
            person_track: {frame: person_box} for one person.
            frames: iterable of frame indices to score over.
            car_tracks: {car_id: {frame: car_box}} for every car.
        Returns:
            (best_id, best_val): the car id and its mean closeness
            (None, -1.0 if there are no cars).
        """
        best_id, best_val = None, -1.0
        for cid, car_frames in car_tracks.items():
            val = self.window_closeness(person_track, frames, car_frames)
            if val > best_val:
                best_id, best_val = cid, val
        return best_id, best_val

    def radial_motion(self, person_track, car_frames) -> float:
        """Net approach of a person toward a car over their shared frames.

        Args:
            person_track: {frame: person_box} for one person.
            car_frames: {frame: car_box} for one car.
        Returns:
            float distance change in car-diagonals; >0 = approached,
            <0 = departed, 0 if fewer than 2 shared frames.
        """
        common = sorted(set(person_track) & set(car_frames))
        if len(common) < 2:
            return 0.0
        d_first = center_distance_in_diagonals(
            person_track[common[0]], car_frames[common[0]]
        )
        d_last = center_distance_in_diagonals(
            person_track[common[-1]], car_frames[common[-1]]
        )
        return d_first - d_last

    def car_speed(self, car_frames, frames, cam_motion) -> float:
        """Mean ground-relative per-frame car displacement over `frames`, in diagonals.

        Args:
            car_frames: {frame: car_box} for one car.
            frames: iterable of frame indices to measure over.
            cam_motion: {frame: 2x3 affine (f-1 -> f)} camera ego-motion.
        Returns:
            float mean per-frame car motion in car-diagonals (0 if <2 frames);
            camera motion is compensated so a parked car reads ~0.
        """
        fs = [f for f in frames if f in car_frames]
        if len(fs) < 2:
            return 0.0
        disp = 0.0
        n = 0
        for a, b in zip(fs, fs[1:]):
            if b != a + 1:
                continue
            ca = box_center(car_frames[a])
            cb = box_center(car_frames[b])
            m = cam_motion.get(b)
            if m is not None:
                px = m[0, 0] * ca[0] + m[0, 1] * ca[1] + m[0, 2]
                py = m[1, 0] * ca[0] + m[1, 1] * ca[1] + m[1, 2]
            else:
                px, py = ca
            disp += math.hypot(cb[0] - px, cb[1] - py)
            n += 1
        if n == 0:
            return 0.0
        return disp / n / box_diag(car_frames[fs[0]])

    def person_drift(self, person_frames, car_frames, frames):
        """Net displacement and straightness of a person over `frames`.
        measures how far a person moved near a car and how straight that movement was — 
        used by the loiter gate to reject people who merely walk past a car (a straight, long path)
        versus those who linger (short, meandering path).

        Args:
            person_frames: {frame: person_box} for one person.
            car_frames: {frame: car_box} (only used to normalize by car size).
            frames: iterable of frame indices to measure over.
        Returns:
            (net, straight): net straight-line displacement in car-diagonals,
            and straightness = net/path in [0, 1] (~1 straight walk-through,
            ~0 meandering). Both 0.0 if fewer than 2 valid frames.
        """
        fs = [f for f in frames if f in person_frames and f in car_frames]
        if len(fs) < 2:
            return 0.0, 0.0
        centers = [box_center(person_frames[f]) for f in fs]
        diag = sum(box_diag(car_frames[f]) for f in fs) / len(fs) or 1.0
        net = math.hypot(centers[-1][0] - centers[0][0],
                         centers[-1][1] - centers[0][1]) / diag
        path = sum(math.hypot(centers[i + 1][0] - centers[i][0],
                              centers[i + 1][1] - centers[i][1])
                   for i in range(len(centers) - 1)) / diag
        straight = net / path if path > 0 else 0.0
        return net, straight

    def estimate_frame_diag(self) -> float:
        """Causal frame diagonal: the fixed per-stream image size.

        Args:
            # person_tracks: {person_id: {frame: box}} (fallback size source).
            # car_tracks: {car_id: {frame: box}} (fallback size source).
        Returns:
            float image diagonal in pixels; uses the known frame size when
            available, else the max box extent seen (never 0).
        """
        if self.frame_w and self.frame_h:
            return math.hypot(self.frame_w, self.frame_h)

    # ---------------------------------------------------------------- #
    # SAM3 tracking + seg cache
    # ---------------------------------------------------------------- #
    def track_video(self, video: Path):
        """Run SAM3 (keeping masks) or load boxes + descriptors from the seg cache.

        Args:
            video: path to the input clip.
        Returns:
            (person_tracks, car_tracks, total_frames), each *_tracks mapping a
            track id to {frame: xyxy+conf box}. Mask descriptors / polygons are
            stored on the detector (person_masks, car_masks, person_polys,
            car_polys) as a side effect.
        """
        cfg = self.cfg
        cache = cfg.cache_dir / f"{video.stem}__sam3seg_s{cfg.sam3_score_th}.json"
        if cfg.use_track_cache and not cfg.refresh_tracks and cache.exists():
            print(f"  [cache] loading seg tracks from {cache.name}")
            pt, ct, pm, cm, pp, cp, fw, fh, tf = self._load_seg(cache)
            self.person_masks, self.car_masks = pm, cm
            self.person_polys, self.car_polys = pp, cp
            if not (fw and fh):
                probe = cv2.VideoCapture(str(video))
                fw = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
                fh = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
                probe.release()
            self.frame_w, self.frame_h = fw, fh
            return pt, ct, tf

        person_tracks, car_tracks, total_frames = self._track_sam3_seg(video)
        if cfg.use_track_cache:
            self._save_seg(cache, person_tracks, car_tracks, total_frames)
            print(f"  [cache] saved seg tracks to {cache.name}")
        return person_tracks, car_tracks, total_frames

    def _sam3_predictor(self):
        """Lazily build (and cache) the SAM3 video predictor.

        Args:
            None.
        Returns:
            The Sam3VideoPredictor instance (created on first call, then reused).
        """
        if self._sam3 is None:
            import sys

            src = str(self.cfg.sam3_src)
            if src not in sys.path:
                sys.path.insert(0, src)
            import torch
            from sam3.model.sam3_video_predictor import Sam3VideoPredictor

            if torch.cuda.is_available():
                torch.cuda.set_device(self.cfg.sam3_gpu)
            self._sam3 = Sam3VideoPredictor(
                checkpoint_path=self.cfg.sam3_checkpoint or None,
                bpe_path=self.cfg.sam3_bpe or None,
            )
        return self._sam3

    def _track_sam3_seg(self, video: Path):
        """Detect + track people and cars with SAM3, recording mask descriptors.

        Args:
            video: path to the input clip.
        Returns:
            (person_tracks, car_tracks, total_frames); mask ellipses and polygons
            are written to the detector's mask/poly stores as a side effect.
        """
        cfg = self.cfg
        predictor = self._sam3_predictor()

        cap = cv2.VideoCapture(str(video))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
        cap.release()
        self.frame_w, self.frame_h = width, height

        person_tracks = defaultdict(dict)
        car_tracks = defaultdict(dict)
        self.person_masks = defaultdict(dict)
        self.car_masks = defaultdict(dict)
        self.person_polys = defaultdict(dict)
        self.car_polys = defaultdict(dict)
        total_frames = 0

        session_id = predictor.handle_request(
            {"type": "start_session", "resource_path": str(video)}
        )["session_id"]
        try:
            for prompt, box_store, mask_store, poly_store in (
                (cfg.sam3_person_prompt, person_tracks, self.person_masks,
                 self.person_polys),
                (cfg.sam3_car_prompt, car_tracks, self.car_masks, self.car_polys),
            ):
                predictor.handle_request(
                    {"type": "reset_session", "session_id": session_id}
                )
                predictor.handle_request(
                    {"type": "add_prompt", "session_id": session_id,
                     "frame_index": 0, "text": prompt}
                )
                for resp in predictor.handle_stream_request(
                    {"type": "propagate_in_video", "session_id": session_id,
                     "propagation_direction": "forward"}
                ):
                    f_idx = int(resp["frame_index"])
                    total_frames = max(total_frames, f_idx + 1)
                    self._store_seg_frame(resp["outputs"], box_store, mask_store,
                                          poly_store, f_idx, width, height)
        finally:
            predictor.handle_request(
                {"type": "close_session", "session_id": session_id}
            )
        return person_tracks, car_tracks, total_frames

    @staticmethod
    def _as_list(value):
        """Convert a numpy array or any iterable to a plain Python list.

        Args:
            value: a numpy array (has .tolist) or any iterable.
        Returns:
            A plain Python list of the values.
        """
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    def _store_seg_frame(self, out, box_store, mask_store, poly_store,
                         f_idx, width, height):
        """Record one SAM3 frame: box5, mask ellipse and mask silhouette polygons.

        Args:
            out: one SAM3 frame output (boxes_xywh, obj_ids, probs, binary_masks).
            box_store: {track_id: {frame: box5}} to append boxes into.
            mask_store: {track_id: {frame: ellipse}} to append mask ellipses into.
            poly_store: {track_id: {frame: polygons}} to append silhouettes into.
            f_idx: frame index being stored.
            width: frame width in pixels (to un-normalize boxes).
            height: frame height in pixels (to un-normalize boxes).
        Returns:
            None (writes into the passed stores in place).
        """
        boxes = out.get("out_boxes_xywh")
        ids = out.get("out_obj_ids")
        if boxes is None or ids is None:
            return
        probs = out.get("out_probs")
        masks = out.get("out_binary_masks")
        boxes = self._as_list(boxes)
        ids = self._as_list(ids)
        probs = self._as_list(probs) if probs is not None else None
        for i, obj_id in enumerate(ids):
            p = float(probs[i]) if probs is not None and probs[i] is not None else 1.0
            if p < self.cfg.sam3_score_th:
                continue
            x, y, w, h = (float(v) for v in list(boxes[i])[:4])
            box5 = np.array(
                [x * width, y * height, (x + w) * width, (y + h) * height, p],
                dtype=np.float32,
            )
            store_id = int(obj_id)
            box_store[store_id][f_idx] = box5
            if masks is not None and i < len(masks):
                mask = np.asarray(masks[i])
                ell = mask_ellipse(mask)
                if ell is not None:
                    mask_store[store_id][f_idx] = ell
                polys = mask_polygons(mask)
                if polys:
                    poly_store[store_id][f_idx] = polys

    def _save_seg(self, cache: Path, person_tracks, car_tracks, total_frames):
        """Serialize boxes + mask descriptors to a single JSON cache.

        Args:
            cache: destination JSON path.
            person_tracks: {person_id: {frame: box}}.
            car_tracks: {car_id: {frame: box}}.
            total_frames: number of frames in the clip.
        Returns:
            None (writes the cache file).
        """
        def enc_boxes(tracks):
            """Encode {tid: {frame: box}} tracks to JSON-friendly nested dicts."""
            return {str(tid): {str(f): [float(x) for x in box] for f, box in tr.items()}
                    for tid, tr in tracks.items()}

        def enc_masks(masks):
            """Encode {tid: {frame: ellipse}} mask descriptors to nested dicts."""
            return {str(tid): {str(f): [float(x) for x in ell] for f, ell in tr.items()}
                    for tid, tr in masks.items()}

        def enc_polys(polys):
            """Encode {tid: {frame: polygons}} silhouettes to nested dicts."""
            return {str(tid): {str(f): tr[f] for f in tr} for tid, tr in polys.items()}

        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({
            "detector": "sam3_seg",
            "sam3_score_th": self.cfg.sam3_score_th,
            "total_frames": int(total_frames),
            "frame_w": int(self.frame_w),
            "frame_h": int(self.frame_h),
            "person_tracks": enc_boxes(person_tracks),
            "car_tracks": enc_boxes(car_tracks),
            "person_masks": enc_masks(self.person_masks),
            "car_masks": enc_masks(self.car_masks),
            "person_polys": enc_polys(self.person_polys),
            "car_polys": enc_polys(self.car_polys),
        }))

    @staticmethod
    def _load_seg(cache: Path):
        """Load boxes + mask descriptors from a seg cache.

        Args:
            cache: path to the JSON seg cache written by `_save_seg`.
        Returns:
            (person_tracks, car_tracks, person_masks, car_masks, person_polys,
            car_polys, frame_w, frame_h, total_frames).
        """
        data = json.loads(cache.read_text())

        def dec_boxes(tracks):
            """Decode JSON box tracks back to {int tid: {int frame: np box}}."""
            out = defaultdict(dict)
            for tid, tr in tracks.items():
                for f, box in tr.items():
                    out[int(tid)][int(f)] = np.asarray(box, dtype=np.float32)
            return out

        def dec_masks(masks):
            """Decode JSON mask ellipses back to {int tid: {int frame: list}}."""
            out = defaultdict(dict)
            for tid, tr in masks.items():
                for f, ell in tr.items():
                    out[int(tid)][int(f)] = [float(x) for x in ell]
            return out

        def dec_polys(polys):
            """Decode JSON silhouette polygons back to integer pixel points."""
            out = defaultdict(dict)
            for tid, tr in polys.items():
                for f, ps in tr.items():
                    out[int(tid)][int(f)] = [
                        [[int(x), int(y)] for x, y in poly] for poly in ps
                    ]
            return out

        return (
            dec_boxes(data["person_tracks"]),
            dec_boxes(data["car_tracks"]),
            dec_masks(data.get("person_masks", {})),
            dec_masks(data.get("car_masks", {})),
            dec_polys(data.get("person_polys", {})),
            dec_polys(data.get("car_polys", {})),
            int(data.get("frame_w", 0)),
            int(data.get("frame_h", 0)),
            int(data["total_frames"]),
        )

    # ---------------------------------------------------------------- #
    # Segmentation major-axis gate
    # ---------------------------------------------------------------- #
    def _person_point(self, pid, f, ptrack):
        """Best available position for a person in frame `f` (mask centroid > box).

        Args:
            pid: person track id.
            f: frame index.
            ptrack: {frame: person_box} for that person.
        Returns:
            (x, y) pixel point: the mask centroid if available, else the box center.
        """
        pm = self.person_masks.get(pid, {}).get(f)
        if pm is not None:
            return pm[0], pm[1]
        return box_center(ptrack[f])

    def axis_gate(self, event, person_tracks, car_tracks):
        """Evaluate the major-axis condition for one enter/exit event.

        Args:
            event: the enter/exit Event to test.
            person_tracks: {person_id: {frame: box}}.
            car_tracks: {car_id: {frame: box}} (unused directly; kept for symmetry).
        Returns:
            (ok, frac, detail): whether the event passes, the fraction of window
            frames on the car's major axis, and a human-readable detail string.
        """
        cfg = self.cfg
        ptrack = person_tracks[event.person_id]
        frames = sorted(ptrack)
        window = (frames[: cfg.window]
                  if event.type.startswith("exit")
                  else frames[-cfg.window:])
        cmasks = self.car_masks.get(event.car_id, {})

        considered = 0
        passed = 0
        for f in window:
            car_ell = cmasks.get(f)
            if car_ell is None or f not in ptrack:
                continue
            considered += 1
            px, py = self._person_point(event.person_id, f, ptrack)
            if on_major_axis(px, py, car_ell, cfg.seg_major_reach, cfg.seg_minor_band):
                passed += 1

        if considered == 0:
            ok = not cfg.seg_strict
            return ok, 0.0, f"no car-mask frames in window (seg_strict={cfg.seg_strict})"
        frac = passed / considered
        ok = frac >= cfg.seg_axis_frac
        detail = (f"on_major_axis {passed}/{considered} = {frac:.2f} "
                  f">= seg_axis_frac {cfg.seg_axis_frac} "
                  f"(reach {cfg.seg_major_reach}, band {cfg.seg_minor_band})")
        return ok, frac, detail
