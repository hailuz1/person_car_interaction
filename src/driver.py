"""Causal, provisional-only streaming driver.

Runs the interaction detector online, one frame at a time, and emits each state
at its earliest causal moment as a **provisional** signal -- no confirm/retract
lifecycle:

  * **loiter** fires the frame the near-count first crosses ``loiter_min_frames``
    (while the person is still on screen -- zero latency).
  * **exit**   fires the frame a trailing window first shows the departure
    (closeness / DIoU drop).
  * **enter**  is only knowable once the person has been gone past the
    ``min_absence`` horizon so no future frame can change the verdict; it fires
    at that earliest causal moment (its irreducible minimum latency).

All three states are produced by the provisional gates. When a track has been
gone long enough it is *finalized* -- no re-classification is run, the track is
just closed and a compact per-track ``diagnostics`` record (built from the events
it emitted) is appended. Provisional signals are never re-emitted as "confirmed"
nor retracted: the low-latency hypothesis is kept (the accepted trade-off of a
pure real-time system).

The driver is fed camera motion (per-frame f-1 -> f transform) incrementally
through ``push_frame``.
"""

from collections import defaultdict
from dataclasses import asdict

from .detector import Detector
from .events import Event
from .geometry import box_center


class ProvisionalDriver:
    """Online provisional driver"""

    _ABSENT = object()  # sentinel: no camera transform provided for this frame

    def __init__(self, detector: Detector):
        """Initialize the online driver state.

        Args:
            detector: the Detector providing the primitives and per-track logic.
        Returns:
            None.
        """
        self.det = detector
        self.cfg = detector.cfg
        self.person_tracks = defaultdict(dict)
        self.car_tracks = defaultdict(dict)
        self.starts = {}   # pid -> (first_frame, center, box)
        self.ends = {}     # pid -> (last_frame, center, box), updated live
        self.last_seen = {}
        self.finalized = set()
        self.events = []          # every emitted signal (provisional loiter/exit + enter)
        self.diagnostics = []
        self.total_frames = 0
        self.cam_motion = {}      # frame -> 2x3 affine (f-1 -> f), fed via push_frame
        # A track is final once the person has been gone long enough that an
        # enter's "vanishes_mid_clip" delay can no longer change its verdict.
        # how many frames the driver must wait after a person was last seen before its track can be finalized (enter emitted)
        self.finalize_delay = self.cfg.min_absence + 1

        self.near_frames = defaultdict(list)  # (pid, cid) -> near-frame indices so far
        self.exit_ctx = {}                    # pid -> fixed start-window exit context (or False)
        self.prov_emitted = defaultdict(dict)  # pid -> {type: Event} already raised

    # ---------------------------------------------------------------- #
    # Frame intake
    # ---------------------------------------------------------------- #
    def push_frame(self, f_idx, persons, cars, cam_transform=_ABSENT):
        """Feed one frame of detections; return the events newly emitted this frame.

        Args:
            f_idx: the current frame index.
            persons: iterable of (person_id, box) for this frame.
            cars: iterable of (car_id, box) for this frame.
            cam_transform: optional 2x3 affine (f-1 -> f); omitted = none stored.
        Returns:
            list[Event] emitted this frame (provisional loiter/exit + any enter
            finalized now).
        """
        self.total_frames = max(self.total_frames, f_idx + 1)
        if cam_transform is not self._ABSENT:
            self.cam_motion[f_idx] = cam_transform
        for cid, box in cars:
            self.car_tracks[cid][f_idx] = box
        for pid, box in persons:
            if pid not in self.starts:
                self.starts[pid] = (f_idx, box_center(box), box)
            self.person_tracks[pid][f_idx] = box
            self.ends[pid] = (f_idx, box_center(box), box)
            self.last_seen[pid] = f_idx

        emitted = self._check_state(f_idx)
        for pid in list(self.person_tracks):
            if pid in self.finalized:
                continue
            if f_idx - self.last_seen[pid] >= self.finalize_delay:
                self._finalize_track(pid)
        return emitted

    # ---------------------------------------------------------------- #
    # loiter / exit scan
    # ---------------------------------------------------------------- #
    def _check_state(self, f_idx):
        """Grow near-history and emit any new provisional loiter/exit signals.

        Args:
            f_idx: the current frame index.
        Returns:
            list[Event] of loiter/exit signals first raised this frame.
        """
        cfg = self.cfg
        out = []
        # Grow each active person's per-car "near" history with this frame only.
        for pid, ptrack in self.person_tracks.items():
            if pid in self.finalized:
                continue
            pbox = ptrack.get(f_idx)
            if pbox is None:
                continue
            for cid, ctrack in self.car_tracks.items():
                cbox = ctrack.get(f_idx)
                if cbox is None:
                    continue
                # Check if the person is "near" this car this frame, and if so, record it.
                if self.det.closeness(pbox, cbox) > cfg.loiter_near_th:
                    self.near_frames[(pid, cid)].append(f_idx)

        for pid in list(self.person_tracks):
            if pid in self.finalized:
                continue
            if "loiter" not in self.prov_emitted[pid]:
                e = self._prov_loiter(pid)
                if e is not None:
                    self.prov_emitted[pid]["loiter"] = e
                    self.events.append(e)
                    out.append(e)
            if "exit" not in self.prov_emitted[pid]:
                e = self._prov_exit(pid, f_idx)
                if e is not None:
                    self.prov_emitted[pid]["exit"] = e
                    self.events.append(e)
                    out.append(e)
            if "enter" not in self.prov_emitted[pid]:
                e = self._prov_enter(pid, f_idx)
                if e is not None:
                    self.prov_emitted[pid]["enter"] = e
                    self.events.append(e)
                    out.append(e)
        return out

    def _prov_loiter(self, pid):
        """ loiter once the near count first crosses the threshold.

        Args:
            pid: the person track id.
        Returns:
            An Event of type "loiter", or None if the loiter gate is not (yet) met.
        """
        cfg = self.cfg
        ptrack = self.person_tracks[pid]
        if len(ptrack) < cfg.min_track_len:
            return None
        best_cid, best_nf, best_key = None, [], None
        # q=person id, cid=car id, nf=near frames
        for (q, cid), nf in self.near_frames.items():
            if q != pid or not nf:
                continue
            cframes = self.car_tracks[cid]
            # the average closeness between a person and a car over the frames where they were near each other
            mean_clos = sum(self.det.closeness(ptrack[f], cframes[f])
                            for f in nf) / len(nf)
            key = (len(nf), mean_clos)
            # keep the car with the longest near-history, breaking ties by mean closeness
            if best_key is None or key > best_key:
                best_cid, best_nf, best_key = cid, nf, key
        if best_cid is None or len(best_nf) < cfg.loiter_min_frames:
            return None
        cframes = self.car_tracks[best_cid]
        v_car = self.det.car_speed(cframes, best_nf, self.cam_motion)
        if v_car > cfg.car_max_speed:
            return None
        p_net, p_straight = self.det.person_drift(ptrack, cframes, best_nf)
        if p_net > cfg.loiter_max_net_drift and p_straight > cfg.loiter_straight_th:
            return None
        f0, f1 = best_nf[0], best_nf[-1]
        return Event(
            person_id=int(pid), car_id=int(best_cid), type="loiter", frame=int(f0),
            frame_start=int(f0), frame_end=int(f1),
            c_start=round(float(self.det.closeness(ptrack[f0], cframes[f0])), 3),
            c_end=round(float(self.det.closeness(ptrack[f1], cframes[f1])), 3),
            motion=0.0, duration=int(len(best_nf)),
            metrics={
                "near_frames": {"value": int(len(best_nf)),
                                "th": int(cfg.loiter_min_frames), "pass": True},
                "mean_closeness": round(float(best_key[1]), 4),
                "car_speed": {"value": round(float(v_car), 4),
                              "th": float(cfg.car_max_speed),
                              "pass": bool(v_car <= cfg.car_max_speed)},
                "person_net_drift": {"value": round(float(p_net), 4),
                                     "th": float(cfg.loiter_max_net_drift)},
                "person_straightness": {"value": round(float(p_straight), 4),
                                        "th": float(cfg.loiter_straight_th)},
                "not_walking_past": {
                    "pass": bool(not (p_net > cfg.loiter_max_net_drift
                                      and p_straight > cfg.loiter_straight_th)),
                    "detail": "rejected only if net_drift AND straightness both "
                              "exceed their thresholds",
                },
            },
        )

    def _exit_base(self, pid, frames):
        """Fixed start-window half of the exit gate, computed once per track.

        Args:
            pid: the person track id.
            frames: the sorted list of this person's frame indices.
        Returns:
            dict {scid, c_start, diou_start} when the start-window emergence gate
            passes, or False when this track can never be an exit.
        """
        cfg = self.cfg
        ptrack = self.person_tracks[pid]
        start_win = frames[: cfg.window]
        scid, c_start = self.det.best_car(ptrack, start_win, self.car_tracks)
        if scid is None or c_start <= cfg.near_th:
            return False
        f_first = frames[0]
        if f_first <= cfg.edge_margin:
            return False
        cframes = self.car_tracks[scid]
        frame_diag = self.det.estimate_frame_diag()
        start_overlap = self.det.window_overlap(ptrack, start_win, cframes)
        car_frac = self.det.mean_car_diag(start_win, cframes) / frame_diag
        overlap_th = (cfg.on_car_overlap_th_big
                      if car_frac >= cfg.big_car_frame_ratio
                      else cfg.on_car_overlap_th)
        emerged = start_overlap > overlap_th or c_start >= cfg.on_car_near_th
        if not emerged:
            return False
        v_car = self.det.car_speed(cframes, start_win, self.cam_motion)
        if v_car > cfg.car_max_speed:
            return False
        return {
            "scid": scid,
            "c_start": c_start,
            "diou_start": self.det.window_diou(ptrack, start_win, cframes),
        }

    def _prov_exit(self, pid, f_idx):
        """exit the frame a trailing window first shows the departure.

        Args:
            pid: the person track id to test.
            f_idx: the current frame index.
        Returns:
            An Event of type "exit" (already passed the axis gate), or None if no
            departure is visible yet.
        """
        cfg = self.cfg
        ptrack = self.person_tracks[pid]
        frames = sorted(ptrack)
        if len(frames) < max(cfg.min_track_len, cfg.window):
            return None
        ctx = self.exit_ctx.get(pid)
        if ctx is None:
            ctx = self._exit_base(pid, frames)
            self.exit_ctx[pid] = ctx
        if not ctx:
            return None
        cframes = self.car_tracks[ctx["scid"]]
        end_win = frames[-cfg.window:]
        c_end = self.det.window_closeness(ptrack, end_win, cframes)
        diou_end = self.det.window_diou(ptrack, end_win, cframes)
        departed = ((ctx["c_start"] - c_end) > cfg.delta
                    or (ctx["diou_start"] - diou_end) > cfg.exit_diou_drop)
        if not departed:
            return None
        ev = Event(
            person_id=int(pid), car_id=int(ctx["scid"]), type="exit",
            frame=int(frames[0]), frame_start=int(frames[0]), frame_end=int(frames[-1]),
            c_start=round(float(ctx["c_start"]), 3),
            c_end=round(float(c_end), 3),
            motion=round(float(-self.det.radial_motion(ptrack, cframes)), 3),
            metrics={
                "c_start": round(float(ctx["c_start"]), 4),
                "c_end": round(float(c_end), 4),
                "diou_start": round(float(ctx["diou_start"]), 4),
                "diou_end": round(float(diou_end), 4),
                "closeness_drop": {"value": round(float(ctx["c_start"] - c_end), 4),
                                   "th": float(cfg.delta)},
                "diou_drop": {"value": round(float(ctx["diou_start"] - diou_end), 4),
                              "th": float(cfg.exit_diou_drop)},
                "departed": bool(departed),
            },
        )
        # Apply the seg major-axis gate now: its window (the fixed start window)
        # has its car masks already revealed, so its verdict here matches finalize.
        ok, _frac, _detail = self.det.axis_gate(ev, self.person_tracks, self.car_tracks)
        if not ok:
            return None
        return ev

    def _prov_enter(self, pid, f_idx):
        """enter once the person has been absent long enough to prove they got in.

        Enter has an irreducible latency: while the person is still on screen a
        "standing next to the car" track is indistinguishable from one that just
        got in. It is only knowable once the person has been gone past the
        ``min_absence`` horizon, so this fires at that earliest causal moment.

        Args:
            pid: the person track id to test.
            f_idx: the current frame index.
        Returns:
            An Event of type "enter" (already passed the axis gate), or None if
            the enter verdict is not (yet) provable.
        """
        cfg = self.cfg
        # Enter and exit are mutually exclusive; a track that already exited a car
        # cannot also be an enter.
        if "exit" in self.prov_emitted[pid]:
            return None
        ptrack = self.person_tracks[pid]
        frames = sorted(ptrack)
        if len(frames) < max(cfg.min_track_len, cfg.window):
            return None
        f_last = frames[-1]
        # Causal gate: the person must have really vanished (absent for the whole
        # min_absence horizon) before enter can be proven.
        if f_idx - f_last < cfg.min_absence:
            return None
        end_win = frames[-cfg.window:]
        start_win = frames[: cfg.window]
        ecid, c_end = self.det.best_car(ptrack, end_win, self.car_tracks)
        if ecid is None:
            return None
        cframes = self.car_tracks[ecid]
        c_start = self.det.window_closeness(ptrack, start_win, cframes)
        motion = self.det.radial_motion(ptrack, cframes)  # >0 => approach
        v_car = self.det.car_speed(cframes, end_win, self.cam_motion)
        end_overlap = self.det.window_overlap(ptrack, end_win, cframes)
        frame_diag = self.det.estimate_frame_diag()
        car_frac = self.det.mean_car_diag(end_win, cframes) / frame_diag
        overlap_th = (cfg.on_car_overlap_th_big
                      if car_frac >= cfg.big_car_frame_ratio
                      else cfg.on_car_overlap_th)
        reached = end_overlap > overlap_th or c_end >= cfg.on_car_near_th
        entered = (c_end > cfg.near_th
                   and reached
                   and (c_end - c_start) > cfg.delta
                   and motion > cfg.move_min
                   and v_car <= cfg.car_max_speed)
        if not entered:
            return None
        ev = Event(
            person_id=int(pid), car_id=int(ecid), type="enter",
            frame=int(f_last), frame_start=int(frames[0]), frame_end=int(f_last),
            c_start=round(float(c_start), 3),
            c_end=round(float(c_end), 3), motion=round(float(motion), 3),
            metrics={
                "c_start": round(float(c_start), 4),
                "c_end": {"value": round(float(c_end), 4),
                          "th": float(cfg.near_th), "pass": bool(c_end > cfg.near_th)},
                "closeness_rise": {"value": round(float(c_end - c_start), 4),
                                   "th": float(cfg.delta),
                                   "pass": bool((c_end - c_start) > cfg.delta)},
                "motion": {"value": round(float(motion), 4),
                           "th": float(cfg.move_min), "pass": bool(motion > cfg.move_min)},
                "car_speed": {"value": round(float(v_car), 4),
                              "th": float(cfg.car_max_speed),
                              "pass": bool(v_car <= cfg.car_max_speed)},
                "end_overlap": round(float(end_overlap), 4),
                "car_frac": round(float(car_frac), 4),
                "overlap_th": float(overlap_th),
                "reached": {"value": bool(reached),
                            "detail": f"end_overlap {end_overlap:.4f} > "
                                      f"{overlap_th} or c_end {c_end:.4f} >= "
                                      f"{cfg.on_car_near_th}"},
            },
        )
        ok, _frac, _detail = self.det.axis_gate(ev, self.person_tracks, self.car_tracks)
        if not ok:
            return None
        return ev

    # ---------------------------------------------------------------- #
    # finalize (records the per-track diagnostics; no re-classification)
    # ---------------------------------------------------------------- #
    def _finalize_track(self, pid):
        """Close a finished track and record its diagnostics entry.

        All enter/exit/loiter events are already emitted by the provisional
        gates, so finalizing runs no classification: it only marks the track
        done and appends a compact per-track diagnostics record built from the
        events that fired.

        Args:
            pid: the person track id to finalize.
        Returns:
            None.
        """
        self.finalized.add(pid)
        emitted = self.prov_emitted[pid]
        loiter = emitted.get("loiter")
        if loiter is not None:
            near = self.near_frames.get((pid, loiter.car_id))
            if near and near[-1] > loiter.frame_end:
                loiter.frame_end = int(near[-1])
        fired = [emitted[t] for t in ("loiter", "exit", "enter") if t in emitted]
        self.diagnostics.append({
            "person_id": int(pid),
            "n_frames": len(self.person_tracks[pid]),
            "result": [e.type for e in fired] or "none",
            "events": [asdict(e) for e in fired],
        })

    def flush(self):
        """Finalize every still-open track (end of stream); return all events.

        Args:
            None.
        Returns:
            list[Event] of every emitted signal, sorted by (frame, person_id).
        """
        for pid in list(self.person_tracks):
            if pid not in self.finalized:
                self._finalize_track(pid)
        self.events.sort(key=lambda e: (e.frame, e.person_id))
        self.det.last_diagnostics = self.diagnostics
        return self.events
