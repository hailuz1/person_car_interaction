# Interaction States — Gates & Conditions

Every person track is evaluated independently for each state below. A state is
emitted only when **all** of its gates pass. All three states are produced
**online** by `ProvisionalDriver`. The events
that schoosen (with their per-gate values) are recorded to `diagnostics.json`.

All time-based thresholds are configured in **seconds** and converted to a
per-clip **frame** count using the clip's FPS (`Config.resolve_for_fps`), so a
threshold means the same real-world duration regardless of frame rate.

## Event types summary

| Type | State | Notes |
| --- | --- | --- |
| `enter` | Enter | Got into the car (emitted after `min_absence` absence). |
| `exit` | Exit | Emerged from the car and moved off. |
| `loiter` | Loiter | Lingered near a parked car. |


## Shared quantities

| Symbol | Meaning |
| --- | --- |
| `c_start` | Mean closeness to the car over the **start** window (first `window` frames). |
| `c_end` | Mean closeness to the car over the **end** window (last `window` frames). |
| `motion` | Net radial motion toward (`+`) / away (`-`) from the car. |
| `start_overlap` / `end_overlap` | Mean fraction of the person box overlapping the car over the start / end window. |
| `car_frac` | Car box diagonal ÷ frame diagonal (how large / close the car appears). |
| `diou_start` / `diou_end` | Mean person↔car DIoU over the start / end window. |
| `v_car` | Mean ground-relative car speed (camera-motion compensated). |
| `near_frames` | Number of frames the person is "near" the car (closeness `> loiter_near_th`). |
| `net_drift`, `straightness` | Total person displacement and how straight the path is (loiter vs. walk-through). |

## Key thresholds (defaults)

| Config field | Default | Used by |
| --- | --- | --- |
| `min_track_sec` → `min_track_len` | 0.20 s | Track prerequisite |
| `edge_margin_sec` → `edge_margin` | 0.10 s | Exit |
| `window_sec` → `window` | 0.17 s | Start/end windows |
| `near_th` | 0.35 | Enter, Exit |
| `delta` | 0.18 | Enter, Exit |
| `move_min` | 0.35 | Enter, Exit subtype |
| `on_car_overlap_th` | 0.15 | Enter, Exit |
| `on_car_overlap_th_big` | 0.03 | Enter, Exit (near/big car) |
| `big_car_frame_ratio` | 0.45 | Enter, Exit (near/big car) |
| `on_car_near_th` | 0.60 | Enter, Exit (emerged/reached) |
| `exit_diou_drop` | 0.045 | Exit |
| `car_max_speed_per_sec` → `car_max_speed` | 0.25 diag/s | Enter, Exit, Loiter |
| `min_absence_sec` → `min_absence` | 0.20 s | Enter |
| `loiter_min_sec` → `loiter_min_frames` | 2.0 s | Loiter |
| `loiter_near_th` | 0.60 | Loiter |
| `loiter_max_net_drift` | 0.6 | Loiter |
| `loiter_straight_th` | 0.85 | Loiter |
| `seg_major_reach`, `seg_minor_band`, `seg_axis_frac` | 1.35, 1.4, 0.40 | Enter, Exit (axis gate) |
| `seg_strict` | False | Enter, Exit (axis gate) |

---

## Prerequisite (all states)

| Gate | Condition | Purpose |
| --- | --- | --- |
| `min_track_len` | `n_frames >= min_track_len` (exit/enter also require `>= window`) | Skip tracks too short to judge reliably. |

---

## Exit

The person is **at** the car at the start of the track and then **emerges** from
it and leaves. The event type stored is always `exit`.

| Gate | Condition | Purpose |
| --- | --- | --- |
| `not_at_clip_edge` | `f_first > edge_margin` | Ignore tracks that begin at the very start of the clip. |
| `at_car_at_start` | `c_start > near_th` | The person is genuinely at the car when first seen. |
| `emerged_from_car` | `start_overlap > overlap_th` **or** `c_start >= on_car_near_th` | The person physically overlapped the car (came out of it), relaxed for big/near cars. |
| `closeness_drop` | `c_start - c_end > delta` **or** `diou_start - diou_end > exit_diou_drop` | Overall proximity decreased over the track. |
| `car_stationary` | `v_car <= car_max_speed` | The car was parked, not driving. |
| `axis_gate` | Person on the car's major-axis region in `>= seg_axis_frac` of the start-window frames | Reject events off the car body (see below). |

**`overlap_th` selection (near/big car):**

| Case | Condition | `overlap_th` |
| --- | --- | --- |
| Near / big car | `car_frac >= big_car_frame_ratio` | `on_car_overlap_th_big` (0.03) |
| Normal car | `car_frac <  big_car_frame_ratio` | `on_car_overlap_th` (0.15) |

**Reported motion:** `motion` is stored as `-radial_motion` (departure ≥ 0), a
larger value means the person walked further from the car after emerging.

---

## Enter

The person ends the track **at** a car while vanishing well before the clip
ends (they got inside). Only considered when the track was **not** already
emitted as an exit and a nearby car exists. Enter is the only state with an
irreducible latency: it fires once the person has stayed absent for the whole
`min_absence` horizon (`f_idx - f_last >= min_absence`).

| Gate | Condition | Purpose |
| --- | --- | --- |
| `vanishes_mid_clip` | Person absent for the whole `min_absence` horizon before the clip ends | The person disappears mid-clip (into the car), not merely at the end. |
| `at_car_at_end` | `c_end > near_th` | The person is at the car when last seen. |
| `reached_car` | `end_overlap > overlap_th` **or** `c_end >= on_car_near_th` | Physically reached the car body (same big-car relaxation as exit). |
| `closeness_rise` | `c_end - c_start > delta` | Overall closeness increased over the track. |
| `moved_closer` | `motion > move_min` | A real net approach toward the car. |
| `car_stationary` | `v_car <= car_max_speed` | The car was parked, not driving. |
| `axis_gate` | Person on the car's major-axis region in `>= seg_axis_frac` of the end-window frames | Reject events off the car body (see below). |

---

## Loiter

The person stays **near** a stationary car for a long time, without clearly
entering or leaving. Evaluated independently of enter/exit. The candidate car is
the one with the most "near" frames (closeness `> loiter_near_th`), ties broken
by higher mean closeness. Emitted at the first near-frame with zero latency.

| Gate | Condition | Purpose |
| --- | --- | --- |
| `near_long_enough` | `near_frames >= loiter_min_frames` | Lingered near the car long enough to count. |
| `car_stationary` | `v_car <= car_max_speed` | The car was parked, not driving. |
| `not_walking_past` | `not (net_drift > loiter_max_net_drift and straightness > loiter_straight_th)` | Reject pass-throughs: a long, straight path is "walking by", not loitering. |

---

## Segmentation major-axis gate (Enter & Exit)

Final filter: **rejects events where the person is NOT at the car's side/door.**

- Person must be aligned with car's length (not in front/behind) in ≥40% of frames
- If no car mask exists: keep event.
- **Why?** Prevents false events from people standing in front/behind the car

---

