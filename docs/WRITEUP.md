# Person-Car Interaction Detection — Write-up

## 1. Problem & Goal
I detect three interaction types per person track relative to a nearby car:

| Type       | Meaning                                                  |
|------------|-----------------------------------------------------------|
| **enter**  | Person approaches and gets into the car.                  |
| **exit**   | Person emerges from the car and moves away.               |
| **loiter** | Person stays near a stationary car without entering/leaving. |

## 2. Approach

The pipeline runs one clip at a time and produces `event.json` (the interaction
list) plus optional annotated MP4 and `diagnostics.json`.

**a. Detection & tracking (SAM3).** Each clip is segmented and tracked with SAM3. This yields per-track boxes, mask ellipses, and polygons per frame. Results are cached to
`track_cache/<clip>__sam3seg_s<θ>.json`, so re-runs are CPU-only and fully
reproducible.

**b. Geometry primitives.** All distances are normalised by the car's box diagonal,
making them scale-invariant to how large or near the car appears. The core
**closeness** score combines person-over-car overlap with a distance term:
`closeness = max(overlap, dist_score) ∈ [0,1]`. A **DIoU** metric (intersection
over *union*, plus a centre-distance penalty) captures a person separating from a
car over time.

**c. Camera ego-motion.** Frame-to-frame camera motion is estimated by optical flow
(`goodFeaturesToTrack` → `calcOpticalFlowPyrLK` → `estimateAffinePartial2D`).
This lets me compute **ground-relative car speed**: a parked car reads ≈ 0 even
while the camera pans, which is essential for distinguishing a stationary vehicle
from a moving one.

**d. Interaction gates.** Using start/end windows of each person track:
- **exit** fires when the person starts on/near the car (high overlap or closeness)
  and closeness/DIoU drops as they move away.
- **enter** fires when the person approaches, reaches the car, then is absent long
  enough that no future frame can change the verdict.
- **loiter** fires when the person stays near a stationary car long enough, gated
  against walk-throughs by **net drift** and **path straightness** (a person walking
  in a straight line past the car is rejected).

**e. Segmentation major-axis gate.** For exits/enters, I check whether the person
sits on the car's *long axis* (the doors/sides) using the mask's equivalent
ellipse. A configurable fraction of window frames must pass, which suppresses
false positives where a person is near the car's front/rear but not a door.

**f. Causal / provisional emission.** At frame `f` the system uses only frames
`≤ f`. Each interaction is emitted at its **earliest causal moment** and never
retracted, making the method suitable for real-time streaming. `exit` and `loiter`
fire immediately on evidence. `enter` fires at its irreducible latency (once the
person has been absent for `min_absence` frames).

## 3. Output Format

`event.json` is a list of records. Each record is self-describing and includes the
decision metrics that produced it, for auditability:

```json
{
  "person_id": 4, "car_id": 18, "type": "loiter", "frame": 222,
  "c_start": 0.642, "c_end": 0.854, "motion": 0.0, "duration": 60,
  "metrics": { "near_frames": {"value": 60, "th": 60, "pass": true},
               "car_speed": {"value": 0.0011, "th": 0.0083, "pass": true}, ... }
}
```

`frame` is the causal emission frame. `metrics` records each gate's value,
threshold, and pass/fail so results can be inspected without re-running.

**Consolidated summary (`all_events.json`).** A single top-level file merges every
clip's interactions into one human-readable list, produced by
`tools/build_all_events.py`. Each record drops the internal metrics and instead
adds the clip id, a frame *range* + time span in seconds, and a short
plain-language description of the person and vehicle (dominant colour, relative
size, and position) sampled from the actual pixels of a representative frame:

```json
{
  "clip_id": "clip1", "type": "loiter",
  "frame_range": [182, 318], "time_span_s": [6.07, 10.61],
  "person_id": 3, "person_description": "a small person in mostly grey clothing, centre of frame",
  "car_id": 22, "car_description": "a small light-grey vehicle, centre of frame"
}
```

`frame_range`/`time_span_s` give the span both in frames and in seconds (using each
clip's real fps). The descriptions are derived from the cropped detection boxes, so
they are coarse but grounded in the video rather than hand-written.

## 4. Assumptions & Ambiguity Decisions

- **Interaction = enter / exit / loiter.** I scoped it to physical get-in / get-out plus lingering-at-vehicle, and
  explicitly excluded pass-by. *Alternative considered:* treating any proximity as
  interaction - rejected as too permissive (every pedestrian near a car would fire).
- **Loiter is an interaction.** A person deliberately staying at a stationary car
  (e.g. waiting by a door, loading) is plausibly interacting. *Alternative:* only
  enter/exit — rejected because it discards meaningful lingering behaviour. Loiter
  is guarded by drift/straightness gates so genuine walk-throughs don't fire.
- **Only stationary vehicles.** Camera-compensated car speed must be near zero.
  A person "entering" a moving car is out of scope and unreliable to judge.
- **Time-based thresholds.** All temporal thresholds are stored in seconds and
  converted per-clip via fps, so a threshold means the same real duration at any
  frame rate.
- **No inter-clip identity.** Person/car IDs are per-clip only, as permitted.

## 5. Limitations

- **Tracking dependence.** Quality is bounded by SAM3 segmentation/tracking. ID
  switches, false detections or missed detections propagate into the gates.
- **Occlusion / crowding.** Overlapping people near one car can confuse the
  best-car assignment and the major-axis gate.
- **Heuristic thresholds.** Gates are tuned, not learned. Edge cases (partial
  entries) may be missed.
- **Enter latency.** By construction, `enter` cannot be confirmed until the person
  has been absent long enough.
- **Low resolution with large cars.** When a car fills much of a low-resolution
  frame, person-over-car overlap saturates and the car
  diagonal used to normalise distances is large, so proximity scores stay high
  even for people who are merely passing in front of the car. This makes
  pass-by vs. genuine interaction harder to separate.
- **Partial entering.** The system only recognises a *completed* enter/exit — the
  person must fully disappear into (or emerge from) the car. Partial actions, such
  as leaning in to grab something, opening a door without getting in, or a
  half-in/half-out pause, do not cross the absence thresholds and are
  often reported as loiter, even though they are genuine interactions.
- **Enter/exit clipped by the clip boundary.** An `exit` must be seen starting on
  the car, and an `enter` needs the person to vanish before the clip ends. When the
  action straddles the first or last frames, this can misfire: for example, a
  person who simply appears near a car in the opening frames may be misclassified
  as an `exit`, and a person still approaching when the clip ends can be classified as `enter`, because of this we need the start/end window.

## 6. Next Steps & Alternative Ways

- **Quantitative evaluation.** Hand-label a small set of clips and report
  precision/recall per interaction type, in addition, the parameters can be tuned.
- **Depth.** Add a monocular depth prior to recover the real-world size and
  distance of the car and person. Make proximity thresholds metric (meters) instead of pixel/diagonal based.
- **Learned classifier.** Use labled datased to train temporal model
  (e.g. human pose estimation + proximity features) to reduce threshold sensitivity.
- **VLLM.** Use a vision-language model in two ways: (a) as a *complement* to
  verify or label ambiguous clips from natural-language prompts (e.g. "is the
  person getting into the car?"), bootstrapping labels and sanity-checking the
  geometric gates. (b) as a full *replacement*, prompting the model directly on
  the clip to output the interaction list end-to-end, trading the interpretable
  geometric pipeline for broader semantic understanding at higher compute cost.
- **Moving-vehicle handling.** Extend beyond stationary cars with proper relative
  motion modelling.
- **Robustness testing.** Stress-test on low-light, crowded, and fast-camera clips
  to characterise failure modes.
