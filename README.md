# Real-time Person–Car Interaction

A clean, system that detects how person interact with
nearby vehicles — **entering**, **exiting**, or **loitering** — and emits each
interaction at its **earliest causal moment** as a low-latency **provisional**
signal. At frame `f` it uses only information from frames `≤ f`, making it
suitable for real-time / streaming deployments.

It is a faithful refactor of a proven offline SAM3-segmentation pipeline, so the
emitted events match the batch reference (see [Results](#results--verification)).

---

## Table of contents

- [What it detects](#what-it-detects)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Project layout](#project-layout)
- [File-by-file guide](#file-by-file-guide)
- [How it works](#how-it-works)
- [Outputs](#outputs)

---

## What it detects

Per person track, relative to a nearby (stationary) car:

| Type       | Meaning                                                        | Latency |
|------------|----------------------------------------------------------------|---------|
| **enter**  | Person approaches and gets into a car.                          | Emitted once absent past `min_absence` |
| **exit**   | Person emerges from a car at the start of its track, moves away.| Emitted at earliest evidence |
| **loiter** | Person stays near a car long enough without entering/leaving.   | Emitted at earliest evidence |

All three fire provisionally at the earliest causal moment their evidence exists.
`exit` and `loiter` fire immediately; `enter` is only
knowable once the person has been gone long enough that no future frame can change
the verdict, so it is emitted at that irreducible latency
(`min_absence` frames of absence).

## Requirements

- Python 3.12.
- `opencv-python`, `numpy`.
- **Optional (only for uncached clips):** SAM3 source at `../src/sam3-main` and the
  checkpoint at `sam3_checkpoint` (see [Configuration](#configuration)) plus a GPU
  and `torch`.

Cached clips (clip1–clip8 ship with a `sam3seg` cache) run CPU-only.

The core runtime dependencies are pinned in [`requirements.txt`](requirements.txt).
Set up an environment with either conda or plain `pip`:

```bash
# Option A — conda (recommended, uses environment.yml)
conda env create -f environment.yml
conda activate person_car

# Option B — conda env + pip
conda create -n person_car python=3.12 -y
conda activate person_car
pip install -r requirements.txt

# Option C — any Python 3.12 environment
pip install -r requirements.txt
```

## Quick start

```bash
# from the realtime_interaction_noreid/ folder
python run.py --clip clip1                 # live window ('q' quits) + mp4 + json
python run.py --clip clip1 --no-display    # headless (still writes json + mp4)
python run.py --clip clip1 --no-save       # window only, no mp4
python run.py --all                        # every clip in data/given
python run.py --all --data-dir ../data/test  # run the test clips instead
python run.py --clip clip1 --no-json       # skip event.json / diagnostics.json
python run.py --clip clip1 --max-frames 200
```

CLI flags (see [run.py](run.py)):

| Flag            | Default  | Description                                             |
|-----------------|----------|--------------------------------------------------------|
| `--clip`        | `clip1`  | Clip stem in `data/given` (e.g. `clip1`).              |
| `--data-dir`    | none     | Override the input video folder (e.g. `../data/test`). |
| `--all`         | off      | Process every video in `data_dir`.                     |
| `--no-display`  | off      | Do not open a cv2 window (headless).                   |
| `--no-save`     | off      | Do not write the annotated mp4.                        |
| `--no-json`     | off      | Do not write `event.json` / `diagnostics.json`.        |
| `--max-frames`  | none     | Stop after N frames (debugging).                       |

## Project layout

```
realtime_interaction_noreid/
├── run.py                 # CLI entry point
├── README.md              # this file
├── requirements.txt       # core runtime dependencies (pip)
├── environment.yml        # conda environment spec (Python 3.12 + deps)
├── config/
│   ├── __init__.py        # exports Config
│   └── settings.py        # Config dataclass (all thresholds; seconds -> frames)
├── src/
│   ├── __init__.py        # package marker
│   ├── geometry.py        # pure box/mask geometry helpers (stateless)
│   ├── events.py          # Event dataclass (the output record)
│   ├── detector.py        # gates + SAM3 tracking + seg cache + axis gate
│   ├── camera.py          # CamMotion (frame-to-frame ego-motion)
│   ├── driver.py          # ProvisionalDriver (provisional-only streaming)
│   ├── visualizer.py      # Visualizer (boxes + banner + HUD)
│   └── runner.py          # causal frame loop + JSON output
└── runs/                  # per-clip outputs (event.json, diagnostics.json, mp4)
```

Shared workspace resources used from the parent folder:

```
../data/given/<clip>.mp4               # input videos
../track_cache/<clip>__sam3seg_*.json  # cached SAM3-seg tracks (cache-first)
../src/sam3-main/                       # SAM3 source (only if a cache is missing)
```

## File-by-file guide

### `run.py`
CLI entry point. Parses arguments, builds a default `Config`, and calls
`src.runner.run` for one clip or every clip in `data_dir`.

### `config/settings.py`
A single `Config` dataclass holding **every** tunable value: data/IO paths, the
enter/exit/loiter thresholds, robustness gates (min absence, car speed),
loiter parameters, camera-motion parameters, SAM3 settings, and the
segmentation major-axis gate. Time-based thresholds are stored in **seconds** and
converted to a per-clip **frame count** by `resolve_for_fps(fps)`, so a threshold
means the same real-world duration at any frame rate. Paths default to the shared
workspace (`../data/given`, `../track_cache`, `../src/sam3-main`) with outputs in
`runs/`.

### `src/geometry.py`
Stateless, pure functions shared by the detector, driver and visualizer — copied
verbatim from the proven pipeline for byte-identical numerics: `box_center`,
`box_diag`, `center_distance_in_diagonals`, `closeness`, `diou`, `mask_ellipse`,
`axis_offsets`, `on_major_axis`, `mask_polygons`.

### `src/events.py`
The `Event` dataclass — the single output record: `person_id`, `car_id`, `type`
(`enter`/`exit`/`loiter`), `frame`, `c_start`, `c_end`, `motion`, `duration`
(loiter frame count), and `metrics` — a dict of every condition value the gate
evaluated to accept the event (each as `value`/`th`/`pass` where applicable),
serialized into `event.json` and `diagnostics.json` for inspection.

### `src/detector.py`
`Detector` owns everything **per finished track** or **per stream**:
- proximity / motion / continuity primitives (`closeness`, `diou`, `car_speed`,
  `person_drift`, …),
- SAM3 tracking with mask capture plus a JSON **seg-cache reader/writer**
  (`track_video`, `_track_sam3_seg`, `_save_seg`, `_load_seg`),
- the segmentation **major-axis gate** (`axis_gate`) that drops enter/exit events
  off the car's body — used by the streaming driver's provisional gates.

### `src/camera.py`
`CamMotion.step(gray, cur_boxes)` fits the background ego-motion transform
(2×3 affine, frame `f-1 → f`) from the previous and current frames only, masking
out dynamic objects using the previous frame's boxes. This lets the driver
compensate camera motion when judging whether a car is truly stationary.

### `src/driver.py`
`ProvisionalDriver` runs the detector online via `push_frame(...)`, returning the
events newly emitted this frame. **All three states are produced by provisional
gates**, each firing at its earliest causal moment:
- `_prov_loiter` — **loiter** when the near-count first crosses `loiter_min_frames`
  (zero latency, person still on screen),
- `_prov_exit` — **exit** when a trailing window first shows the departure
  (closeness / DIoU drop),
- `_prov_enter` — **enter** once the person has been absent for the whole
  `min_absence` horizon (its irreducible minimum latency); enter and exit are
  mutually exclusive.

`_check_loiter_exit` grows the per-car near-history each frame and runs the three
`_prov_*` gates. `_finalize_track` closes a track once it has been gone
`finalize_delay` frames — it runs **no** re-classification, it only marks the
track done and appends a compact per-track **diagnostics** record built from the
events that already fired. `flush()` finalizes the tail, sorts events, and
publishes `detector.last_diagnostics`. Provisional signals are never re-emitted
or retracted.

### `src/visualizer.py`
`Visualizer.draw(...)` renders one frame at a time: person/car **bounding boxes
with ids**, a persistent banner of the signals emitted so far, and a small HUD.
Kept separate so rendering is fully decoupled from detection.

### `src/runner.py`
`run(stem, cfg, display, save, max_frames, write_json)` ties it together:
resolves fps, runs/loads SAM3 tracks, snapshots the mask descriptors and reveals
them **one frame at a time** (mirroring a live model), then drives
`CamMotion` + `ProvisionalDriver` + `Visualizer` through the
frame loop. Writes `event.json` and `diagnostics.json` and, unless disabled, the
annotated `<clip>.mp4`.

## How it works

```
              ┌──────────────────────── per frame f (only data ≤ f) ────────────────────────┐
 video ──▶ SAM3 tracks (cache-first) ──▶ reveal boxes+masks for frame f
                                              │
                     ┌────────────────────────┴─────────────────────────┐
                     ▼                                                    ▼
             CamMotion                                        ProvisionalDriver
            (f-1 → f affine)                          push_frame(persons, cars, cam)
                     │                                                    │
                     └──────────────── cam motion ───────────────────────┤
                                                                          ▼
                                                        loiter/exit/enter  ──▶ Event
                                                                          │
                                                                          ▼
                                                                    Visualizer.draw
                                                                          │
                                             event.json · diagnostics.json · mp4
```

1. **Tracking (causal by construction).** SAM3 forward propagation is causal;
   the runner loads it from cache (or runs it once) and reveals each frame's boxes
   and mask descriptors only when that frame arrives.
2. **Camera motion.** Fitted from the previous + current frame to tell real car
   motion from ego-motion.
3. **Provisional driver.** Emits loiter, exit, and enter each at its earliest
   causal moment (loiter with zero latency, exit at the first departure window,
   enter once the person has stayed absent past the `min_absence` horizon).
4. **Segmentation axis gate.** Enter/exit events off the car's major axis are
   dropped using the SAM3 mask ellipse.

## Outputs

Per clip, under `runs/<clip>/`:

- **`event.json`** — every emitted interaction. Each record:
  `person_id`, `car_id`, `type`, `frame`, `c_start`, `c_end`, `motion`, `duration`,
  and `metrics` (the condition values — `value`/`th`/`pass` — that accepted it).
- **`diagnostics.json`** — per-track record: the track's `result` (the states it
  emitted, or `none`) and the full `events` list (with their `metrics`). Built
  directly from the emitted events — no re-classification is run at finalize.
- **`<clip>.mp4`** — annotated video (omitted with `--no-save`).


