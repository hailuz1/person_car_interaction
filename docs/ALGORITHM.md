# Person - Car Interaction

This document specifies **every** step, state, condition, and formula in the person–car interaction system.

- Companion overview: [README.md](README.md)
- Gate cheat-sheet: [../STATES_AND_GATES.md](../STATES_AND_GATES.md)

## Table of Contents

1. [Notation](#1-notation)
2. [End-to-end pipeline](#2-end-to-end-pipeline)
3. [Geometry primitives](#3-geometry-primitives)
4. [Mask (segmentation) primitives](#4-mask-segmentation-primitives)
5. [Motion & continuity primitives](#5-motion--continuity-primitives)
6. [Camera ego-motion](#6-camera-ego-motion)
7. [SAM3 tracking & seg cache](#7-sam3-tracking--seg-cache)
8. [Interaction states & gate conditions](#8-interaction-states--gate-conditions)
9. [Segmentation major-axis gate](#9-segmentation-major-axis-gate)
10. [Provisional emission](#10-provisional-emission)
11. [Configuration & fps resolution](#11-configuration--fps-resolution)
12. [Outputs](#12-outputs)

---

## 1. Notation

A **box** is axis-aligned $b=(x_1,y_1,x_2,y_2)$ (optionally with a confidence as a
5th entry). A **track** is a map $\{f \mapsto b_f\}$ from frame index to box.
Person tracks are indexed by $p$, car tracks by $c$.

| Symbol | Definition |
| --- | --- |
| $\mathrm{ctr}(b)$ | Box centre $\big(\tfrac{x_1+x_2}{2},\ \tfrac{y_1+y_2}{2}\big)$. |
| $\mathrm{diag}(b)$ | Box diagonal $\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}+\varepsilon$, $\varepsilon=10^{-6}$. |
| $\mathrm{area}(b)$ | $(x_2-x_1)(y_2-y_1)$. |
| $W$ | Start/end window length in frames (`window`). |
| $\mathrm{start\ win}$ | First $W$ frames of a track, and $\mathrm{end\ win}$ = last $W$ frames. |
| $M_f$ | Camera affine transform mapping frame $f{-}1 \to f$ (2×3). |
| $\Theta$ | Config thresholds (see [§11](#11-configuration--fps-resolution)). |

All distances used inside the interaction gates are **normalised by the car's
box diagonal**, so they are scale-invariant to how large/near the car appears.

---

## 2. End-to-end pipeline

Per clip, [src/runner.py](src/runner.py) `run(...)`:

1. **Resolve fps.** Read the clip fps and convert all seconds-based thresholds to
   frame counts via `Config.resolve_for_fps` ([§11](#11-configuration--fps-resolution)).
2. **Track (cache-first).** `Detector.track_video` loads
   `track_cache/<clip>__sam3seg_s<θ>.json` if present, else runs SAM3 forward
   propagation once (person prompt, then car prompt) and writes the cache
   ([§7](#7-sam3-tracking--seg-cache)). Yields `person_tracks`, `car_tracks`,
   `total_frames`, plus per-frame **mask ellipses** and **silhouette polygons**.
3. **Snapshot masks, then hide them.** Mask descriptors are cached in an "oracle"
   and re-revealed **one frame at a time**, so the causal loop never peeks ahead.
4. **Frame loop** (for $f = 0,1,\dots$):
   1. Reveal this frame's person/car boxes and mask descriptors.
   2. **Camera motion:** $M_f = \texttt{CamMotion.step}$ from frames $f{-}1,f$ only
      ([§6](#6-camera-ego-motion)).
   3. **Driver:** `ProvisionalDriver.push_frame(f, persons, cars, M_f)` ingests the
      frame and returns any events emitted **this** frame
      ([§10](#10-provisional-causal-emission)).
   4. Draw overlays. Optionally write the annotated mp4 frame.
5. **Flush.** Finalize the tail tracks, sort events by $(\text{frame}, \text{person\_id})$,
   write `event.json` and `diagnostics.json`.

---

## 3. Geometry primitives

Source: [src/geometry.py](src/geometry.py).

### 3.1 Overlap (person-over-car)

Intersection area of $p$ and $c$:

$$I(p,c)=\max(0,\ \min(x_2^p,x_2^c)-\max(x_1^p,x_1^c))\cdot\max(0,\ \min(y_2^p,y_2^c)-\max(y_1^p,y_1^c))$$

$$\mathrm{overlap}(p,c)=\frac{I(p,c)}{\max(\varepsilon,\ \mathrm{area}(p))}$$

This is intersection over the **person** area (saturates to 1 when the person
sits fully inside the car).

### 3.2 Centre distance in car diagonals

$$d(p,c)=\frac{\lVert \mathrm{ctr}(p)-\mathrm{ctr}(c)\rVert_2}{\mathrm{diag}(c)}$$

### 3.3 Closeness $\in[0,1]$ — the core proximity score

With $\texttt{near\_dist}$ the distance (in car diagonals) at which the distance
term reaches 0:

$$\mathrm{dist\_score}(p,c)=\max\!\Big(0,\ 1-\tfrac{d(p,c)}{\texttt{near\_dist}}\Big)$$

$$\boxed{\ \mathrm{closeness}(p,c)=\max\big(\mathrm{overlap}(p,c),\ \mathrm{dist\_score}(p,c)\big)\ }$$


### 3.4 Distance-IoU (DIoU) $\in[-1,1]$

Uses intersection over **union** (unlike closeness), so a small person inside a
large car scores low and drops further as they move away:

$$\mathrm{IoU}=\frac{I(p,c)}{\mathrm{area}(p)+\mathrm{area}(c)-I(p,c)+\varepsilon}$$

$$\rho^2=\lVert \mathrm{ctr}(p)-\mathrm{ctr}(c)\rVert_2^2,\qquad
c^2=\text{(diagonal of the smallest box enclosing both)}^2+\varepsilon$$

$$\mathrm{DIoU}(p,c)=\mathrm{IoU}-\frac{\rho^2}{c^2}$$

---

## 4. Mask (segmentation) primitives

Source: [src/geometry.py](src/geometry.py).

### 4.1 Equivalent ellipse of a mask

From binary mask moments $m$ (area $=m_{00}$, centroid $(\bar x,\bar y)$) and the
normalised central moments $\mu_{20},\mu_{02},\mu_{11}$:

$$\text{common}=\sqrt{(\mu_{20}-\mu_{02})^2+4\mu_{11}^2}$$

$$\lambda_1=\tfrac12(\mu_{20}+\mu_{02}+\text{common}),\qquad
\lambda_2=\tfrac12(\mu_{20}+\mu_{02}-\text{common})$$

$$a=2\sqrt{\lambda_1},\quad b=2\sqrt{\lambda_2},\quad
\theta=\tfrac12\operatorname{atan2}(2\mu_{11},\ \mu_{20}-\mu_{02})$$

The descriptor is $[\bar x,\bar y,\text{area},a,b,\theta]$ ($a,b$ = semi-major /
semi-minor lengths, $\theta$ = major-axis orientation).

### 4.2 Offsets in the ellipse frame

For a point $(p_x,p_y)$ with $\Delta=(p_x-\bar x,\ p_y-\bar y)$:

$$u=\Delta_x\cos\theta+\Delta_y\sin\theta,\qquad
v=-\Delta_x\sin\theta+\Delta_y\cos\theta$$

### 4.3 On-major-axis test

With $a'=\max(a,b,1)$, $b'=\max(b,1)$ and config reaches
$\texttt{seg\_major\_reach},\ \texttt{seg\_minor\_band}$:

$$\mathrm{on\_axis}=\big(|u|\le \texttt{seg\_major\_reach}\cdot a'\big)\ \wedge\
\big(|v|\le \texttt{seg\_minor\_band}\cdot b'\big)$$

i.e. the point lies inside the car's elongated major-axis band (its doors/sides).

---

## 5. Motion & continuity primitives

Source: [src/detector.py](src/detector.py).

### 5.1 Window aggregates

Over a frame set $F$ (missing frames skipped in the numerator, but $|F|$ is the
denominator for closeness/overlap):

$$\overline{\mathrm{clos}}(F)=\frac1{|F|}\sum_{f\in F}\mathrm{closeness}(p_f,c_f),\quad
\overline{\mathrm{ov}}(F)=\frac1{|F|}\sum_{f\in F}\mathrm{overlap}(p_f,c_f)$$

$$\overline{\mathrm{DIoU}}(F)=\operatorname{mean}_{f\in F}\mathrm{DIoU}(p_f,c_f),\quad
\overline{\mathrm{diag}}_c(F)=\operatorname{mean}_{f\in F}\mathrm{diag}(c_f)$$

**Best car** for a person over window $F$:
$\ \operatorname{best\_car}(p,F)=\arg\max_{c}\ \overline{\mathrm{clos}}(F)$
and returns that max closeness as $c_\text{start}$ / $c_\text{end}$.

### 5.2 Radial motion (net approach)

Over the common frames $[f_0,\dots,f_n]$ of $p$ and $c$:

$$\mathrm{motion}(p,c)=d(p_{f_0},c_{f_0})-d(p_{f_n},c_{f_n})$$

$\mathrm{motion}>0$ ⇒ approached the car (enter). Departure magnitude is
$-\mathrm{motion}$ (exit).

### 5.3 Car speed (ground-relative, camera-compensated)

For consecutive frames $f,f{+}1$ present in the car track, predict where the
car centre *would* be under pure camera motion and measure the residual:

$$\widehat{\mathrm{ctr}}_{f}=M_{f+1}\cdot \mathrm{ctr}(c_f)\quad(\text{or }\mathrm{ctr}(c_f)\text{ if }M_{f+1}\text{ absent})$$

$$v_\text{car}=\frac{1}{n}\sum_{(f,f+1)}\frac{\big\lVert \mathrm{ctr}(c_{f+1})-\widehat{\mathrm{ctr}}_{f}\big\rVert_2}{\mathrm{diag}(c_{f_0})}$$

Units: car diagonals per frame. A truly parked car has $v_\text{car}\approx 0$
even while the camera moves.

### 5.4 Person drift & straightness (loiter vs. walk-through)

Over near-frame centres $g_i=\mathrm{ctr}(p_{f_i})$ with mean car diagonal
$\bar D=\overline{\mathrm{diag}}_c$:

$$\mathrm{net}=\frac{\lVert g_n-g_0\rVert_2}{\bar D},\qquad
\mathrm{path}=\frac{1}{\bar D}\sum_{i} \lVert g_{i+1}-g_i\rVert_2$$

$$\mathrm{straightness}=\frac{\mathrm{net}}{\mathrm{path}}\in[0,1]$$

A **walk-through** has large `net` *and* high `straightness`. A loiterer stays put
(small net) or wanders (low straightness).

---

## 6. Camera ego-motion

Source: [src/camera.py](src/camera.py). `step(gray_f, boxes_f)`:

1. Build a dynamic-object mask on frame $f{-}1$: each previous box is padded by
   $\texttt{ego\_box\_pad}$ and zeroed out (so features come from the background).
2. `goodFeaturesToTrack` on the masked $f{-}1$ frame
   (`ego_max_corners`, `ego_quality`, `ego_min_dist`).
3. `calcOpticalFlowPyrLK` tracks those points into frame $f$. Keep status-1 pairs.
4. If $\ge \texttt{ego\_min\_pts}$ survive, fit a partial-affine (similarity)
   transform with RANSAC:
   $$M_f=\texttt{estimateAffinePartial2D}(\text{prev},\text{next})\in\mathbb{R}^{2\times3}$$
5. Return $M_f$ (or `None` ⇒ treated as identity). Uses only frames $f{-}1,f$.

$M_f$ feeds `car_speed` (§5.3) so real car motion is separated from panning.

---

## 7. SAM3 tracking & seg cache

Source: [src/detector.py](src/detector.py) `track_video`, `_track_sam3_seg`.

- One SAM3 **session** per clip. Two forward `propagate_in_video` passes: prompt
  `"person"` then `"car"` (`reset_session` between them).
- Per frame, each returned instance with probability $p \ge \texttt{sam3\_score\_th}$
  becomes a pixel-space box $[xW, yH, (x{+}w)W, (y{+}h)H, p]$. Its binary mask is
  reduced to an ellipse descriptor (§4.1) and simplified silhouette polygons.
- Forward propagation is causal. The cache stores boxes + descriptors so
  reruns need no GPU and reproduce identical tracks.
- Cache key: `track_cache/<stem>__sam3seg_s<sam3_score_th>.json`.

---

## 8. Interaction states & gate conditions

Each person track is judged **online** by `ProvisionalDriver` (see [§10](#10-provisional-causal-emission)):
loiter, exit, and enter are each emitted at their earliest causal frame. A state is
reported only when **all** its gates pass. Every gate's pass/fail + detail is written
to `diagnostics.json`.
Let $f_\text{first},f_\text{last}$ be the track's first/last frames, $N$ its length,
and $\mathcal{D}$ the frame diagonal.

**Prerequisite (all states):** $N \ge \texttt{min\_track\_len}$ (else the track is
skipped entirely).

### 8.1 Exit — *person emerges from the car and leaves*

Let $c^\ast=\operatorname{best\_car}(p,\mathrm{start\ win})$ and:

$$c_\text{start}=\overline{\mathrm{clos}}(\mathrm{start}),\quad
c_\text{end}=\overline{\mathrm{clos}}(\mathrm{end}),\quad
\mathrm{motion}=\mathrm{motion}(p,c^\ast)$$

$$\text{car\_frac}=\frac{\overline{\mathrm{diag}}_c(\mathrm{start})}{\mathcal{D}},\quad
\text{overlap\_th}=\begin{cases}\texttt{on\_car\_overlap\_th\_big}&\text{car\_frac}\ge\texttt{big\_car\_frame\_ratio}\\ \texttt{on\_car\_overlap\_th}&\text{otherwise}\end{cases}$$

$$\text{emerged}=\big(\overline{\mathrm{ov}}(\mathrm{start})>\text{overlap\_th}\big)\ \vee\ \big(c_\text{start}\ge\texttt{on\_car\_near\_th}\big)$$

| Gate | Condition | Motivation |
| --- | --- | --- |
| `not_at_clip_edge` | $f_\text{first} > \texttt{edge\_margin}$ | Ignore tracks that exist from frame 0 (no observable arrival). |
| `at_car_at_start` | $c_\text{start} > \texttt{near\_th}$ | Person is genuinely at the car when first seen. |
| `emerged_from_car` | $\text{emerged}$ | Person physically overlapped the car (came out of it), relaxed for big cars. |
| `closeness_drop` | $(c_\text{start}-c_\text{end})>\texttt{delta}\ \vee\ (\overline{\mathrm{DIoU}}(\mathrm{start})-\overline{\mathrm{DIoU}}(\mathrm{end}))>\texttt{exit\_diou\_drop}$ | Overall proximity decreased. |
| `car_stationary` | $v_\text{car}\le \texttt{car\_max\_speed}$ | The car was parked, not driving off. |


### 8.2 Enter — *person reaches the car and gets in*

Only evaluated when the track was **not** an exit and a nearby car exists. Let
$c^\ast=\operatorname{best\_car}(p,\mathrm{end\ win})$ and:

$$c_\text{end}=\overline{\mathrm{clos}}(\mathrm{end}),\quad
c_\text{start}=\overline{\mathrm{clos}}(\mathrm{start}),\quad
\mathrm{motion}=\mathrm{motion}(p,c^\ast)$$

$$\text{reached}=\big(\overline{\mathrm{ov}}(\mathrm{end})>\text{overlap\_th}\big)\ \vee\ \big(c_\text{end}\ge\texttt{on\_car\_near\_th}\big)$$

| Gate | Condition | Motivation |
| --- | --- | --- |
| `vanishes_mid_clip` | $f_\text{last} < \text{total\_frames}-1-\texttt{min\_absence}$ | Person disappears *into* the car mid-clip, not at the clip's end. |
| `at_car_at_end` | $c_\text{end} > \texttt{near\_th}$ | At the car when last seen. |
| `reached_car` | $\text{reached}$ | Physically reached the car body (big-car relaxation as in exit). |
| `closeness_rise` | $(c_\text{end}-c_\text{start})>\texttt{delta}$ | Overall proximity increased — a genuine approach. |
| `moved_closer` | $\mathrm{motion}>\texttt{move\_min}$ | Real net radial approach, not incidental drift. |
| `car_stationary` | $v_\text{car}\le \texttt{car\_max\_speed}$ | Car parked while they approached. |

Emitted at frame $f_\text{last}$. Enter is the only state with irreducible latency
(§10).

### 8.3 Loiter — *person lingers near a parked car*

Candidate car $c^\ast=\arg\max_c \big|\{f:\ \mathrm{closeness}(p_f,c_f)>\texttt{loiter\_near\_th}\}\big|$.
Let $\text{near\_count}$ be that count over near-frames $F^\ast$, and
$(\mathrm{net},\mathrm{straightness})=\text{person\_drift}(F^\ast)$.

$$\text{walk\_through}=\big(\mathrm{net}>\texttt{loiter\_max\_net\_drift}\big)\ \wedge\ \big(\mathrm{straightness}>\texttt{loiter\_straight\_th}\big)$$

| Gate | Condition | Motivation |
| --- | --- | --- |
| `near_long_enough` | $\text{near\_count}\ge \texttt{loiter\_min\_frames}$ | Lingered near the car long enough to count. |
| `car_stationary` | $v_\text{car}\le \texttt{car\_max\_speed}$ | Nearness to a *parked* car. |
| `not_walk_through` | $\neg\,\text{walk\_through}$ | Reject pass-throughs (long, straight paths). |

Emitted at frame $F^\ast_0$ (first near-frame) with `duration` $=\text{near\_count}$.
Note loiter uses the stricter $\texttt{loiter\_near\_th}=0.60$ vs. the enter/exit
$\texttt{near\_th}=0.35$.

---

## 9. Segmentation major-axis gate

Applied **after** an enter/exit passes the box gates (`ProvisionalDriver` →
`Detector.axis_gate`). It can only *remove* events, never add them.

**How it works:**
1. Use the early frames (start window) for exits, late frames (end window) for enters.
2. For each frame with a car mask: check if the person is on the car's long axis (doors/sides).
3. Count how many frames pass: if $\frac{\text{passing frames}}{\text{total frames}} \geq \texttt{seg\_axis\_frac}$ (default 0.40), keep the event.
4. If no car mask exists in the window: keep the event by default.

**Motivation:** box proximity alone can fire on someone *in front of / behind* a
large car. Requiring alignment with the car's long axis keeps events at the
doors/sides where real entering/exiting happens.

---

## 10. Provisional emission

Source: [src/driver.py](src/driver.py) `ProvisionalDriver`. All three states are
emitted at their earliest moment.

**Finalize delay (latency).** A track is *finalized* once the person has been
absent long enough that no future frame can change its verdict:

$$\texttt{finalize\_delay}=\texttt{min\_absence}$$

Finalize runs **no** classification — `_finalize_track` only closes the track and
records its diagnostics from the events that already fired.

| State | When emitted | How |
| --- | --- | --- |
| **loiter** | The frame the near-count first crosses `loiter_min_frames` | `_check_loiter_exit` grows per-(person,car) near-frame lists each frame. `_prov_loiter` picks the car maximising $(\lvert F\rvert,\ \overline{\mathrm{clos}})$, then applies `car_stationary` and `not_walking_past`. Reported `frame`=first near-frame, `duration`=near-count so far. |
| **exit** | The frame a trailing window first shows departure | `_exit_base` computes the fixed **start-window** context once (must pass edge, `at_car_at_start`, `emerged`, `car_stationary`). Each later frame checks $(c_\text{start}-c_\text{end})>\texttt{delta}\ \vee\ (\text{DIoU drop})>\texttt{exit\_diou\_drop}$ over the current end window, then the axis gate on the start window. |
| **enter** | Once the person has been absent for the whole `min_absence` horizon | While the person is on screen, standing at a car is indistinguishable from having just entered. `_prov_enter` waits until they stay gone (`f_idx - f_last ≥ min_absence`), then applies the same box gates as exit's reverse (`at_car_at_end`, `reached`, `closeness_rise`, `moved_closer`, `car_stationary`) and the axis gate. Enter and exit are mutually exclusive. |

`push_frame` returns the events newly emitted for the current frame. `flush`
finalizes any still-open tracks at end of stream and publishes the diagnostics.

---

## 11. Configuration & fps resolution

Source: [config/settings.py](config/settings.py). Fields ending in `_sec` are in
**seconds**. `Config.resolve_for_fps(f)` converts them to per-clip frame counts
(and per-frame speeds) so a threshold means the same real-world duration at any
frame rate:

$$
\begin{aligned}
\texttt{window}&=\max(1,\ \mathrm{round}(\texttt{window\_sec}\cdot f)) &
\texttt{edge\_margin}&=\max(0,\ \mathrm{round}(\texttt{edge\_margin\_sec}\cdot f))\\
\texttt{min\_track\_len}&=\max(2,\ \mathrm{round}(\texttt{min\_track\_sec}\cdot f)) &
\texttt{min\_absence}&=\max(0,\ \mathrm{round}(\texttt{min\_absence\_sec}\cdot f))\\
\texttt{loiter\_min\_frames}&=\max(1,\ \mathrm{round}(\texttt{loiter\_min\_sec}\cdot f)) &
\texttt{car\_max\_speed}&=\dfrac{\texttt{car\_max\_speed\_per\_sec}}{f}
\end{aligned}
$$

**Threshold reference (defaults):**

| Field | Default | State(s) | Meaning |
| --- | --- | --- | --- |
| `min_track_sec` | 0.20 s | all | Minimum track length to judge. |
| `near_dist` | 1.3 | all (closeness) | Distance (car diagonals) at which closeness → 0. |
| `car_max_speed_per_sec` | 0.25 diag/s | all | Max ground-relative car speed (`car_stationary`). |
| `window_sec` | 0.17 s | enter, exit | Start/end window span. |
| `near_th` | 0.35 | enter, exit | "At the vehicle" closeness. |
| `delta` | 0.18 | enter, exit | Min closeness change (rise/drop). |
| `on_car_overlap_th` | 0.15 | exit, enter | Min window overlap (emerged/reached). |
| `on_car_overlap_th_big` | 0.03 | exit, enter | Relaxed overlap for big/near cars. |
| `big_car_frame_ratio` | 0.45 | exit, enter | car/frame diagonal ratio counted "big". |
| `on_car_near_th` | 0.60 | exit, enter | Closeness that alone satisfies emerged/reached. |
| `exit_diou_drop` | 0.045 | exit | Min start→end DIoU drop confirming departure. |
| `edge_margin_sec` | 0.10 s | exit | Ignore exits this soon after clip start. |
| `min_absence_sec` | 0.20 s | enter | Person must vanish this long before clip end. |
| `move_min` | 0.35 | enter | Min net radial approach/departure. |
| `loiter_min_sec` | 2.0 s | loiter | Min time near the car. |
| `loiter_near_th` | 0.60 | loiter | Closeness counted as "near". |
| `loiter_max_net_drift` | 0.6 | loiter | Max net displacement for a loiter. |
| `loiter_straight_th` | 0.85 | loiter | Straightness above which it's a walk-through. |
| `seg_major_reach`, `seg_minor_band`, `seg_axis_frac` | 1.35, 1.4, 0.40 | enter, exit | Segmentation major-axis gate. |
| `seg_strict` | False | enter, exit | Reject events whose window has no mask data. |
| `sam3_score_th` | 0.5 | tracking | Min SAM3 instance probability. |

---

## 12. Outputs

Per clip under `runs/<clip>/`:

- **`event.json`** — every emitted signal. Each record is an `Event`
  ([src/events.py](src/events.py)): `person_id`, `car_id`, `type`
  (`enter`/`exit`/`loiter`), `frame`, `c_start`, `c_end`, `motion`, `duration`,
  and `metrics` (the condition values — `value`/`th`/`pass` — that accepted it).
- **`diagnostics.json`** — per-track: the `result` (states emitted, or `none`) and
  the `events` list with their `metrics`. Built directly from the emitted events.
  no re-classification is run at finalize.
- **`<clip>.mp4`** — annotated video (unless `--no-save`).
