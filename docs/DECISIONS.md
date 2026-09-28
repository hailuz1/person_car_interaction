# Ambiguity Decisions — Per-Clip

Per the task: *"If an ambiguity arises, resolve it and document the decision taken,
together with the alternatives considered."* This document records the ambiguities
that surfaced from the actual clip results in [runs/](../runs) and how each was
resolved. Clips are treated as independent (no inter-clip person/vehicle identity),
as permitted by the task.

Results summary (from each `runs/<clip>/event.json`):

| Clip | Events | Notable pattern |
|------|--------|-----------------|
| clip1 | 2× loiter, 2× enter | Same persons loiter, then later enter. |
| clip2 | 1× exit | Single clean exit. |
| clip3 | exit + loiter (same person/frame), loiter | Person emerges and lingers. |
| clip4 | loiter + exit + loiter | Person already at car at frame 0. |
| clip5 | (none) | Two people, only pass-by. |
| clip6 | 2× loiter, 1× enter | Loiter escalates to enter. |
| clip7 | exit + loiter (same person/frame) ×2 | Emerge-then-linger. |
| clip8 | 2× loiter | Lingering only, no entry/exit. |

---

## Ambiguity 1 — A person both **loiters** and **enters/exits** (clip1, clip3, clip4, clip6, clip7)

**Observed:** In clip1, persons 3 and 4 each emit a `loiter` and later an `enter`.
In clip3/clip7, a person emits `exit` **and** `loiter` on the same frame. In clip6,
person 1 loiters then enters.

**Ambiguity:** Is this double-counting the same interaction?

**Decision:** **No — keep both.** `loiter`, `enter`, and `exit` are treated as
**distinct, non-exclusive** signals about the same person–car pair. Loitering
(lingering nearby) and entering (getting in) are genuinely different events, and a
person often does both in sequence (linger, then get in). Each is emitted once, at
its own earliest causal moment, and never retracted.

**Alternative considered:** Collapse to a single "final" label per person (mutually
exclusive states). *Rejected* because it would (a) discard the earlier provisional
loiter signal that is correct at the time it fires, and (b) break the streaming /
causal guarantee — we cannot know the final label without peeking ahead.

---

## Ambiguity 2 — **Exit** and **loiter** on the *same frame* (clip3 person 0, clip7)

**Observed (clip3 person 0):** both fire at frame 9. Diagnostics: `c_start = 1.0`
(fully on the car), `exit` has `c_end = 0.809, motion = 0.077` (moving away);
`loiter` has `c_end = 1.0`.

**Ambiguity:** A person emerging from a car is momentarily *on* the car (looks like
loiter) while also *leaving* it (exit). Which one is it?

**Decision:** **Emit both.** At the causal moment, the evidence for *emerged-and-
departing* (exit) and *present-nearby* (loiter) both hold. Suppressing one would
require a future look-ahead to decide the "true" intent.

**Alternative considered:** Priority rule (exit suppresses loiter). *Rejected* as
brittle — the ordering depends on threshold tuning and would hide a valid signal.
Downstream consumers can apply their own priority if they want a single label.

---

## Ambiguity 3 — Person **already at the car at frame 0** (clip4 person 0, clip8 person 0)

**Observed:** clip4 person 0 emits `loiter` at frame 0; clip8 person 0 at frame 0.

**Ambiguity:** The track starts mid-interaction. We never saw them approach, so we
cannot tell if they just exited, are about to enter, or are simply standing there.

**Decision:** **Emit `loiter` only.** With no approach/departure history, the only
defensible claim is "present near a stationary car," i.e. loiter. Enter/exit require
observed motion into/out of the car, which is absent here.

**Alternative considered:** Guess `exit` (assume they just got out). *Rejected* — no
evidence supports it; it would fabricate an interaction that may never have happened.

---

## Ambiguity 4 — **No interaction** despite people near a car (clip5)

**Observed:** clip5 has two person tracks (196 and 300 frames) but emits **zero**
events; diagnostics mark both `result: none`.

**Ambiguity:** Nearby people could be interacting or just walking past.

**Decision:** **Emit nothing (pass-by).** The loiter gate rejected them via the
**net-drift + straightness** test (walking in a fairly straight line through the
scene), and neither crossed the enter/exit evidence thresholds. Correctly emitting
an empty list is the intended behaviour for pass-by-only clips.

**Alternative considered:** Lower thresholds so *something* fires. *Rejected* — it
would trade a correct empty result for false positives on every passer-by.

---

## Ambiguity 5 — **Which car** a person interacts with (clip1: cars 18 & 22, clip4: cars 16 & 38)

**Observed:** Multiple cars are present; each event names one `car_id`.

**Ambiguity:** When several cars are nearby, which one is the interaction partner?

**Decision:** **Best car by mean closeness** over the relevant window
(`best_car = argmax closeness`). The car the person is most consistently closest to
wins. This yielded clean, stable pairings (person 3 ↔ car 22, person 4 ↔ car 18 in
clip1).

**Alternative considered:** Nearest-centre at a single frame. *Rejected* — a single
frame is noisy under occlusion and camera motion; a window mean is far more robust.

---

## Ambiguity 6 — **Moving vs. stationary** vehicle (all clips)

**Ambiguity:** "Entering a vehicle" is only well-defined for a vehicle that is
essentially parked; judging entry into a moving car is unreliable.

**Decision:** **Gate on camera-compensated car speed ≈ 0.** Using optical-flow
ego-motion, a truly parked car reads near zero even while the camera pans, so all
interaction gates require a stationary car. This scoped the problem to the cases we
can judge confidently.

**Alternative considered:** Allow moving vehicles. *Rejected for now* — it needs
relative-motion modelling (listed under Next Steps in the write-up) and would add
false positives without it.
