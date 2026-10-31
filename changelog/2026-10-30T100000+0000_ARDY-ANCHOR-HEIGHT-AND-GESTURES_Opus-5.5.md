# ARDY service: anchor only a real standing pose; gestures from standing don't take

Model name: **Opus 5.5**. Checkpoint **2026-10-30 10:00 UTC**. Status: **fixed and verified;
one finding left open**.

Follows `2026-10-30T085000+0000_ARDY-STRESS-IDLE-ANCHOR-AND-GET-UP_Opus-5.5.md`.

## New stress scenarios (`stress_service.py`)

- **Sit, then walk, then idle** (two seeds): she stands up and walks at the asked speed (1.19
  to 1.20 m/s), though 25 to 35° off her facing while walking.
- **The bug:** in seed 0 she then idled at hips 0.75 m, a half crouch, and the idle anchor
  captured it. An anchor needed only the guard's 0.70 m, so a transitional crouch could be held
  for as long as she idled.
- **Wave, lean and point after 3 s of standing:** none of them happened (joint energy 0.05 to
  0.13 rad/s, the hand never near the head, no lean).

## Fixes (`service.py`)

- **An anchor is only taken from a real standing height:** hips at least 0.82 m
  (`ANCHOR_MIN_HIPS`). Her rest is 0.894 m; ARDY's idle stands at 0.89 to 0.92 m.
- **Below that height under an idle-like key, she is helped up** (the get-up target from the
  last entry). Before, only hips under 0.70 m counted as down.
- **Result:** after sitting, seed 0 now ends standing at hips 0.90 m. Every earlier scenario is
  unchanged or better:
  - 2-minute idle: worst pop 13.5°/frame, acceleration 1.8× the recordings' 99th percentile,
    3% outside the limits;
  - lying starts get up as before;
  - `test_service.py` passes.

## Gestures: tried, not fixed (`gesture_test.py`, `outputs/gesture-test.json`)

Text guidance on its own (the service's `cfg` is 2.0 for text) was set to 2.0, 3.5 and 5.0, two
seeds each, for 3 s of idle and then 6 s of the gesture:
- **Wave:**
  - 1 of 6 runs raised the hand above the head (cfg 5.0, arm energy 0.86 rad/s);
  - in the rest the hand stayed 0.44 to 0.48 m below the head.
- **Point:** erratic. The arm moved in 3 of 6 runs, but the hand reached head height only once.
- **Lean:** the spine never bent more than in idle (8 to 12°).

Stronger text guidance is unreliable and may degrade everything else, so it is unchanged.

**Where gestures belong:**
- **In SORA:** her existing gaze, reach and point overrides, layered on top of ARDY's pose (the
  2026-10-27 SORA changelog notes they may be lost while ARDY drives her);
- **or in ARDY:** its end-effector (hand) constraints, which the 2026-10-26 notes found
  dependable.

**"Lean" matters to SORA:** her player sends it, and with ARDY she won't lean.

Nothing committed.
