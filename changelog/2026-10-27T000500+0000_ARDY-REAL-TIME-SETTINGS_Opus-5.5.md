# ARDY: the real-time setting on the owner's GPU (4 diffusion steps, replanning every 0.4 s)

Model name: **Opus 5.5**. Checkpoint **2026-10-27 00:05 UTC** (continued after a usage-limit
pause). Status: **measured in motion-lab; SORA integration not started**.

## Changes

- **`ardy-runtime/stream.py`:**
  - `--keep N` plays N frames of each generation before replanning;
  - the latency report states the real-time budget and how many steps missed it.
- **`ardy-runtime/README.md`:** the diffusion-steps table and the gesture caveat.

## Measured

The 60 s scenario (walk, stand, wave, run while turning), fp16, DirectML, 4 frames of history,
a replan every 8 frames (0.4 s):

- **10 steps:** 721 ms median per replan; late on 150 of 151.
- **4 steps:** 313 ms median (376 max), never late.
  - walk 1.04 to 1.18 m/s against 1.2;
  - run 2.0 to 2.7 against 2.5;
  - clean stop;
  - 0.12% of joint-frames outside the limits;
  - foot slide 0.059 m/s.
- **2 steps:** 155 ms, but walking misses its target (about 0.5 m/s) and 1.6% is outside the
  limits.
- **Gesture prompts after standing are unreliable** (a weak wave here, a strong one in
  `wave_test.py`). A hand target is the dependable route for a specific gesture.

## Next

The SORA side: the motion service, Unity, then settings and the GUI. The owner decides the open
choices first.

Nothing committed.
