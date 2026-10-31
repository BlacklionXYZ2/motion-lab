# Motion core v5: a flaw in the tracking term (root jumps, then divergence), found and fixed

Model name: **Opus 5.5**. Checkpoint **2026-10-30 17:35 UTC**. Status: **fixed; retraining**.

Follows `2026-10-30T152800+0000_MOTION-V5-FREE4-FREE5-RESULTS_Opus-5.5.md`. The session's
usage limit paused the work for about two hours this afternoon; `latent-vel-free6` trained
through it.

## What happened

- **`latent-vel-free6`** (free5 + counterfactual commands at tracking weight 2 + free-run
  skate) **diverged**:
  - validation 7.8 at step 1,000, then 28, 71, 653 and 828;
  - tracking error 179 m/s;
  - the feet sliding at 15 m/s.

  Even its best checkpoint (step 1,000) is worse than free5. It is kept in
  `runs/latent-vel-free6` and not used.
- **The cause:** the tracking term measured her achieved velocity as the displacement between
  two frames 0.5 s apart.
  - Fed-back positions are detached in training, so only the last tick before each mark got
    gradient, and it was asked to make up the whole half second's error.
  - That teaches single-tick root jumps. Counterfactual commands (far from what she was doing)
    made the error large and it ran away.
- **The same flaw hid in free5,** the best model of the last entry. A 60-second free run over 16
  test seeds measured the root's speed per frame:

  | | median | 99th percentile | fastest frame |
  |---|---|---|---|
  | recordings | 0.64 | 2.55 | 4.93 m/s |
  | v4 | 0.08 | 0.64 | 2.39 |
  | free4 (no tracking) | 0.12 | 0.32 | 2.21 |
  | **free5** (tracking) | 0.07 | 1.45 | **10.52** |

  free5's root occasionally jumps at twice the recordings' fastest frame. **No stress measure
  caught it**, so free5 is no longer the recommended model.

## Fixes (`versions/v5`)

- **`m1.step_velocity`:** each predicted tick's ground velocity from its own input. Tracking
  (`_tracking` and the counterfactual runs) now averages these per 0.5 s. It is the same value,
  but every tick carries its share of the gradient.
  - A test checks all 60 ticks of a half second receive gradient.
- **The stress battery now measures root jumps:** the root speed's 99th percentile, its fastest
  frame, and the share of frames over 5 m/s (faster than any recorded frame). They are in
  `stress.summary` and the comparison table.
- **Tests:** 82 pass.

## Retraining

**`latent-vel-free7`** continues from free4 (no tracking, so no jumps: fastest frame 2.2 m/s):
- the fixed tracking at weight 1;
- every second free run counterfactual;
- the recording-free skate term (0.5).

Nothing committed.
