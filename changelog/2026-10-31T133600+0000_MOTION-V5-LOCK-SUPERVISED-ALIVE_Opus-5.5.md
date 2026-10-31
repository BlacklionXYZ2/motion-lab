# Motion core v5: learning under the locks from the data alone gives the first live, non-gliding model

Model name: **Opus 5.5**. Checkpoint **2026-10-31 13:36 UTC**. Status: **interim result; the
run is finishing**.

Follows `2026-10-31T124500+0000_MOTION-V5-FREE13-FREE14-LOCKS-COLLAPSE_Opus-5.5.md`.

## `latent-vel-lock-sup`

- **Setup:** free4's weights, the foot and ground locks on, and **only the supervised losses**:
  no free runs, no counterfactual commands. Checkpoints kept at every validation.
- **Speed:** about 1.6 s a step, half the cost of a run with free runs.
- **Validation:** root-velocity error 0.25 → 0.21 m/s and hips-height error 3.4 → 1.7 cm over
  the first 1,000 steps. Her legs carry more of the recorded travel.

## The step-1,000 checkpoint under the battery (run-time sampling, 16 seeds)

| | recordings | free7 (glider) | free12b | free14 | **lock-sup at 1,000** |
|---|---|---|---|---|---|
| Seeds frozen at 5 min (keyword / 0 m/s) | 0 | 0 / – | all / all | 94% / all | **0 / 0** |
| Energy over 5 min, rad/s | 1.03 | 0.68 → 0.59 | 0.13 → 0.02 | 0.09 → 0.04 | **0.30 → 0.24** |
| Carried feet (walk, 5 min) | 3.4% | 99.7% | 51% | 0 | **0.04%** |
| Hovering (idle) | 0.07% | 0 | 93% | 0 | **0** |
| Legs crossed (walk) | 10% | 0 | 3% | 1% | 0.6% |
| Hips at idle, m | 0.81 | 0.65 | 0.57 | 0.44 | 0.64 |
| Outside the limits (walk) | 0.5% | 6% | 2.4% | 8.9% | 9.7% |
| Steering: walk 1.2 / run 3 / stand, m/s reached | – | 1.30 / 3.07 / 0.02 | 0.22 / 1.66 / 0.05 | 0.01 / 0.25 / 0.02 | 0.21 / 0.22 / **0.21** |
| Fastest root frame, m/s | 5.1 | 2.0 | 2.9 | 2.3 | **8.4** |

**Reading:**
- **The first foot-locked model that stays alive:** energy holds steady over 5 minutes, at rest
  and walking.
- **Its feet are really planted** (carried 0.04% against the gliders' 99.7%), with no hovering.
- **But it ignores the velocity command.** It shuffles at about 0.21 m/s whether asked to walk,
  run or stand.
- **The best checkpoint, and whether velocity starts to matter later in training,** come in
  the next entry (steps 2,000 and 3,000 are being tested).

## A cap on the lock's root step (`m1.LOCK_MAX_SPEED`, 6 m/s)

- lock-sup's root reached 8.4 m/s in one frame.
- **Replaying recordings through the lock** (`scripts/lock_check.py`, now reporting this):
  per-frame errors against the recorded root reach 4.5 m/s (99.9th percentile 2.7 m/s). When the
  support switches sharply between feet, the root briefly moves with the swinging foot.
- **The fix:** the lock's per-tick root step is now capped at 6 m/s, a little above the
  recordings' fastest frame (5.1). It doesn't touch the recordings (their largest derived frame
  is 5.55 m/s) but stops the model's extreme spikes.
- **Not fixed:** the milder support-switching jitter. That needs the support smoothed over time.
- **Note:** the running training used the uncapped lock. Evaluations from now on, and future
  runs, use the cap.
- Tests: 88 pass.

Nothing committed.
