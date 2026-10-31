# Motion core v5: damped feedback windows; the foot lock checked on real motion; commands ahead

Model name: **Opus 5.5**. Checkpoint **2026-10-31 05:45 UTC**. Status: **diagnosed; a new run
training**.

Follows `2026-10-31T042000+0000_MOTION-V5-FLIGHT-LIMIT-AND-FEEDBACK-WINDOWS_Opus-5.5.md`.

## Feedback windows: NaN, then damped

- **`latent-vel-free12`** (8-tick feedback windows, undamped) went to NaN at step 85. Credit
  through fed-back ticks compounds, and one infinite gradient turns every weight into NaN.
- **Two fixes:**
  - **`m1._damped`:** each fed-back tick keeps its value, but its gradient is scaled by
    `free_run.feedback_decay` (0.3), so credit through earlier ticks decays geometrically;
  - **`train_m1.py` skips any optimiser step whose gradient norm isn't finite** (and counts
    them).
  - Tests check that a window sends more gradient back, and that damping scales it.
- **`latent-vel-free12b`** (damped):
  - stable, with no skipped steps in 1,000;
  - much livelier: the energy-floor shortfall is 0.013, against free11's 0.13;
  - **counterfactual tracking still about 1.1 m/s.**
- **`latent-vel-free11`** (the same recipe from free4, undamped, cut every tick) stayed at
  about 1.0 m/s to step 2,250 and was stopped. It is kept.

## Is the foot lock even able to travel? (`scripts/lock_check.py`)

Recorded test windows (256 × 5 s) were replayed through `m1.foot_locked`, each root derived from
the recorded poses alone:

| Recorded speed, m/s | recorded | derived through the lock | lock engaged |
|---|---|---|---|
| 0 to 0.2 | 0.07 | 0.08 | 100% |
| 0.2 to 0.8 | 0.50 | 0.44 | 100% |
| 0.8 to 1.6 | 1.10 | 0.92 | 98% |
| 1.6 to 3 | 2.08 | 1.68 | 97% |
| over 3 | 3.57 | 2.57 | 99% |

- **The lock reproduces 84 to 88% of real travel at walking speeds** (72% sprinting), and the
  lowest foot sits at floor height (median lift 0.0 m).
- So a model that moves its legs as the recordings do can travel under it. The stall is in the
  training.
- The 12 to 16% shortfall probably comes from the soft minimum averaging a rolling foot's ankle
  and toe. It is left as is for now.

## The likely cause, and the change

- **The likely cause:** counterfactual commands pointed anywhere, so three-quarters of them
  needed her to turn or step sideways first. Facing hasn't been learned under the lock. free7
  could strafe-glide sideways, but stepping sideways is far harder.
- **The change:** `random_commands(heading=…, ahead_share=…)`, config
  `free_run.cf_ahead_share`. A share of commands now points roughly ahead of her (normal, 30°
  spread around her facing at the start of the free run); the rest stay uniform.
- Tests: 87 pass.

## Running

**`latent-vel-free13`:** free12b's weights at step 1,000, with 75% of commands ahead. free12b
continues to 3,000 with uniform commands, as the control.

Nothing committed.
