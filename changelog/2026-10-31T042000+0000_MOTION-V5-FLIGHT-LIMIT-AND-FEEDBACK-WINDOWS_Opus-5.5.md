# Motion core v5: closing the flight loophole stalls tracking; truncated feedback gradients next

Model name: **Opus 5.5**. Checkpoint **2026-10-31 04:20 UTC**. Status: **two runs inconclusive;
a third (the fix) training**.

Follows `2026-10-31T022000+0000_MOTION-V5-FREE9-FOOT-LOCK-RESULT_Opus-5.5.md`.

## Runs (GPU, side by side, about 8 s a step together)

**`latent-vel-free10`** (free9 + the 0.3 s flight limit + energy floor 1 + facing 1) was paused
at step 2,016. Validation:

| Step | Counterfactual tracking error, m/s | Facing, 1 − cos | Energy-floor shortfall | Excess foot lift, m |
|---|---|---|---|---|
| 500 | 0.77 | 1.03 | 0.15 | 0.036 |
| 1,000 | 1.11 | 0.98 | 0.07 | 0.040 |
| 1,500 | 0.99 | 0.95 | 0.06 | 0.030 |
| 2,000 | 1.00 | 0.92 | 0.04 | 0.033 |

**`latent-vel-free11`** (the same recipe from free4, not free9) at step 1,000: tracking error
1.07 m/s, facing 0.87. It is still training.

**Reading:**
- **With the loophole closed** (no gliding in flight), neither model learns to travel to a
  command. The tracking error stays near 1 m/s, about what standing still scores.
- **The energy floor is working:** the freezing eases.
- **The starting point doesn't matter:** free4 and free9 stall alike.

## Why: stepping needs credit over several ticks

- In free runs the model's output is fed back as its next input. The gradient was cut at every
  tick: v3 found it grew to about 10⁹ through 32 ticks, because the state carries angular
  velocity × 60.
- So each loss only teaches the tick it falls on.
- **Gliding was learnable**, because one tick's root velocity does the job.
- **Stepping isn't:** lifting a foot now only pays off when it lands ahead a few ticks later,
  and that credit never arrived.

## The fix (`m1.rollout(feedback_window=k)`, config `free_run.feedback_window`)

- The gradient now runs back through k consecutive fed-back ticks and is cut between windows.
  Gradient clipping (1.0) stays.
- **Test:** with a window of 8, the last prediction's loss sends more gradient into the decoder
  (through the earlier ticks) than with the cut every tick. Tests: 86 pass.

**`latent-vel-free12`** continues free10's weights (its step-2,016 state) with 8-tick windows in
free runs, 3,000 steps. free11 continues alongside.

Nothing committed.
