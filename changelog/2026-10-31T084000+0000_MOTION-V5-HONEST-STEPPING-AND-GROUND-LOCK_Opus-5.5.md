# Motion core v5: the first honest stepping (free12b), with hovering and freezing; a ground lock

Model name: **Opus 5.5**. Checkpoint **2026-10-31 08:40 UTC**. Status: **free12b done and
tested; free13 and the new free14 training**.

Follows `2026-10-31T054500+0000_MOTION-V5-LOCK-CHECK-AND-COMMAND-CURRICULUM_Opus-5.5.md`.

## The battery gains SORA-like long runs

A velocity model is always sent a velocity by SORA, so `long_runs` now adds two closed-loop
5-minute runs (`stress_m1.py`, reported in the table):
- standing at **0 m/s**;
- **walking ahead at 1.2 m/s**.

The keyword-only runs remain.

## Results (run-time sampling, 16 seeds; `outputs/stress-v5-*`)

| | recordings | free7 (glider) | free9 | free12b at 2,000 | **free12b** | free13 at 1,000 |
|---|---|---|---|---|---|---|
| Carried feet while steering at 1.2 m/s | – | 97% | 95% | 42.5% | **42%** | 72% |
| Steering: walk 1.2 / run 3 m/s, reached | – | 1.30 / 3.07 | 1.01 / 3.10 | 0.46 / 1.37 | 0.22 / 1.66 | 0.78 / 2.32 |
| **5 min walking at 1.2 m/s:** speed reached | – | – | – | – | 0.27 → 0.28 | – |
| **5 min walking at 1.2 m/s:** carried feet | – | – | – | – | **20%** | – |
| **5 min at 0 m/s:** seeds frozen | – | – | – | – | all | – |
| Hovering (idle) | 0.07% | 0 | 2% | 90% | **93%** | 6% |
| Hips at idle, m | 0.81 | 0.65 | 0.78 | 0.58 | 0.57 | 0.67 |
| Legs crossed (walk) | 10% | 0 | 15% | 6% | 3% | **74%** |
| Outside the limits (walk) | 0.5% | 6% | 16% | 1.4% | 2.4% | 1.9% |
| Seeds frozen at 5 min, keyword only | 0 | 0 | all | all | all | all |
| Push of 1 rad never recovered | – | 6% | 12.5% | – | 37.5% | – |

**Reading:**
- **free12b steps honestly:** walking under command, its feet are carried only 20% of the time
  (recordings 3.4%, gliders 95 to 99%). But it is slow (0.27 m/s when asked for 1.2), and it
  freezes at rest.
- **It hovers in a crouch at idle** (93%, hips 0.57 m). The lock fixed the root's *horizontal*
  travel to the feet, but the hips' *height* is still the model's own output. So she can fold
  her legs up under hips that keep their height.
- **free13** (commands mostly ahead) is faster mid-training but crosses its legs 74% of the
  time. Its run continues.
- **free7 is still the best functional model** for steering and liveliness, and it glides.

## The ground lock (`ModelConfig.ground_lock`, `m1.foot_locked(ground_height=True)`)

- **What it does:** while the foot lock holds, the root's height is lowered so the lowest
  contact joint (a soft minimum) rests on its floor height. Hovering becomes impossible by
  construction, as gliding did.
- **In the lock's fade band** it applies partly, so a raised body settles over a few ticks. In
  flight the model's height applies, and the 0.3 s flight limit brings her down.
- **Tests** (88 pass): 2 cm up is grounded in one tick; 5 cm up settles within 10 ticks while
  the model asks to stay up.

## Running

- **`latent-vel-free14`:** free12b's best with the ground lock and mostly-ahead commands,
  3,000 steps.
- **`latent-vel-free13`** continues alongside.

Nothing committed.
