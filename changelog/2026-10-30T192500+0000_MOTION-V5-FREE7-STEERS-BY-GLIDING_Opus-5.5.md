# Motion core v5: free7 follows commands to within 10%, but glides there; stepping and facing next

Model name: **Opus 5.5**. Checkpoint **2026-10-30 19:25 UTC**. Status: **trained and
stress-tested; the next run is training**.

Follows `2026-10-30T173500+0000_MOTION-V5-TRACKING-FLAW-AND-ROOT-JUMPS_Opus-5.5.md`.

## `latent-vel-free7` (from free4; 3,000 steps on the GPU; best at step 2,500)

- **Recipe:** tracking measured per tick, every second free run counterfactual (a random world
  velocity), and the recording-free skate term.
- **Training was stable.** Counterfactual tracking error fell from 0.18 to 0.06 m/s, free-run
  tracking from 0.17 to 0.09.

**Stress battery** (16 test seeds, run-time sampling: temperature 1, noise correlation 0.98;
`outputs/stress-v5-latent-vel-free7-rho0.98-reset0.json`):

| | recordings | v4 | free5 | **free7** |
|---|---|---|---|---|
| Steering: walk 1.2 m/s, reached | – | – | 0.09 | **1.30** |
| Steering: run 3 m/s, reached | – | – | 0.16 | **3.07** |
| Steering: stand, residual m/s | – | – | 0.07 | **0.017** |
| Steering: travel direction error, 90° / 180° | – | – | – | **2° / 1°** |
| Steering: turns to face the new direction | – | – | never | **never** |
| Gliding while steering | – | – | 66% | **98%** |
| Fastest root frame, m/s | 5.1 | 2.3 | 10.6 | **1.95** |
| Frozen at 5 min | 0 | all | 0 | 0 |
| Hovering / legs crossed / wrist in torso | 0.07% / 10% / 0.08% | 43% / 32% / 2.8% | 1.4% / 0 / 0 | **0 / 0 / 0** |
| Outside the limits (walk, 5 min) | 0.5% | 3.3% | 0.9% | 6.1% |
| Energy over 5 min under walk, rad/s | 1.03 | 0.04 → 0.03 | 0.83 → 0.89 | 0.67 → 0.26 |
| Push of 1 rad recovered | – | 94% | all | 94%, in 0.6 s |

**60 s free run** (`outputs/sampling-sweep-free7.log`):

| | recordings | free7 |
|---|---|---|
| Jitter, m/s³ | 231 | 22 |
| Energy, rad/s | 0.88 | 0.42 to 0.74 |
| Planted-foot slide, m/s | 0.067 | **0.28** |

**Reading:**
- **For the first time one of motion-lab's own models goes where it is told:**
  - at the asked speed and in the asked direction;
  - it stops when asked;
  - there are no root jumps (the fixed tracking).
- **But it glides there and never faces where it goes.** The feet are carried along with the
  root (98% of moving frames) and she crabs or backs her way.
- **These are two loopholes in the losses:**
  - `free_skate` only counted feet within 3 cm of the floor, so feet skimming a little higher
    escape it;
  - nothing asked her to face her direction of travel.
- **Also worse:** limits 6.1%, and the energy under walk fades over 5 minutes.

## Next: `latent-vel-free8` (from free7; training)

- **`stance`:** the slowest foot joint's ground speed above `0.1 + 0.15 × root speed` m/s, in
  every free run.
  - Measured on the recordings first: the slowest foot moves at 0.13 m/s on average while
    walking (0.5 to 1.5 m/s) and 0.37 while running (1.5 to 3 m/s).
  - The allowance matches their 90th percentile at walking pace.
  - When gliding, every foot moves with the root.
- **`facing`:** in counterfactual runs faster than 0.5 m/s, 1 − cos of the angle between her
  heading and the command's direction, from 1.5 s in.
- **Weights:** stance 2, facing 0.5.

Tests: 84 pass (stance zero standing and catching a glide; facing zero along the command, 1 at
90°, 2 at 180°).

Nothing committed.
