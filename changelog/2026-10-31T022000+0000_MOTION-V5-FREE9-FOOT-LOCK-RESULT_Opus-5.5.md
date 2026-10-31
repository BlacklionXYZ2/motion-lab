# Motion core v5: the foot lock dents the gliding, and the model finds a loophole

Model name: **Opus 5.5**. Checkpoint **2026-10-31 02:20 UTC**. Status: **trained and
stress-tested; a fix for the loophole follows**.

Follows `2026-10-30T224500+0000_MOTION-V5-FREE8-FACING-AND-FOOT-LOCK_Opus-5.5.md`.

## A sharper gliding measure (`stress.body`: `carried_share_when_moving`)

- **The old measure was too loose.** "Gliding" (no foot faster than 1.2× the root) also counts
  the body moving over planted feet, so the recordings scored 29%.
- **The new measure, "carried feet":** the share of moving frames in which even the slowest foot
  joint travels at more than 0.1 m/s + half the root's speed, so no foot is planted.

| Carried feet, 5-minute walk runs | recordings | v4 | free7 |
|---|---|---|---|
| share of moving frames | **3.4%** | 95.6% | 99.7% |

## The foot lock alone, on free7, untrained

With the lock switched on at run time (`stress_m1.py --foot-lock`), free7 can no longer reach its
commanded speed (walk 0.34 of 1.2 m/s, run 2.26 of 3), because its legs never learned to step.

## `latent-vel-free9` (free7 + the foot lock + facing 0.5; 3,000 steps, 3.2 h on the GPU)

**Training:**
- It ran at 3.9 s a step, since the lock adds forward kinematics on every tick.
- Counterfactual tracking error went from 0.42 to about 0.2 m/s.
- Facing stayed at about 0.92.

**Stress battery** (run-time sampling, 16 seeds, `outputs/stress-v5-latent-vel-free9-rho0.98.json`):

| | recordings | v4 | free7 | free9 |
|---|---|---|---|---|
| **Carried feet, 5-minute walk runs** | 3.4% | 96% | 99.7% | **37%** |
| Carried feet while steering | – | – | 97% | 95% |
| Steering: walk 1.2 / run 3 m/s, reached | – | – | 1.30 / 3.07 | 1.01 / 3.10 |
| Hips at idle, m | 0.81 | 0.67 | 0.65 | **0.78** |
| Seeds frozen at 5 min | 0 | all | 0 | **all** |
| Energy over 5 min, rad/s | 1.03 | 0.04 → 0.02 | 0.66 → 0.64 | 0.09 → 0.03 |
| Outside the limits (walk) | 0.5% | 3.3% | 6.1% | 16% |
| Push of 1 rad: never recovered | – | 6% | 6% | 12.5% |
| Turns to face a new heading | – | – | never | never |

**Reading:**
- **The lock works where she isn't being pushed:** carried feet fell from 99.7% to 37% in long
  runs, and the hips came up.
- **While steering she found a loophole.** The lock fades out between 3 and 8 cm of lift (to
  allow running flight). The ground term only objects above 10 cm. So she keeps every foot
  skimming in that band, is "in flight", and drives the root as before.
- **Freezing is back** in plain long runs. Stepping is harder than gliding, and the energy
  floor wasn't enough.

## Next

- **Flight gets a time limit:** after 0.3 s with every foot off the floor, the lock re-engages
  on the lowest foot wherever it is. Real running flight lasts about 0.1 to 0.15 s.
- The energy floor and facing get more weight.

**Done at 02:19:**
- **The flight limit is built:** `m1.AIR_LIMIT_S`. The model keeps a flight timer between ticks,
  as it keeps the noise process. Every run loop resets it at its start (`MotionPrior.reset_noise`).
- **Tests (85 pass):** one tick in the air is timed; after 0.3 s the lock holds her; touching
  down resets the timer.
- **`latent-vel-free10` is training** from free9, with energy floor 1 and facing 1.

Nothing committed.
