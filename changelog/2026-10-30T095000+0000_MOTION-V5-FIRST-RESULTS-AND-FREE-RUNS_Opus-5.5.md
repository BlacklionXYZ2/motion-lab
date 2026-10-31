# Motion core v5: the latent and the foot losses didn't fix long runs; free-run training next

Model name: **Opus 5.5**. Checkpoint **2026-10-30 09:50 UTC**. Status: **two models trained and
stress-tested (a negative result); two more training**.

Follows `2026-10-30T075000+0000_MOTION-V5-STRESS-BATTERY-AND-LATENT_Opus-5.5.md`.

## Training

The GPU came free at about 08:10 (the owner: "The GPU is free now"). `latent-b01` was paused on
the CPU at step 604 (its STOP file) and resumed on the GPU. An ablation ran beside it.

| Run | What | Steps | Time | Best validation | Power (logged, whole PC) |
|---|---|---|---|---|---|
| `latent-b01` | v4 + latent (32, KL 0.1, free bits 0.02) + foot losses | 6,000 | 1.7 h (20 min on the CPU) | 0.71 (ELBO, at 64 ticks) | about 0.36 kWh at the wall |
| `feet-only` | v4 + foot losses, no latent | 6,000 | 1.2 h | 0.69 | about 0.28 kWh |

The two shared the GPU, so their power overlaps. The latent's KL grew from about 0.3 to 0.7
nats a tick once rollouts reached 32 ticks.

## Stress battery (16 test seeds; `versions/v5/outputs/stress-v5-*.json`)

| 5-minute runs unless stated | recordings | v4 | latent-b01 | feet-only |
|---|---|---|---|---|
| Frozen in the last 30 s (unknown) | 0 | all | all | 94% |
| Energy over 5 min, rad/s | 1.03 | 0.04 → 0.02 | 0.12 → 0.06 | 0.10 → 0.04 |
| Gliding (walk) | 30% | 90% | 81% | **61%** |
| Hovering (idle) | 0.07% | 43% | 61% | 68% |
| Legs crossed (walk) | 10% | 32% | 69% | 37% |
| Outside the data's limits (walk) | 0.5% | 3.3% | **23%** | **19%** |
| Push of 1 rad: never recovered | – | 6% | **88%** | **81%** |
| Flicker every tick: jerk against steady | – | 4.3× | **1.0×** | 2.1× |
| Speed in the switch test, walk / run, m/s | 0.56 / 1.03 (data means) | 0.24 / 0.17 | 0.09 / 0.11 | 0.07 / 0.04 |

**Reading:**
- **Neither fix worked.** Energy is 2 to 3× v4's but still about 10× below the recordings, and
  every keyword still winds down.
- Gliding fell from 90% to 61% with the foot losses alone. Flicker is calmer.
- Everything about pose quality in long runs got worse, by similar amounts in both models:
  limits, hovering, recovery from a push. So the common cause is the fine-tune, not the latent:
  - rollouts of only 64 ticks (v4 trained to 128);
  - losses that all compare against a recording, which a long run leaves behind after a
    second or two.
- **Supervised rollouts never visit the states a long free run drifts into.** In the latent
  model the posterior, which sees the recording, steers every training rollout. So nothing
  teaches either model to come back.
- **v4 stays the best of motion-lab's own models for long-run pose quality.** ARDY remains far
  ahead of all of them.

## Next: free-run training (`m1.free_run_losses`, `m1.plausibility`; running)

Each training step adds a batch that runs alone for 4 s from 0.5 s of recording, as at run time
(a latent model draws z from its prior). It is scored only on what holds at any horizon:
- **`ground`:** the lowest foot joint within 10 cm above (flight) and 2 cm below its floor
  height, in the support frame;
- **`self_contact`:** no wrist within 10 cm of the spine, chest or upper chest;
- **`leg_cross`:** no ankle more than 8 cm past the other sideways;
- **the joint and hips limits;**
- **`energy_floor`:** at least half the recorded window's own joint speed.

The free-run terms also count towards the best checkpoint.

Both runs grow last round's models with the velocity input, `MotionSignal.velocity` (the
recording's mean velocity over the next 0.5 s, in her heading frame):

| Run | From | Note |
|---|---|---|
| `latent-vel-free2` | `latent-b01` | 4,000 steps; KL as before; ground, self-contact and leg-cross 4.0, energy floor 0.5; a 3 s free run every third step |
| `feet-vel-free2` | `feet-only` | the same without the latent |

They are second starts. The first (`latent-vel-free`, `feet-vel-free`) ran a 4 s free run every
step at 3.5 s a step, about 5 hours for 5,000 steps. They were paused at steps 230 and 263 and
are kept.

Tests: 76 pass (plausibility zero when standing at rest and catching a hover, a sink, a wrist on
the chest and crossed legs; free runs training the prior and not the posterior).

Nothing committed.
