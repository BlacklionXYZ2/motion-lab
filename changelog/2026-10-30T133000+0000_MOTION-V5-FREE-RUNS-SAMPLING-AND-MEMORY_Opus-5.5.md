# Motion core v5: the first model that doesn't freeze, and why the others did

Model name: **Opus 5.5**. Checkpoint **2026-10-30 13:30 UTC**. Status: **trained and
stress-tested; the next run is training**.

Follows `2026-10-30T095000+0000_MOTION-V5-FIRST-RESULTS-AND-FREE-RUNS_Opus-5.5.md`. The
session's usage limit paused this work for about two hours around midday. The training runs
carried on and finished in that time.

## Runs (all on the GPU, `versions/v5/runs/`)

| Run | What | Steps | Note |
|---|---|---|---|
| `latent-vel-free2` | latent-b01 + velocity input + 3 s free runs (energy floor only) | 1,362 | **stopped**: its free runs ran away (below) |
| `latent-vel-free3` | that run's weights continued, with an energy **band** (0.5× to 1.5× the recorded window) | 3,000 | done; best validation 0.79 |
| `feet-vel-free2` | the deterministic twin (no latent) | 4,000 | done; it never met the energy floor (free-run loss 0.46 at the end) |
| `latent-vel-free4` | `latent-vel-free3` continued (see Next) | 3,000 | training |

**What went wrong with `latent-vel-free2`:** free runs from validation windows had a joint speed
of 0.8 → 0.9 → 1.5 → 2.0 → 2.4 rad/s per half second, against the recordings' 1.0. A floor
alone pushed it into running away, so a ceiling was added (`LossWeights.energy_ceiling`).

## Stress battery, 5-minute runs (16 test seeds; `outputs/stress-v5-*-reset*.json`)

| | recordings | v4 | latent-vel-free3 | + memory reset 2.5 s | feet-vel-free2 | + memory reset |
|---|---|---|---|---|---|---|
| Seeds frozen in the last 30 s | 0 | all | **0** | **0** | 75 to 88% | **0** |
| Energy, rad/s | 1.03 | 0.04 → 0.02 | 1.9 → 2.2 | 1.7 → 1.8 | 0.11 → 0.04 | 0.17 → 0.11 |
| Gliding (walk) | 30% | 90% | 59% | 68% | 100% | 75% |
| Hovering (idle) | 0.07% | 43% | 39% | 23% | 0.3% | 0.1% |
| Legs crossed (walk) | 10% | 32% | 19% | **0.2%** | 0.04% | 0.04% |
| Outside the limits (walk) | 0.5% | 3.3% | 7.4% | **0%** | 31% | 21% |
| Hips at idle, mean m | 0.81 | 0.67 | 0.58 | 0.50 | 0.44 | 0.44 |
| Push of 1 rad, never recovered | – | 6% | **0** | 6% | 25% | 19% |
| Steering, walk 1.2 / run 3 m/s, reached | – | – | 0.24 / 0.58 | 0.10 / 0.29 | 0.30 / 0.38 | 0.13 / 0.22 |
| Turning to a new heading | – | – | never | never | never | never |

**Reading:**
- **latent-vel-free3 is the first of motion-lab's own models that doesn't freeze.**
  - Free runs plus the latent did what neither did alone. The deterministic twin with the same
    training still freezes and holds a bad posture.
  - Run is now faster than walk (0.58 against 0.24 m/s). That is still a fifth of what was
    asked, and she never turns to face a new heading.
- **Its energy is mostly shake** (the 60 s evaluation, `outputs/eval-latent-vel-free3.json`):
  - jitter 3,666 m/s³, against the recordings' 471 and v4's 64;
  - mean hips 0.57 m (a crouch);
  - foot slide 0.18 m/s;
  - error at 0.5 s doubled, to 20°.

## Why: two separate causes, found by diagnosis

1. **Shake comes from sampling the noise independently every tick** (`scripts/sampling_sweep.py`,
   60 s, unknown keyword, `outputs/sampling-sweep-latent-vel-free3.json`). The posterior's z,
   which the decoder trained on, changes smoothly from tick to tick.

   | Prior temperature, noise correlation | Jitter, m/s³ | Energy over 60 s, rad/s |
   |---|---|---|
   | recordings | 231 | 0.88 |
   | 1.0, independent | 3,654 | 1.6 → 1.9 |
   | 1.0, 0.98 | 533 | 0.71 to 0.80, steady |
   | 0.5, 0.98 | 264 | about 0.47 |
   | 0, the prior's mean | 3 | 0.17 → 0.03 (freezes, as v4 did) |

   `MotionPrior.noise_correlation` (an AR(1) process, still standard normal per tick) is new.
   - **ρ 0.98 matches the recordings' energy for the whole minute, with jitter 2.3× theirs.**
   - The randomness is what keeps her alive: without it she freezes.
   - The crouch, the limits and the foot slide don't change with sampling; they are in the
     weights.
2. **Hovering and limb drift come from the GRU's memory drifting** (`scripts/hidden_drift.py`).
   - Training windows are at most 2.8 s. Over 5 minutes v4's memory grew from a mean size of
     0.20 to 0.32, and its saturated units from 2.5% to 7.7%.
   - **The fix tried:** staggered memories (`stress.drive(reset=…)`, the battery's
     `--reset`). A second memory restarts from zero and takes over every 2.5 s, so the model
     never runs on more memory than training gave it.
   - **On v4:** hovering 26% → 0.05%, outside the limits 3.3% → 1.2%, and the memory stays
     bounded. Energy is unchanged, because the freezing is the separate averaging problem.
   - **On latent-vel-free3:** outside the limits 7.4% → 0% and legs crossed 19% → 0.2%, but she
     travels less (walking 0.2 → 0.05 m/s). The memory carries locomotion.

## Next (`latent-vel-free4`, running)

`latent-vel-free3` continued for 3,000 steps with:
- **free runs sampled as at run time:** noise correlation 0.98 (`free_run.noise_correlation`),
  4 s, every third step;
- **`posture`:** mean hips within 8 cm of the recorded window's;
- **`jerk_ceiling`:** at most 2× the recorded window's jerk;
- **`free_limits` 4:** the joint and hips limits weigh 4× in free runs.

Tests: 77 pass (the correlated noise keeps N(0, 1) per tick and changes slowly; the new terms
train the prior).

Not done: viewing any of this on Alice; the Unity tests. Nothing committed.
