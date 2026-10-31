# v5: stress-tested; a latent, free-run training, steering and the foot lock (from 2026-10-30)

The working version: new work happens in this folder. It started from v4's trained model
(copied to `runs/from-v4/best.pt`) and its data (`data/processed/`, hard links to v4's files,
never written). The full story, round by round, is in motion-lab's changelog (2026-10-30 and
2026-10-31 entries).

## Status (2026-10-31 16:00 UTC)

| Problem in v4 | State | How |
|---|---|---|
| Freezes in every 5-minute run | **solved** | a per-tick latent + free-run training + correlated run-time noise (free3 to free7) |
| Shakes once alive | **solved** | correlated noise (jitter 15× the recordings' → 1.6×) |
| Hovers with the feet tucked up | **solved by construction** | the ground lock (free14 onwards) |
| Root jumps (found in free5) | **solved** | tracking measured per tick; a 6 m/s cap on the lock's root step |
| Doesn't go where she's sent | solved for gliding models (free7: speed within 10%, direction within 2°); **open under the foot lock** | counterfactual commands; under the lock no model has learned it yet |
| Doesn't face where she goes | solved in free8 (turns in about 4 s); **open under the foot lock** | the facing reward |
| **Glides** (the feet carried with the body) | **solved by construction** | the foot lock (lock-sup: carried feet 0.04%, the recordings 3.4%) |

**Where it stands:**
- **`latent-vel-lock-sup`:** the first model that is alive *and* doesn't glide or hover. It
  walks on planted feet and stays lively for 5 minutes, but ignores the velocity command (about
  0.21 m/s whatever is asked).
- **`latent-vel-lock-free`** (free runs added on top of it, to teach steering) and
  **`latent-vel-lock-sup2`** (more of the same supervised training) are training.

**Best model for each job so far:**

| Job | Model | Its weak point |
|---|---|---|
| Steering precisely, lively | `latent-vel-free7` | glides (99.7% carried feet), never faces where it goes |
| Turning to face where it goes | `latent-vel-free8` | freezes in long runs; glides |
| Walking on planted feet, alive | `latent-vel-lock-sup` (step 1,000) | ignores the velocity command; 9.7% outside the limits |
| A clean body at idle | `latent-vel-free4` | crouches a little; no steering |

**Run-time settings for a latent model:**
- `temperature` 1.0 and `noise_correlation` 0.98. With the prior's mean alone (no noise) every
  model freezes.
- **No memory resets** for free-run-trained models: they lowered energy and the hips.
- Resets every 2.5 s (`stress.drive(reset=150)`) only help models trained without free runs.

## Why: what the stress battery found in v4

v4's evaluation reported averages over calm 60 s runs. The **stress battery**
(`motionlab/stress.py`, `scripts/stress_m1.py`) looks for what averages hide, with the
recordings measured the same way. On v4 (`outputs/stress-v4-combined.json`, 16 test seeds,
5-minute runs):

| | recordings | v4 |
|---|---|---|
| Seeds frozen in the last 30 s of 5 min | 0 | **all**, under every keyword |
| Feet carried with the body (moving frames) | 3.4% | **96%** |
| Hovering (every foot joint over 15 cm up), idle | 0.07% | **43%** |
| Legs crossed | 10% | **32%** |
| Wrist inside the torso | 0.08% | **2.8%** |
| Turning, rad/s | 0.89 | **0.012** |
| Outside the data's joint limits | 0.5% | 3.3% |

No NaN, no mirror bias, recovers from pushes of up to 0.5 rad in under a second, and a held
T-pose start is safe. Run is no faster than walk.

## What v5 changes

**The model (`motionlab/m1.py`)**
1. **A per-tick latent** (`ModelConfig.latent`, VRNN-style after Motion VAEs):
   - in training a posterior that sees the recorded next frame draws z; at run time a learned
     prior draws it from the GRU's memory and the current state;
   - KL with free bits (`LossWeights.kl`, `kl_free_bits`);
   - validation is the ELBO. Prior samples scored against the recording would reward the
     calmest model again.
2. **A velocity input** (`MotionSignal.velocity`, a v5 draft; the `locomotion` encoder): where
   SORA wants her to go, in her heading frame. In training it is the recording's own mean
   velocity over the next 0.5 s (`"velocity_lookahead": 30`). It is optional per model.
3. **Correlated run-time noise** (`MotionPrior.noise_correlation`, an AR(1) process, still
   N(0, 1) per tick). Independent draws every tick made the first live model shake.
4. **The foot lock** (`ModelConfig.foot_lock`, `m1.foot_locked`): while a foot is near the
   floor, the root moves so that foot stays put, so she travels only by stepping. It runs the
   same in training and at run time (the roadmap's M2 foot lock). It comes with:
   - **support choice:** a soft minimum (5 mm) over the contact joints, each counted by its
     higher position over two frames;
   - **flight:** the lock fades out between 3 and 8 cm of lift, and flight is limited to
     0.3 s (`AIR_LIMIT_S`, a timer kept between ticks);
   - **the ground lock** (`ModelConfig.ground_lock`): the lowest foot also rests on the
     floor, so hovering is impossible;
   - **a cap:** the lock's root step is limited to 6 m/s (`LOCK_MAX_SPEED`).
5. **Growing a model** (`MotionPrior.load_deterministic`, `--init-from`): a latent, a new
   encoder or a lock can be added to a checkpoint, which then behaves exactly like it until
   trained.

**The training (`motionlab/m1.py`, `scripts/train_m1.py`)**
6. **Feet in the world** (`foot_height`, `foot_skate`): foot heights with the root included,
   and planted feet held still.
7. **Free runs** (`m1.free_run_losses`, the config's `free_run`): batches that run alone,
   sampling the prior as at run time. They are scored only on what holds at any horizon:
   - `plausibility` (feet on the floor, no wrist in the torso, no hard leg crossing);
   - the joint and hips limits (`free_limits`);
   - an energy band (0.5× to 1.5× the recorded window);
   - `posture` (hips within 8 cm of the window);
   - `jerk_ceiling`;
   - `track` (following the velocity input, measured per tick: `m1.step_velocity`);
   - `free_skate`.
8. **Counterfactual runs** (`m1.counterfactual_losses`, `free_run.counterfactual`): a random
   world velocity, given in her heading frame each tick (`m1.with_command`). They are scored on
   following it (`track`), `facing`, and the recording-free terms.
   - With `free_run.cf_ahead_share`, a share of commands points roughly ahead of her.
9. **Feedback windows** (`free_run.feedback_window`, `feedback_decay`): the gradient runs back
   through k fed-back ticks, damped per tick. Stepping is a multi-tick skill. Steps with a
   non-finite gradient are skipped.
10. **`keep_checkpoints`:** a checkpoint at every validation, for the battery to choose from.
    Validation's 4-second free runs didn't predict long runs.
11. **Half-precision clips in RAM**, so CPU training fits the owner's free memory.

**The stress battery (`motionlab/stress.py`, `scripts/stress_m1.py`, `scripts/stress_table.py`)**
- **Probes:**
  - long runs, under keywords and as SORA drives it (standing at 0 m/s, walking at 1.2 m/s);
  - keyword switches and flicker;
  - pushes;
  - cold starts;
  - mirror bias;
  - steering (walk, run, stand, turn 90° and 180°).
- **Measures:** frozen seeds, energy, carried feet (`carried_share_when_moving`, the honest
  gliding measure), hovering, crossed legs, wrist in torso, toes in the floor, limits, hips,
  root jumps (99th percentile, fastest frame, frames over 5 m/s).
- **Options:** `--rho`, `--temperature`, `--reset`, `--foot-lock`.

## Runs

| Run | What | Result |
|---|---|---|
| `latent-b01` | v4 + latent + foot losses | no better than v4 |
| `feet-only` | v4 + foot losses (ablation) | no better than v4 |
| `latent-vel-free`, `feet-vel-free` | first starts of free2 (a 4 s free run every step) | too slow; stopped at steps 230 and 263 |
| `latent-vel-free2` | latent-b01 + velocity + free runs, energy floor only | ran away (2.4 rad/s); stopped at 1,362 |
| `latent-vel-free3` | free2 + an energy band | **the first that doesn't freeze**; shakes, crouches |
| `feet-vel-free2` | the same without the latent | still freezes |
| `latent-vel-free4` | free3 + run-time noise in free runs, posture, jerk ceiling, limits 4× | a clean body; no steering |
| `latent-vel-free5` | free4 + `track` (0.5 s displacement) | sane, but **root jumps to 10.5 m/s** |
| `latent-vel-free6` | free5 + counterfactual runs | **diverged** (the same flaw) |
| `latent-vel-free7` | free4 + `track` per tick + counterfactual + skate | **steers precisely; glides** |
| `latent-vel-free8`, `free8b` | free7 + `stance` + `facing` | **faces where it goes**; stance brought freezing back |
| `latent-vel-free9` | free7 + the foot lock + facing | carried feet 99.7% → 37% in walks; skimmed its feet to glide "in flight" |
| `latent-vel-free10` | free9 + the 0.3 s flight limit | tracking stalled (1.0 m/s error); paused at 2,016 |
| `latent-vel-free11` | the free10 recipe from free4 | stalled the same; stopped at 2,250 |
| `latent-vel-free12` | free10 + 8-tick feedback windows | NaN at step 85 |
| `latent-vel-free12b` | free12 + damping 0.3 + the skip guard | **first honest stepping** (20% carried feet walking) but slow (0.27 m/s of 1.2) and hovers at idle (93%) |
| `latent-vel-free13` | free12b + commands mostly ahead | best tracking in training, but its final model crosses its legs (85%) and hovers (96%) |
| `latent-vel-free14` | free12b + the ground lock + commands ahead | no hovering or carried feet, but barely moves (0.01 m/s) in a deep crouch (hips 0.44 m) |
| `latent-vel-lock-sup` | free4 + the foot and ground locks, **supervised losses only** | **alive and on planted feet** (energy steady 5 min, carried feet 0.04%); ignores the velocity command |
| `latent-vel-lock-sup2` | lock-sup continued, supervised only | training |
| `latent-vel-lock-free` | lock-sup + free runs (gentle weights, commands ahead, feedback windows) | training |
| `smoke-latent`, `smoke-free` | smoke runs | kept |

Every run's config is `configs/<run>.json`. Its notes say why it was made. Power is logged in
each run folder (`power.csv`, `power-summary.json`).

### Selected battery results (run-time sampling, 16 seeds)

| | recordings | v4 | free5 | free7 | free12b | free14 | lock-sup @ 1,000 |
|---|---|---|---|---|---|---|---|
| Seeds frozen at 5 min | 0 | all | 0 | 0 | all | 94% | **0** |
| Energy over 5 min, rad/s | 1.03 | 0.04 → 0.02 | 0.82 → 0.95 | 0.68 → 0.59 | 0.13 → 0.02 | 0.09 → 0.04 | 0.30 → 0.24 |
| Carried feet (walk) | 3.4% | 96% | – | 99.7% | 51% | 0 | **0.04%** |
| Hovering (idle) | 0.07% | 43% | 1.4% | 0 | 93% | 0 | 0 |
| Legs crossed (walk) | 10% | 32% | 0.01% | 0 | 3% | 1% | 0.6% |
| Outside the limits (walk) | 0.5% | 3.3% | 0.9% | 6% | 2.4% | 8.9% | 9.7% |
| Hips at idle, m | 0.81 | 0.67 | 0.71 | 0.65 | 0.57 | 0.44 | 0.64 |
| Fastest root frame, m/s | 5.1 | 2.3 | 10.6 | 2.0 | 2.9 | 2.3 | 8.4 (before the cap) |
| Steering: walk 1.2 / run 3 m/s | – | – | 0.09 / 0.16 | **1.30 / 3.07** | 0.22 / 1.66 | 0.01 / 0.25 | 0.21 / 0.22 |
| Faces a new heading | – | – | never | never | never | never | never |

## Tools in this folder

| Script | What |
|---|---|
| `scripts/train_m1.py` | training (resume, grow a model, free runs, keep checkpoints) |
| `scripts/stress_m1.py` | the stress battery |
| `scripts/stress_table.py` | stress reports side by side |
| `scripts/evaluate_m1.py` | v4's 60 s evaluation |
| `scripts/sampling_sweep.py` | jitter, energy and limits across prior temperature and noise correlation |
| `scripts/hidden_drift.py` | the GRU's memory over a long run, with or without resets |
| `scripts/lock_check.py` | replays recordings through the foot lock: does it reproduce their travel? (84 to 88% at walking speeds) |

## Run it (from this folder)

```powershell
cd C:\Users\oscar\Desktop\motion-lab\versions\v5
..\..\.venv\Scripts\python scripts\train_m1.py configs\latent-vel-lock-sup.json
..\..\.venv\Scripts\python scripts\stress_m1.py runs\latent-vel-lock-sup\step-1000.pt --rho 0.98
..\..\.venv\Scripts\python scripts\stress_table.py outputs\stress-v4-combined.json outputs\stress-v5-lock-sup-step1000.json
..\..\.venv\Scripts\python -m pytest tests          # 88 tests
```
