# Motion core v5: a stress battery that finds what v4 hides, and the first fixes

Model name: **Opus 5.5**. Checkpoint **2026-10-30 07:50 UTC**. Status: **built and tested;
the first v5 model is training** (results in a later entry).

The owner (2026-10-30): "keep iteratively improving the motion core by research, training and
testing. Your tests should aim to expose unusual or undesirable behaviour and you should then
look to fix it."

## The GPU and memory on the day

LM Studio's `llama-server` held 13.5 GB of the 16 GB card (a 35B Qwen model), and about 6.5 GB of
system RAM was free. That is the owner's app, so it was left alone. Everything below ran on the
CPU (i5-14600K). This affects how much can be trained, not the method.

## v5 (`versions/v5/`, the new working version)

- A code copy of v4 (no data, runs or outputs). Its `data/processed/` is **hard links** to v4's
  files (3,390 files, about 4.5 GB, unchanged and never written), so v5 is self-contained
  without a second copy on disk. v4's best model is copied to `runs/from-v4/` as v5's start.
- **Half-precision clips in RAM, on the CPU too** (`datasets/m1_data.py`): about 2.3 GB instead
  of 4.5, so a CPU run fits the free memory. Everything that computes with them converts first.

## The stress battery (`motionlab/stress.py`, `scripts/stress_m1.py`)

v4's evaluation reports averages over calm 60 s runs. The battery looks for what averages hide,
and measures the recordings the same way for reference:

- **long runs** (5 minutes, under unknown, idle, walk and run): freezing, spinning, wandering;
- **gliding:** the root moving over 0.3 m/s while no foot joint moves faster than 1.2× the root;
- **floor:** toes below the recordings' lowest 0.5%, or every foot joint more than 15 cm up;
- **body:** legs crossed (left ankle more than 2 cm right of the right one), a wrist within
  8 cm of the spine, the head below the hips;
- **switches:** the keyword changes every 4 s (idle, walk, run, walk, sit, idle, wave, run,
  idle): the speed reached, and the jolt at each switch against the recordings' 99th percentile;
- **flicker:** walk and idle alternating every tick, run and sit every 3 ticks;
- **pushes:** every joint knocked 0.2, 0.5 and 1 rad off at once, as the physics body may do;
- **cold starts:** a held T-pose (her player's start) and a frozen copy of one frame;
- **mirror bias:** a seed and its mirror image should turn by opposite amounts.

### v4 under the battery (`versions/v5/outputs/stress-v4-combined.json`, 16 test seeds)

| | v4 | recordings |
|---|---|---|
| Seeds frozen (energy under 0.1 rad/s) in the last 30 s of 5 min | **100%**, every keyword | 0% |
| Energy over 5 min, rad/s | 0.04 → 0.02 | about 1 |
| Gliding, share of moving frames | **82 to 92%** | 30% |
| Every foot joint over 15 cm up (hovering in a tuck) | **33 to 40%** (unknown, idle); 13 to 19% (walk, run) | 0.07% |
| Legs crossed | **30 to 38%** | 10% |
| Wrist inside the torso | **3%** | 0.08% |
| Turning, rad/s | **0.011** (she only goes straight) | 0.89 |
| Outside the data's joint limits, 5 min | 3.6 to 4.8% | 0.5% |
| Run against walk in the switch test | run **slower** than walk (0.17 against 0.24 m/s) | |
| Flicker every tick | 4.3× the jerk of a steady keyword; stable | |

**Fine:** no NaN anywhere; pushes of 0.2 and 0.5 rad recover within 0.8 s (1 rad: 6% of seeds
don't recover in 5 s); cold starts stay upright with no floor problems; no mirror bias
(asymmetry 0.001 rad/s); keyword switches cause no pops (v4 barely moves, so nothing pops).

**Reading:**
- The **freezing** is the averaging problem v4's README describes: a deterministic model trained
  on the average next pose drifts to the calmest pose.
- **Gliding and hovering have a separate cause.** v4's losses score the legs only relative to the
  root, and the root (velocity and hips height) on its own. Nothing ties the feet to the floor,
  so in long runs the root slides and the legs fold up under hips that keep their height.

## Fixes in v5 (`motionlab/m1.py`)

1. **A per-tick latent** (`ModelConfig.latent`, VRNN-style; Chung et al. 2015, and Ling et al.
   2020's Motion VAE, which uses a fixed prior):
   - in training, a posterior that sees the recorded next frame draws a 32-number z;
   - at run time a learned prior draws it from the GRU's memory and the current state;
   - z enters the GRU's input and the decoder, so the decoder needn't average the possible next
     frames into one calm frame.
   - KL against the prior with free bits (`LossWeights.kl`, `kl_free_bits`).
   - A latent model can start from a v4 checkpoint (`load_deterministic`) and then, with zero
     noise, behaves exactly like it (tested).
   - The runtime still calls a pure function; its extra input is the noise.
2. **Feet in the world** (`LossWeights.foot_height`, `foot_skate`): foot-joint heights with the
   root included, against the recording, and a foot planted in the recording mustn't move over
   the ground.
3. `train_m1.py` reads `loss_weights` from the config and starts a latent model from a v4
   checkpoint. A latent model's validation is its ELBO (the posterior's reconstruction plus the
   weighted KL). Scoring prior samples against the recording would reward the calmest
   checkpoint, which is the averaging bias again.

Tests: 71 pass (v4's 66, plus 5 for the latent, the foot terms and the battery's driver).

## Training (running)

`configs/latent-b01.json`: v4's model plus the latent and feet, fine-tuned for 6,000 steps on the
CPU with batch 64 and rollouts to 64 ticks; KL weight 0.1 with 0.02 free nats per dimension. A
30-step smoke run took 71 s. Power is logged (`runs/latent-b01/power.csv`).

Nothing committed.
