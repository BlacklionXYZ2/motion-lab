# Motion core M1: the training pipeline, first version

Model name: **Opus 5.5**. Checkpoint **2026-10-23 17:15:00 +01:00**, Europe/London. Status:
**the pipeline runs end to end; component-tested; no real training run**.

The owner: "Could you start setting up the training pipeline for the motion core? I'll validate
the physics model later."

All the code is in `~/Desktop/motion-lab` (uncommitted), described in its `docs/M1.md`.

## Changes (motion-lab)

- **Data.** LAFAN1 is downloaded to `data/lafan1/` (git-ignored; `data/README.md` notes what's
  there and its licence: CC BY-NC-ND 4.0, so retargeted copies are never shared).
  - `datasets/bvh.py`: a BVH reader.
  - `datasets/lafan1.py`: retargeting onto the canonical skeleton and resampling to 60 Hz.
    - The calibration is in `assets/lafan1_alignment.json`: the hips and bone directions from
      the offsets, twist from the takes' first-frame T-poses.
    - The root is scaled by standing hips height (0.973).
  - `datasets/contacts.py`: foot contacts.
  - `datasets/m1_data.py`: splits by performer (1 to 3 train, 4 validation, 5 test), window
    sampling, mirror and facing augmentation, and normalisation.
  - `scripts/build_lafan1.py` builds and checks all 77 takes.
- **Model and training.**
  - `m1.py`: a GRU motion prior (533k parameters) in M0's export shape. It has a rotation and an
    ω head, and predicts root velocity, hips height and contacts.
  - It also holds the rollout and the losses.
  - `scripts/train_m1.py` runs the 1 → 8 → 32 rollout curriculum from
    `configs/m1/baseline.json`.
- **Evaluation.** `m1_eval.py` and `scripts/evaluate_m1.py` run 60 s rollouts against the
  recordings.
- **Tests.** `tests/test_m1_pipeline.py` has 9 tests: BVH, retargeting helpers, the batched
  states against M0's definition, augmentation, contacts, rollout gradients, and ONNX export
  parity.

## Problems found and fixed while building

- **Retarget scale.** Leg-length scaling put her hips 12 cm too low; it now scales by standing
  hips height.
- **Gradient explosion.** Backpropagating through fed-back outputs blew the gradient up to 4×10⁹
  at 32-tick rollouts. It's now off by default.
- **Sinking hips.** An integrated vertical root velocity sank her 14 m in 60 s; hips height is
  now predicted directly.

## Verification

- **LAFAN1 build (77 takes):**
  - retargeted bones are within 0.0003° of the source;
  - planted ankles sit at 0.09 to 0.10 m on flat ground (rest: 0.092 m);
  - standing hips are at about 0.85 m.
- **Tests and lint:** motion-lab pytest 44 passed (35 before, plus 9 new); ruff clean.
- **Smoke training, 600 steps on the RX 9070 XT (about 0.3 s per step):**
  - validation at a 32-tick horizon: 5.3° per joint and 3.3 cm;
  - in 60 s rollouts it is stable and upright but too still (0.29 rad/s against the recordings'
    1.12) and jittery. That's expected this early.
- **Not run:** the full baseline (about 1.6 hours of GPU time), and the head and size comparisons.

## Open (owner)

- **Intent labels:** MotionSignal v1 lacks run, fall and get-up.
- **AMASS and BABEL registration.**
- **The viewer.**

Nothing committed.
