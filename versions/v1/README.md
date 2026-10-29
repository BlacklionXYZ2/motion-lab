# v1: LAFAN1 + 100STYLE + CMU, bone-direction-matched retargeting (2026-10-24)

The owner's kept model: `runs/combined/` (`best.pt` from step 27,000; `last.pt` and `state.pt`
from step 40,000). Everything that produced it is in this folder.

## What's here

| Path | What |
|---|---|
| `motionlab/` | the package: skeleton, rotations, FK, formats, `MotionSignal`, adapter, export, retargeting (`datasets/`), the model (`m1.py`), evaluation (`m1_eval.py`) |
| `scripts/` | data builds (`build_lafan1.py`, `build_datasets.py style100\|cmu`, `build_amass.py`), training (`train_m1.py`), evaluation (`evaluate_m1.py`), checks (`compare_datasets.py`, `check_amass_sample.py`), M0's tools |
| `configs/combined.json` | the run's config |
| `data/processed/` | the converted data it trained on: LAFAN1 (4.5 h), 100STYLE (22.1 h), CMU (10.8 h); git-ignored |
| `runs/combined/` | the model, its log and its saved state; `runs/combined-final/` is a copy of the step-40,000 model for evaluation; `runs/smoke-combined/` the 300-step test before the run |
| `outputs/` | data build reports and the 60 s evaluations (`eval-combined.json`, `eval-combined-final.json`) |
| `tests/` | this version's tests (53) |

**About the code.** This folder's code was copied from the working code on 2026-10-25, with the
retargeting switched back to v1's behaviour: `match_bone_directions` defaults to `True` in
`datasets/retarget.py` and `datasets/lafan1.py`. `motionlab/assets/lafan1_alignment.json` is
v1's LAFAN1 calibration, and the same file is also kept as `lafan1_alignment_v1_bone_matched.json`.
- **What matches the run exactly:** the model architecture, the training script, the losses and
  the data.
- **What's newer than the run:** the docstrings, and v2's switch itself.

## Run it (from this folder)

```powershell
cd C:\Users\oscar\Desktop\motion-lab\versions\v1
..\..\.venv\Scripts\python -m pytest tests
..\..\.venv\Scripts\python scripts\evaluate_m1.py runs\combined\best.pt     # rewrites outputs\eval-combined.json
..\..\.venv\Scripts\python scripts\train_m1.py --resume runs\combined         # it finished; nothing to resume
```

## Model

The `MotionPrior` (828,643 parameters):
- **Encoders:** a two-layer pose encoder, and a separate two-layer encoder for each `MotionSignal`
  group (intent, conversation, emotion, attention, audio, context).
- **Mixing:** FiLM (the signals scale and shift the pose features), then a fusion network.
- **Core:** a GRU (1 × 256), then a two-layer decoder.
- **Activations:** GELU.
- **Outputs:** the rotation head, root horizontal velocity, absolute hips height, and foot contacts.

Training: 40,000 steps, batch 128, 72-tick windows, and a rollout curriculum of 1 → 8 → 32. Every
signal was neutral, with no labels.

## Retargeting (the flaw)

Each canonical bone was turned, at rest and per frame, to point exactly where the source's bone
points. That copied each dataset's **skeleton construction** into the joint angles. Typical
angles:
- **collarbones:** about 25° (LAFAN1), 18° (100STYLE), 38 to 41° (CMU);
- **upper arms:** 79°, 82°, 118°;
- **spine:** 26°, 7°, 9°.

## Results (60 s free rollouts on the three test splits)

| | best (27k) | final (40k) | recordings |
|---|---|---|---|
| Jitter | 47 m/s³ | 257 m/s³ | 390 m/s³ |
| Planted foot slide | 0.16 m/s | 0.23 m/s | 0.10 m/s |
| Joints beyond limits | 12% | 14% | 0.3% |
| Motion energy over 60 s | 0.09 → 0.03 rad/s | 0.09 → 0.05 rad/s | 1.12 rad/s |

- **What got better:** the hips buzz from v0 is fixed, and foot slide is lower.
- **What went wrong:** the model drifts between the datasets' different "neutral" poses (the
  collarbones reach 48 to 92° against a 40° limit), and it winds down to near stillness.
- **The fix is v2:** retargeting relative to each dataset's own T-pose.

Details: the changelogs
[`2026-10-24T013000+0100_MOTION-M1-MORE-DATA-AND-COMBINED-TRAINING_Opus-5.5.md`](../../changelog/2026-10-24T013000+0100_MOTION-M1-MORE-DATA-AND-COMBINED-TRAINING_Opus-5.5.md)
and [`2026-10-25T090000+0100_MOTION-M1-COMBINED-RUN-RESULT_Opus-5.5.md`](../../changelog/2026-10-25T090000+0100_MOTION-M1-COMBINED-RUN-RESULT_Opus-5.5.md).
