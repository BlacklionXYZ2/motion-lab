# v4: keywords and the hips above the support (from 2026-10-26; the working version)

The current version: new work happens in this folder. It's v3 (clean data, limit losses,
rollouts to 128 ticks) with two changes aimed at what v3 left:

1. **Keywords (MotionSignal v2, a draft):** the model is told what she's doing and how she
   feels, from the datasets' own labels. v3 was given nothing, so its safest guess was to barely
   move.
2. **Hips height above what she stands on** (the support), not above the room's floor. v3 couldn't
   see the floor, so starting on a box left its hips floating in 2 of 16 test runs.

**Status (2026-10-26): trained and evaluated.** The owner approved the vocabularies for this run
("Start the training"); MotionSignal v2 is still a draft as far as SORA goes.
- **Run:** `runs/combined/`. Best model at step 38,000; 40,000 steps took 9.2 h and finished
  cleanly.
- **Power** (`runs/combined/power-summary.json`, 8.2 h logged): CPU 39.5 W and GPU 38.4 W on
  average, about 1.3 kWh at the wall for the whole run (estimate).

**60 s free rollouts, keywords unknown** (v3's numbers on the same cleaned test set):

| | v3 | v4 | recordings |
|---|---|---|---|
| Outside the measured limits | 3.6% | **2.1%** | 0.4% |
| Hips above the support: mean / highest, m | (floor) 0.93 / 2.27 | **0.83 / 0.96** | 0.83 / 1.23 |
| Error at 0.5 / 1 / 2 s, degrees | 9.6 / 10.5 / 14.0 | **9.5 / 10.0 / 13.4** | |
| Planted foot slide, m/s | 0.11 | 0.10 | 0.08 |
| Energy over 60 s, rad/s | 0.05 → 0.02 | 0.05 → 0.04 | 1.04 |

**30 s under each keyword** (same seeds):

| Condition | Root speed, m/s | Energy, rad/s |
|---|---|---|
| unknown | 0.11 | 0.05 → 0.02 |
| idle | 0.09 | 0.06 → 0.03 |
| walk | 0.20 | 0.05 → 0.03 |
| run | 0.22 | 0.05 → 0.02 |
| sit / wave / look | 0.09 / 0.15 / 0.11 | about 0.05 → 0.03 |
| walk + cheerful, sad, frustrated or confident (high) | 0.31 to 0.33 | up to 0.07 → 0.06 |

The recordings walk at 0.56 m/s and run at 1.03 m/s on average.

- **Fixed:** the hips. They never float now (highest 0.96 m above the support, against v3's
  2.27 m). The limits are better still, and the short-horizon error is the lowest yet.
- **The keywords steer, but weakly.** Walk and run roughly double her speed over unknown or
  idle, and an emotion adds more. But she shuffles at a third of real walking speed, run is no
  faster than walk, and every condition still winds down towards stillness.
- **Reading:** the keywords help at the start, but the averaging problem remains. A
  deterministic model trained on the average next pose drifts to the calmest motion when it runs
  on its own. The next step is probabilistic output: the model samples one plausible next
  pose instead of predicting the mean, as in Motion VAEs (Ling et al. 2020), which hold lively,
  steerable motion through long autoregressive runs.

## The keywords (`motionlab/motion_signal.py`, `specs/motion_signal.v2.schema.json`)

- **Intent:** v1's idle, wave, sit, lean, climb, stand, walk, point, look and present, plus run,
  fall and get_up, or unknown. In v1, unknown and idle were the same value; now they differ.
- **Emotion:** neutral, calm, warm, cheerful, excited, confident, focused, surprised, concerned,
  embarrassed, frustrated or sad, or unknown. Each has a fixed valence/arousal anchor, so emotions
  with little data start near similar ones.
- **Intensity** of the emotion: low, medium or high, or unknown.
- Conversation, attention, audio and context are unchanged.

## Where the labels come from (`motionlab/datasets/labels.py`, `scripts/add_labels.py`)

Minutes of training data per keyword (`outputs/labels-v4.txt`):

| Intent | Minutes | From |
|---|---|---|
| walk | 702 | LAFAN1 walk takes, 100STYLE walking gaits, BABEL |
| run | 380 | LAFAN1 run and sprint, 100STYLE running gaits, BABEL |
| idle | 167 | 100STYLE idle, BABEL "stand", and any walk or run slower than 0.15 m/s |
| look | 16 | BABEL |
| climb | 9 | BABEL (stairs and inclines) |
| sit, lean, wave, stand | 2 to 4 each | BABEL |
| fall, get_up | 2 each | LAFAN1's falls, split by the hips' height |
| point | 0.5 | BABEL |
| present | 0 | none |
| unknown | 958 | mixed takes (LAFAN1's obstacles, dance, fights), 100STYLE transitions, unsegmented CMU |

- **Emotion:** only 100STYLE's acted styles: Depressed → sad (15 min), Neutral → neutral (13),
  Proud → confident (12), Elated → cheerful (10) and Angry → frustrated (9). All are walking or
  running, and all but neutral are at high intensity. The other seven emotions have no data yet;
  they wait for BEAT or similar (roadmap M3 and M5).
- **Rare intents** are sampled more often (`rare_intent_boost` 20: a clip made entirely of them is
  drawn 21× as often).
- **Labels are dropped** to unknown in 30% of windows, per group, so unknown stays meaningful.

## The support height

Per frame: the height of the lowest planted foot or toe joint above its rest height. It holds
while she's in the air, and is filtered over 1 s (median) and 0.25 s (mean) so a slow-swinging
foot doesn't count. The hips' training range above the support is 0.0 to 1.06 m, against 0.06 to
1.96 m above the floor in v3, so the hips limit now catches floating.

## Evaluation

As v3, plus every seed rerun for 30 s under each keyword condition (unknown, idle, walk, run,
sit, wave, look, and walking with four emotions). It reports root speed, energy, limits and hips
per condition, plus the recordings' speed per intent for reference. It's for v4 models only:
earlier models have the v1 signal.

## What's here

| Path | What |
|---|---|
| `motionlab/` | v3's package plus `motion_signal.py` v2 and `datasets/labels.py`; the sampler carries labels |
| `scripts/add_labels.py` | adds labels and support heights to this version's data (done) |
| `configs/combined.json` | v4's run: v3's settings, `support_frame`, `rare_intent_boost` |
| `data/processed/` | a copy of v3's data with labels and support added (git-ignored) |
| `outputs/` | the label coverage, the quality report, evaluations |
| `runs/smoke-v4/` | a 40-step smoke run |
| `tests/` | 66 tests |

## Run it (from this folder)

```powershell
cd C:\Users\oscar\Desktop\motion-lab\versions\v4
..\..\.venv\Scripts\python scripts\add_labels.py                           # once (done)
..\..\.venv\Scripts\python scripts\joint_limits.py                         # once (done)
..\..\.venv\Scripts\python scripts\train_m1.py configs\combined.json       # about 10 h; pause: New-Item runs\combined\STOP
..\..\.venv\Scripts\python scripts\train_m1.py --resume runs\combined
..\..\.venv\Scripts\python scripts\evaluate_m1.py runs\combined\best.pt
..\..\.venv\Scripts\python -m pytest tests
```
