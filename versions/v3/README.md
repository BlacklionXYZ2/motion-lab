# v3: clean data, limit losses, longer rollouts (2026-10-25)

Superseded by [v4](../v4/README.md) on 2026-10-26; this folder stays as it was.

It's v2's model, data and settings, with
three changes aimed at v2's drift into impossible limb poses in long free runs
([docs/M1.md](docs/M1.md) has the detail):

1. **Quality filter:** broken CMU frames (a hand, head or foot flipped about 180°) are cut out,
   and 361 CMU takes stored twice under two subjects are dropped to one copy. 4.4% of frames go.
2. **Limit losses:** every joint's range and the hips' height range are measured from the
   cleaned training data (`motionlab/assets/joint_limits.json`). Training is penalised for going
   past them.
3. **Longer rollouts:** the model practises running on its own for up to 128 ticks (about 2 s)
   rather than 32.

**Status (2026-10-26): trained and evaluated.** `runs/combined/` holds the best model (step
39,000) and `last.pt` (40,000); `runs/combined-final/` is a copy of the final for evaluation.
40,000 steps took 9.4 h and finished cleanly.

**60 s free rollouts, all on v3's cleaned test set (16 seeds):**

| | v2 final | v3 best | v3 final | recordings |
|---|---|---|---|---|
| Outside the measured limits | 21.4% | **3.6%** | 3.7% | 0.4% |
| Beyond the old placeholders | 16.6% | **0.6%** | 0.7% | 0.0% |
| Planted foot slide, m/s | 0.21 | **0.11** | 0.11 | 0.08 |
| Error at 0.5 / 1 / 2 s, degrees | 9.6 / 11.3 / 15.3 | 9.6 / 10.5 / **14.0** | 9.6 / 10.5 / 14.0 | |
| Energy over 60 s, rad/s | 0.10 → 0.06 | **0.05 → 0.02** | 0.05 → 0.02 | 1.04 |
| Hips height mean / lowest / highest, m | 0.75 / 0.40 / 1.64 | 0.93 / 0.60 / **2.27** | 0.92 / 0.60 / 2.27 | 0.85 / 0.09 / 1.56 |

- **Fixed:** the drift into impossible limb poses. Forearms, hands and toes are back inside
  the data's ranges. The worst remaining joint is the left collarbone (23 to 30% of frames just
  outside), then the upper chest (14 to 17%). Foot slide halved and long-horizon error fell.
- **Worse: it moves even less.** Energy fell from v2's 0.10 → 0.06 to 0.05 → 0.02 rad/s, about
  50× calmer than the recordings. The penalty and the longer runs both reward standing still
  when nothing says what to do. It's the averaging problem, now stronger: keyword conditioning
  (and possibly probabilistic output) is the fix, not more training of this kind.
- **Hips floating, in 2 of 16 seeds:** one hovers at about 1.4 m for the whole minute; one starts
  at 1.24 m (standing on something) and climbs to 2.27 m. The other 14 stay between 0.59 and
  0.99 m. The hips height is absolute above the floor, and the model can't know whether it's
  standing on a box, so a raised start stays raised. Next: a height relative to the feet, or
  floor-contact information.

**The starting point:** v2's final model, measured by this version's evaluator on the cleaned test
set (`outputs/eval-v2-final-on-v3-test.json`):
- joints outside the measured limits: 21% of joint-frames (forearms 51 to 75%, hands 34 to 56%,
  toes 28 to 57%); the recordings: 0.4%;
- beyond the old placeholder limits: 16.6% (on v2's own test set, 12.1%);
- energy over 60 s: 0.10 → 0.06 rad/s (recordings 1.04).

**What v3 should show** (a prediction to test): far fewer joints outside the limits, no hips
spikes and smaller long-horizon error. Energy may rise a little but will stay well below the
data's until keyword conditioning.

## What's here

| Path | What |
|---|---|
| `motionlab/` | the package: v2's, plus `datasets/quality.py`, the limit losses in `m1.py` and the per-joint report in `m1_eval.py` |
| `motionlab/assets/joint_limits.json` | the measured limits (`scripts/joint_limits.py`) |
| `scripts/` | as v2, plus `joint_limits.py`; `evaluate_m1.py --clean` tests an earlier model on v3's test set |
| `configs/combined.json` | v3's run: v2's settings, `clean_data`, the five-phase curriculum |
| `configs/combined-17m.json` | the 17.5M size test on v3's setup (not run) |
| `data/processed/` | a copy of v2's converted data, unchanged (git-ignored); the filter runs as it loads |
| `docs/M1.md` | the M1 pipeline in full, with v3's changes at the top |
| `outputs/` | the filter's report and evaluations |
| `runs/smoke-v3/` | a 40-step smoke run (checks the phases, validation and saving) |
| `tests/` | this version's tests (59) |

## Run it (from this folder)

```powershell
cd C:\Users\oscar\Desktop\motion-lab\versions\v3
..\..\.venv\Scripts\python scripts\joint_limits.py                         # remeasure the limits (already done)
..\..\.venv\Scripts\python scripts\train_m1.py configs\combined.json       # about 9.5 h; pause: New-Item runs\combined\STOP
..\..\.venv\Scripts\python scripts\train_m1.py --resume runs\combined
..\..\.venv\Scripts\python scripts\evaluate_m1.py runs\combined\best.pt
..\..\.venv\Scripts\python -m pytest tests
```

The raw downloads are shared, in `motion-lab/data/`. The data builds are v2's, unchanged
(`scripts/build_*.py`); v3's own data copy came from v2's folder.
