# v2: T-pose-relative retargeting (2026-10-25)

Superseded by [v3](../v3/README.md) on 2026-10-25; this folder stays as it was.

It's the same model and training code as
v1. The one change is the **retargeting**: each dataset's own T-pose becomes her T-pose, and only
motion away from it carries over, as in UniVRM's retargeting and SORA's `AvatarFit`. v1 matched
every bone's direction instead, which copied each dataset's skeleton construction into the joint
angles; v1's `README.md` has the numbers.

**Status (2026-10-26): trained and evaluated.** `runs/combined/` holds the best model (step
30,000), `last.pt` the final one (40,000), and `runs/combined-final/` a copy of the final for
evaluation. It's the no-labels control for the keyword runs to follow.

**Data:** the typical joint angles now agree across datasets (`outputs/compare-v2-data.txt` against
`compare-v1-data.txt`):
- spine spread 21° → 5°;
- neck 20° → 3°;
- forearms 27° → 7°;
- feet 25° → 7°;
- collarbones 24° → 16°.

**60 s free rollouts against v1:**

| | v1 best / final | v2 best / final | recordings |
|---|---|---|---|
| Joints beyond limits | 12.2% / 14.3% (collarbones) | 11.8% / 12.1% (toes, hands, forearm) | 0.0% |
| Motion energy over 60 s, rad/s | 0.09→0.03 / 0.09→0.05 | 0.13→0.06 / 0.11→0.08 | 1.12 |
| Error at 0.5 / 1 / 2 s | 11.8° / 16.5° / 19.6° (final) | 11.8° / 15.9° / 18.9° (final) | |

- **What v2 fixed:** the collarbone drift (v1's dataset inconsistency). It also winds down less.
- **What's left:**
  1. **Drift into extreme limb poses over long free runs** (toes 81 to 88° against a 60° limit;
     hands 104° against 90°). Training rolls out at most 32 ticks and has no joint-limit loss.
  2. **It's about 10× calmer than the data**, the average-the-futures problem. That's for
     keyword conditioning, and possibly probabilistic output.
  3. **Occasional hips spikes** to 1.8 to 1.9 m.

**Next:** v3 (a joint-limit loss and longer rollouts), the 17.5M size test, then keywords.

**Found later, in v3 (2026-10-25):** about 1.8% of CMU's frames here have a hand, head or foot
flipped about 180°, and 361 CMU takes are stored twice under two subjects, so some motion is in
two splits. v3 filters both; v2's data and model are unchanged.

## What's here

| Path | What |
|---|---|
| `motionlab/` | the package (as in v1, with v2's retargeting as the default) |
| `scripts/` | data builds, training, evaluation, checks, M0's tools |
| `configs/combined.json` | LAFAN1 + 100STYLE (weight 0.3) + CMU, 40,000 steps: identical to v1's except the data |
| `data/processed/` | v2's converted data (git-ignored) |
| `docs/M1.md` | the M1 pipeline in full (moved from `docs/M1.md`; it also covers v0 and v1's history) |
| `outputs/` | build reports, the cross-dataset angle comparison and evaluations |
| `tests/` | this version's tests (53) |

## Run it (from this folder)

```powershell
cd C:\Users\oscar\Desktop\motion-lab\versions\v2
..\..\.venv\Scripts\python scripts\build_lafan1.py
..\..\.venv\Scripts\python scripts\build_datasets.py style100
..\..\.venv\Scripts\python scripts\build_datasets.py cmu
..\..\.venv\Scripts\python scripts\compare_datasets.py data\processed\lafan1 data\processed\100style data\processed\cmu
..\..\.venv\Scripts\python scripts\train_m1.py configs\combined.json       # about 7.5 h; pause: New-Item runs\combined\STOP
..\..\.venv\Scripts\python scripts\train_m1.py --resume runs\combined
..\..\.venv\Scripts\python scripts\evaluate_m1.py runs\combined\best.pt
..\..\.venv\Scripts\python -m pytest tests
```

The raw downloads are shared, in `motion-lab/data/`; the scripts find them there.
